"""Warm ONE-expert placement microbenchmark, never language-model TPS.

Same officially sampled FP8 expert decoded to FP16 for all three policies:
resident GPU floats, CPU predecoded float file + H2D, CPU FP8 file/decode + H2D.
No pruning/model routing. Tiny warm file working set; not cold185GB streaming.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import subprocess
import time

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from qualify_flash_next_fp8_expert import original_decoder, PREFIX
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_expert_cache import CachedExpertStore
from venturi.models.flash_next_cuda_fp8 import CUDABlockFP8ExpertStore
from venturi.ops.flash_next_fp8 import NativeBlockFP8Decoder
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_moe import MoEGeometry, FusedFloatingExpertStore


def mlp(x, weights):
    gate, up = F.linear(x, weights[0]).chunk(2, -1)
    return F.linear(F.silu(gate)*up, weights[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trace-only', action='store_true')
    p.add_argument('--native-library')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device, dtype = torch.device('cuda:0'), torch.float16
    props = torch.cuda.get_device_properties(device)
    if props.major != 7 or props.minor != 0:
        raise ValueError('Pinned V100/SM70 placement comparison only')
    torch.cuda.set_per_process_memory_fraction(1024*2**20/props.total_memory, device)
    g = MoEGeometry(2560, 1, 1, 640, 640)
    cp = inspect_checkpoint(a.checkpoint)
    coded = BlockFP8ExpertStore(cp, g, PREFIX)
    source = load_file(str(Path(a.checkpoint)/'one-expert-codec.safetensors'))
    decode, source_sha = original_decoder(a.reference)
    expected = [decode(source[f'{PREFIX}.0.{name}.weight'], source[f'{PREFIX}.0.{name}.weight_scale_inv'],
                       [128, 128], dtype) for name in ('gate_proj', 'up_proj', 'down_proj')]
    baseline = out/'predecoded-baseline'
    baseline.mkdir()
    save_file({'expert.gate_up_proj': torch.cat(expected[:2]).unsqueeze(0),
               'expert.down_proj': expected[2].unsqueeze(0)}, baseline/'model.safetensors')
    floating = FusedFloatingExpertStore(inspect_checkpoint(baseline), g, 'expert')
    resident = (torch.cat(expected[:2]).to(device), expected[2].to(device))
    readers = {'gpu-resident': None, 'cpu-predecoded-file': floating, 'cpu-fp8-decode': coded}
    vectorized = BlockFP8ExpertStore(cp, g, PREFIX, decode_strategy='vectorized')
    size = 3*g.hidden_size*g.intermediate_size*2
    ram_cache = CachedExpertStore(vectorized, max_resident_bytes=size)
    vram_cache = CachedExpertStore(vectorized, max_resident_bytes=size)
    readers.update({'cpu-fp8-vectorized': vectorized, 'cpu-fp8-ram-cache': ram_cache, 'gpu-fp8-vram-cache': vram_cache})
    native_cache = None
    if a.native_library:
        native_store = CUDABlockFP8ExpertStore(cp, g, PREFIX, decoder=NativeBlockFP8Decoder(a.native_library))
        native_cache = CachedExpertStore(native_store, max_resident_bytes=size)
        readers.update({'gpu-fp8-native-decode': native_store, 'gpu-fp8-native-vram-cache': native_cache})
    direct_gpu = {'gpu-fp8-vram-cache', 'gpu-fp8-native-decode', 'gpu-fp8-native-vram-cache'}
    records, cold_fills = [], []
    repetitions, iterations, warmup = (1, 8, 2) if a.trace_only else (3, 32, 4)
    monitor = None
    with (out/'gpu-samples.csv').open('w') as gpu, (out/'gpu-samples.stderr').open('w') as errors:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu',
                                    '--format=csv,noheader,nounits', '--loop-ms=250'], stdout=gpu, stderr=errors)
        try:
            with torch.inference_mode():
                for batch in (1, 8):
                    x = (torch.randn(batch, 2560, generator=torch.Generator().manual_seed(313))*.1).to(device=device, dtype=dtype)
                    wanted = mlp(x, resident)
                    # Correctness BEFORE timing; same decoded weights and GEMM
                    # shapes. Compare exact FP16 outputs, not BF16 model quality.
                    for name, reader in readers.items():
                        if reader is None:
                            continue
                        where = device if name in direct_gpu else 'cpu'
                        start, cpu = time.perf_counter(), time.process_time()
                        fetched = reader.read(0, device=where, dtype=dtype)
                        torch.cuda.synchronize()
                        cold_fills.append(dict(policy=name, batch=batch, read_ready_ms=(time.perf_counter()-start)*1000,
                            parent_cpu_ms=(time.process_time()-cpu)*1000, scope='Read/cache fill only; CPU paths exclude H2D'))
                        weights = tuple(w.to(device=device) for w in fetched)
                        for got, ref in zip(weights, resident):
                            torch.testing.assert_close(got, ref, atol=0, rtol=0)
                        torch.testing.assert_close(mlp(x, weights), wanted, atol=0, rtol=0)
                    for repeat in range(repetitions):
                        names = list(readers)
                        names = names[repeat%len(names):]+names[:repeat%len(names)]  # paired balanced order
                        for name in names:
                            reader = readers[name]
                            for _ in range(warmup):
                                if reader is None:
                                    weights = resident
                                elif name in direct_gpu:
                                    weights = reader.read(0, device=device, dtype=dtype)
                                else:
                                    weights = tuple(w.to(device=device) for w in reader.read(0, device='cpu', dtype=dtype))
                                mlp(x, weights)
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            read_ms, copy_ms, compute_ms = [], [], []
                            wall, cpu = time.perf_counter(), time.process_time()
                            for _ in range(iterations):
                                nvtx = torch.cuda.nvtx.range(name) if a.trace_only else nullcontext()
                                with nvtx:
                                    begin = time.perf_counter()
                                    if reader is None:
                                        direct, host = resident, None
                                    elif name in direct_gpu:
                                        direct, host = reader.read(0, device=device, dtype=dtype), None
                                    else:
                                        direct, host = None, reader.read(0, device='cpu', dtype=dtype)
                                    read_ms.append((time.perf_counter()-begin)*1000)
                                    c0, c1, k0, k1 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                                    c0.record()
                                    weights = direct if host is None else tuple(w.to(device=device) for w in host)
                                    c1.record()
                                    k0.record()
                                    result = mlp(x, weights)
                                    k1.record()
                                    k1.synchronize()
                                    copy_ms.append(c0.elapsed_time(c1))
                                    compute_ms.append(k0.elapsed_time(k1))
                            elapsed, process = time.perf_counter()-wall, time.process_time()-cpu
                            torch.testing.assert_close(result, wanted, atol=0, rtol=0)
                            records.append(dict(policy=name, batch=batch, repeat=repeat, iterations=iterations, warmup=warmup,
                                wall_seconds=elapsed, parent_cpu_seconds=process,
                                mean_wall_ms=elapsed*1000/iterations, expert_input_rows_per_second=batch*iterations/elapsed,
                                cpu_read_decode_ms=read_ms, copy_phase_stream_ms=copy_ms, compute_phase_stream_ms=compute_ms,
                                p50_cpu_read_decode_ms=statistics.median(read_ms), p50_copy_phase_stream_ms=statistics.median(copy_ms),
                                p50_compute_phase_stream_ms=statistics.median(compute_ms),
                                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), exact_output=True,
                                timing='Parent clocks and CUDA events, no profiler/counter replay' if not a.trace_only else 'PROFILE ONLY, do not score throughput'))
                            # Preserve completed paired records even if an
                            # external timeout interrupts a later policy.
                            (out/'progress.json').write_text(json.dumps(dict(completed=False, profile_only=a.trace_only,
                                cases=records, read_fills_outside_timing=cold_fills), indent=2)+'\n')
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait(timeout=10)
    report = dict(scope='ONE sampled official FP8 expert, same decoded FP16 weights/MLP, warm file placement microbenchmark',
        completed=True, profile_only=a.trace_only, cases=records, read_fills_outside_timing=cold_fills,
        cache_stats=dict(ram=ram_cache.stats, vram=vram_cache.stats,
                         native_vram=None if native_cache is None else native_cache.stats),
        cache_scope='Hot ONE-expert working set; not real-model routing/cache hit rates or OOM admission', dtype=str(dtype), gpu=props.name, torch=str(torch.__version__),
        sglang_codec_sha256=source_sha, model_tps_measured=False, model_quality_measured=False,
        whole_model_tested=False, routing_tested=False, cold_full_checkpoint_io_tested=False,
        production='Running co-resident service, not an isolated full-model benchmark',
        event_scope='GPU stream phase elapsed includes host launch gaps; NOT raw copy-engine bandwidth/kernel-only time',
        fetch_scope='Direct GPU policies include H2D/decode/error-sync in fetch wall time, NOT the later no-op copy phase',
        gpu_counters='Achieved occupancy/SM/DRAM counters not collected; no kernel efficiency claim',
        memory_scope='Torch peak excludes CUDA contexts/libraries/production; raw total-device samples separate')
    (out/'scores.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}, indent=2))


if __name__ == '__main__':
    main()
