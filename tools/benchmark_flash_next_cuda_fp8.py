"""Resident-input SM70 decoder kernel comparison; not PCIe/model throughput."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from safetensors.torch import load_file
import torch

from qualify_flash_next_fp8_expert import PREFIX
from venturi.ops.flash_next_fp8 import NativeBlockFP8Decoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--linear', required=True)
    p.add_argument('--tiled', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile-one', choices=['linear','tiled'])
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(0).total_memory)
    sources = load_file(str(Path(a.checkpoint)/'one-expert-codec.safetensors'))
    operators = dict(linear=NativeBlockFP8Decoder(a.linear), tiled=NativeBlockFP8Decoder(a.tiled))
    records = []
    with torch.inference_mode():
        for projection in (('gate_proj',) if a.profile_one else ('gate_proj', 'down_proj')):
            base = f'{PREFIX}.0.{projection}'
            codes = sources[base+'.weight'].view(torch.uint8).to('cuda:0')
            scales = sources[base+'.weight_scale_inv'].view(torch.int16).to('cuda:0')
            expected = operators['linear'].decode(codes, scales, dtype=torch.float16) if not a.profile_one else None
            for repeat in range(1 if a.profile_one else 3):
                order = (a.profile_one,) if a.profile_one else (('linear','tiled') if repeat%2 == 0 else ('tiled','linear'))
                for name in order:
                    operator = operators[name]
                    errors = torch.zeros((), dtype=torch.int32, device='cuda:0')
                    result = torch.empty(codes.shape, dtype=torch.float16, device='cuda:0')
                    stream = torch.cuda.current_stream()
                    iterations = 1 if a.profile_one else 200
                    if not a.profile_one:
                        torch.testing.assert_close(operator.decode(codes, scales, dtype=torch.float16), expected, atol=0, rtol=0)
                        for _ in range(10):
                            status = operator._decode(codes.data_ptr(), scales.data_ptr(), result.data_ptr(), errors.data_ptr(),
                                *codes.shape, 1, stream.cuda_stream)
                            assert status == 0
                        torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall, cpu = time.perf_counter(), time.process_time()
                    start.record()
                    for _ in range(iterations):
                        status = operator._decode(codes.data_ptr(), scales.data_ptr(), result.data_ptr(), errors.data_ptr(),
                            *codes.shape, 1, stream.cuda_stream)
                        if status:
                            raise RuntimeError(f'Native decoder failed: {status}')
                    end.record()
                    end.synchronize()
                    elapsed, process = time.perf_counter()-wall, time.process_time()-cpu
                    assert errors.item() == 0
                    if expected is not None:
                        torch.testing.assert_close(result, expected, atol=0, rtol=0)
                    records.append(dict(projection=projection, shape=list(codes.shape), variant=name, repeat=repeat,
                        iterations=iterations, stream_interval_ms=start.elapsed_time(end), wall_seconds=elapsed,
                        parent_cpu_seconds=process, mean_stream_us=start.elapsed_time(end)*1000/iterations,
                        mean_wall_us=elapsed*1e6/iterations, exact_output=True))
                    (out/'progress.json').write_text(json.dumps(dict(completed=False, cases=records), indent=2)+'\n')
    (out/'gpu-after.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader,nounits'], text=True))
    report = dict(completed=True, cases=records, profile_only=bool(a.profile_one), dtype='FP16',
        scope='Resident-code standalone weight decode, repeated warm working set; no H2D/MLP/whole model',
        event_scope='Stream span includes host launch gaps; hardware profiler durations reported separately',
        model_tps_measured=False, model_quality_measured=False)
    (out/'scores.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
