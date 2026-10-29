"""Prepared PLE reader ablation: synthetic warm CPU gather + blocking H2D only.

Four controls retain the same rows, precision and one pageable H2D per step.
No full-model inference, cold 102GB access, pinned overlap or scheduler claim.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import resource
import subprocess
from time import perf_counter

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from benchmark_flash_next_ngram import make_fixture
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_ngram import (
    FP8ShardedNGramTable, NGramRequest, PreparedNGramTable, ShardedNGramTable,
)


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cpu_seconds=r.ru_utime + r.ru_stime, minor_faults=r.ru_minflt,
                major_faults=r.ru_majflt)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encode_fp8(root, config):
    """Synthetic official PLE encoding/scale contract, not official weight data."""
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    for path in sorted(root.glob("*.safetensors")):
        tensors = load_file(path)
        for name in tensors:
            if ".ngram_embedding.shard_" in name:
                tensors[name] = (tensors[name] / scale).to(torch.float8_e4m3fn)
        if any(n.endswith(".layer_multipliers") for n in tensors):
            prefix = next(n.removesuffix(".layer_multipliers") for n in tensors
                          if n.endswith(".layer_multipliers"))
            tensors[prefix + ".ngram_embedding.weight_scale"] = scale
        save_file(tensors, path, metadata={"scope": "synthetic official FP8 PLE contract"})
    mapping, total = {}, 0
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                mapping[name] = path.name
                total += tensor.numel() * tensor.element_size()
    (root / "model.safetensors.index.json").write_text(json.dumps(
        dict(weight_map=mapping, metadata=dict(total_size=total))))
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoding", choices=["bf16", "fp8"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.steps <= 256 or not 1 <= a.repeats <= 10 or any(
        not 1 <= b <= 128 for b in a.batch_sizes
    ):
        p.error("Bounded work sizes required")
    torch.set_num_threads(1)
    root, output = Path(a.fixture), Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    config, source_bytes = make_fixture(root, json.loads(Path(a.config).read_text()))
    if a.encoding == "fp8":
        source_bytes = encode_fp8(root, config)
    checkpoint = inspect_checkpoint(root)
    cls = FP8ShardedNGramTable if a.encoding == "fp8" else ShardedNGramTable
    store = cls(checkpoint, config, max_tokens=max(a.batch_sizes))
    prep_cpu, start = usage(), perf_counter()
    prepared = PreparedNGramTable(store, max_open_files=32).prepare()
    preparation = dict(wall_seconds=perf_counter() - start,
                       cpu_seconds=usage()["cpu_seconds"] - prep_cpu["cpu_seconds"],
                       unique_files=len({str(s.file) for s in store.parts}), slices=len(store.parts))
    chunks = []
    for spec in store.parts:
        with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
            chunks.append(reader.get_tensor(spec.name).to(torch.bfloat16))
    resident = torch.cat(chunks)
    if a.encoding == "fp8":
        resident *= store.weight_scale
    resident = resident.to(a.device)
    del chunks
    telemetry = (output / "nvml.csv").open("w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
        "--format=csv", "--loop-ms=200",
    ], stdout=telemetry, stderr=subprocess.DEVNULL)
    report = dict(scope="synthetic warm PLE CPU gather + one blocking pageable H2D per step; NOT model TPS",
                  encoding=a.encoding, output_dtype="BF16", profile_only=a.profile_only,
                  real_model_tested=False, fixture_total_bytes=source_bytes,
                  preparation=preparation, batch_results=[], torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(a.device), cpu_threads=torch.get_num_threads(),
                  telemetry_scope="Whole validation/warmup/scored process; use explicit score intervals; short cells may have zero samples",
                  counter_scope="Copy-only GPU path; no compute kernel occupancy/DRAM counters collected",
                  fixture_scope="16 heads x 160; vocab base65521; multipliers7/13/19; deterministic synthetic values; synchronized EOS reuse; warm small table")
    try:
        for batch in a.batch_sizes:
            rng = random.Random(4100 + batch)
            requests = [NGramRequest(config, store.multipliers) for _ in range(batch)]
            plans = []
            for step in range(a.steps):
                row = [r.preview([config.eos_token_id if step % 11 == 0 else rng.randrange(config.vocab_size)])
                       for r in requests]
                plans.append(row)
                for r, proposal in zip(requests, row):
                    r.commit(proposal)
            operations = {
                "serial": lambda row: torch.cat([store.lookup(p) for p in row]).to(a.device),
                "coalesced": lambda row: torch.cat(store.lookup_many(row)).to(a.device),
                "prepared_serial": lambda row: torch.cat([prepared.lookup(p) for p in row]).to(a.device),
                "prepared_coalesced": lambda row: torch.cat(prepared.lookup_many(row)).to(a.device),
            }
            for row in plans:
                ids = torch.tensor([p.rows[0] for p in row], dtype=torch.long, device=a.device)
                expected = torch.nn.functional.embedding(ids, resident).flatten(-2).view(torch.int16)
                for operation in operations.values():
                    if not torch.equal(operation(row).view(torch.int16), expected):
                        raise AssertionError("PLE staging differs in BF16 output bits")
            for operation in operations.values():
                for row in plans[:4]:
                    operation(row)
            torch.cuda.synchronize(a.device)
            values = {name: [] for name in operations}
            if a.profile_only:
                for name, operation in operations.items():
                    with torch.cuda.nvtx.range(f"PREP_{a.encoding}_{name}_B{batch}"):
                        for row in plans:
                            operation(row)
                        torch.cuda.synchronize(a.device)
            else:
                for repeat in range(a.repeats):
                    names = list(operations)
                    # Alternate both orientation and the start position.
                    if repeat % 2:
                        names.reverse()
                    names = names[repeat % 4:] + names[:repeat % 4]
                    for order, name in enumerate(names):
                        torch.cuda.synchronize(a.device)
                        interval_start, before, start = stamp(), usage(), perf_counter()
                        for row in plans:
                            operations[name](row)
                        torch.cuda.synchronize(a.device)
                        wall, after, interval_end = perf_counter() - start, usage(), stamp()
                        delta = {k: after[k] - before[k] for k in before}
                        values[name].append(dict(repeat=repeat, order=order,
                            interval_start_utc=interval_start, interval_end_utc=interval_end,
                            wall_seconds=wall, step_ms=wall * 1000 / a.steps,
                            request_rows_per_second=batch * a.steps / wall,
                            requested_token_rows=batch * a.steps,
                            h2d_payload_bytes=batch * a.steps * config.ple_embed_dim * 2, **delta))
            report["batch_results"].append(dict(batch=batch, steps=a.steps, output_bits_exact=True,
                                                compared_token_rows=batch * a.steps, results=values))
    finally:
        prepared.close()
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
    report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
