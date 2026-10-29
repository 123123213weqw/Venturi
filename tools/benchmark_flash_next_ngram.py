"""Synthetic warm PLE row-gather + H2D microbenchmark, NOT model throughput.

Serial and coalesced CPU lookups use identical per-request plans, one pageable
H2D transfer per step, and bit-exact resident GPU lookup controls. The fixture
has the real 16-head/160-element geometry but a much smaller row vocabulary.
It does not predict random access to a real 102.4 GB table or full-model TPS.
"""

import argparse
import json
from pathlib import Path
import random
import resource
import subprocess
from time import perf_counter

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next import FlashNextConfig
from venturi.models.flash_next_ngram import NGramRequest, ShardedNGramTable


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def make_fixture(root, raw):
    root.mkdir(parents=True, exist_ok=False)
    raw["text_config"]["ngram_vocab_size_base"] = 65521
    c = FlashNextConfig.from_dict(raw)
    prefix = f"model.language_model.layers.{c.ple_layers[0] - 1}.ple.ple_embedding"
    _, _, padded = c.ngram_geometry()
    rows, width = padded // c.split_ngram_parts, c.ple_embed_dim // c.ngram_heads
    mapping, total = {}, 0
    for start in range(0, c.split_ngram_parts, 4):
        tensors, filename = {}, f"ngram-{start // 4:03d}.safetensors"
        for part in range(start, min(start + 4, c.split_ngram_parts)):
            # Exactly representable BF16 values, bounded temporary allocation.
            data = ((torch.arange(rows * width, dtype=torch.int32) + part) % 127 - 63)
            tensors[f"{prefix}.ngram_embedding.shard_{part}.weight"] = data.to(
                torch.bfloat16).reshape(rows, width) * (1 / 128)
        if start == 0:
            tensors[prefix + ".layer_multipliers"] = torch.tensor([7, 13, 19], dtype=torch.long)
        save_file(tensors, root / filename, metadata={"scope": "synthetic PLE-only fixture"})
        mapping.update({k: filename for k in tensors})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
    (root / "config.json").write_text(json.dumps(raw))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        dict(weight_map=mapping, metadata=dict(total_size=total))))
    return c, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--output", required=True)
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
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    c, source_bytes = make_fixture(Path(a.fixture), json.loads(Path(a.config).read_text()))
    checkpoint = inspect_checkpoint(a.fixture)
    store = ShardedNGramTable(checkpoint, c, max_tokens=max(a.batch_sizes))
    chunks = []
    for spec in store.parts:
        with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
            chunks.append(reader.get_tensor(spec.name))
    # Resident table is a correctness oracle, not a viable placement for the
    # real 95.37 GiB table on a V100. Keep it out of the timed operation.
    resident = torch.cat(chunks).to(a.device)
    del chunks
    telemetry = (output / "nvml.csv").open("w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
        "--format=csv", "--loop-ms=200",
    ], stdout=telemetry, stderr=subprocess.DEVNULL)
    report = dict(scope="synthetic warm PLE-only CPU row lookup plus one blocking pageable H2D copy; not model TPS",
                  real_model_tested=False, profile_only=a.profile_only,
                  fixture_table_bytes=source_bytes - 24, ngram_heads=c.ngram_heads,
                  head_width=store.width, batch_results=[],
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(a.device),
                  cpu_threads=torch.get_num_threads(),
                  caveat="335 MB synthetic warm page-cache working set is not the 102.4 GB original table",
                  counters="No changed CUDA compute kernel; occupancy/SM counters are not row-copy efficiency metrics")
    try:
        for batch in a.batch_sizes:
            rng = random.Random(4100 + batch)
            requests = [NGramRequest(c, store.multipliers) for _ in range(batch)]
            plans = []
            for step in range(a.steps):
                row = [r.preview([c.eos_token_id if step % 11 == 0 else rng.randrange(c.vocab_size)])
                       for r in requests]
                plans.append(row)
                for request, proposal in zip(requests, row):
                    request.commit(proposal)

            def serial(row):
                return torch.cat([store.lookup(p) for p in row]).to(a.device)

            def coalesced(row):
                return torch.cat(store.lookup_many(row)).to(a.device)

            # Check every scored input, including distinct private EOS histories.
            for row in plans:
                ids = torch.tensor([p.rows[0] for p in row], dtype=torch.long, device=a.device)
                expected = torch.nn.functional.embedding(ids, resident).flatten(-2)
                for operation in (serial, coalesced):
                    if not torch.equal(operation(row), expected):
                        raise AssertionError("PLE staging differs from resident GPU table lookup")
            for operation in (serial, coalesced):
                for row in plans[:4]:
                    operation(row)
            torch.cuda.synchronize(a.device)
            values = {"serial": [], "coalesced": []}
            if a.profile_only:
                for name, operation in (("serial", serial), ("coalesced", coalesced)):
                    with torch.cuda.nvtx.range(f"PLE_{name}_B{batch}"):
                        for row in plans:
                            operation(row)
                torch.cuda.synchronize(a.device)
            else:
                for repeat in range(a.repeats):
                    order = [("serial", serial), ("coalesced", coalesced)]
                    if repeat % 2:
                        order.reverse()
                    for name, operation in order:
                        torch.cuda.synchronize(a.device)
                        cpu_start, start = cpu_seconds(), perf_counter()
                        for row in plans:
                            operation(row)
                        torch.cuda.synchronize(a.device)
                        wall, cpu = perf_counter() - start, cpu_seconds() - cpu_start
                        values[name].append(dict(repeat=repeat, wall_seconds=wall,
                                                 cpu_seconds=cpu, step_ms=wall * 1000 / a.steps,
                                                 requested_token_rows=batch * a.steps,
                                                 h2d_payload_bytes=batch * a.steps * c.ple_embed_dim * 2))
            report["batch_results"].append(dict(batch=batch, steps=a.steps, exact=True,
                                                compared_token_rows=batch * a.steps,
                                                results=values))
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
