"""Pinned upstream GR oracle plus bounded real BF16 gated-residual weights.

Only the GR/RMS class definitions from an explicitly hashed reference execute.
This is component correctness, not full model quality or throughput validation.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

from collect_hf_headers import fetch_range
from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next import FlashNextConfig
from venturi.models.flash_next_residual import FlashNextGatedResidual


UPSTREAM_SHA256 = "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def reference_class(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Pinned upstream GR source hash differs")
    names = {"Qwen4ExpTextRMSNorm", "Qwen4ExpTextGatedResidual"}
    selected = [node for node in ast.parse(data).body
                if isinstance(node, ast.ClassDef) and node.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned GR definitions missing")
    # Strip kernel-dispatch decorators only; execute the floating source bodies.
    for node in selected:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    scope = dict(torch=torch, nn=nn, F=F)
    exec(compile(module, str(path), "exec"), scope)
    return scope["Qwen4ExpTextGatedResidual"]


def compare(checkpoint, raw, reference, device, dtypes):
    c = FlashNextConfig.from_dict(raw)
    rc = SimpleNamespace(**raw["text_config"])
    rows = []
    for prefix, combine in (("model.language_model.layers.0.attn_hyper_connection", True),
                            ("model.language_model.hyper_connection_mixer", False)):
        loaded = FlashNextGatedResidual.from_checkpoint(checkpoint, c, prefix,
                    eps=rc.rms_norm_eps, use_combine=combine)
        for dtype in dtypes:
            # Explicitly label casts; FP32/FP16 tests do not imply source BF16 GPU support.
            candidate = loaded.to(device=device, dtype=dtype)
            oracle = reference(rc, use_combine=combine).to(device=device, dtype=dtype)
            oracle.load_state_dict(candidate.state_dict(), strict=True)
            for batch, length in ((1, 1), (8, 1), (2, 4)):
                generator = torch.Generator(device=device).manual_seed(913 + batch + length)
                x = torch.randn((batch, length, c.hidden_size * c.hc_count),
                                dtype=dtype, device=device, generator=generator)
                original = x.clone()
                with torch.inference_mode():
                    actual, expected = candidate(x), oracle(x)
                    aa, ee = (actual, expected) if combine else ((actual,), (expected,))
                    integer = torch.int32 if dtype == torch.float32 else torch.int16
                    exact = all(torch.equal(a.view(integer), e.view(integer)) for a, e in zip(aa, ee))
                    if not exact:
                        raise AssertionError("Gated-residual output bits differ from pinned floating reference")
                    if combine:
                        block = torch.randn((batch, length, c.hidden_size), dtype=dtype,
                                            device=device, generator=generator)
                        injected = candidate.inject(block, actual[1], actual[2])
                        expected_injected = expected[1] + (block.unsqueeze(-2) * expected[2].unsqueeze(-1)).flatten(-2)
                        if not torch.equal(injected.view(integer), expected_injected.view(integer)):
                            raise AssertionError("Residual injection bits differ")
                    if not torch.equal(x.view(integer), original.view(integer)):
                        raise AssertionError("Residual input was mutated")
                rows.append(dict(prefix=prefix, combine=combine, dtype=str(dtype), device=str(device),
                                 batch=batch, sequence_length=length, output_bits_exact=exact))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--headers", required=True)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", help="Reuse an already sampled GR-only checkpoint; no downloads")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-cuda-memory-mib", type=int, default=1024)
    p.add_argument("--allow-range-samples", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((Path(a.metadata) / "config.json").read_text())
    reference = reference_class(a.upstream_modeling)
    samples = []
    if a.checkpoint:
        checkpoint = inspect_checkpoint(a.checkpoint)
    else:
        if not a.allow_range_samples or a.device != "cpu":
            p.error("Sampling requires explicit --allow-range-samples on CPU first")
        targets = ("model.language_model.layers.0.attn_hyper_connection.",
                   "model.language_model.hyper_connection_mixer.")
        tensors = {}
        for path in sorted(Path(a.headers).glob("*.json")):
            record = json.loads(path.read_text())
            if record["repo"] != "Qwen/Qwen3.8-Flash-Next" or record["revision"] != REVISION:
                raise ValueError("Original official BF16 source pin differs")
            for name, spec in record["header"].items():
                if name == "__metadata__" or not name.startswith(targets):
                    continue
                if spec["dtype"] != "BF16":
                    raise ValueError("Expected original BF16 GR weight")
                lo, hi = spec["data_offsets"]
                offset = 8 + record["header_bytes"]
                url = f"https://huggingface.co/{record['repo']}/resolve/{REVISION}/{record['filename']}"
                data, size = fetch_range(url, offset + lo, offset + hi - 1)
                if size != record["file_bytes"]:
                    raise ValueError("GR sample source length changed")
                tensor = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(spec["shape"]).clone()
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError("Nonfinite original GR tensor")
                tensors[name] = tensor
                filename = name + ".bin"
                (output / filename).write_bytes(data)
                samples.append(dict(tensor=name, shard=record["filename"], dtype=spec["dtype"],
                    shape=spec["shape"], absolute_range=[offset + lo, offset + hi - 1],
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), sample_file=filename))
        if len(tensors) != 7:
            raise ValueError("Missing gated-residual or final-mixer sampled weights")
        partial = output / "gr-only-checkpoint"
        partial.mkdir()
        save_file(tensors, partial / "model.safetensors", metadata={"scope": "GR-only official BF16 range samples"})
        (partial / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        checkpoint = inspect_checkpoint(partial)
    device = torch.device(a.device)
    if device.type == "cuda":
        if not 128 <= a.max_cuda_memory_mib <= 2048:
            p.error("Explicit bounded CUDA component allocation required")
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(a.max_cuda_memory_mib * 2**20 / total, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    rows = compare(checkpoint, raw, reference, device, dtypes)
    result = dict(scope="GR read/inject/final mixer component correctness only; NOT model quality/TPS",
                  revision=REVISION, upstream_sha256=UPSTREAM_SHA256, samples=samples,
                  source_sampled_bytes=sum(s["bytes"] for s in samples), cases=rows,
                  whole_model_tested=False, native_kernel_implemented=False,
                  gpu_precision_note="V100 has no native BF16 Tensor Cores; CUDA tests explicitly cast BF16 source weights to FP32/FP16",
                  torch=str(torch.__version__))
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
