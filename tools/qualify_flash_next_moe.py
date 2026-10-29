"""Pinned floating MoE oracle; bounded original router/shared/selected experts.

Official router still selects from all512 experts/top10. Real cases deliberately
reuse a seed direction at positive scales; sample only their bounded expert
union (at most16). This is NOT arbitrary-input coverage, quality or model TPS.
Synthetic cases independently cover varied/tied routes and normalization modes.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from collect_hf_headers import fetch_range
from venturi.models.flash_next_moe import FlashNextMoE, FlashNextTopKRouter, MoEGeometry, TensorExpertStore


UPSTREAM_SHA256 = "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.0.mlp"


def reference_class(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Pinned upstream MoE source hash differs")
    names = {"Qwen4ExpTextMLP", "Qwen4ExpTextExperts", "Qwen4ExpTextTopKRouter", "Qwen4ExpTextSparseMoeBlock"}
    selected = [n for n in ast.parse(data).body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned MoE definitions missing")
    for node in selected:
        node.decorator_list = []  # Floating bodies only, no external expert kernels.
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    scope = dict(torch=torch, nn=nn, F=F, ACT2FN={"silu": F.silu})
    exec(compile(module, str(path), "exec"), scope)
    return scope["Qwen4ExpTextSparseMoeBlock"]


class ExpertIndex:
    """Replace meta parameter indexing, not routing/math of the pinned oracle."""

    def __init__(self, store, position, dtype, device):
        self.store, self.position, self.dtype, self.device = store, position, dtype, device

    def __getitem__(self, expert):
        return self.store.read(int(expert), device=self.device, dtype=self.dtype)[self.position]


def make_oracle(reference, candidate, dtype, device):
    g = candidate.geometry
    config = SimpleNamespace(hidden_size=g.hidden_size, num_experts=g.num_experts,
        num_experts_per_tok=g.top_k, norm_topk_prob=g.normalize_topk,
        moe_intermediate_size=g.intermediate_size, shared_expert_intermediate_size=g.shared_intermediate_size,
        hidden_act="silu")
    # Never allocate the full official512-expert bank just to test selected rows.
    with torch.device("meta"):
        oracle = reference(config)
    loaded = oracle.load_state_dict(candidate.state_dict(), strict=False, assign=True)
    if set(loaded.missing_keys) != {"experts.gate_up_proj", "experts.down_proj"} or loaded.unexpected_keys:
        raise ValueError("Reference core binding differs")
    for key, position in (("gate_up_proj", 0), ("down_proj", 1)):
        del oracle.experts._parameters[key]
        setattr(oracle.experts, key, ExpertIndex(candidate.expert_store, position, dtype, device))
    return oracle.eval()


def compare(reference, geometry, core, bank, inputs, dtype, device, *, scope):
    candidate = FlashNextMoE(geometry, bank).to(device=device, dtype=dtype).eval()
    candidate.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in core.items()}, strict=True)
    oracle = make_oracle(reference, candidate, dtype, device)
    integer = torch.int32 if dtype == torch.float32 else torch.int16
    rows = []
    for name, x in inputs:
        x = x.to(device=device, dtype=dtype)
        before = x.clone()
        with torch.inference_mode():
            actual_route, expected_route = candidate.gate(x), oracle.gate(x)
            route_exact = all(torch.equal(a.view(integer) if a.dtype != torch.int64 else a,
                                          b.view(integer) if b.dtype != torch.int64 else b)
                              for a, b in zip(actual_route, expected_route))
            selected = sorted(set(actual_route[2].flatten().tolist()))
            if not set(selected) <= set(bank.experts):
                raise ValueError("Real input selected an unsampled expert; no pruning allowed")
            actual, expected = candidate(x), oracle(x)
            exact = torch.equal(actual.view(integer), expected.view(integer))
            if not exact or not route_exact or not torch.equal(x.view(integer), before.view(integer)):
                raise AssertionError(f"MoE route/output/input bits differ: {scope}/{name}/{dtype}")
        rows.append(dict(scope=scope, case=name, shape=list(x.shape), device=str(device), dtype=str(dtype),
                         router_bits_exact=route_exact, output_bits_exact=exact,
                         selected_expert_ids=selected, normalize_topk=geometry.normalize_topk))
    return rows


def synthetic_cases(reference, device, dtypes):
    result = []
    for normalize in (True, False):
        g = MoEGeometry(8, 4, 2, 6, 5, normalize)
        torch.manual_seed(420)
        bank = TensorExpertStore(g, {e: (torch.randn(12, 8) * 0.2, torch.randn(8, 6) * 0.2) for e in range(4)})
        module = FlashNextMoE(g, bank)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2)
        core = {k: v.detach().clone() for k, v in module.state_dict().items()}
        inputs = [(f"B{b}S{s}", torch.randn(b, s, 8)) for b, s in ((1, 1), (8, 1), (2, 4), (0, 1))]
        for dtype in dtypes:
            result.extend(compare(reference, g, core, bank, inputs, dtype, device, scope="synthetic-varied-route"))
            tied = dict(core, **{"gate.weight": torch.zeros_like(core["gate.weight"])})
            result.extend(compare(reference, g, tied, bank, inputs[:2], dtype, device, scope="synthetic-tied-router"))
    return result


def sample_real(metadata, headers, output):
    raw = json.loads((metadata / "config.json").read_text())
    g = MoEGeometry.from_dict(raw)
    records = {}
    for path in sorted(headers.glob("*.json")):
        record = json.loads(path.read_text())
        if record["repo"] != "Qwen/Qwen3.8-Flash-Next" or record["revision"] != REVISION:
            raise ValueError("Original official BF16 source pin differs")
        for name, spec in record["header"].items():
            if name != "__metadata__" and name.startswith(PREFIX + "."):
                records[name] = record, spec
    samples, tensors = [], {}

    def read(name, expert=None):
        record, spec = records[name]
        if spec["dtype"] != "BF16":
            raise ValueError("Original BF16 MoE sample required")
        lo, hi = spec["data_offsets"]
        shape = spec["shape"]
        if expert is not None:
            if len(shape) != 3 or shape[0] != g.num_experts:
                raise ValueError("Original fused expert geometry required")
            per_expert = (hi - lo) // g.num_experts
            lo += expert * per_expert
            hi, shape = lo + per_expert, shape[1:]
        offset = 8 + record["header_bytes"]
        url = f"https://huggingface.co/{record['repo']}/resolve/{REVISION}/{record['filename']}"
        data, size = fetch_range(url, offset + lo, offset + hi - 1)
        if size != record["file_bytes"]:
            raise ValueError("MoE sample backing file size changed")
        value = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).clone()
        if not torch.isfinite(value.float()).all():
            raise ValueError("Nonfinite original MoE weight")
        samples.append(dict(tensor=name, expert_id=expert, dtype="BF16", shape=shape,
            shard=record["filename"], absolute_range=[offset + lo, offset + hi - 1],
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return value

    for key in ("gate.weight", "shared_expert_gate.weight", "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight", "shared_expert.down_proj.weight"):
        tensors[key] = read(PREFIX + "." + key)
    generator = torch.Generator().manual_seed(8913)
    direction = torch.randn(1, 1, g.hidden_size, generator=generator)
    # Actual512-way router for all planned CPU/cast precisions, never a fake
    # reduced expert router. BF16 input rounding can change the boundary expert.
    selected = set()
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        router = FlashNextTopKRouter(g).to(dtype)
        with torch.no_grad():
            router.weight.copy_(tensors["gate.weight"].to(dtype))
            for b, s in ((1, 1), (8, 1), (2, 4)):
                values = direction.expand(b, s, -1) * torch.linspace(0.7, 1.3, b * s).reshape(b, s, 1)
                selected.update(router(values.to(dtype))[2].flatten().tolist())
    ids = sorted(selected)
    if len(ids) > 16:
        raise ValueError("Planned real MoE cases exceed16 sampled experts; refuse an unbounded download")
    for expert in ids:
        tensors[f"sampled_experts.{expert}.gate_up_proj"] = read(PREFIX + ".experts.gate_up_proj", expert)
        tensors[f"sampled_experts.{expert}.down_proj"] = read(PREFIX + ".experts.down_proj", expert)
    root = output / "official-moe-samples"
    root.mkdir()
    save_file(tensors, root / "model.safetensors", metadata={"scope": "partial MoE component samples; NOT a complete model/layer"})
    save_file({"direction": direction}, root / "input.safetensors")
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (root / "ranges.json").write_text(json.dumps(dict(revision=REVISION, samples=samples,
        selected_expert_ids=ids, source_bytes=sum(s["bytes"] for s in samples)), indent=2) + "\n")
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata")
    p.add_argument("--headers")
    p.add_argument("--samples", help="Reuse partial official MoE samples, never a full checkpoint")
    p.add_argument("--allow-range-samples", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic-only", action="store_true")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    reference = reference_class(a.upstream_modeling)
    device = torch.device(a.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(1536 * 2**20 / total, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    cases = synthetic_cases(reference, device, dtypes)
    real = None
    if not a.synthetic_only:
        if a.samples:
            root = Path(a.samples)
        elif a.allow_range_samples and device.type == "cpu" and a.metadata and a.headers:
            root = sample_real(Path(a.metadata), Path(a.headers), output)
        else:
            p.error("Explicit existing samples or bounded original CPU sampling required")
        raw = json.loads((root / "config.json").read_text())
        ranges = json.loads((root / "ranges.json").read_text())
        if ranges["revision"] != REVISION:
            raise ValueError("Original MoE sample source pin differs")
        g = MoEGeometry.from_dict(raw)
        tensors = load_file(root / "model.safetensors")
        core = {k: v for k, v in tensors.items() if not k.startswith("sampled_experts.")}
        bank = TensorExpertStore(g, {e: (tensors[f"sampled_experts.{e}.gate_up_proj"],
                                        tensors[f"sampled_experts.{e}.down_proj"]) for e in ranges["selected_expert_ids"]})
        direction = load_file(root / "input.safetensors")["direction"]
        inputs = [(f"B{b}S{s}-scaled-direction", direction.expand(b, s, -1) *
                   torch.linspace(0.7, 1.3, b * s).reshape(b, s, 1)) for b, s in ((1, 1), (8, 1), (2, 4))]
        for dtype in dtypes:
            cases.extend(compare(reference, g, core, bank, inputs, dtype, device, scope="official-sampled-top10"))
        real = dict(source_bytes=ranges["source_bytes"], expert_ids=ranges["selected_expert_ids"],
                    caveat="Only one seed direction at positive scales and a <=16-expert union; actual512-way router/top10 retained; not arbitrary input coverage")
    report = dict(scope="Floating MoE component correctness only; no model TPS/quality or native kernel claim",
                  upstream_sha256=UPSTREAM_SHA256, original_revision=REVISION, official_samples=real,
                  cases=cases, torch=str(torch.__version__), native_kernel_implemented=False,
                  whole_model_tested=False, cuda_allocator_cap_mib=1536 if device.type == "cuda" else None)
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
