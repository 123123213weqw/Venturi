"""Pinned QSA indexer mask/raw-state oracle; no full sparse attention or TPS.

Synthetic masks exercise holes, padding, tails, fully masked rows and cached
decode. Optional3 actual indexer weights use synthetic activations/raw past,
including >2048 visible tokens; this is not a whole-model context/quality test.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from collect_hf_headers import fetch_range
from venturi.models.flash_next_qsa import FlashNextQSAIndexer, QSAIndexerGeometry


UPSTREAM_SHA256 = "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.3.self_attn.indexer"
INDEX_ABI_SHIM = "Only cast scatter_indices int32->int64 for Torch2.5 scatter ABI; mask/math/selection unchanged"


def reference_class(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Pinned QSA reference source hash differs")
    names = {"Qwen4ExpTextRMSNorm", "rotate_half", "apply_rotary_pos_emb", "Qwen4ExpTextQSAIndexer"}
    selected = [n for n in ast.parse(data).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned QSA definitions missing")
    for n in selected:
        n.decorator_list = []

    class ScatterABI(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "scatter" and len(node.args) == 3 and (
                isinstance(node.args[1], ast.Name) and node.args[1].id == "scatter_indices"
            ):
                node.args[1] = ast.Call(func=ast.Attribute(value=node.args[1], attr="long", ctx=ast.Load()), args=[], keywords=[])
                self.count += 1
            return node

    shim = ScatterABI()
    selected = [shim.visit(n) for n in selected]
    if shim.count != 1:
        raise ValueError("Pinned reference scatter ABI call differs")
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    scope = dict(torch=torch, nn=nn, math=math)
    exec(compile(module, str(path), "exec"), scope)
    return scope["Qwen4ExpTextQSAIndexer"]


class RawCache:
    def __init__(self, past=None):
        self.raw = None if past is None else past.clone()

    def update_indexer(self, new, layer):
        self.raw = new.clone() if self.raw is None else torch.cat((self.raw, new), 1)
        return self.raw


def positions(batch, length, width):
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
    freq = torch.arange(length, dtype=torch.float32).view(1, length, 1) * inv_freq
    return torch.cat((freq.cos(), freq.cos()), -1).expand(batch, -1, -1), torch.cat(
        (freq.sin(), freq.sin()), -1).expand(batch, -1, -1)


def compare(reference, geometry, weights, fixture_cases, device, dtypes, scope):
    g = geometry
    config = SimpleNamespace(hidden_size=g.hidden_size, indexer_n_heads=g.query_heads, indexer_kv_heads=1,
        indexer_head_dim=g.head_dim, indexer_budget=g.token_budget, indexer_compress_ratio=g.compress_ratio,
        rms_norm_eps=g.eps)
    report = []
    for dtype in dtypes:
        candidate = FlashNextQSAIndexer(g).to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(weights, strict=True)
        oracle = reference(config, 3).to(device=device, dtype=dtype).eval()
        oracle.load_state_dict(candidate.state_dict(), strict=True)
        integer = torch.int32 if dtype == torch.float32 else torch.int16
        for name, x, past, visible, width in fixture_cases:
            x = x.to(device=device, dtype=dtype)
            past = None if past is None else past.to(device=device, dtype=dtype)
            original = None if past is None else past.clone()
            cos, sin = (v.to(device=device, dtype=dtype) for v in positions(x.shape[0], visible.shape[-1], width))
            for floating in (False, True):
                visibility = visible.to(device=device)
                mask = torch.where(visibility, torch.zeros((), device=device, dtype=dtype),
                                   torch.finfo(dtype).min) if floating else visibility
                cache = RawCache(past)
                with torch.inference_mode():
                    actual, raw = candidate(x, (cos, sin), mask, raw_past=past)
                    expected = oracle(x, (cos, sin), mask, cache)
                same_mask = torch.equal(actual, expected) if not floating else torch.equal(
                    actual.view(integer), expected.view(integer))
                same_keys = torch.equal(raw.view(integer), cache.raw.view(integer))
                if not same_mask or not same_keys or past is not None and not torch.equal(past, original):
                    raise AssertionError(f"QSA mask/raw-state/input differs: {name}/{dtype}/{floating}")
                if not torch.all(~(actual == 0 if floating else actual) | visibility):
                    raise AssertionError("QSA selected a masked token")
                report.append(dict(scope=scope, case=name, shape=list(x.shape), past_length=0 if past is None else past.shape[1],
                    dtype=str(dtype), device=str(device), additive_mask=floating, rotary_width=width,
                    selected_mask_exact=same_mask, raw_state_bits_exact=same_keys,
                    max_selected_tokens=int((actual == 0 if floating else actual).sum(-1).max().item())))
    return report


def cases(g, real=False):
    generator = torch.Generator().manual_seed(1526)
    cells = []
    shapes = ((1, 1, 0), (2, 3, 0), (2, 9, 0), (2, 1, 7), (2, 3, 5)) if not real else (
        (1, 8, 0), (1, 1, 4096), (2, 1, 4100))
    for batch, length, past_length in shapes:
        x = torch.randn(batch, length, g.hidden_size, generator=generator)
        past = torch.randn(batch, past_length, g.head_dim, generator=generator) if past_length else None
        key_positions = torch.arange(past_length + length)
        visible = key_positions.view(1, 1, 1, -1) <= (torch.arange(length) + past_length).view(1, 1, -1, 1)
        visible = visible.expand(batch, 1, -1, -1).clone()
        width = 64 if real else 2
        name = f"B{batch}S{length}P{past_length}"
        cells.append((name + "-causal", x, past, visible, width))
        if batch == 2:
            irregular = visible.clone()
            irregular[0, :, :, 1::3] = False
            irregular[1, :, -1] = False
            cells.append((name + "-holes-and-fully-masked", x, past, irregular, width))
    if not real:
        name, x, past, visible, _ = cells[-2]
        cells.append(("ties-zero-input-full-rotary", torch.zeros_like(x), None if past is None else torch.zeros_like(past), visible, 4))
    return cells


def sample(metadata, headers, output):
    raw = json.loads((metadata / "config.json").read_text())
    weights, samples = {}, []
    for path in headers.glob("*.json"):
        r = json.loads(path.read_text())
        if r["repo"] != "Qwen/Qwen3.8-Flash-Next" or r["revision"] != REVISION:
            raise ValueError("Official QSA sample pin differs")
        for name, spec in r["header"].items():
            if not name.startswith(PREFIX + "."):
                continue
            if spec["dtype"] != "BF16":
                raise ValueError("Original BF16 QSA weights required")
            offset = 8 + r["header_bytes"]
            lo, hi = spec["data_offsets"]
            data, size = fetch_range(f"https://huggingface.co/{r['repo']}/resolve/{REVISION}/{r['filename']}", offset + lo, offset + hi - 1)
            if size != r["file_bytes"]:
                raise ValueError("QSA sample backing length differs")
            value = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(spec["shape"]).clone()
            if not torch.isfinite(value.float()).all():
                raise ValueError("Nonfinite official QSA sample")
            weights[name.removeprefix(PREFIX + ".")] = value
            samples.append(dict(tensor=name, shape=spec["shape"], source_range=[offset + lo, offset + hi - 1],
                shard=r["filename"], bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    if len(weights) != 3:
        raise ValueError("QSA indexer sample must contain exactly3 weights")
    root = output / "official-indexer-samples"
    root.mkdir()
    save_file(weights, root / "model.safetensors", metadata={"scope": "3 actual indexer weights, not full attention/model"})
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (root / "ranges.json").write_text(json.dumps(dict(revision=REVISION, samples=samples), indent=2) + "\n")
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata")
    p.add_argument("--headers")
    p.add_argument("--samples")
    p.add_argument("--allow-range-samples", action="store_true")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1024 * 2**20 / torch.cuda.get_device_properties(device).total_memory, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    reference = reference_class(a.upstream_modeling)
    torch.manual_seed(516)
    g = QSAIndexerGeometry(8, 2, 4, 4, 2)
    module = FlashNextQSAIndexer(g)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.15)
    report = compare(reference, g, module.state_dict(), cases(g), device, dtypes, "synthetic-indexer")
    sample_info = None
    if not a.synthetic_only:
        if a.samples:
            root = Path(a.samples)
        elif a.allow_range_samples and a.metadata and a.headers and device.type == "cpu":
            root = sample(Path(a.metadata), Path(a.headers), output)
        else:
            p.error("Explicit samples or bounded CPU source ranges required")
        raw, ranges = json.loads((root / "config.json").read_text()), json.loads((root / "ranges.json").read_text())
        if ranges["revision"] != REVISION:
            raise ValueError("QSA source sample revision differs")
        g = QSAIndexerGeometry.from_dict(raw)
        report.extend(compare(reference, g, load_file(root / "model.safetensors"), cases(g, True), device, dtypes,
                              "official-indexer-weights-synthetic-state"))
        sample_info = dict(source_bytes=sum(s["bytes"] for s in ranges["samples"]), tensors=len(ranges["samples"]),
                           caveat="Actual indexer weights; synthetic current activations/raw past, not a model sequence or context-limit test")
    result = dict(scope="QSA indexer-only correctness; no main attention, native kernel, full generation or TPS claim",
                  cases=report, official_samples=sample_info, upstream_sha256=UPSTREAM_SHA256, original_revision=REVISION,
                  reference_index_abi_shim=INDEX_ABI_SHIM,
                  torch=str(torch.__version__), whole_model_tested=False, native_kernel_implemented=False)
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
