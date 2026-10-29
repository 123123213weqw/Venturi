"""Pinned eager QSA main-attention oracle; component correctness, not TPS.

Uses original attention/eager/normalization/indexer definitions. Only the same
documented int64 scatter ABI cast as the indexer qualification is applied.
Actual block weights (when requested) still use synthetic activations/state.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file, save_file
import torch

from collect_hf_headers import fetch_range
from qualify_flash_next_qsa import INDEX_ABI_SHIM, REVISION, UPSTREAM_SHA256, positions, reference_class
from venturi.models.flash_next_attention import FlashNextQSAAttention, QSAAttentionGeometry, QSAState
from venturi.models.flash_next_qsa import QSAIndexerGeometry


PREFIX = "model.language_model.layers.3.self_attn"


def attention_reference(path):
    indexer = reference_class(path)
    scope = dict(indexer.__init__.__globals__)
    names = {"repeat_kv", "eager_attention_forward", "Qwen4ExpTextAttention"}
    selected = [n for n in ast.parse(Path(path).read_bytes()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned main attention definitions missing")
    for n in selected:
        n.decorator_list = []
    class EagerOnly:
        @staticmethod
        def get_interface(name, default):
            if name != "eager":
                raise ValueError("Only original eager implementation qualified")
            return default
    scope["ALL_ATTENTION_FUNCTIONS"] = EagerOnly
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["Qwen4ExpTextAttention"]


class Cache:
    def __init__(self, past):
        self.raw = None if past is None else past.raw_indexer_keys.clone()
        self.keys = None if past is None else past.keys.clone()
        self.values = None if past is None else past.values.clone()

    def update_indexer(self, keys, layer):
        self.raw = keys.clone() if self.raw is None else torch.cat((self.raw, keys), 1)
        return self.raw

    def update(self, keys, values, layer):
        self.keys = keys.clone() if self.keys is None else torch.cat((self.keys, keys), 2)
        self.values = values.clone() if self.values is None else torch.cat((self.values, values), 2)
        return self.keys, self.values


def cases(g, real=False):
    rng = torch.Generator().manual_seed(116)
    shapes = ((1, 1, 0), (2, 7, 0), (2, 1, 11), (2, 3, 5)) if not real else (
        (1, 6, 0), (1, 1, 4096), (2, 1, 4100), (8, 1, 16))
    for batch, length, past_length in shapes:
        x = torch.randn(batch, length, g.indexer.hidden_size, generator=rng) * 0.3
        past = None if not past_length else QSAState(
            torch.randn(batch, past_length, g.indexer.head_dim, generator=rng),
            torch.randn(batch, g.kv_heads, past_length, g.head_dim, generator=rng),
            torch.randn(batch, g.kv_heads, past_length, g.head_dim, generator=rng))
        visible = torch.arange(length + past_length).view(1, 1, 1, -1) <= torch.arange(
            past_length, past_length + length).view(1, 1, -1, 1)
        visible = visible.expand(batch, 1, -1, -1).clone()
        yield f"B{batch}S{length}P{past_length}-causal", x, past, visible
        if batch > 1:
            holes = visible.clone()
            holes[0, :, :, 1::3] = False
            holes[1, :, -1] = False
            yield f"B{batch}S{length}P{past_length}-holes-empty-row", x, past, holes


def compare(reference, geometry, weights, device, dtypes, real=False):
    g, report = geometry, []
    config = SimpleNamespace(hidden_size=g.indexer.hidden_size, indexer_n_heads=g.indexer.query_heads,
        indexer_kv_heads=1, indexer_head_dim=g.indexer.head_dim, indexer_budget=g.indexer.token_budget,
        indexer_compress_ratio=g.indexer.compress_ratio, rms_norm_eps=g.indexer.eps,
        num_attention_heads=g.query_heads, num_key_value_heads=g.kv_heads, head_dim=g.head_dim,
        attention_dropout=0.0, attention_bias=False, _attn_implementation="eager")
    for dtype in dtypes:
        # Separate long-history B1/B2 and short-history B8 checks. Eager repeated
        # GQA K/V at B8/P4100 exceeded the isolated V1001.5GiB allocator cap in
        # v1; preserve that failure instead of stopping production or hiding it.
        candidate = FlashNextQSAAttention(g).to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(weights, strict=True)
        oracle = reference(config, 3).to(device=device, dtype=dtype).eval()
        oracle.load_state_dict(candidate.state_dict(), strict=True)
        for name, x, past, visible in cases(g, real):
            x = x.to(device=device, dtype=dtype)
            if past is not None:
                past = QSAState(*(t.to(device=device, dtype=dtype) for t in (past.raw_indexer_keys, past.keys, past.values)))
            original = None if past is None else past.truncate(past.raw_indexer_keys.shape[1])
            cos, sin = (v.to(device=device, dtype=dtype) for v in positions(x.shape[0], visible.shape[-1], 64 if real else 2))
            mask = torch.where(visible.to(device=device), torch.zeros((), dtype=dtype, device=device), torch.finfo(dtype).min)
            cache = Cache(past)
            with torch.inference_mode():
                actual, state, probs = candidate(x, (cos, sin), mask, past=past)
                expected, expected_probs = oracle(x, (cos, sin), mask, cache)
            for a, b in ((actual, expected), (probs, expected_probs), (state.raw_indexer_keys, cache.raw),
                         (state.keys, cache.keys), (state.values, cache.values)):
                torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
            if past is not None:
                for a, b in zip((past.raw_indexer_keys, past.keys, past.values),
                                (original.raw_indexer_keys, original.keys, original.values)):
                    assert torch.equal(a, b)
            finite = torch.isfinite(expected)
            report.append(dict(case=name, dtype=str(dtype), device=str(device), official_weights=real,
                output_and_probabilities_exact_equal_nan=True, all_three_states_exact=True,
                nan_output_elements=int(torch.isnan(actual).sum().item()),
                finite_output_elements=int(finite.sum().item()),
                max_finite_abs_error=float((actual[finite] - expected[finite]).abs().max().item()) if finite.any() else None))
    return report


def sample(headers, metadata, output):
    """Bounded resumable per-tensor chunks; never overwrite a mismatched piece."""
    root = output / "official-attention-samples"
    root.mkdir(exist_ok=True)
    specs = {}
    for path in headers.glob("*.json"):
        record = json.loads(path.read_text())
        if record["repo"] != "Qwen/Qwen3.8-Flash-Next" or record["revision"] != REVISION:
            raise ValueError("Original official attention sample pin differs")
        for name, spec in record["header"].items():
            if name.startswith(PREFIX + "."):
                specs[name] = (record, spec)
    if len(specs) != 9 or any(s[1]["dtype"] != "BF16" for s in specs.values()):
        raise ValueError("Exactly9 original BF16 attention/indexer tensors required")
    tensors, segments = {}, []
    for name, (record, spec) in sorted(specs.items()):
        lo, hi = spec["data_offsets"]
        offset, data = 8 + record["header_bytes"], bytearray()
        for start in range(lo, hi, 8 * 2**20):
            end = min(start + 8 * 2**20, hi)
            piece = root / (name.removeprefix(PREFIX + ".") + f".{start-lo}-{end-lo}.bin")
            proof = piece.with_suffix(".json")
            if piece.exists():
                payload = piece.read_bytes()
                p = json.loads(proof.read_text())
                if p["sha256"] != hashlib.sha256(payload).hexdigest() or p["source_range"] != [offset + start, offset + end - 1]:
                    raise ValueError("Existing range piece differs")
            else:
                payload, total = fetch_range(f"https://huggingface.co/{record['repo']}/resolve/{REVISION}/{record['filename']}",
                                             offset + start, offset + end - 1)
                if total != record["file_bytes"]:
                    raise ValueError("Source shard size changed")
                p = dict(tensor=name, shard=record["filename"], source_range=[offset + start, offset + end - 1],
                         bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                piece.write_bytes(payload)
                proof.write_text(json.dumps(p, indent=2) + "\n")
            if len(payload) != end - start:
                raise ValueError("Wrong range size")
            data.extend(payload)
            segments.append(p)
            print(f"sampled {name} {end-lo}/{hi-lo}", flush=True)
        tensors[name.removeprefix(PREFIX + ".")] = torch.frombuffer(data, dtype=torch.bfloat16).reshape(spec["shape"]).clone()
    target = root / "model.safetensors"
    if target.exists():
        raise FileExistsError("Use --samples to reuse a completed source sample")
    save_file(tensors, target, metadata={"scope": "9 actual QSA attention weights, not a complete layer/model"})
    (root / "config.json").write_bytes((metadata / "config.json").read_bytes())
    (root / "ranges.json").write_text(json.dumps(dict(revision=REVISION, segments=segments,
        total_bytes=sum(t.numel() * t.element_size() for t in tensors.values()),
        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2) + "\n")
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples")
    p.add_argument("--headers")
    p.add_argument("--metadata")
    p.add_argument("--allow-range-samples", action="store_true")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1536 * 2**20 / torch.cuda.get_device_properties(device).total_memory, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    reference = attention_reference(a.upstream_modeling)
    torch.manual_seed(116)
    g = QSAAttentionGeometry(QSAIndexerGeometry(8, 2, 4, 4, 2), 4, 2, 4)
    m = FlashNextQSAAttention(g)
    with torch.no_grad():
        for t in m.parameters():
            t.copy_(torch.randn_like(t) * 0.15)
    report = compare(reference, g, m.state_dict(), device, dtypes)
    sample_info = None
    if not a.synthetic_only:
        if a.samples:
            root = Path(a.samples)
        elif a.allow_range_samples and a.headers and a.metadata and device.type == "cpu":
            root = sample(Path(a.headers), Path(a.metadata), output)
        else:
            p.error("Explicit existing samples or bounded CPU ranges required")
        raw = json.loads((root / "config.json").read_text())
        ranges = json.loads((root / "ranges.json").read_text())
        target = root / "model.safetensors"
        if ranges["revision"] != REVISION or hashlib.sha256(target.read_bytes()).hexdigest() != ranges["artifact_sha256"]:
            raise ValueError("Official source sample identity differs")
        report.extend(compare(reference, QSAAttentionGeometry.from_dict(raw), load_file(target), device, dtypes, True))
        sample_info = dict(source_bytes=ranges["total_bytes"], tensors=9,
                           caveat="Actual block weights but synthetic activations/past, no generated sequence/quality/context-limit test")
    result = dict(scope="Floating eager QSA attention correctness only, additive masks; no native kernel/TPS/full generation",
        cases=report, official_samples=sample_info, upstream_sha256=UPSTREAM_SHA256, original_revision=REVISION,
        reference_index_abi_shim=INDEX_ABI_SHIM, torch=str(torch.__version__),
        explicit_candidate_state_budget_bytes=128 * 2**20,
        explicit_expanded_kv_budget_bytes=512 * 2**20,
        batch_history_scope="B8/P16 and separate B1/P4096,B2/P4100; not B8/long-history validation",
        empty_rows="Pinned finite-min mask combination may produce -inf then NaNs; matched with equal_nan, not silently repaired",
        whole_model_tested=False, native_kernel_implemented=False)
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
