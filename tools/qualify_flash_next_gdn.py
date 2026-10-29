"""Pinned full floating GDN block/cache oracle; no native/model TPS claim.

Actual9 original BF16 block weights, if requested, use synthetic activations
and paired raw-conv/FP32 recurrent past. Cache methods execute pinned upstream
code, not a guessed convolution-history adapter. No cache crop/record-past.
"""

import argparse
import ast
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import torch

from collect_hf_headers import fetch_range
from qualify_flash_next_delta import oracle
from qualify_flash_next_qsa import REVISION, UPSTREAM_SHA256
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_gdn import FlashNextGDN, GDNGeometry, GDNState


PREFIX = "model.language_model.layers.0.linear_attn"
CACHE_SHA256 = "36c4cd52d66f39fdb51fa2fc92cfc43e2e8b152e07f7c2a17186eac1e81237e7"


def references(modeling, cache_source):
    scope = oracle(modeling)
    names = {"apply_mask_to_padding_states", "causal_conv1d_fn", "causal_conv1d_update", "Qwen4ExpTextGatedDeltaNet"}
    selected = [n for n in ast.parse(Path(modeling).read_bytes()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned GDN block definitions missing")
    for node in selected:
        for n in ast.walk(node):
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
                n.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    exec(compile(module, str(modeling), "exec"), scope)
    data = Path(cache_source).read_bytes()
    if hashlib.sha256(data).hexdigest() != CACHE_SHA256:
        raise ValueError("Pinned upstream cache source hash differs")
    classes = []
    methods = {"get_conv_state", "get_recurrent_state", "update_conv_state", "update_recurrent_state", "is_recording_past"}
    for n in ast.parse(data).body:
        if isinstance(n, ast.ClassDef) and n.name in {"LinearAttentionCacheLayerMixin", "LinearAttentionLayer"}:
            classes.append(n)
        elif isinstance(n, ast.ClassDef) and n.name == "Cache":
            picked = [method for method in n.body if isinstance(method, ast.FunctionDef) and method.name in methods]
            if {m.name for m in picked} != methods:
                raise ValueError("Pinned Cache state methods missing")
            classes.append(ast.ClassDef(name="ReferenceCache", bases=[], keywords=[], body=picked, decorator_list=[]))
    if len(classes) != 3:
        raise ValueError("Pinned linear cache classes missing")
    scope.update(ABC=ABC, abstractmethod=abstractmethod, is_torchdynamo_compiling=lambda: False)
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[]))
    exec(compile(module, str(cache_source), "exec"), scope)
    return scope


def make_cache(reference, past, kernel):
    cache = reference["ReferenceCache"]()
    cache.layers = [reference["LinearAttentionLayer"]()]
    if past is not None:
        cache.update_conv_state(past.conv.clone(), 0, conv_kernel_size=kernel)
        cache.update_recurrent_state(past.recurrent.clone(), 0)
    return cache


def fixtures(g, real):
    rng = torch.Generator().manual_seed(351)
    shapes = ((1, 1, False), (2, 2, False), (2, 7, False), (2, 1, True), (2, 3, True), (1, 65, False))
    if real:
        shapes = ((1, 1, False), (1, 5, False), (1, 65, False), (8, 1, False), (8, 1, True), (2, 3, True))
    for batch, length, cached in shapes:
        x = torch.randn(batch, length, g.hidden_size, generator=rng) * 0.15
        past = None if not cached else GDNState(
            torch.randn(batch, g.conv_channels, g.conv_kernel, generator=rng) * 0.1,
            torch.randn(batch, g.value_heads, g.key_dim, g.value_dim, generator=rng) * 0.1, 11)
        for masked in (False, True):
            mask = None
            if masked:
                mask = torch.ones(batch, length, dtype=torch.bool)
                mask[0, ::2] = False
                if batch > 1:
                    mask[-1] = False
            yield f"B{batch}S{length}-{'cached' if cached else 'prefill'}-{'padding' if masked else 'unmasked'}", x, past, mask


def compare(reference, g, model, device, dtypes, real):
    report = []
    config = SimpleNamespace(hidden_size=g.hidden_size, linear_num_value_heads=g.value_heads,
        linear_num_key_heads=g.key_heads, linear_key_head_dim=g.key_dim, linear_value_head_dim=g.value_dim,
        linear_conv_kernel_dim=g.conv_kernel, hidden_act="silu", output_gate_type="sigmoid",
        rms_norm_eps=g.eps, layer_types=["linear_attention"])
    for dtype in dtypes:
        candidate = FlashNextGDN(g).to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(model.state_dict(), strict=True)
        expected_module = reference["Qwen4ExpTextGatedDeltaNet"](config, 0).to(device=device, dtype=dtype).eval()
        expected_module.load_state_dict(candidate.state_dict(), strict=True)
        for name, x, past, mask in fixtures(g, real):
            x = x.to(device=device, dtype=dtype)
            if past is not None:
                past = GDNState(past.conv.to(device=device, dtype=dtype), past.recurrent.to(device=device), past.tokens_seen)
            preserved = None if past is None else past.snapshot()
            mask = None if mask is None else mask.to(device=device)
            cache = make_cache(reference, past, g.conv_kernel)
            with torch.inference_mode():
                actual, proposed = candidate(x, past=past, attention_mask=mask)
                expected = expected_module(x, cache_params=cache, attention_mask=mask)
            if not torch.isfinite(actual).all():
                raise AssertionError(f"Nonfinite GDN output: {name}/{dtype}")
            for a, b in ((actual, expected), (proposed.conv, cache.get_conv_state(0)),
                         (proposed.recurrent, cache.get_recurrent_state(0))):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            if past is not None:
                assert torch.equal(past.conv, preserved.conv) and torch.equal(past.recurrent, preserved.recurrent)
            assert proposed.tokens_seen == (0 if past is None else past.tokens_seen) + x.shape[1]
            report.append(dict(case=name, dtype=str(dtype), device=str(device), actual_official_weights=real,
                core_mode="recurrent" if past is not None and x.shape[1] == 1 else "chunk64",
                output_exact=True, all_output_finite=True, conv_state_exact=True, fp32_recurrent_state_exact=True,
                input_past_unchanged=True, padding_semantics="masked inputs still update convolution/RNN state"))
    return report


def sample(headers, metadata, output):
    root = output / "official-gdn-samples"
    root.mkdir(exist_ok=True)
    specs = {}
    for path in headers.glob("*.json"):
        record = json.loads(path.read_text())
        if record["repo"] != "Qwen/Qwen3.8-Flash-Next" or record["revision"] != REVISION:
            raise ValueError("Original official GDN sample pin differs")
        for name, spec in record["header"].items():
            if name.startswith(PREFIX + "."):
                specs[name] = (record, spec)
    if len(specs) != 9 or any(s[1]["dtype"] != "BF16" for s in specs.values()):
        raise ValueError("Exactly9 original BF16 GDN tensors required")
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
                if p["sha256"] != hashlib.sha256(payload).hexdigest() or p["source_range"] != [offset+start, offset+end-1]:
                    raise ValueError("Existing GDN source piece differs")
            else:
                payload, total = fetch_range(f"https://huggingface.co/{record['repo']}/resolve/{REVISION}/{record['filename']}",
                                             offset+start, offset+end-1)
                if total != record["file_bytes"]:
                    raise ValueError("Source GDN shard size changed")
                p = dict(tensor=name, shard=record["filename"], source_range=[offset+start, offset+end-1],
                         bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                piece.write_bytes(payload)
                proof.write_text(json.dumps(p, indent=2) + "\n")
            if len(payload) != end-start:
                raise ValueError("Wrong GDN range size")
            data.extend(payload)
            segments.append(p)
            print(f"sampled {name} {end-lo}/{hi-lo}", flush=True)
        tensors[name] = torch.frombuffer(data, dtype=torch.bfloat16).reshape(spec["shape"]).clone()
    target = root / "model.safetensors"
    if target.exists():
        raise FileExistsError("Reuse finished GDN sample explicitly with --samples")
    save_file(tensors, target, metadata={"scope": "9 actual GDN block tensors, not complete decoder/model"})
    (root / "config.json").write_bytes((metadata / "config.json").read_bytes())
    (root / "ranges.json").write_text(json.dumps(dict(revision=REVISION, segments=segments,
        source_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2) + "\n")
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--upstream-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples")
    p.add_argument("--metadata")
    p.add_argument("--headers")
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
    reference = references(a.upstream_modeling, a.upstream_cache)
    torch.manual_seed(351)
    g = GDNGeometry(8, 2, 6, 4, 3, 4)
    model = FlashNextGDN(g)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter)*0.1 + (1 if name == "norm.weight" else 0))
    report = compare(reference, g, model, device, dtypes, False)
    info = None
    if not a.synthetic_only:
        if a.samples:
            root = Path(a.samples)
        elif a.allow_range_samples and a.headers and a.metadata and device.type == "cpu":
            root = sample(Path(a.headers), Path(a.metadata), output)
        else:
            p.error("Explicit samples or bounded CPU source ranges required")
        provenance = json.loads((root / "ranges.json").read_text())
        if provenance["revision"] != REVISION or hashlib.sha256((root/"model.safetensors").read_bytes()).hexdigest() != provenance["artifact_sha256"]:
            raise ValueError("Actual GDN sample provenance differs")
        checkpoint = inspect_checkpoint(root)
        g = GDNGeometry.from_dict(dict(checkpoint.config))
        model = FlashNextGDN.from_checkpoint(checkpoint, g, PREFIX)
        report.extend(compare(reference, g, model, device, dtypes, True))
        info = dict(tensors=9, bytes=provenance["source_bytes"], binding_tested=True,
                    caveat="Actual layer0 GDN weights, synthetic activations/conv/recurrent history, not a decoder/model sequence")
    result = dict(scope="Floating full GDN block and pinned ordinary cache correctness only", cases=report,
        official_samples=info, upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256,
        original_revision=REVISION, torch=str(torch.__version__),
        mode_contract="chunk64 prefill/multi-token cached, recurrent cached-single; no recording/export branch",
        cache_contract="Pinned actual LinearAttentionLayer plus five original Cache methods; only layer-list construction is a harness",
        state_layout="raw conv [B,C,K], FP32 recurrent [B,HV,DK,DV]; no automatic commit/crop",
        native_kernel_implemented=False, whole_model_tested=False, throughput_measured=False)
    (output/"qualification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
