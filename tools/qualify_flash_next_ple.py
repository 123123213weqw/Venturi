"""Pinned floating PLE conditioning/dilated-conv oracle; not full-model TPS.

Embedding producer is supplied bounded rows, not a95GiB table allocation.
Hash/CPU reader and EOS/commit behavior have separate tests; no full real table
or generated-token integration is claimed by synthetic embedding-row fixtures.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import torch
from torch import nn

from collect_hf_headers import fetch_range
from qualify_flash_next_gdn import CACHE_SHA256, references
from qualify_flash_next_qsa import REVISION, UPSTREAM_SHA256
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_ple import FlashNextPLE, PLEGeometry, PLEState


PREFIX = "model.language_model.layers.1.ple"


class SuppliedRows(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.rows = None

    def forward(self, input_ids, cache):
        if self.rows is None or self.rows.shape[:2] != input_ids.shape:
            raise ValueError("PLE oracle requires explicitly supplied matching rows")
        return self.rows


def reference(modeling, cache):
    scope = references(modeling, cache)
    names = {"Qwen4ExpTextRMSNorm", "Qwen4ExpTextPLELayer"}
    selected = [n for n in ast.parse(Path(modeling).read_bytes()).body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned PLE definitions missing")
    for n in selected:
        n.decorator_list = []
    scope.update(math=math, Qwen4ExpTextNGramEmbedding=SuppliedRows)
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    exec(compile(module, str(modeling), "exec"), scope)
    return scope


def make_cache(ref, past, state_length):
    cache = ref["ReferenceCache"]()
    cache.layers = [ref["LinearAttentionLayer"](number_of_states=3)]
    if past is not None:
        cache.update_conv_state(past.conv.clone(), 0, state_idx=1, conv_kernel_size=state_length)
    return cache


def fixtures(g, real=False):
    rng = torch.Generator().manual_seed(1251)
    shapes = ((1, 1, False), (2, 3, False), (2, 13, False), (2, 1, True), (2, 4, True))
    if real:
        shapes = ((1, 1, False), (1, 5, False), (1, 13, False), (8, 1, False), (8, 1, True), (2, 3, True))
    for batch, length, cached in shapes:
        x = torch.randn(batch, length, g.streams*g.hidden_size, generator=rng)*0.1
        rows = torch.randn(batch, length, g.embedding_dim, generator=rng)*0.1
        past = None if not cached else PLEState(torch.randn(batch, g.streams*g.hidden_size, g.state_length,
                                                           generator=rng)*0.1, 17)
        for masked in (False, True):
            mask = None
            if masked:
                mask = torch.ones(batch, length, dtype=torch.bool)
                mask[0, ::2] = False
                if batch > 1:
                    mask[-1] = False
            yield f"B{batch}S{length}-{'cached' if cached else 'prefill'}-{'masked' if masked else 'unmasked'}", x, rows, past, mask


def compare(ref, g, model, device, dtypes, real=False):
    report = []
    config = SimpleNamespace(hidden_size=g.hidden_size, hc_count=g.streams, ple_embed_dim=g.embedding_dim,
                             ple_conv_kernel_size=g.conv_kernel, ngram_size=g.dilation, rms_norm_eps=g.eps)
    for dtype in dtypes:
        candidate = FlashNextPLE(g).to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(model.state_dict(), strict=True)
        oracle = ref["Qwen4ExpTextPLELayer"](config, 0, 0).to(device=device, dtype=dtype).eval()
        oracle.load_state_dict(candidate.state_dict(), strict=True)
        for name, x, rows, past, mask in fixtures(g, real):
            x, rows = (t.to(device=device, dtype=dtype) for t in (x, rows))
            past = None if past is None else PLEState(past.conv.to(device=device, dtype=dtype), past.tokens_seen)
            retained = None if past is None else past.snapshot()
            cache = make_cache(ref, past, g.state_length)
            oracle.ple_embedding.rows = rows
            mask = None if mask is None else mask.to(device=device)
            with torch.inference_mode():
                actual, proposed = candidate(x, rows, past=past, conv_mask=mask)
                expected = oracle(x, torch.zeros(x.shape[:2], device=device, dtype=torch.long), cache, conv_mask=mask)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(proposed.conv, cache.get_conv_state(0, state_idx=1), atol=0, rtol=0)
            if not torch.isfinite(actual).all() or past is not None and not torch.equal(retained.conv, past.conv):
                raise AssertionError("PLE nonfinite output or mutated past")
            report.append(dict(case=name, dtype=str(dtype), device=str(device), actual_official_weights=real,
                output_exact=True, conv_state_exact=True, all_output_finite=True, input_past_unchanged=True,
                embedding_rows="synthetic supplied rows; hash/reader separately qualified"))
    return report


def sample(headers, metadata, output):
    root = output/"official-ple-samples"
    root.mkdir(exist_ok=True)
    specs = {}
    for path in headers.glob("*.json"):
        record = json.loads(path.read_text())
        if record["repo"] != "Qwen/Qwen3.8-Flash-Next" or record["revision"] != REVISION:
            raise ValueError("Original PLE source pin differs")
        for name, spec in record["header"].items():
            if name.startswith(PREFIX+".") and not name.startswith(PREFIX+".ple_embedding."):
                specs[name] = record, spec
    if len(specs) != 6 or any(v[1]["dtype"] != "BF16" for v in specs.values()):
        raise ValueError("Exactly6 original BF16 PLE conditioning tensors required")
    tensors, segments = {}, []
    for name, (record, spec) in sorted(specs.items()):
        lo, hi = spec["data_offsets"]
        offset, data = 8+record["header_bytes"], bytearray()
        for start in range(lo, hi, 8*2**20):
            end = min(start+8*2**20, hi)
            piece = root/(name.removeprefix(PREFIX+".")+f".{start-lo}-{end-lo}.bin")
            proof = piece.with_suffix(".json")
            if piece.exists():
                payload = piece.read_bytes()
                p = json.loads(proof.read_text())
                if p["sha256"] != hashlib.sha256(payload).hexdigest() or p["source_range"] != [offset+start, offset+end-1]:
                    raise ValueError("Existing PLE source piece differs")
            else:
                payload, total = fetch_range(f"https://huggingface.co/{record['repo']}/resolve/{REVISION}/{record['filename']}",
                                             offset+start, offset+end-1)
                if total != record["file_bytes"]:
                    raise ValueError("PLE source shard size differs")
                p = dict(tensor=name, shard=record["filename"], source_range=[offset+start, offset+end-1],
                         bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                piece.write_bytes(payload)
                proof.write_text(json.dumps(p, indent=2)+"\n")
            if len(payload) != end-start:
                raise ValueError("Incorrect PLE source range length")
            data.extend(payload)
            segments.append(p)
            print(f"sampled {name} {end-lo}/{hi-lo}", flush=True)
        tensors[name] = torch.frombuffer(data, dtype=torch.bfloat16).reshape(spec["shape"]).clone()
    target = root/"model.safetensors"
    if target.exists():
        raise FileExistsError("Reuse completed PLE sample explicitly")
    save_file(tensors, target, metadata={"scope": "6 PLE conditioning weights, excludes ngram table/full decoder"})
    (root/"config.json").write_bytes((metadata/"config.json").read_bytes())
    (root/"ranges.json").write_text(json.dumps(dict(revision=REVISION, segments=segments,
        source_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2)+"\n")
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
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1536*2**20/torch.cuda.get_device_properties(device).total_memory, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    ref = reference(a.upstream_modeling, a.upstream_cache)
    torch.manual_seed(1251)
    g = PLEGeometry(8, 2, 16, 4, 3)
    model = FlashNextPLE(g)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter)*0.1)
    report = compare(ref, g, model, device, dtypes)
    if a.samples:
        root = Path(a.samples)
    elif a.allow_range_samples and a.headers and a.metadata and device.type == "cpu":
        root = sample(Path(a.headers), Path(a.metadata), output)
    else:
        p.error("Explicit samples or bounded CPU ranges required")
    provenance = json.loads((root/"ranges.json").read_text())
    if provenance["revision"] != REVISION or hashlib.sha256((root/"model.safetensors").read_bytes()).hexdigest() != provenance["artifact_sha256"]:
        raise ValueError("PLE source sample provenance differs")
    cp = inspect_checkpoint(root)
    g = PLEGeometry.from_dict(dict(cp.config))
    model = FlashNextPLE.from_checkpoint(cp, g, PREFIX)
    report.extend(compare(ref, g, model, device, dtypes, True))
    result = dict(scope="Floating PLE conditioning and short-conv correctness only, supplied embedding rows",
        cases=report, original_revision=REVISION, upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256,
        official_source_bytes=provenance["source_bytes"], official_tensor_count=6,
        embedding_harness="Original PLE forward, only huge embedding producer replaced by explicit rows; no token-history slot2 assertion",
        cache_contract="Pinned LinearAttentionLayer(3 states) and five Cache methods, short-conv slot1", torch=str(torch.__version__),
        whole_table_tested=False, whole_model_tested=False, native_kernel_implemented=False, throughput_measured=False)
    (output/"qualification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
