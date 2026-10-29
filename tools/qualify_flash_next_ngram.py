"""Compare PLE hashes to a pinned upstream CPU oracle and read bounded real rows.

Only selected class/functions from the explicitly pinned source are executed.
The enormous upstream embedding is constructed on meta, replaced by an index
recorder, and never materialized. Samples are not a verified full checkpoint.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace

import torch
from torch import nn

from collect_hf_headers import fetch_range
from venturi.flash_next import FlashNextManifest
from venturi.models.flash_next_ngram import NGramRequest


UPSTREAM_SHA256 = "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"


def reference_class(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Explicit upstream reference hash differs")
    tree = ast.parse(data)
    names = {"_splitmix64", "_build_layer_multipliers", "_is_prime", "_find_nth_prime_after",
             "Qwen4ExpTextNGramEmbedding"}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned reference definitions missing")
    scope = dict(torch=torch, nn=nn, math=math, _MASK64=2**64 - 1,
                 _SPLITMIX_GAMMA=0x9E3779B97F4A7C15, _SPLITMIX_M1=0xBF58476D1CE4E5B9,
                 _SPLITMIX_M2=0x94D049BB133111EB, _PRIME_1=10007)
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["Qwen4ExpTextNGramEmbedding"]


class IndexRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.zeros(1))

    def forward(self, ids):
        return ids.unsqueeze(-1)


class ContextCache:
    def __init__(self):
        self.state = None

    def get_conv_state(self, *args, **kwargs):
        return self.state

    def update_conv_state(self, ids, *args, conv_kernel_size, **kwargs):
        history = ids if self.state is None else torch.cat([self.state, ids], dim=-1)
        self.state = history[:, -conv_kernel_size:].clone()
        return self.state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--headers", required=True)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-range-samples", action="store_true")
    a = p.parse_args()
    if not a.allow_range_samples:
        p.error("Real source sampling needs explicit --allow-range-samples")
    root, headers, output = Path(a.metadata), Path(a.headers), Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = FlashNextManifest.from_headers(root, headers)
    c = manifest.config
    records = {record["filename"]: record for record in (
        json.loads(f.read_text()) for f in headers.glob("*.json"))}
    tensors = {name: (r, spec) for r in records.values() for name, spec in r["header"].items()
               if name != "__metadata__"}

    def sample(name, row=None):
        record, spec = tensors[name]
        lo, hi = spec["data_offsets"]
        if row is not None:
            width = spec["shape"][1] * 2
            if spec["dtype"] != "BF16" or not 0 <= row < spec["shape"][0]:
                raise ValueError("Invalid sample geometry")
            lo += row * width
            hi = lo + width
        offset = 8 + record["header_bytes"]
        url = f"https://huggingface.co/{manifest.repo}/resolve/{manifest.revision}/{record['filename']}"
        raw, size = fetch_range(url, offset + lo, offset + hi - 1)
        if size != record["file_bytes"]:
            raise ValueError("Sample backing file size differs")
        return raw, dict(tensor=name, row=row, shard=record["filename"],
                         absolute_range=[offset + lo, offset + hi - 1], bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest())

    prefix = f"model.language_model.layers.{c.ple_layers[0] - 1}.ple.ple_embedding"
    raw, multiplier_record = sample(prefix + ".layer_multipliers")
    (output / "layer_multipliers.bin").write_bytes(raw)
    multipliers = struct.unpack("<" + "q" * c.ngram_size, raw)
    reference = reference_class(a.upstream_modeling)
    raw_config = json.loads((root / "config.json").read_text())["text_config"]
    raw_config.setdefault("seed", 1234)
    ref_config = SimpleNamespace(**raw_config)

    def make_reference():
        with torch.device("meta"):
            ref = reference(ref_config, c.ple_embed_dim, c.ple_layers[0] - 1, 0)
        ref.layer_multipliers = nn.Buffer(torch.tensor(multipliers, dtype=torch.long))
        sizes, offsets, _ = c.ngram_geometry()
        ref.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(sizes, dtype=torch.long))
        ref.ngram_heads_offsets = nn.Buffer(torch.tensor(offsets, dtype=torch.long))
        ref.ngram_embedding = IndexRecorder()
        return ref

    rng = random.Random(8912)
    cases = [[0], [c.eos_token_id], [0, c.eos_token_id, c.eos_token_id, c.vocab_size - 1, 2]]
    cases.append([c.eos_token_id if i % 7 == 0 else rng.randrange(c.vocab_size) for i in range(67)])
    comparisons = []
    for case, ids in enumerate(cases):
        for chunk in (1, 2, 3, 8, len(ids)):
            ref, cache = make_reference(), ContextCache()
            request, actual, expected = NGramRequest(c, multipliers), [], []
            for start in range(0, len(ids), chunk):
                part = ids[start:start + chunk]
                proposal = request.preview(part)
                actual.extend(proposal.rows)
                request.commit(proposal)
                expected.extend(ref(torch.tensor([part]), cache)[0].tolist())
            if actual != [tuple(row) for row in expected]:
                raise AssertionError("Native intake hash differs from pinned reference")
            comparisons.append(dict(case=case, chunk=chunk, tokens=len(ids), exact=True))

    proposal = NGramRequest(c, multipliers).preview([c.vocab_size - 1])
    _, _, padded = c.ngram_geometry()
    rows_per_part = padded // c.split_ngram_parts
    samples = []
    for head, global_row in enumerate(proposal.rows[0]):
        part, row = divmod(global_row, rows_per_part)
        raw, record = sample(f"{prefix}.ngram_embedding.shard_{part}.weight", row)
        values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        if not torch.isfinite(values.float()).all():
            raise AssertionError("Nonfinite sampled original embedding row")
        filename = f"head-{head:02d}-row-{global_row}.bin"
        (output / filename).write_bytes(raw)
        samples.append(dict(record, global_row=global_row, output=filename,
                            finite=True, elements=values.numel()))
    report = dict(repo=manifest.repo, revision=manifest.revision,
                  upstream_modeling_sha256=UPSTREAM_SHA256,
                  multipliers=list(multipliers), multiplier_sample=multiplier_record,
                  hash_comparisons=comparisons, sampled_rows=samples,
                  reference_hash_exact=True, full_model_tested=False,
                  payload_scope="24-byte multipliers and 16 BF16 rows only; full checkpoint not downloaded/verified",
                  gpu_used=False)
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(reference_hash_exact=True, comparisons=len(comparisons),
                         real_rows=len(samples), full_model_tested=False), indent=2))


if __name__ == "__main__":
    main()
