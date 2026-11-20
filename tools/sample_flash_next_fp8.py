"""Bounded official FP8 PLE row/scale audit, not full-checkpoint qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from collect_hf_headers import fetch_range


def e4m3fn(code):
    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (math.ldexp(mantissa, -9) if exponent == 0 else
                   math.ldexp(1 + mantissa / 8, exponent - 7))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp8-metadata", required=True)
    p.add_argument("--bf16-samples", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source, original, output = Path(a.fp8_metadata), Path(a.bf16_samples), Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    api = json.loads((source / "hf-model-api.json").read_text())
    records = [json.loads(f.read_text()) for f in (source / "selected-headers").glob("*.json")]
    tensors = {n: (r, t) for r in records for n, t in r["header"].items() if n != "__metadata__"}
    scale_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    record, tensor = tensors[scale_name]
    if tensor["dtype"] != "BF16" or tensor["shape"] != [1]:
        raise ValueError("Expected official scalar BF16 PLE scale")
    offset = 8 + record["header_bytes"]
    lo, hi = tensor["data_offsets"]
    raw, size = fetch_range(f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{record['filename']}",
                           offset + lo, offset + hi - 1)
    if size != record["file_bytes"]:
        raise ValueError("Scale file size differs")
    (output / "weight_scale.bin").write_bytes(raw)
    scale = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
    if not torch.isfinite(scale.float()).all() or scale.item() <= 0:
        raise ValueError("Invalid FP8 PLE scale")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    base = json.loads((original / "qualification.json").read_text())
    samples = []
    differences, originals = [], []
    for sample in base["sampled_rows"]:
        name, row = sample["tensor"], sample["row"]
        filename = index[name]
        cache = source / "selected-headers" / (filename + ".json")
        if not cache.exists():
            prefix, size = fetch_range(f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{filename}", 0, 7)
            length = int.from_bytes(prefix, "little")
            if not 2 <= length <= 16 * 2**20:
                raise ValueError("Invalid header length")
            header, again = fetch_range(f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{filename}",
                                       8, 7 + length)
            if size != again:
                raise ValueError("Header backing changed")
            cache.write_text(json.dumps(dict(repo=api['id'], revision=api['sha'], filename=filename,
                                             file_bytes=size, header_bytes=length,
                                             header_sha256=hashlib.sha256(header).hexdigest(),
                                             header=json.loads(header)), indent=2) + "\n")
        record = json.loads(cache.read_text())
        tensor = record["header"][name]
        if tensor["dtype"] != "F8_E4M3" or tensor["shape"] != [2500012, 160]:
            raise ValueError("Unexpected official FP8 ngram geometry")
        lo = 8 + record["header_bytes"] + tensor["data_offsets"][0] + row * 160
        raw, size = fetch_range(f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{filename}", lo, lo + 159)
        if size != record["file_bytes"]:
            raise ValueError("Row backing file size differs")
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).to(torch.bfloat16) * scale
        independent = torch.tensor([e4m3fn(c) for c in raw], dtype=torch.bfloat16) * scale
        if not torch.isfinite(decoded.float()).all() or not torch.equal(
            decoded.view(torch.int16), independent.view(torch.int16)
        ):
            raise AssertionError("FP8 row decode differs from independent E4M3FN formula")
        name_out = sample["output"]
        (output / name_out).write_bytes(raw)
        bf16 = torch.frombuffer(bytearray((original / name_out).read_bytes()), dtype=torch.bfloat16).float()
        differences.append(decoded.float() - bf16)
        originals.append(bf16)
        samples.append(dict(tensor=name, row=row, filename=filename, absolute_range=[lo, lo + 159],
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            decoded_bf16_bits_exact=True, output=name_out))
    delta, baseline = torch.cat(differences), torch.cat(originals)
    report = dict(repo=api['id'], revision=api['sha'], encoding="F8_E4M3 + one BF16 weight_scale",
                  weight_scale=scale.item(), row_samples=samples, decoded_bf16_bits_exact=True,
                  source_ngram_code_bytes=51200245760, source_ngram_scale_bytes=2,
                  raw_bf16_ngram_bytes=102400491520, full_model_tested=False,
                  cross_artifact_sample_error=dict(elements=delta.numel(), max_abs=delta.abs().max().item(),
                                                  rms=delta.square().mean().sqrt().item(),
                                                  relative_rms=(delta.square().mean() / baseline.square().mean()).sqrt().item()),
                  caveat="16 corresponding rows only; not full-source verification, perplexity or model-quality parity")
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "row_samples"}, indent=2))


if __name__ == "__main__":
    main()
