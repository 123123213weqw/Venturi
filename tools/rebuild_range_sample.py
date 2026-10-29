"""Rebuild a bounded exact local sample artifact from pinned HF byte ranges.

Transport helper only, not a model downloader/exporter. Preserve the supplied
safetensors header byte-for-byte; authenticate every bounded segment and final
artifact against the local qualification hashes. Completed segments can resume.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct

from collect_hf_headers import fetch_range


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            sha.update(block)
    return sha.hexdigest()


def rebuild(manifest, output):
    raw = manifest.read_bytes()
    m = json.loads(raw)
    header = base64.b64decode(m["header_base64"], validate=True)
    if len(header) < 10 or struct.unpack("<Q", header[:8])[0] != len(header) - 8:
        raise ValueError("Exact safetensors header required")
    if not len(header) < m["bytes"] <= 256 * 2**20 or not re.fullmatch(r"[0-9a-f]{64}", m["sha256"]):
        raise ValueError("Bounded sample artifact/whole SHA required")
    specs = json.loads(header[8:])
    expected = {k for k in specs if k != "__metadata__"}
    if len(m["segments"]) > 128 or {s["key"] for s in m["segments"]} != expected or len(m["segments"]) != len(expected):
        raise ValueError("Exactly one source range per sample tensor required")
    spans = []
    for s in m["segments"]:
        spec = specs[s["key"]]
        lo, hi = spec["data_offsets"]
        if s["target_range"] != [len(header) + lo, len(header) + hi - 1] or hi <= lo or hi - lo > 16 * 2**20:
            raise ValueError("Bounded sample destination geometry mismatch")
        if s["source_range"][1] - s["source_range"][0] != hi - lo - 1:
            raise ValueError("Sample source range length differs")
        if not re.fullmatch(r"https://huggingface\.co/[\w.-]+/[\w.-]+/resolve/[0-9a-f]{40}/[\w.-]+\.safetensors", s["url"]):
            raise ValueError("Explicit pinned HF safetensors source required")
        if not re.fullmatch(r"[0-9a-f]{64}", s["sha256"]):
            raise ValueError("Source range SHA required")
        spans.append((len(header) + lo, len(header) + hi))
    previous = len(header)
    for start, end in sorted(spans):
        if start != previous:
            raise ValueError("Sample payload spans must cover the artifact without gaps/overlaps")
        previous = end
    if previous != m["bytes"]:
        raise ValueError("Sample artifact length differs")
    if output.exists():
        if digest(output) != m["sha256"]:
            raise ValueError("Existing sample artifact differs; refuse to replace")
        print("existing sample verified", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".building")
    if not partial.exists():
        with partial.open("wb") as f:
            f.write(header)
            f.truncate(m["bytes"])
    with partial.open("r+b") as f:
        if partial.stat().st_size != m["bytes"] or f.read(len(header)) != header:
            raise ValueError("Existing partial sample differs")
        for s in m["segments"]:
            start, end = s["target_range"]
            f.seek(start)
            if hashlib.sha256(f.read(end - start + 1)).hexdigest() == s["sha256"]:
                print("cached segment", s["key"], flush=True)
                continue
            data, source_size = fetch_range(s["url"], *s["source_range"])
            if source_size != s["source_file_bytes"] or hashlib.sha256(data).hexdigest() != s["sha256"]:
                raise ValueError("Source segment size/SHA mismatch")
            f.seek(start)
            f.write(data)
            f.flush()
            print("verified segment", s["key"], len(data), flush=True)
    if digest(partial) != m["sha256"]:
        raise ValueError("Rebuilt sample whole-file SHA mismatch")
    partial.replace(output)
    print("exact sample rebuilt", m["bytes"], m["sha256"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    rebuild(a.manifest, a.output)


if __name__ == "__main__":
    main()
