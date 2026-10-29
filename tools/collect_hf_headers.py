"""Collect pinned HF safetensors headers, never complete weight payloads.

This is an intake utility, not a verified checkpoint downloader. Range responses
are bounded and validated; header hashes do not authenticate tensor payloads.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import urllib.request


def fetch_range(url, start, end, *, timeout=40):
    length = end - start + 1
    if start < 0 or not 0 < length <= 16 * 2**20:
        raise ValueError("Invalid/beyond-limit metadata range")
    # A distinct query prevents intermediary caches from mixing range requests.
    url += f"?venturi_header_range={start}-{end}"
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or not match:
                    raise ValueError("Server did not honor a bounded byte range")
                lo, hi, size = map(int, match.groups())
                if (lo, hi) != (start, end) or hi >= size:
                    raise ValueError("Mismatched HTTP Content-Range")
                data = response.read(length + 1)
                if len(data) != length:
                    raise ValueError("Truncated/overlong metadata range")
                return data, size
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def collect(repo, revision, index, output, *, workers=4):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Expected an explicit Hugging Face repo id")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin a full immutable HF revision")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Nonempty safetensors weight_map required")
    names = sorted(set(mapping.values()))
    if any(not re.fullmatch(r"[\w.-]+\.safetensors", n) for n in names):
        raise ValueError("Only direct safetensors shard filenames are supported")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def one(name):
        target = output / (name + ".json")
        if target.exists():
            record = json.loads(target.read_text())
            if record.get("repo") != repo or record.get("revision") != revision:
                raise ValueError("Cached header provenance differs")
            return name
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        prefix, file_bytes = fetch_range(url, 0, 7)
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= 16 * 2**20 or 8 + header_bytes > file_bytes:
            raise ValueError("Invalid safetensors header length")
        header, again = fetch_range(url, 8, 7 + header_bytes)
        if again != file_bytes:
            raise ValueError("Shard length changed across requests")
        raw = json.loads(header)
        if not isinstance(raw, dict):
            raise ValueError("Safetensors header must be an object")
        record = dict(
            repo=repo, revision=revision, filename=name,
            file_bytes=file_bytes, header_bytes=header_bytes,
            header_sha256=hashlib.sha256(header).hexdigest(), header=raw,
            scope="HTTP header ranges only; tensor payloads not downloaded/verified",
        )
        temp = target.with_suffix(".json.part")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(target)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in pool.map(one, names):
            print(name, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    a = p.parse_args()
    collect(a.repo, a.revision, json.loads(Path(a.index).read_text()), a.output,
            workers=a.workers)


if __name__ == "__main__":
    main()
