"""Bounded raw-tensor storage checks, independent of Qwen roles and file aliases."""

import hashlib, json, random
from venturi.container.safetensors import SafetensorsContainer


def verify_checkpoint(path, *, sample_tensors=7, seed=20261113):
    if type(sample_tensors) is not int or sample_tensors < 0:
        raise ValueError("Sample count must be a nonnegative integer")
    source = SafetensorsContainer(path)
    names = source.list_tensors()
    rng = random.Random(seed)
    chosen = sorted(rng.sample(names, min(sample_tensors, len(names))))
    report = {"checkpoint": str(source.path), "sampled": [], "verified_bytes": 0}
    for name in chosen:
        info = source.tensor_info(name)
        h = hashlib.sha256()
        for start in range(0, info.byte_size, 8 << 20):
            data = source.read_bytes(
                name, start, min(8 << 20, info.byte_size - start), max_bytes=8 << 20
            )
            h.update(data)
            report["verified_bytes"] += len(data)
        report["sampled"].append(
            {
                "tensor": name,
                "shape": info.shape,
                "dtype": info.dtype,
                "bytes": info.byte_size,
                "sha256": h.hexdigest(),
            }
        )
    source.check_source()
    report["status"] = "storage-readable"
    report["scope"] = (
        "Header/schema and bounded read integrity; observed digests are not compared with a trusted hash"
    )
    return report


def write_evidence(report, path):
    from pathlib import Path

    Path(path).write_text(json.dumps(report, indent=2) + "\n")
