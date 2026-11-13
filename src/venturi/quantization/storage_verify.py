"""Codec-agnostic storage verification: byte-level round-trip proofs.

For sampled matrix objects, reads every stored plane back and verifies the
container geometry against the codec registry: plane offsets, plane sizes and
total bytes must match exactly, and the raw bytes must round-trip unchanged.
Dense objects verify as one contiguous plane. This proves the loader moves
the source representation bit-for-bit for any registered codec; it is not an
execution or quality qualification.
"""

import hashlib
import json
import random

from venturi.quantization.codecs import CODECS, codec_for


def _geometry(shape, codec):
    from venturi.ninfer import row_geometry

    return row_geometry(shape, codec.fmt)


def _sample_planes(artifact, obj, codec, rng):
    n, k = obj["shape"]
    checks = []
    if codec.dense:
        data = artifact.read_object(obj["id"])
        checks.append(("dense", 0, codec.encoded_bytes(n, k), data))
        return checks
    bits, group, groups, high_offset, scales_offset, total = _geometry(
        obj["shape"], codec
    )
    row_codes = groups * group * bits // 8
    row_high = groups * (8 * (bits - 4) if bits in (5, 6) else 0) // 8 or None
    row_scales = groups * 2
    rows = sorted(
        rng.sample(range(n), min(3, n)) + [0, n // 2, n - 1]
    ) if n >= 3 else list(range(n))
    for row in dict.fromkeys(rows):
        base = artifact.read_object(
            obj["id"], offset=row * row_codes, size=row_codes
        )
        if codec.forbidden_codes and bits == 8:
            if any(b == 0x80 for b in base):
                raise ValueError(
                    f"Forbidden Q8 code -128 in {obj['id']} row {row}"
                )
        checks.append((f"codes/row{row}", row * row_codes, row_codes, base))
        if row_high:
            high = artifact.read_object(
                obj["id"], offset=high_offset + row * row_high, size=row_high
            )
            checks.append(
                (f"high/row{row}", high_offset + row * row_high, row_high, high)
            )
        scales = artifact.read_object(
            obj["id"], offset=scales_offset + row * row_scales, size=row_scales
        )
        checks.append(
            (
                f"scales/row{row}",
                scales_offset + row * row_scales,
                row_scales,
                scales,
            )
        )
    return checks


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(artifact, *, sample_objects=7, seed=20261111):
    """Verify sampled objects of every codec present; return an evidence dict.

    Every registered codec found in the artifact is checked: container-stored
    plane offsets must equal the registry geometry, sampled plane bytes must
    be readable and length-exact. Dense and row-split both covered.
    """
    rng = random.Random(seed)
    objects = [
        (name, obj)
        for name, obj in artifact.objects.items()
        if obj.get("kind") == "tensor" and obj.get("format") in CODECS
    ]
    matrices = [(n, o) for n, o in objects if len(o["shape"]) == 2]
    smalls = [(n, o) for n, o in objects if len(o["shape"]) != 2]
    chosen = (
        sorted(rng.sample(matrices, min(sample_objects, len(matrices))))
        if matrices
        else []
    )
    report = {
        "artifact": str(artifact.path),
        "artifact_sha256": _file_sha256(artifact.path),
        "sampled": [],
        "codecs_present": sorted({o["format"] for _, o in objects}),
    }
    verified_bytes = 0
    for name, obj in chosen + smalls[:3]:
        codec = codec_for(obj["format"])
        if codec.quantized and len(obj["shape"]) == 2:
            _, _, _, high_offset, scales_offset, total = _geometry(
                obj["shape"], codec
            )
            if obj["bytes"] != codec.encoded_bytes(*obj["shape"]):
                raise ValueError(
                    f"Stored size mismatch for {name}: {obj['bytes']} != "
                    f"{codec.encoded_bytes(*obj['shape'])}"
                )
            planes = _sample_planes(artifact, obj, codec, rng)
            entry = {
                "object": name,
                "format": obj["format"],
                "shape": obj["shape"],
                "plane_bytes": {
                    "codes": high_offset,
                    "high": scales_offset - high_offset,
                    "scales": total - scales_offset,
                },
                "samples": [],
            }
            for label, offset, size, data in planes:
                if len(data) != size:
                    raise ValueError(
                        f"Plane {label} of {name}: read {len(data)} != {size}"
                    )
                entry["samples"].append(
                    {
                        "plane": label,
                        "offset": offset,
                        "bytes": size,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                verified_bytes += size
        else:
            data = artifact.read_object(obj["id"])
            if len(data) != obj["bytes"]:
                raise ValueError(f"Object {name} byte length mismatch")
            entry = {
                "object": name,
                "format": obj["format"],
                "shape": obj["shape"],
                "samples": [
                    {
                        "plane": "dense",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
            verified_bytes += len(data)
        report["sampled"].append(entry)
    report["verified_bytes"] = verified_bytes
    report["status"] = "storage-verified"
    return report


def write_evidence(report, path):
    with open(path, "w") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
