"""Explicit byte-preserving artifact staging, e.g. to a Linux tmpfs directory.

This trades host RAM and one up-front copy for faster repeated native loads.
It does not change GPU weight representation or speed up resident decoding.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from .ninfer import NInferArtifact


def _identity(artifact):
    result = []
    for file in artifact.files:
        stat = file.path.stat()
        result.append(
            {
                "path": str(file.path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "device": stat.st_dev,
            }
        )
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def stage_artifact(checkpoint, directory, *, reuse=False):
    started = perf_counter()
    artifact = NInferArtifact(checkpoint)
    if any(file.path.name == "staging.json" for file in artifact.files):
        raise ValueError("Artifact filename staging.json is reserved for the staging report")
    identity = _identity(artifact)
    parent = Path(directory).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = parent / f"venturi-ninfer-{artifact.artifact_id}-{key}"
    report_path = destination / "staging.json"
    if destination.exists():
        if not reuse:
            raise ValueError(
                "Staged artifact already exists; pass reuse=True for verified reuse"
            )
        report = json.loads(report_path.read_text())
        if report["source_identity"] != identity:
            raise ValueError("Staged artifact source identity changed")
        for file, expected in report["sha256"].items():
            if (
                Path(file).name != file
                or file in (".", "..")
                or _sha256(destination / file) != expected
            ):
                raise ValueError("Staged artifact checksum mismatch")
        NInferArtifact(destination / artifact.path.name)
        return {
            **report,
            "reused": True,
            "validation_seconds": perf_counter() - started,
        }
    total = sum(f.path.stat().st_size for f in artifact.files)
    if shutil.disk_usage(parent).free < total + 64 * 2**20:
        raise ValueError("Staging directory lacks artifact size plus 64 MiB headroom")
    temporary = Path(tempfile.mkdtemp(prefix=".venturi-stage-", dir=parent))
    try:
        hashes = {}
        for file in artifact.files:
            target = temporary / file.path.name
            digest = hashlib.sha256()
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with file.path.open("rb") as source, os.fdopen(fd, "wb") as output:
                while block := source.read(8 * 2**20):
                    output.write(block)
                    digest.update(block)
                output.flush()
                os.fsync(output.fileno())
            hashes[target.name] = digest.hexdigest()
            if _sha256(target) != hashes[target.name]:
                raise ValueError("Staging copy failed byte-level verification")
        if _identity(artifact) != identity:
            raise ValueError("Source artifact changed during staging")
        staged = NInferArtifact(temporary / artifact.path.name)
        if staged.artifact_id != artifact.artifact_id:
            raise ValueError("Staged artifact identity mismatch")
        report = {
            "source": str(artifact.path),
            "path": str(destination / artifact.path.name),
            "source_identity": identity,
            "sha256": hashes,
            "bytes": total,
            "stage_seconds": perf_counter() - started,
            "reused": False,
            "scope": "Byte-exact copy into explicitly selected storage; host RAM/disk cost and staging time are additional. No quantization change, no resident-decode speedup, and no automatic eviction.",
        }
        (temporary / "staging.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
