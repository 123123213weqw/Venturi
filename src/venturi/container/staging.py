"""Explicit verified copy of a standard checkpoint; no format-specific cache ABI."""

import hashlib, os, shutil, tempfile
from pathlib import Path
from time import perf_counter
from .safetensors import SafetensorsContainer


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_checkpoint(checkpoint, directory, *, reuse=False):
    started = perf_counter()
    source = SafetensorsContainer(checkpoint)
    parent = Path(directory).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Staging parent must be a directory")
    digest = _sha256(source.path)
    source.check_source()
    destination = parent / ("venturi-checkpoint-" + digest)
    target = destination / source.path.name
    if destination.exists():
        if not reuse:
            raise FileExistsError(destination)
        staged = SafetensorsContainer(target)
        if _sha256(target) != digest:
            raise ValueError("Staging checksum mismatch")
        staged.check_source()
        return dict(
            source=str(source.path),
            path=str(target),
            sha256=digest,
            bytes=target.stat().st_size,
            reused=True,
            validation_seconds=perf_counter() - started,
        )
    if shutil.disk_usage(parent).free < source.path.stat().st_size + (64 << 20):
        raise ValueError("Insufficient staging space")
    temporary = Path(tempfile.mkdtemp(prefix=".venturi-stage-", dir=parent))
    try:
        output = temporary / source.path.name
        with (
            source.path.open("rb") as src,
            os.fdopen(
                os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as dst,
        ):
            for block in iter(lambda: src.read(8 << 20), b""):
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        source.check_source()
        SafetensorsContainer(output)
        if _sha256(output) != digest:
            raise ValueError("Staging copy failed checksum verification")
        temporary.rename(destination)
        return dict(
            source=str(source.path),
            path=str(target),
            sha256=digest,
            bytes=target.stat().st_size,
            reused=False,
            stage_seconds=perf_counter() - started,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
