"""Immutable loaded descriptors, distinct from the mutable checkpoint loader.

Native model creation already copies constants and owns encoded weight mappings.
Seal the detached Python descriptor once so each step need not JSON-serialize a
whole model directory. Source-file identity/size/timestamps are still checked on
every entry, in addition to the native mapping's existing source guards.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from ..ninfer import PayloadFile


def source_digest(config, manifest):
    return hashlib.sha256(json.dumps([config, manifest], sort_keys=True).encode()).digest()


def _freeze(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Loaded source directory requires JSON string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError("Loaded source directory requires JSON metadata values")


@dataclass(frozen=True, slots=True)
class _ArtifactMetadata:
    path: object
    manifest: object
    objects: object
    files: tuple
    artifact_id: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LoadedQwenSource:
    """Private immutable metadata capsule; not a checkpoint/weight reader."""
    config: object
    artifact: _ArtifactMetadata
    digest: bytes

    @classmethod
    def capture(cls, weights, expected_digest):
        encoded = json.dumps([weights.config, weights.artifact.manifest], sort_keys=True).encode()
        actual = hashlib.sha256(encoded).digest()
        if actual != expected_digest:
            raise RuntimeError("Native Qwen metadata changed during creation; recreate model")
        source = weights.artifact
        config, directory = json.loads(encoded)
        manifest = _freeze(directory)
        # Fresh immutable values, not aliases to the loader's mutable lists/maps.
        files = tuple(PayloadFile(f.path, f.logical_start, f.payload_start, f.payload_bytes) for f in source.files)
        artifact = _ArtifactMetadata(
            source.path, manifest,
            MappingProxyType({obj["id"]: obj for obj in manifest["objects"]}),
            files, source.artifact_id, source.payload_bytes,
        )
        return cls(_freeze(config), artifact, actual)
