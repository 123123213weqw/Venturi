"""Immutable loaded source identity, not an alternate weight container."""

from dataclasses import dataclass
import hashlib, json
from types import MappingProxyType


def source_digest(weights):
    return hashlib.sha256(
        json.dumps(weights.signature(), sort_keys=True).encode()
    ).digest()


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    path: object
    stamp: tuple

    def check_source(self):
        s = self.path.stat()
        if (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) != self.stamp:
            raise RuntimeError("Weight source changed; recreate and verify model")


@dataclass(frozen=True, slots=True)
class LoadedQwenSource:
    config: object
    source: SourceIdentity
    digest: bytes

    @classmethod
    def capture(cls, weights, expected_digest):
        actual = source_digest(weights)
        if actual != expected_digest:
            raise RuntimeError("Qwen metadata changed during creation")
        config = _freeze(json.loads(json.dumps(weights.config)))
        return cls(
            config, SourceIdentity(weights.source.path, weights.source.stamp), actual
        )
