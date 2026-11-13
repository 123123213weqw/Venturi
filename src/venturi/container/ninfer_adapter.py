"""Legacy .ninfer container adapter (implements WeightContainer Protocol).

Wraps the existing NInferArtifact to satisfy the WeightContainer interface.
This allows .ninfer files to work alongside safetensors/GGUF without any
caller knowing which format is in use.
"""

from pathlib import Path
from typing import Any

from ..ninfer import NInferArtifact
from .base import ByteSpan, TensorInfo


class NInferContainer:
    """Adapter: NInferArtifact → WeightContainer interface."""

    def __init__(self, path: str | Path):
        self.artifact = NInferArtifact(Path(path))
        self._tensors: dict[str, TensorInfo] = {}
        self._logical: dict[str, int] = {}
        self._total = 0
        self._build()

    def _build(self):
        for name, obj in self.artifact.objects.items():
            self._tensors[name] = TensorInfo(
                name=name,
                dtype=obj.get("format", "unknown"),
                shape=tuple(obj["shape"]),
                byte_offset=obj["offset"],
                byte_size=obj["bytes"],
            )
        # Compute logical offsets from the artifact's file layout.
        logical = 0
        for name in sorted(self._tensors, key=lambda n: self._tensors[n].byte_offset):
            self._logical[name] = logical
            logical += self._tensors[name].byte_size
        self._total = logical

    def list_tensors(self) -> list[str]:
        return list(self._tensors.keys())

    def tensor_info(self, name: str) -> TensorInfo:
        return self._tensors[name]

    def metadata(self) -> dict[str, Any]:
        return self.artifact.manifest

    def byte_spans(self) -> list[ByteSpan]:
        spans = []
        for f in self.artifact.files:
            spans.append(ByteSpan(
                path=str(f.path),
                logical_start=f.logical_start,
                file_offset=f.payload_start,
                bytes=f.payload_bytes,
                file_bytes=f.path.stat().st_size,
            ))
        return spans

    def read_bytes(self, tensor_name: str, offset: int, size: int) -> bytes:
        return self.artifact.read_object(tensor_name, offset=offset, size=size)

    def total_payload_bytes(self) -> int:
        return self._total

    def logical_offset(self, name: str) -> int:
        return self._logical[name]

    def model_config(self) -> dict:
        return self.artifact.manifest["components"]["text"]["config"]

    def binding_map(self) -> dict[str, str]:
        """Logical role name → tensor (object) name."""
        return {
            role: binding["object"]
            for role, binding in self.artifact.manifest["bindings"].items()
        }
