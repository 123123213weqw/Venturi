"""Cross-shard GLM5 tensor directory with strict per-layer validation."""

from dataclasses import dataclass

from venturi.models.glm5.geometry import (
    GLOBAL_TENSORS,
    Glm5TextGeometry,
    layer_tensor_names,
)


@dataclass(frozen=True)
class Glm5TensorEntry:
    name: str
    type_id: int
    shape: tuple
    shard: str
    offset: int = 0
    nbytes: int = 0


class Glm5TensorDirectory:
    """Merged tensor table over all shards; validates the full expected set."""

    def __init__(self, geometry: Glm5TextGeometry, shards):
        """shards: iterable of (path, tensors) as produced by inspect_gguf."""
        if not isinstance(geometry, Glm5TextGeometry):
            raise TypeError("Glm5TextGeometry required")
        self.geometry = geometry
        self._entries: dict[str, Glm5TensorEntry] = {}
        for path, tensors in shards:
            for tensor in tensors:
                if tensor.name in self._entries:
                    raise ValueError(f"Duplicate tensor across shards: {tensor.name}")
                self._entries[tensor.name] = Glm5TensorEntry(
                    tensor.name, tensor.type_id, tuple(tensor.shape), str(path),
                    tensor.offset, tensor.bytes)

    def validate(self):
        """Every expected tensor exists exactly once; returns self or raises."""
        geometry = self.geometry
        missing = []
        for name in GLOBAL_TENSORS:
            if name not in self._entries:
                missing.append(name)
        for layer in range(geometry.num_layers + geometry.nextn_layers):
            for name in layer_tensor_names(geometry, layer):
                if name not in self._entries:
                    missing.append(name)
        if missing:
            raise ValueError(
                f"GLM5 tensor directory is missing {len(missing)} tensors: "
                f"{sorted(missing)[:6]}..."
            )
        return self

    def entry(self, name: str) -> Glm5TensorEntry:
        if name not in self._entries:
            raise KeyError(name)
        return self._entries[name]

    def expert_bank_entries(self, layer: int):
        """(gate, up, down) fused-bank entries for one sparse layer."""
        return tuple(
            self.entry(f"blk.{layer}.ffn_{role}_exps.weight")
            for role in ("gate", "up", "down")
        )

    def core_entries(self, layer: int):
        """All non-expert-bank entries for one layer (shared experts included)."""
        names = [n for n in layer_tensor_names(self.geometry, layer)
                 if not n.endswith("_exps.weight")]
        return {n: self.entry(n) for n in names}
