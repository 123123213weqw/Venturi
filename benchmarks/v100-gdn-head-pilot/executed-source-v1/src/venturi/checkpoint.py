"""Architecture-independent checkpoint inspection without materializing weights."""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from safetensors import SafetensorError, safe_open

from .dtypes import DTYPE_INFO


class CheckpointError(ValueError):
    """A checkpoint's storage is incomplete, unsupported, or inconsistent."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, ValueError) as error:
        raise CheckpointError(f"Cannot read {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise CheckpointError(f"{path.name} must contain a JSON object")
    return data


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    file: Path

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_INFO[self.dtype][1]


@dataclass(frozen=True)
class Checkpoint:
    directory: Path
    files: tuple[Path, ...]
    tensors: Mapping[str, TensorSpec]
    config: Mapping[str, Any]
    file_metadata: Mapping[str, Mapping[str, str]]
    index_metadata: Mapping[str, Any]

    @property
    def tensor_bytes(self) -> int:
        return sum(spec.nbytes for spec in self.tensors.values())

    def summary(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.directory),
            "config": dict(self.config),
            "files": [path.name for path in self.files],
            "tensor_count": len(self.tensors),
            # Physical storage elements are not model parameter counts when
            # several quantized values are packed into each integer element.
            "storage_elements": sum(spec.numel for spec in self.tensors.values()),
            "tensor_bytes": self.tensor_bytes,
            "dtype_counts": dict(Counter(spec.dtype for spec in self.tensors.values())),
            "file_metadata": {
                name: dict(metadata) for name, metadata in self.file_metadata.items()
            },
            "index_metadata": dict(self.index_metadata),
        }


def inspect_checkpoint(path: str | Path) -> Checkpoint:
    """Inspect a checkpoint directory or an explicitly selected safetensors file.

    Config and quantization descriptions are opaque, optional metadata. No model
    code is executed, and no architecture-specific tensor schema is assumed.
    Indexed shards are authoritative. HF cache symlinks are supported.
    """
    # Preserve a checkpoint file's symlink name and sibling config location:
    # HF cache blobs themselves need not have a .safetensors extension.
    source = Path(path).expanduser().absolute()
    if not source.exists():
        raise CheckpointError(f"Checkpoint path does not exist: {source}")
    if not source.is_dir() and (
        not source.is_file() or source.suffix != ".safetensors"
    ):
        raise CheckpointError("Expected a checkpoint directory or a .safetensors file")
    root = source.resolve() if source.is_dir() else source.parent.resolve()
    source = root if source.is_dir() else root / source.name
    config_path = root / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    weight_map: dict[str, str] | None = None
    index_metadata: dict[str, Any] = {}
    if source.is_file():
        files = (source,)
    else:
        canonical_index = root / "model.safetensors.index.json"
        indexes = (
            [canonical_index]
            if canonical_index.exists()
            else sorted(root.glob("*.safetensors.index.json"))
        )
        if len(indexes) > 1:
            raise CheckpointError(
                "Multiple safetensors indexes found; select an unambiguous checkpoint directory"
            )
        if indexes:
            index = _read_json(indexes[0])
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise CheckpointError(
                    "The shard index must contain a nonempty weight_map"
                )
            for name, filename in weight_map.items():
                if (
                    not name
                    or not isinstance(filename, str)
                    or not filename.endswith(".safetensors")
                    or "/" in filename
                    or "\\" in filename
                ):
                    raise CheckpointError(
                        f"Invalid shard index entry: {name!r}: {filename!r}"
                    )
            index_metadata = index.get("metadata", {})
            if not isinstance(index_metadata, dict):
                raise CheckpointError("Shard index metadata must be a JSON object")
            files = tuple(root / name for name in sorted(set(weight_map.values())))
        elif (root / "model.safetensors").is_file():
            files = (root / "model.safetensors",)
        else:
            files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        raise CheckpointError(f"No safetensors checkpoint files found in {root}")

    specs: dict[str, TensorSpec] = {}
    file_metadata: dict[str, Mapping[str, str]] = {}
    for shard in files:
        if not shard.is_file():
            raise CheckpointError(f"Missing checkpoint shard: {shard.name}")
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as reader:
                file_metadata[shard.name] = MappingProxyType(reader.metadata() or {})
                keys = reader.keys()
                if not keys:
                    raise CheckpointError(
                        f"Checkpoint shard contains no tensors: {shard.name}"
                    )
                for name in keys:
                    if name in specs:
                        raise CheckpointError(f"Duplicate tensor across shards: {name}")
                    if weight_map is not None and weight_map.get(name) != shard.name:
                        raise CheckpointError(
                            f"Shard index disagrees with {shard.name} for tensor {name}"
                        )
                    view = reader.get_slice(name)
                    dtype = view.get_dtype()
                    if dtype not in DTYPE_INFO:
                        raise CheckpointError(
                            f"Unsupported storage dtype {dtype} for {name}; native sub-byte formats are not supported"
                        )
                    specs[name] = TensorSpec(
                        name, tuple(view.get_shape()), dtype, shard
                    )
        except (OSError, SafetensorError) as error:
            raise CheckpointError(f"Cannot inspect {shard.name}: {error}") from error
    if weight_map is not None and set(weight_map) != set(specs):
        missing = sorted(set(weight_map) - set(specs))
        raise CheckpointError(f"Shard index references missing tensors: {missing[:5]}")
    return Checkpoint(
        root,
        files,
        MappingProxyType(specs),
        MappingProxyType(config),
        MappingProxyType(file_metadata),
        MappingProxyType(index_metadata),
    )
