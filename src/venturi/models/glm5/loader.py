"""GLM5 plan and core-tensor loading over the verified GGUF directory.

The three fused expert banks per sparse layer stay on disk behind
GGUFExpertStore; everything else (norms, router, attention/KDA/indexer/HC
matrices, shared experts, embedding, head) decodes eagerly into device
tensors through the pinned per-type decoders.
"""

from dataclasses import dataclass
from pathlib import Path

from venturi.container.gguf import gguf_quantization as quant
from venturi.container.gguf.gguf_quant_rows import QUANT_TABLES
from venturi.models.glm5.binding import Glm5TensorDirectory
from venturi.models.glm5.geometry import GLOBAL_TENSORS, Glm5TextGeometry


@dataclass(frozen=True)
class Glm5Plan:
    geometry: Glm5TextGeometry
    layers: list          # per-layer dicts: kind, core entries, expert banks
    core_bytes: int
    expert_bank_bytes: int

    def summary(self) -> dict:
        geometry = self.geometry
        kinds = {}
        for info in self.layers:
            kinds[f"{info['attention']}/{info['mlp']}"] = kinds.get(f"{info['attention']}/{info['mlp']}", 0) + 1
        return {
            "blocks": geometry.num_layers,
            "nextn_blocks": geometry.nextn_layers,
            "sparse_mla_layers": list(geometry.sparse_mla_layers),
            "layer_kinds": kinds,
            "core_bytes": self.core_bytes,
            "expert_bank_bytes": self.expert_bank_bytes,
            "core_gib": round(self.core_bytes / 2**30, 2),
            "expert_bank_gib": round(self.expert_bank_bytes / 2**30, 2),
        }


def plan_glm5_text(directory: Glm5TensorDirectory) -> Glm5Plan:
    """Metadata-only structural plan across every block."""
    directory.validate()
    geometry = directory.geometry
    layers, core_bytes, bank_bytes = [], 0, 0
    for layer in range(geometry.num_layers + geometry.nextn_layers):
        core = directory.core_entries(layer)
        attention = geometry.attention_kind(layer)
        mlp = geometry.mlp_kind(layer)
        info = {
            "layer": layer, "attention": attention, "mlp": mlp,
            "core": core, "expert_banks": None,
        }
        if mlp == "sparse":
            info["expert_banks"] = directory.expert_bank_entries(layer)
        layers.append(info)
    # Core payload accounting is decoded-elements x 4 (F32 target).
    core_bytes = _core_payload_bytes(directory, geometry)
    bank_bytes = _expert_payload_bytes(directory, geometry)
    return Glm5Plan(geometry, layers, core_bytes, bank_bytes)


def _core_payload_bytes(directory, geometry):
    total = 0
    names = list(GLOBAL_TENSORS)
    for layer in range(geometry.num_layers + geometry.nextn_layers):
        names.extend(n for n in directory.core_entries(layer))
    for name in names:
        shape = directory.entry(name).shape
        elements = 1
        for dimension in shape:
            elements *= dimension
        total += elements * 4
    return total


def _expert_payload_bytes(directory, geometry):
    total = 0
    for layer in range(geometry.num_layers + geometry.nextn_layers):
        if geometry.mlp_kind(layer) != "sparse":
            continue
        for bank in directory.expert_bank_entries(layer):
            total += bank.nbytes  # authoritative on-disk payload from the inventory
    return total


def decode_tensor(entry, raw_path_bytes, *, max_values=2**26):
    """Decode one tensor's raw quantized payload to a CPU float32 tensor."""
    decoder_name = QUANT_TABLES[entry.type_id][0]
    decoder = getattr(quant, decoder_name)
    return decoder(raw_path_bytes, entry.shape[-1], max_values=max_values)


def read_entry_bytes(entry) -> bytes:
    """Bounded pread of one tensor's full quantized payload from its shard."""
    import os
    block_values, block_bytes = QUANT_TABLES[entry.type_id][1], QUANT_TABLES[entry.type_id][2]
    width = entry.shape[-1]
    row_bytes = width // block_values * block_bytes
    rows = 1
    for dimension in entry.shape[:-1]:
        rows *= dimension
    fd = os.open(entry.shard, os.O_RDONLY)
    try:
        remaining, pieces = rows * row_bytes, []
        offset = entry.offset
        while remaining:
            data = os.pread(fd, remaining, offset + (rows * row_bytes - remaining))
            if not data:
                raise OSError(f"Short payload read for {entry.name}")
            pieces.append(data)
            remaining -= len(data)
        return b"".join(pieces)
    finally:
        os.close(fd)


def load_core_tensors(directory: Glm5TensorDirectory, *, device="cpu", dtype=None,
                      layers=None, include_globals=True):
    """Decode non-expert-bank tensors to a dict of tensors on the target device."""
    import torch
    geometry = directory.geometry
    target = torch.device(device)
    names = []
    if include_globals:
        names.extend(GLOBAL_TENSORS)
    for layer in (layers if layers is not None else range(geometry.num_layers + geometry.nextn_layers)):
        names.extend(directory.core_entries(layer))
    out = {}
    for name in names:
        entry = directory.entry(name)
        tensor = decode_tensor(entry, read_entry_bytes(entry))
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite decoded tensor: {name}")
        if dtype is not None:
            tensor = tensor.to(dtype)
        out[name] = tensor.to(target)
    return out
