"""GGUF-backed tiered expert store over fused 3-D expert banks.

Binds a layer's `ffn_{gate,up,down}_exps.weight` GGUF tensors (any quant type
with an in-tree decoder) to the universal expert-store contract: rows of one
expert are its slice of the fused bank, read coalesced and decoded on demand.
Admission is the VerifiedQuantRows full-SHA discipline; nothing is cached or
prefetched here — wrap with the LRU cache for hot residency.
"""

from venturi.container.gguf.gguf_quant_rows import QUANT_TABLES, VerifiedQuantRows
from venturi.models.moe.geometry import ExpertGeometry


class GGUFExpertStore:
    """read(expert) -> (gate_up, down) float32 CPU tensors, fused-bank layout."""

    def __init__(self, geometry: ExpertGeometry, root, shard_assets, *, layer,
                 gate_name, up_name, down_name, max_values=2**20):
        """shard_assets: {tensor_name: (ModelAsset, gguf_type_id)} for the banks.

        max_values is the per-lookup decode budget forwarded to every bank;
        read() chunks each expert fetch to stay within it.
        """
        if not isinstance(geometry, ExpertGeometry):
            raise TypeError("ExpertGeometry required")
        self.geometry = geometry
        specs = {
            "gate": (gate_name, (geometry.num_experts, geometry.intermediate_size, geometry.hidden_size)),
            "up": (up_name, (geometry.num_experts, geometry.intermediate_size, geometry.hidden_size)),
            "down": (down_name, (geometry.num_experts, geometry.hidden_size, geometry.intermediate_size)),
        }
        self._banks = {}
        for role, (name, shape) in specs.items():
            entry = shard_assets.get(name)
            if entry is None:
                raise ValueError(f"Expert bank tensor absent from shard assets: {name}")
            asset, type_id = entry
            self._banks[role] = VerifiedQuantRows(asset, root, name, shape,
                                                  type_id=type_id, max_values=max_values)
        self.layer = layer
        self.max_values = max_values

    def read(self, expert, *, device=None, dtype=None):
        if type(expert) is not int or not 0 <= expert < self.geometry.num_experts:
            raise ValueError("Expert id outside the bank")
        geometry = self.geometry
        gate = self._read_rows("gate", expert, geometry.intermediate_size)
        up = self._read_rows("up", expert, geometry.intermediate_size)
        gate_up = _cat(gate, up)
        down = self._read_rows("down", expert, geometry.hidden_size)
        if device is not None:
            gate_up = gate_up.to(device=device, dtype=dtype)
            down = down.to(device=device, dtype=dtype)
        return gate_up, down

    def _read_rows(self, role, expert, rows_per_expert):
        """One expert's row block, chunked to the bank's bounded decode budget."""
        import torch
        bank = self._banks[role]
        width = bank.width
        start = expert * rows_per_expert
        rows_per_chunk = max(1, self.max_values // width)
        pieces = []
        for offset in range(0, rows_per_expert, rows_per_chunk):
            count = min(rows_per_chunk, rows_per_expert - offset)
            ids = torch.arange(start + offset, start + offset + count, dtype=torch.int64)
            pieces.append(bank.lookup(ids))
        return _cat_rows(pieces)

    def close(self):
        for bank in self._banks.values():
            bank.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def closed(self):
        return all(bank.closed for bank in self._banks.values())


def _cat(first, second):
    import torch
    return torch.cat([first, second], dim=0)


def _cat_rows(pieces):
    import torch
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
