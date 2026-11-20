"""Type-aware verified GGUF quantized row storage (pread, no mmap/cache).

Generalizes the IQ4_NL row-store admission and coalesced-read discipline to
any GGUF quant type with an in-tree decoder: full-file and header SHA
admission, file identity fingerprinting, bounded budgets, poison-on-error
leases. One table per instance, one declared quant type — mixed-type stores
are explicit per-table constructions, never implicit.
"""

import hashlib
import math
import os
import stat as stat_module
import threading
from pathlib import Path

from venturi.container.gguf.gguf_inventory import inspect_gguf
from venturi.container.model_assets import ModelAsset

# type_id -> (decoder name, block values, block bytes)
QUANT_TABLES = {
    0: ("decode_f32_rows", 1, 4),
    10: ("decode_q2_k_rows", 256, 84),
    11: ("decode_q3_k_rows", 256, 110),
    13: ("decode_q5_k_rows", 256, 176),
    14: ("decode_q6_k_rows", 256, 210),
    18: ("decode_iq3_xxs_rows", 256, 98),
    20: ("decode_iq4_nl_rows", 32, 18),
    8: ("decode_q8_0_rows", 32, 34),
    21: ("decode_iq3_s_rows", 256, 110),
    22: ("decode_iq2_s_rows", 256, 82),
    23: ("decode_iq4_xs_rows", 256, 136),
    30: ("decode_bf16_rows", 1, 2),
}


def quant_row_bytes(type_id, width):
    """Row payload bytes for one quant type at a given table width."""
    if type_id not in QUANT_TABLES:
        raise ValueError(f"Unsupported quant type id {type_id}")
    block_values, block_bytes = QUANT_TABLES[type_id][1], QUANT_TABLES[type_id][2]
    if type(width) is not int or width <= 0 or width % block_values:
        raise ValueError(f"Width must be divisible by block size {block_values}")
    return width // block_values * block_bytes


def _fingerprint(status):
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


class VerifiedQuantRows:
    """Verified pread row reader for one quantized GGUF table.

    lookup_many merges row IDs across caller requests, deduplicates and
    coalesces adjacent rows, decodes once, and scatters into independent
    float32 results. No persistent cache, no prefetch, no GPU.
    """

    def __init__(
        self,
        asset,
        root,
        table_name,
        expected_shape,
        *,
        type_id,
        max_values=2**20,
    ):
        if not isinstance(asset, ModelAsset):
            raise TypeError("Explicit ModelAsset required")
        if type_id not in QUANT_TABLES:
            raise ValueError(f"Unsupported quant type id {type_id}")
        if (not isinstance(expected_shape, (tuple, list)) or not 2 <= len(expected_shape) <= 3
                or any(type(x) is not int for x in expected_shape)
                or not 0 < expected_shape[0] < 2**63):
            raise ValueError("Bounded [rows, width] (or [experts, rows, width]) shape required")
        # A fused 3-D bank [experts, rows_per_expert, width] is addressed as a
        # flat 2-D row space of experts * rows_per_expert rows.
        width = expected_shape[-1]
        rows = 1
        for dimension in expected_shape[:-1]:
            rows *= dimension
        self.experts_per_bank = expected_shape[0] if len(expected_shape) == 3 else None
        self.rows_per_expert = expected_shape[1] if len(expected_shape) == 3 else None
        self.row_bytes = quant_row_bytes(type_id, width)
        if type(max_values) is not int or not 0 < max_values <= 2**20:
            raise ValueError("Explicit bounded lookup output budget required")
        if not hasattr(os, "pread"):
            raise RuntimeError("POSIX pread required; no fallback backend")
        self.table_name, self.type_id = table_name, type_id
        self.rows, self.width, self._max_values = rows, width, max_values
        self._lock = threading.RLock()
        self._last_read = None
        self._poisoned = False
        path = Path(root).resolve(strict=True) / asset.filename
        if path.is_symlink():
            raise ValueError("Symlinked GGUF source refused")
        stream = os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                   | getattr(os, "O_NONBLOCK", 0)), "rb", buffering=0)
        try:
            before = os.fstat(stream.fileno())
            if not stat_module.S_ISREG(before.st_mode) or before.st_size != asset.bytes:
                raise ValueError("Full regular source with exact manifest size required")
            inventory = inspect_gguf(path, expected_bytes=asset.bytes)
            directory = {tensor.name: tensor for tensor in inventory.tensors}
            tensor = directory.get(table_name)
            # The bank may be declared 3-D [experts, rows_per_expert, width];
            # its flat row space is (rows, width), so either representation of
            # the same extent is admissible.
            shape_ok = tuple(tensor.shape) == (rows, width) or (
                len(expected_shape) == 3
                and tuple(tensor.shape) == tuple(expected_shape))
            if tensor is None or tensor.type_id != type_id or not shape_ok:
                raise ValueError(f"Quant table name/dtype/shape differs: {table_name}")
            fingerprint = _fingerprint(before)
            if fingerprint != _fingerprint(os.fstat(stream.fileno())):
                raise ValueError("Source changed during header inventory")
            whole = hashlib.sha256()
            header = hashlib.sha256()
            offset = 0
            while offset < asset.bytes:
                raw = stream.read(min(8 * 2**20, asset.bytes - offset))
                if not raw:
                    raise ValueError("Source truncated during full SHA scan")
                whole.update(raw)
                if offset < inventory.header_bytes_read:
                    header.update(raw[:inventory.header_bytes_read - offset])
                offset += len(raw)
            if stream.read(1) or fingerprint != _fingerprint(os.fstat(stream.fileno())):
                raise ValueError("Source changed during full SHA scan")
            if whole.hexdigest() != asset.sha256:
                raise ValueError("Full source SHA256 mismatch; row store NOT admitted")
            if header.hexdigest() != inventory.header_sha256:
                raise ValueError("Inventory header differs from verified source descriptor")
            self.asset = asset
            self.header_sha256 = inventory.header_sha256
            self._tensor = tensor
            self._path = path
            self._fingerprint = fingerprint
            self._stream = stream
        except BaseException:
            stream.close()
            raise

    def _check_lease(self):
        if _fingerprint(os.stat(self._path)) != self._fingerprint:
            raise OSError("Verified row source changed; explicit re-admission required")

    def lookup_many(self, requests):
        """Merged, deduplicated, coalesced row decode into float32 outputs."""
        import torch
        from venturi.container.gguf import gguf_quantization as quant

        with self._lock:
            if getattr(self, "_stream", None) is None:
                raise RuntimeError("Row store closed")
            if self._poisoned:
                raise RuntimeError("Row store lease poisoned; rebuild explicitly")
            requests = list(requests)
            if not 1 <= len(requests) <= 64:
                raise ValueError("1..64 requests required")

            total = 0
            snapshots = []
            for request in requests:
                if (not isinstance(request, torch.Tensor) or request.device.type != "cpu"
                        or request.dtype != torch.int64 or request.dim() == 0):
                    raise ValueError("Requests must be CPU int64 tensors")
                total += request.numel() * self.width
                if total > self._max_values:
                    raise ValueError("Request batch exceeds bounded budget")
                snapshots.append(request.clone())
            for ids in snapshots:
                if any(not 0 <= row < self.rows for row in ids.reshape(-1).tolist()):
                    raise ValueError("Row id out of table bounds")
            unique = sorted({row for ids in snapshots for row in ids.reshape(-1).tolist()})
            pieces, payload, calls = [], 0, 0
            self._check_lease()
            start = 0
            while start < len(unique):
                end = start + 1
                while end < len(unique) and unique[end] == unique[end - 1] + 1:
                    end += 1
                span = (unique[end - 1] - unique[start] + 1) * self.row_bytes
                offset = self._tensor.offset + unique[start] * self.row_bytes
                remaining, done = span, 0
                while remaining:
                    data = os.pread(self._stream.fileno(), remaining, offset + done)
                    if not data:
                        raise OSError("Short quant row payload read")
                    pieces.append(data)
                    done += len(data)
                    remaining -= len(data)
                    calls += 1
                payload += span
                start = end
            decoder = getattr(quant, QUANT_TABLES[self.type_id][0])
            try:
                decoded = decoder(b"".join(pieces), self.width, max_values=self._max_values)
            except (ValueError, OSError):
                self._poisoned = True
                raise
            self._check_lease()
            positions = {row: index for index, row in enumerate(unique)}
            outputs = []
            for request in requests:
                index = torch.tensor([positions[row] for row in request.reshape(-1).tolist()])
                outputs.append(decoded.index_select(0, index).reshape(*request.shape, self.width))
            self._last_read = (len(unique), payload, calls)
            return outputs

    def lookup(self, row_ids):
        return self.lookup_many([row_ids])[0]

    def close(self):
        with self._lock:
            if getattr(self, "_stream", None) is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def closed(self):
        return getattr(self, "_stream", None) is None

    @property
    def last_read(self):
        return self._last_read
