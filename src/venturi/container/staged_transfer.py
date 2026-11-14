"""Venturi-owned pinned-staged host-to-device transfer for checkpoint payloads.

A small ring of pinned host slots feeds non-blocking device copies on a
dedicated stream: while one slot's DMA is in flight, the host reads the next
disk chunk into another slot, so sequential disk reads overlap with earlier
transfers. No model, shard or file-format policy lives here; byte offsets and
sizes come from the caller (SafetensorsContainer tensor metadata).
"""

from __future__ import annotations

import ctypes
import os
from typing import Callable

import torch

DEFAULT_SLOT_BYTES = 32 * 1024 * 1024
DEFAULT_SLOTS = 4

_PREADV = hasattr(os, "preadv")


def _writable_region(bytes_tensor: torch.Tensor):
    """Writable memoryview over a contiguous CPU uint8 tensor slice."""
    count = bytes_tensor.numel()
    if count == 0:
        return memoryview(b"")
    return memoryview((ctypes.c_char * count).from_address(bytes_tensor.data_ptr()))


def plan_chunks(total: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Split [0, total) bytes into (offset, count) pieces of at most chunk_bytes."""
    if type(total) is not int or type(chunk_bytes) is not int:
        raise ValueError("total and chunk_bytes must be integers")
    if total < 0 or chunk_bytes <= 0:
        raise ValueError("total must be nonnegative and chunk_bytes positive")
    return [
        (start, min(chunk_bytes, total - start))
        for start in range(0, total, chunk_bytes)
    ]


def pread_fill(fd: int, position: int, count: int, dest) -> int:
    """Read up to count bytes from fd at position into writable buffer dest.

    preadv writes straight into the pinned slot where available; otherwise a
    pread plus one host copy is used. Returns the number of bytes read.
    """
    view = memoryview(dest)
    filled = 0
    while filled < count:
        if _PREADV:
            got = os.preadv(fd, [view[filled:count]], position + filled)
        else:
            block = os.pread(fd, count - filled, position + filled)
            if not block:
                break
            view[filled : filled + len(block)] = block
            got = len(block)
        if got <= 0:
            break
        filled += got
    return filled


class StagedTransfer:
    """Ring of pinned host slots overlapping sequential reads with device copies.

    upload() fills one device tensor at a time. Each chunk is read into the
    next free slot, copied on a dedicated stream with non_blocking=True, and
    the slot is not reused on the host until its previous copy completed. The
    target tensor is safe to consume on the device's current stream as soon
    as upload() returns.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        slot_bytes: int = DEFAULT_SLOT_BYTES,
        slots: int = DEFAULT_SLOTS,
    ):
        if type(slot_bytes) is not int or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive integer")
        if type(slots) is not int or slots <= 0:
            raise ValueError("slots must be a positive integer")
        self.device = torch.device(device)
        self.slot_bytes = slot_bytes
        self.slots = slots
        self._pinned = torch.empty(slots * slot_bytes, dtype=torch.uint8, pin_memory=True)
        self._stream = torch.cuda.Stream(device=self.device)
        self._slot_done = [torch.cuda.Event() for _ in range(slots)]
        self._used = [False] * slots
        self._cursor = 0

    def upload(
        self,
        target: torch.Tensor,
        read_chunk: Callable[[int, int, object], int],
    ) -> None:
        """Fill the device tensor via read_chunk(offset, count, dest) -> int."""
        payload = target.reshape(-1).view(torch.uint8)
        chunks = plan_chunks(payload.numel(), self.slot_bytes)
        with torch.cuda.stream(self._stream):
            for offset, count in chunks:
                index = self._cursor % self.slots
                self._cursor += 1
                if self._used[index]:
                    # Host-side guard: never overwrite a slot whose previous
                    # device copy may still be reading it.
                    self._slot_done[index].synchronize()
                base = index * self.slot_bytes
                slot = self._pinned[base : base + count]
                got = read_chunk(offset, count, _writable_region(slot))
                if got != count:
                    raise OSError(
                        f"Short read: got {got} of {count} bytes at payload offset {offset}"
                    )
                self._used[index] = True
                payload[offset : offset + count].copy_(
                    self._pinned[base : base + count], non_blocking=True
                )
                self._slot_done[index].record(self._stream)
        if chunks:
            done = torch.cuda.Event()
            done.record(self._stream)
            # The consumer runs on the current stream: order it after every
            # copy of this tensor issued on the staging stream.
            torch.cuda.current_stream(self.device).wait_event(done)
