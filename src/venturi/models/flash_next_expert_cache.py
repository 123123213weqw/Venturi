"""Opt-in bounded floating expert LRU, independent of the source weight codec.

This cache retains decoded weight matrices, not native packed Q4/Q8/FP8 kernels.
Budget counts cache-owned storage only: live callers, allocator reservations,
miss scratch and CUDA libraries are ADDITIONAL. Not an OOM/placement controller.
Returned weights are immutable by contract. Source weights must remain immutable.
"""

from collections import OrderedDict
from dataclasses import dataclass
import threading

import torch

from .flash_next_moe import MoEGeometry


@dataclass
class _Entry:
    weights: tuple
    bytes: int
    ready: object


class CachedExpertStore:
    """Serialize same-key misses, explicit per-device/dtype LRU and CUDA ordering.

    Eviction releases only cache ownership; outstanding callers keep valid
    tensors. GPU hits wait for the fill event on the caller's current stream;
    record_stream protects asynchronous consumers from premature storage reuse.
    No empty_cache, process-wide stream, async prefetch or automatic fallback.
    """

    def __init__(self, source, *, max_resident_bytes):
        if not isinstance(getattr(source, 'geometry', None), MoEGeometry) or not callable(getattr(source, 'read', None)):
            raise ValueError('Explicit qualified expert source required')
        if type(max_resident_bytes) is not int or max_resident_bytes < 0:
            raise ValueError('Explicit nonnegative retained-weight byte budget required')
        self.source, self.geometry, self.max_resident_bytes = source, source.geometry, max_resident_bytes
        self._entries, self._bytes, self._lock = OrderedDict(), 0, threading.RLock()
        self._counts = dict(hits=0, misses=0, evictions=0, bypasses=0)

    @property
    def stats(self):
        with self._lock:
            return dict(self._counts, entries=len(self._entries), retained_bytes=self._bytes,
                        max_resident_bytes=self.max_resident_bytes)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    @staticmethod
    def _lease(entry, device):
        if device.type == 'cuda':
            stream = torch.cuda.current_stream(device)
            stream.wait_event(entry.ready)
            for weight in entry.weights:
                weight.record_stream(stream)
        return entry.weights

    def _fetch(self, expert, device, dtype):
        g = self.geometry
        values = self.source.read(expert, device=device, dtype=dtype)
        shapes = ((2*g.intermediate_size, g.hidden_size), (g.hidden_size, g.intermediate_size))
        if not isinstance(values, tuple) or len(values) != 2 or any(not isinstance(w, torch.Tensor) or
                tuple(w.shape) != shape or w.dtype != dtype or w.device != device for w, shape in zip(values, shapes)):
            raise ValueError('Expert source returned incompatible matrices/dtype/device')
        return values

    def read(self, expert, *, device, dtype):
        g = self.geometry
        if type(expert) is not int or not 0 <= expert < g.num_experts or dtype not in (
                torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('Valid expert and explicit floating execution dtype required')
        device = torch.device(device)
        if device.type == 'cuda':
            device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        elif device.type != 'cpu' or device.index is not None:
            raise ValueError('Only explicit CPU or CUDA decoded expert cache is qualified')
        size = 3*g.hidden_size*g.intermediate_size*torch.empty((), dtype=dtype, device='cpu').element_size()
        key = (expert, device, dtype)
        with self._lock, torch.no_grad():
            if key in self._entries:
                entry = self._entries[key]
                values = self._lease(entry, device)
                self._entries.move_to_end(key)
                self._counts['hits'] += 1
                return values
            self._counts['misses'] += 1
            if size > self.max_resident_bytes:
                self._counts['bypasses'] += 1
                return self._fetch(expert, device, dtype)
            while self._bytes+size > self.max_resident_bytes:
                _, old = self._entries.popitem(last=False)
                self._bytes -= old.bytes
                self._counts['evictions'] += 1
            values = self._fetch(expert, device, dtype)
            # Own contiguous storage: caching a tiny view into an entire expert
            # bank would invalidate the retained-byte budget. Cold fill includes
            # this copy, never hidden in the hot-hit performance claim.
            owned = tuple(w.clone(memory_format=torch.contiguous_format) for w in values)
            ready = None
            if device.type == 'cuda':
                stream = torch.cuda.current_stream(device)
                for weight in values:
                    weight.record_stream(stream)
                ready = torch.cuda.Event()
                ready.record(stream)
            entry = _Entry(owned, size, ready)
            self._entries[key] = entry
            self._bytes += size
            return self._lease(entry, device)
