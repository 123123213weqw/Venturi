"""Explicit native TP1 Q8/static-state adapter for same-process PD workers.

Model execution remains C++/CUDA. Export leases existing state; transport copies
valid KV, FP32 recurrent state and convolution history without Torch math or CPU
materialization. Host staging and peer transport are explicit, never fallbacks.
"""
from contextlib import ExitStack
import ctypes as C
from dataclasses import dataclass
import threading
from time import perf_counter

from ..runtime.state import RequestState, StateComponent, StateIdentity, TransferReceipt
from .qwen_batch import NativeQwenBatchSession
from .qwen_native import NativeQwenModel


@dataclass(frozen=True)
class _Lease:
    owner: object
    slot: int
    epoch: int


class NativeQwenPDWorker:
    def __init__(self, batch):
        if type(batch) is not NativeQwenBatchSession or type(batch.model) is not NativeQwenModel:
            raise TypeError("PD requires an ordinary native Qwen batch, not PP/MTP")
        m = batch.model
        if (m._mode != "resident" or m._tensor_parallel_device is not None
                or getattr(m, "_kv_cache_mode", "static") != "static"):
            raise ValueError("PD v1 requires resident TP1 weights and static request state")
        if not hasattr(m._backend._native, "venturi_qwen_batch_handoff"):
            raise RuntimeError("Native PD handoff export absent; rebuild explicitly")
        m._metadata()
        self.batch = batch
        self.capacities = batch.capacities
        self.max_token_rows = batch.max_token_rows
        self.vocab_size = batch.vocab_size
        self.state_identity = StateIdentity(
            m._digest.hex(), "qwen-static-kv-gdn-conv-v1", str(m.dtype), m.projection_profile,
        )
        self._lock = threading.RLock()
        self._epochs = [0] * len(self.capacities)

    def reset(self, *, slots):
        with self._lock:
            self.batch.reset(slots=slots)
            for slot in slots:
                self._epochs[slot] += 1

    def step(self, prompts, *, slots, sample_slots, decode_slots=()):
        with self._lock:
            result = self.batch.step(prompts, slots=slots, sample_slots=sample_slots,
                                     decode_slots=decode_slots)
            # Appending tokens invalidates all previously exported views.
            for slot in slots:
                self._epochs[slot] += 1
            return result

    def export_state(self, slot, *, position):
        self.batch._slots([slot])
        with self._lock, self.batch._lock:
            self.batch._check()
            info = self.batch.info()["requests"][slot]
            if (type(position) is not int or position <= 0 or info["position"] != position
                    or info["failed"]):
                raise ValueError("PD source position is not a valid committed prefix")
            cfg = self.batch.model._cfg
            dtype = self.state_identity.precision.removeprefix("torch.")
            components = []
            packed = (2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
                      + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"])
            for layer, kind in enumerate(cfg["layer_types"]):
                if kind == "linear_attention":
                    planes = (("history", dtype, (cfg["linear_conv_kernel_dim"] - 1, packed), 2),
                              ("recurrent", "float32", (cfg["linear_num_value_heads"],
                               cfg["linear_value_head_dim"], cfg["linear_key_head_dim"]), 4))
                else:
                    shape = (position, cfg["num_key_value_heads"], cfg["head_dim"])
                    planes = (("key", dtype, shape, 2), ("value", dtype, shape, 2))
                for name, precision, shape, size in planes:
                    count = 1
                    for n in shape:
                        count *= n
                    components.append(StateComponent(f"layer.{layer}.{name}", precision,
                                                     shape, count * size))
            return RequestState(self.state_identity, position, tuple(components),
                                _Lease(self, slot, self._epochs[slot]))


class NativeQwenStateTransport:
    def __init__(self, *, mode):
        if mode not in ("host", "peer"):
            raise ValueError("Explicit PD transport must be host or peer")
        self.mode = mode

    def transfer(self, state, target, *, slot):
        if (not isinstance(state, RequestState) or not isinstance(state.payload, _Lease)
                or type(target) is not NativeQwenPDWorker):
            raise TypeError("Native Qwen transport requires a leased native state and target")
        source = state.payload.owner
        if type(source) is not NativeQwenPDWorker or source is target:
            raise ValueError("PD requires distinct producer and consumer workers")
        target.batch._slots([slot])
        if source.state_identity != state.identity or target.state_identity != state.identity:
            raise ValueError("PD checkpoint/layout/precision/arithmetic identity mismatch")
        a, b = source.batch, target.batch
        if a.model._backend._native._handle != b.model._backend._native._handle:
            raise ValueError("PD v1 requires the same loaded native library, not foreign handles")
        start = perf_counter()
        with ExitStack() as stack:
            for worker in sorted((source, target), key=id):
                stack.enter_context(worker._lock)
            for batch in sorted((a, b), key=id):
                stack.enter_context(batch._lock)
            a._check()
            b._check()
            lease = state.payload
            if source._epochs[lease.slot] != lease.epoch:
                raise ValueError("PD source lease is stale after reset or append")
            fresh = source.export_state(lease.slot, position=state.position)
            if fresh.components != state.components:
                raise ValueError("PD state component descriptors were altered")
            info = b.info()["requests"][slot]
            if info["position"] or info["failed"] or state.position > info["capacity"]:
                raise ValueError("PD destination must be fresh and fit the imported prefix")
            function = a.model._backend._native.venturi_qwen_batch_handoff
            function.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                                 C.c_int, C.c_int, C.POINTER(C.c_uint64)]
            function.restype = C.c_int
            output = (C.c_uint64 * 4)()
            a.model._backend._check(function(a._handle, lease.slot, b._handle, slot,
                                             state.position, int(self.mode == "peer"), output))
            if list(output[:3]) != [state.nbytes, state.position, len(state.components)]:
                raise RuntimeError("PD native transfer receipt differs from exported state")
            target._epochs[slot] += 1
        return TransferReceipt(state.position, state.nbytes, perf_counter() - start,
                               self.mode, int(output[3]))
