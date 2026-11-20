"""Explicit native two-GPU pipeline with stage-local resident weights and state.

One native call owns each complete request generation loop. Stage model locks
are released at boundaries, allowing different requests on different GPUs.
Portable pinned host DMA is explicit; no peer-copy or CPU numerical fallback.
"""

import ctypes as C
import threading
from types import SimpleNamespace
import weakref

import torch

from .qwen_native import NativeQwenModel, NativeQwenSession
from .qwen_weights import QwenQ8Weights
from ..ops import CudaBackend


def _pipeline_bind(backend):
    n = backend._native
    ptr, integer, u64 = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "create": ([ptr, ptr], ptr),
        "destroy": ([ptr], None),
        "session": ([ptr, integer], ptr),
        "session_destroy": ([ptr], None),
        "forward": ([ptr, ptr, integer, ptr, u64, integer, ptr, u64], integer),
        "generate": ([ptr, ptr, integer, integer, ptr, integer, ptr, ptr], integer),
        "info": ([ptr, C.POINTER(u64)], integer),
        "parallel_info": ([ptr, C.POINTER(u64)], integer),
        "details": ([ptr, C.POINTER(u64)], integer),
        "reset": ([ptr], integer),
        "snapshot": ([ptr, integer, integer, ptr, u64], integer),
        "debug_failure": ([ptr, integer, integer], integer),
    }
    functions = {}
    for suffix, (args, result) in signatures.items():
        f = getattr(n, "venturi_qwen_pipeline_" + suffix)
        f.argtypes, f.restype = args, result
        functions["venturi_qwen_" + suffix] = f
    # Only the C handle protocol is reused. Every operation, including the
    # generation loop and layer-state ownership, is the native pipeline API.
    return SimpleNamespace(**functions)


class NativeQwenPipelineModel(NativeQwenModel):
    """Two resident, contiguous layer stages; no implicit TP/PP hybrid."""

    @classmethod
    def from_q8(
        cls,
        weights,
        backend,
        *,
        weight_mode,
        pipeline_device,
        split_layer=None,
        transport="host",
        dtype=torch.bfloat16,
        device="cuda:0",
        max_tokens=128,
        reserve_bytes=512 * 2**20,
    ):
        if not isinstance(weights, QwenQ8Weights) or not isinstance(
            backend, CudaBackend
        ):
            raise TypeError(
                "Native pipeline requires Q8 weights and explicit CudaBackend"
            )
        if weight_mode != "resident" or transport != "host":
            raise ValueError(
                "Native pipeline requires explicit resident weights and host transport"
            )
        layers = weights.config["num_hidden_layers"]
        split = layers // 2 if split_layer is None else split_layer
        if type(split) is not int or not 0 < split < layers:
            raise ValueError(
                "Pipeline split must leave at least one layer in each stage"
            )
        a, b = backend._device(dtype, device), backend._device(dtype, pipeline_device)
        if a == b:
            raise ValueError("Pipeline requires two different CUDA devices")
        stages = []
        handle = None
        try:
            for target, span in ((a, (0, split)), (b, (split, layers))):
                stages.append(
                    NativeQwenModel.from_q8(
                        weights,
                        backend,
                        weight_mode="resident",
                        dtype=dtype,
                        device=target,
                        max_tokens=max_tokens,
                        reserve_bytes=reserve_bytes,
                        _stage_range=span,
                    )
                )
            native = _pipeline_bind(backend)
            handle = native.venturi_qwen_create(stages[0]._handle, stages[1]._handle)
            if not handle:
                backend._check(-1)
            self = cls()
            self._handle, self._native, self._backend = handle, native, backend
            self._stages = tuple(stages)
            self._cfg, self._weights = stages[0]._cfg, stages[0]._weights
            self._device, self._dtype = a, dtype
            self._max_tokens, self._mode = max_tokens, "resident"
            self._tensor_parallel_device, self._tensor_parallel_transport = None, "host"
            self.pipeline_device, self.split_layer = b, split
            self._lock = threading.RLock()
            self._metadata()
            self._finalizer = weakref.finalize(
                self, native.venturi_qwen_destroy, handle
            )
            return self
        except Exception:
            if handle:
                native.venturi_qwen_destroy(handle)
            raise
        finally:
            # The native pipeline and its sessions retain shared ownership.
            for stage in stages:
                stage.close()

    def _metadata(self):
        for stage in self._stages:
            stage._metadata()

    def create_session(self, capacity):
        if (
            type(capacity) is not int
            or not 0 < capacity <= self._cfg["max_position_embeddings"]
        ):
            raise ValueError("Native Qwen capacity exceeds context")
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Native Qwen pipeline model closed")
            self._metadata()
            h = self._native.venturi_qwen_session(self._handle, capacity)
            if not h:
                self._backend._check(-1)
            return NativeQwenPipelineSession(self, h, capacity)


class NativeQwenPipelineSession(NativeQwenSession):
    """Independent state per stage; concurrent calls on different requests."""

    def info(self):
        with self._lock:
            result = super().info()
            detail = (C.c_uint64 * 4)()
            self.model._backend._check(
                self.model._native.venturi_qwen_details(self._handle, detail)
            )
            result.update(
                parallelism="pipeline",
                pipeline_transport="host",
                pipeline_split_layer=int(detail[0]),
                peer_request_device_bytes=int(detail[1]),
                pipeline_handoffs=int(detail[2]),
                pipeline_activation_payload_bytes=result["peer_input_copy_bytes"],
                pipeline_selection_d2h_bytes=int(detail[3]),
                scheduling="Native stage locks; distinct requests may overlap across stages, no continuous batching",
            )
            return result
