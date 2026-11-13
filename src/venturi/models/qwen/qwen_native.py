"""Owned C++/CUDA Qwen forward/cache/greedy loop, with explicit weight residency.

Python preflights checkpoint metadata and prepares small constants at load time.
Native code owns copies, mapped Q8 planes, buffers, state, cursor and generation.
Streaming and resident modes never substitute for one another. A model currently
serializes calls on its private stream; this is not continuous/paged batching or
whole-model CUDA graphs. Diagnostic forward returns owned CPU FP32 logits.
"""

import copy
import ctypes as C
import threading
import weakref

import torch

from venturi.ninfer import NInferArtifact, row_geometry
from venturi.ops import CudaBackend, DenseWeight
from venturi.ops.vocabulary import _ids
from venturi.models.qwen.gdn_frontend import GDNFrontendWeights
from venturi.models.qwen.qwen_weights import QwenQ8Weights
from venturi.models.qwen.qwen_metadata import LoadedQwenSource, source_digest
from venturi.models.qwen.qwen_rope import QwenTextRoPE
from venturi.models.qwen.qwen_model import _memory_guard, qwen_resident_bytes
from venturi.models.decoder_layer import _outside_capture

from venturi.models.qwen_load.abi import Config, Span, Matrix, Layer, _bind, _digest
from venturi.models.qwen_load.budget import _head_gdn_workspace, plan_budget
from venturi.models.qwen_load.spec import LoadSpec, validate_load_spec
from venturi.models.qwen_load.planner import prepare_source
from venturi.models.qwen_load.descriptors import build_descriptors
from venturi.models.qwen_load.verify import normalized_profile, projection_info, reconcile


class NativeQwenModel:
    """Immutable loaded model; native sessions keep native-owned weights alive.

    Small GPU constants are copied into native ownership before creation returns.
    Read-only mapped source still must stay unchanged and available. Closing this
    Python model prevents new sessions, but existing sessions remain valid.
    """

    @classmethod
    def from_q8(
        cls,
        weights,
        backend,
        *,
        weight_mode,
        dtype=torch.bfloat16,
        device="cuda:0",
        max_tokens=128,
        reserve_bytes=512 * 2**20,
        tensor_parallel_device=None,
        tensor_parallel_transport="peer",
        projection_profile="fp16",
        _stage_range=None,
        experimental_flags=0,
        kv_cache="static",
        kv_gpu_budget_bytes=0,
        kv_ram_budget_bytes=0,
        metadata_mode="checked-json",
    ):
        """Orchestrate the staged load pipeline (qwen_load); order unchanged.

        spec -> planner -> budget -> descriptors -> native create -> verify,
        with the historical validation order, messages and accounting.
        """
        spec = validate_load_spec(
            weights,
            backend,
            weight_mode=weight_mode,
            dtype=dtype,
            device=device,
            max_tokens=max_tokens,
            reserve_bytes=reserve_bytes,
            tensor_parallel_device=tensor_parallel_device,
            tensor_parallel_transport=tensor_parallel_transport,
            projection_profile=projection_profile,
            _stage_range=_stage_range,
            experimental_flags=experimental_flags,
            kv_cache=kv_cache,
            kv_gpu_budget_bytes=kv_gpu_budget_bytes,
            kv_ram_budget_bytes=kv_ram_budget_bytes,
            metadata_mode=metadata_mode,
        )
        owned, cfg = prepare_source(spec)
        budget = plan_budget(spec, owned, cfg)
        desc = build_descriptors(spec, owned, cfg, budget)
        torch.cuda.current_stream(spec.target).synchronize()
        create = (
            desc.native.venturi_qwen_stage_create
            if spec.stage_range is not None
            else desc.native.venturi_qwen_create
        )
        if spec.profiled_create is not None:
            create = spec.profiled_create
        handle = create(
            C.byref(desc.config),
            desc.spans,
            len(desc.spans),
            desc.layers,
            C.byref(desc.embedding),
            C.byref(desc.head),
            desc.final_norm,
            desc.rope.inverse_frequency.data_ptr(),
            *(
                (spec.experimental_flags,)
                if spec.experimental_flags
                else (2,)
                if spec.profiled_create is not None and not spec.column_profile
                else (spec.stage_range or ())
            ),
        )
        if not handle:
            backend._check(-1)
        if spec.tier_enable is not None:
            try:
                backend._check(
                    spec.tier_enable(
                        handle, spec.kv_gpu_budget_bytes, spec.kv_ram_budget_bytes
                    )
                )
            except BaseException:
                desc.native.venturi_qwen_destroy(handle)
                raise
        self = cls()
        self._kv_cache_mode = spec.kv_cache
        self._handle, self._native, self._backend = handle, desc.native, backend
        self._weights, self._cfg, self._dtype, self._device = (
            owned,
            cfg,
            spec.dtype,
            spec.target,
        )
        self._tensor_parallel_device = spec.tensor_parallel_device
        self._tensor_parallel_transport = (
            spec.tensor_parallel_transport
            if spec.tensor_parallel_device is not None
            else None
        )
        self._projection_profile = normalized_profile(spec)
        self._projection_info = projection_info(
            spec, cfg, budget.quant_bytes, desc.native
        )
        reconcile(spec, backend, handle, cfg, owned, budget, self._projection_info)
        self._max_tokens, self._mode, self._digest, self._stamps = (
            spec.max_tokens,
            spec.weight_mode,
            desc.initial_digest,
            desc.initial_stamps,
        )
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, desc.native.venturi_qwen_destroy, handle)
        try:
            self._metadata_seal = None
            if spec.metadata_mode == "sealed":
                self._metadata_seal = LoadedQwenSource.capture(
                    owned, desc.initial_digest
                )
                self._weights = self._metadata_seal
            self._metadata()
        except Exception:
            self.close()
            raise
        return self


    @property
    def config(self):
        return copy.deepcopy(self._cfg)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def projection_profile(self):
        return getattr(
            self,
            "_projection_profile",
            "bf16" if getattr(self, "_dtype", None) == torch.bfloat16 else "fp16",
        )

    @property
    def projection_info(self):
        return dict(
            getattr(
                self,
                "_projection_info",
                dict(
                    projection_profile=self.projection_profile,
                    activation_quantization_passes=0,
                    primary_projection_workspace_bytes=0,
                    peer_projection_workspace_bytes=0,
                ),
            )
        )

    def kv_cache_info(self):
        """Model-wide KV pool stats, not per-request bytes or NVML VRAM/RSS."""
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Native Qwen model closed")
            if getattr(self, "_kv_cache_mode", "static") == "static":
                return dict(mode="static")
            query = self._native.venturi_qwen_kv_tier_info
            query.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
            query.restype = C.c_int
            values = (C.c_uint64 * 18)()
            self._backend._check(query(self._handle, values))
            names = ("enabled", "page_bytes", "gpu_pool_bytes", "gpu_mapped_bytes",
                     "virtual_reserved_bytes", "cpu_backing_bytes", "logical_valid_bytes",
                     "evictions", "reloads", "device_to_host_bytes", "host_to_device_bytes",
                     "allocation_ooms", "preflights", "peak_gpu_pool_bytes", "peak_cpu_backing_bytes",
                     "gpu_budget_bytes", "ram_budget_bytes", "entries")
            return dict(mode="vram-first", **dict(zip(names, values)))

    def _kv_cache_info_for_request(self):
        # A native request retains shared Model ownership after the public model
        # handle closes. Never query its freed handle, including close/query races.
        with self._lock:
            if self._finalizer.alive:
                return self.kv_cache_info()
            return dict(mode=getattr(self, "_kv_cache_mode", "static"),
                        model_owner_closed=True, stats_available=False,
                        scope="Request retains native KV ownership; model-handle diagnostics unavailable after owner close")

    def _metadata(self):
        # The detached loaded descriptor has no mutable nested maps/lists. Check
        # its identity rather than reserializing hundreds of roles per token.
        if (self._metadata_seal is None and _digest(self._weights) != self._digest) or (
            self._metadata_seal is not None and self._weights is not self._metadata_seal
        ):
            raise RuntimeError("Native Qwen metadata changed; recreate model")
        stamps = []
        for file in self._weights.artifact.files:
            stat = file.path.stat()
            stamps.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        stamps = tuple(stamps)
        if stamps != self._stamps:
            raise RuntimeError("Native Qwen source changed; recreate and verify model")
        _outside_capture(self.device)

    def _binding_weights(self):
        """Detached metadata reader for one-time cross-artifact binding (MTP).

        Loaded capsules intentionally are not mutable checkpoint readers. Reopen
        only the metadata, preserving the exact loaded directory digest and file
        guards; no model weight planes are materialized or reread here.
        """
        self._metadata()
        if isinstance(self._weights, QwenQ8Weights):
            return self._weights
        weights = QwenQ8Weights.from_ninfer(NInferArtifact(self._weights.artifact.path))
        if _digest(weights) != self._weights.digest:
            raise RuntimeError("Native Qwen binding metadata differs from loaded source")
        self._metadata()
        return weights

    def create_session(self, capacity):
        if (
            type(capacity) is not int
            or not 0 < capacity <= self._cfg["max_position_embeddings"]
        ):
            raise ValueError("Native Qwen capacity exceeds context")
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Native Qwen model closed")
            self._metadata()
            h = self._native.venturi_qwen_session(self._handle, capacity)
            if not h:
                self._backend._check(-1)
            return NativeQwenSession(self, h, capacity)

    def create_batch(self, capacities):
        """Explicit grouped target execution with independent request caches."""
        from venturi.models.qwen.qwen_batch import NativeQwenBatchSession

        return NativeQwenBatchSession.create(self, capacities)

    def close(self):
        with self._lock:
            self._finalizer()


class NativeQwenSession:
    """Independent native cursor, GDN history/state and dense KV; no Python loop."""

    def __init__(self, model, handle, capacity):
        self.model, self.capacity, self._handle = model, capacity, handle
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(
            self, model._native.venturi_qwen_session_destroy, handle
        )

    def _check(self):
        if not self._finalizer.alive:
            raise RuntimeError("Native Qwen session closed")
        self.model._metadata()

    def info(self):
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Native Qwen session closed")
            out = (C.c_uint64 * 8)()
            self.model._backend._check(
                self.model._native.venturi_qwen_info(self._handle, out)
            )
            result = dict(
                zip(
                    (
                        "position",
                        "failed",
                        "model_device_bytes",
                        "request_device_bytes",
                        "pinned_host_bytes",
                        "mapped_payload_bytes",
                        "model_weight_upload_bytes",
                        "streaming_embedding_upload_bytes",
                    ),
                    out,
                )
            )
            peer = (C.c_uint64 * 4)()
            self.model._backend._check(
                self.model._native.venturi_qwen_parallel_info(self._handle, peer)
            )
            result.update(
                peer_device_index=int(peer[0]) - 1 if peer[0] else None,
                peer_transport=self.model._tensor_parallel_transport,
                peer_device_bytes=int(peer[1]),
                peer_input_copy_bytes=int(peer[2]),
                peer_output_copy_bytes=int(peer[3]),
                **self.model.projection_info,
            )
            if result.get("head_gdn_enabled"):
                peer_request = (C.c_uint64 * 1)()
                self.model._backend._check(
                    self.model._native.venturi_qwen_head_gdn_session_info(
                        self._handle, peer_request
                    )
                )
                result["peer_request_device_bytes"] = int(peer_request[0])
                result["request_memory_scope"] = (
                    "request_device_bytes is primary; peer_request_device_bytes is additional private distributed GDN cache, excluding shared model workspaces"
                )
            if getattr(self.model, "_kv_cache_mode", "static") != "static":
                result["shared_model_kv_cache"] = self.model._kv_cache_info_for_request()
            return result

    def generation_timings(self):
        """Native-call timestamps, including model queues but excluding Python/load."""
        with self._lock:
            self._check()
            values = (C.c_double * 2)()
            self.model._backend._check(
                self.model._native.venturi_qwen_timings(self._handle, values)
            )
            return dict(
                native_time_to_first_token_seconds=values[0],
                native_generation_seconds=values[1],
            )

    @property
    def position(self):
        return self.info()["position"]

    @property
    def failed(self):
        return bool(self.info()["failed"])

    def forward(
        self,
        token_ids,
        *,
        all_logits=False,
        trace=None,
        max_cpu_output_bytes=512 * 2**20,
    ):
        _ids(token_ids, self.model._cfg["vocab_size"])
        if type(all_logits) is not bool or (
            trace is not None and not isinstance(trace, list)
        ):
            raise ValueError("Use boolean all_logits and optional explicit trace list")
        with self._lock:
            self._check()
            cfg = self.model._cfg
            n, initial = len(token_ids), self.position
            if n > self.capacity - initial:
                raise ValueError("Native Qwen input exceeds capacity")
            required = (n if all_logits else 1) * cfg["vocab_size"] * 4
            if trace is not None:
                required += cfg["num_hidden_layers"] * n * cfg["hidden_size"] * 2
            if type(max_cpu_output_bytes) is not int or max_cpu_output_bytes <= 0:
                raise ValueError("max_cpu_output_bytes must be positive")
            if required > max_cpu_output_bytes:
                raise MemoryError(
                    "Native Qwen explicit CPU diagnostic output exceeds budget"
                )
            out = torch.empty(
                (n, cfg["vocab_size"]) if all_logits else (cfg["vocab_size"],),
                dtype=torch.float32,
                device="cpu",
            )
            hidden = (
                torch.empty(
                    cfg["num_hidden_layers"],
                    n,
                    cfg["hidden_size"],
                    dtype=self.model.dtype,
                    device="cpu",
                )
                if trace is not None
                else None
            )
            ids = (C.c_int64 * n)(*token_ids)
            self.model._backend._check(
                self.model._native.venturi_qwen_forward(
                    self._handle,
                    ids,
                    n,
                    out.data_ptr(),
                    out.numel(),
                    int(all_logits),
                    hidden.data_ptr() if hidden is not None else 0,
                    hidden.numel() if hidden is not None else 0,
                )
            )
            if trace is not None:
                for start in range(0, n, self.model._max_tokens):
                    for i in range(cfg["num_hidden_layers"]):
                        trace.append(
                            dict(
                                position=initial + start,
                                layer=i,
                                hidden=hidden[
                                    i, start : start + self.model._max_tokens
                                ].clone(),
                            )
                        )
            return out

    def generate(self, prompt_token_ids, *, max_new_tokens, eos_token_ids=()):
        return self._generate_with(
            None,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    def _generate_with(
        self, entry, prompt_token_ids, *, max_new_tokens, eos_token_ids=()
    ):
        vocab = self.model._cfg["vocab_size"]
        _ids(prompt_token_ids, vocab)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < 2**31:
            raise ValueError("max_new_tokens must be positive int32")
        if not isinstance(eos_token_ids, (tuple, list)):
            raise ValueError("EOS IDs must be an explicit list/tuple")
        if eos_token_ids:
            _ids(eos_token_ids, vocab)
        with self._lock:
            self._check()
            if len(prompt_token_ids) + max_new_tokens > self.capacity - self.position:
                raise ValueError("Native Qwen generation exceeds capacity")
            prompt = (C.c_int64 * len(prompt_token_ids))(*prompt_token_ids)
            eos = (C.c_int64 * len(eos_token_ids))(*eos_token_ids)
            out = (C.c_int64 * max_new_tokens)()
            count = C.c_int()
            if entry is None:
                entry = self.model._native.venturi_qwen_generate
            self.model._backend._check(
                entry(
                    self._handle,
                    prompt,
                    len(prompt),
                    max_new_tokens,
                    eos,
                    len(eos),
                    out,
                    C.byref(count),
                )
            )
            return list(out[: count.value])

    def snapshot(self):
        """Explicit CPU diagnostic copies, never used by native generation."""
        with self._lock:
            self._check()
            cfg, position = self.model._cfg, self.position
            result = dict(gdn=[], kv=[])
            packed = (
                2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
                + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            )
            for i, kind in enumerate(cfg["layer_types"]):
                tensors = []
                shapes = (
                    (
                        (cfg["linear_conv_kernel_dim"] - 1, packed),
                        (
                            cfg["linear_num_value_heads"],
                            cfg["linear_value_head_dim"],
                            cfg["linear_key_head_dim"],
                        ),
                    )
                    if kind == "linear_attention"
                    else ((position, cfg["num_key_value_heads"], cfg["head_dim"]),) * 2
                )
                for j, shape in enumerate(shapes):
                    tensor = torch.empty(
                        shape,
                        dtype=torch.float32
                        if kind == "linear_attention" and j == 1
                        else self.model.dtype,
                        device="cpu",
                    )
                    self.model._backend._check(
                        self.model._native.venturi_qwen_snapshot(
                            self._handle,
                            i,
                            j if kind == "linear_attention" else 2 + j,
                            tensor.data_ptr(),
                            tensor.numel() * tensor.element_size(),
                        )
                    )
                    tensors.append(tensor)
                result["gdn"].append(
                    tuple(tensors) if kind == "linear_attention" else None
                )
                result["kv"].append(
                    tuple(tensors) if kind == "full_attention" else None
                )
            return result

    def reset(self):
        with self._lock:
            self._check()
            self.model._backend._check(
                self.model._native.venturi_qwen_reset(self._handle)
            )

    def _inject_failure(self, *, layer, position):
        with self._lock:
            self._check()
            self.model._backend._check(
                self.model._native.venturi_qwen_debug_failure(
                    self._handle, layer, position
                )
            )

    def close(self):
        with self._lock:
            self._finalizer()
