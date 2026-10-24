"""Experimental native greedy MTP, explicit one/two proposals and private requests.

Loading uses Python metadata only. Drafting, verification, GDN checkpoint/restore,
replay and greedy selection remain native. Generation requires fresh/reset state;
continuation remains unsupported. An explicit two-proposal path preserves the
one-proposal default.
"""

import copy
import ctypes as C
import threading
from types import SimpleNamespace
import weakref

import torch

from ..ninfer import NInferArtifact, row_geometry
from ..ops.vocabulary import _ids
from .qwen_native import (
    Config,
    Span,
    Matrix,
    Layer,
    NativeQwenModel,
    NativeQwenSession,
    _digest,
)
from .qwen_pipeline import NativeQwenPipelineModel
from .qwen_mtp import QwenMTPWeights
from .qwen_rope import QwenTextRoPE
from .qwen_model import _memory_guard


def _bind(backend):
    n = backend._native
    p, i, u = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "create": (
            [
                p,
                i,
                C.POINTER(Config),
                C.POINTER(Span),
                i,
                C.POINTER(Layer),
                C.POINTER(Matrix),
                p,
                p,
                p,
                p,
            ],
            p,
        ),
        "destroy": ([p], None),
        "session": ([p, i], p),
        "session_destroy": ([p], None),
        "generate": ([p, p, i, i, p, i, p, p], i),
        "reset": ([p], i),
        "info": ([p, C.POINTER(u)], i),
        "timings": ([p, C.POINTER(C.c_double)], i),
        "parallel_info": ([p, C.POINTER(u)], i),
        "snapshot": ([p, i, i, p, u], i),
        "debug_failure": ([p, i, i], i),
        "stats": ([p, C.POINTER(u), C.POINTER(C.c_double)], i),
        "debug_proposals": ([p, p, i], i),
        "draft": ([p, p, i, p, p, p], i),
        "draft_snapshot": ([p, i, p, u], i),
    }
    result = {}
    for suffix, (args, restype) in signatures.items():
        fn = getattr(n, "venturi_qwen_mtp_" + suffix)
        fn.argtypes, fn.restype = args, restype
        result["venturi_qwen_" + suffix] = fn
    for suffix, args, restype in (
        ("create_profiled", signatures["create"][0] + [i], p),
        ("projection_info", [p, C.POINTER(u)], i),
    ):
        fn = getattr(n, "venturi_qwen_mtp_" + suffix, None)
        if fn is not None:
            fn.argtypes, fn.restype = args, restype
        result["venturi_qwen_" + suffix] = fn
    # Additive exports: an older ABI16 library still supports depth one.
    for suffix, args in (
        ("generate2", signatures["generate"][0]),
        ("chain_stats", [p, C.POINTER(u)]),
        ("generate2_prefix", signatures["generate"][0]),
        ("prepare_prefix", [p]),
        ("prefix_info", [p, C.POINTER(u)]),
        ("head_gdn_session_info", [p, C.POINTER(u)]),
        ("head_gdn_memory", [p, C.POINTER(u)]),
        ("layer_graph", [p, i]),
        ("layer_graph_info", [p, C.POINTER(u)]),
        ("verification_trace", [p, i]),
        ("verification_data", [p, p, u, p, u, C.POINTER(u)]),
        ("graph_pause", [p, i, i]),
        ("graph_pause_status", [p, C.POINTER(i)]),
    ):
        fn = getattr(n, "venturi_qwen_mtp_" + suffix, None)
        if fn is not None:
            fn.argtypes, fn.restype = args, i
        result["venturi_qwen_" + suffix] = fn
    return SimpleNamespace(**result)


def _stamps(weights):
    return tuple(
        (
            f.path.stat().st_dev,
            f.path.stat().st_ino,
            f.path.stat().st_size,
            f.path.stat().st_mtime_ns,
            f.path.stat().st_ctime_ns,
        )
        for f in weights.artifact.files
    )


class NativeQwenMTPModel(NativeQwenModel):
    @classmethod
    def from_q8(
        cls,
        target,
        weights,
        *,
        reserve_bytes=512 * 2**20,
        draft_projection_profile="fp16",
    ):
        if (
            not isinstance(target, NativeQwenModel)
            or isinstance(target, cls)
            or not isinstance(weights, QwenMTPWeights)
        ):
            raise TypeError(
                "MTP requires an existing native target and explicit QwenMTPWeights"
            )
        if draft_projection_profile not in ("fp16", "q8-residual"):
            raise ValueError("MTP draft profile must be explicit fp16/q8-residual")
        residual = draft_projection_profile == "q8-residual"
        if residual:
            if target.dtype != torch.float16 or isinstance(
                target, NativeQwenPipelineModel
            ):
                raise ValueError("Residual MTP draft requires FP16 non-PP target")
            if any(
                getattr(target._backend._native, "venturi_qwen_mtp_" + suffix, None)
                is None
                for suffix in ("create_profiled", "projection_info")
            ):
                raise RuntimeError(
                    "Explicit MTP draft profile exports absent; rebuild blocks"
                )
        if target._mode != "resident" or target._max_tokens < 2:
            raise ValueError("MTP target requires resident weights and max_tokens >= 2")
        weights = QwenMTPWeights.from_ninfer(NInferArtifact(weights.artifact.path))
        weights.validate_target(target._weights)
        cfg, backend = target.config, target._backend
        device = (
            target.pipeline_device
            if isinstance(target, NativeQwenPipelineModel)
            else target.device
        )
        native = _bind(backend)
        if residual and torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("Residual MTP draft requires SM70")
        rope = QwenTextRoPE.from_config(cfg)
        q8_bytes = sum(
            s.nbytes for s in weights.plan.bindings if s.format == "q8_g32_fp16"
        )
        quant_bytes = (
            target._max_tokens
            * (
                (
                    max(
                        s.shape[1]
                        for s in weights.plan.bindings
                        if s.format == "q8_g32_fp16"
                    )
                    + 31
                )
                // 32
            )
            * 72
            if residual
            else 0
        )
        _memory_guard(
            device,
            q8_bytes
            + 4 * 2**20
            + quant_bytes
            + 2 * (8 * cfg["hidden_size"] + 2 * cfg["head_dim"]) * 4,
            reserve_bytes,
            None,
        )
        initial_digest, initial_stamps = _digest(weights), _stamps(weights)
        keep = []

        def norm(role):
            r = backend.upload_zero_centered_rmsnorm(
                weights.read_float("mtp/" + role),
                dtype=target.dtype,
                device=device,
                epsilon=cfg["rms_norm_eps"],
            )
            keep.append(r)
            return r.storage_tensors[0].data_ptr()

        def matrix(role):
            obj = weights._object("mtp/" + role)
            offset = row_geometry(obj["shape"], obj["format"])[4]
            return Matrix(obj["offset"], obj["offset"] + offset, *obj["shape"])

        layer = Layer()
        layer.kind, layer.norm, layer.post = (
            1,
            norm("input_norm"),
            norm("post_attention_norm"),
        )
        qk = backend.upload_qwen_qk_norm(
            weights.read_float("mtp/attention/q_norm"),
            weights.read_float("mtp/attention/k_norm"),
            dtype=target.dtype,
            device=device,
            epsilon=cfg["rms_norm_eps"],
        )
        keep.append(qk)
        layer.qgamma, layer.kgamma = qk.query_gamma.data_ptr(), qk.key_gamma.data_ptr()
        for j, role in enumerate(
            (
                "attention/q",
                "attention/k",
                "attention/v",
                "attention/o",
                "mlp/gate",
                "mlp/up",
                "mlp/down",
            )
        ):
            layer.matrix[j] = matrix(role)
        fc = matrix("input_projection")
        output_norm, embedding_norm, hidden_norm = (
            norm(s) for s in ("output_norm", "embedding_norm", "hidden_norm")
        )
        c = Config(
            device.index,
            int(target.dtype == torch.bfloat16),
            1,
            cfg["hidden_size"],
            cfg["intermediate_size"],
            cfg["vocab_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            rope.rotary_dim,
            cfg["linear_num_key_heads"],
            cfg["linear_num_value_heads"],
            cfg["linear_key_head_dim"],
            cfg["linear_value_head_dim"],
            cfg["linear_conv_kernel_dim"],
            target._max_tokens,
            cfg["max_position_embeddings"],
            1,
            cfg["rms_norm_eps"],
            cfg["linear_key_head_dim"] ** -0.5,
            cfg["head_dim"] ** -0.5,
            -1,
            0,
            reserve_bytes,
        )
        spans = (Span * len(weights.artifact.files))(
            *(
                Span(
                    str(f.path.resolve()).encode(),
                    f.logical_start,
                    f.payload_start,
                    f.payload_bytes,
                    f.path.stat().st_size,
                )
                for f in weights.artifact.files
            )
        )
        torch.cuda.current_stream(device).synchronize()
        with target._lock:
            if not target._finalizer.alive:
                raise RuntimeError("MTP target model closed")
            target._metadata()
            create = (
                native.venturi_qwen_create_profiled
                if residual
                else native.venturi_qwen_create
            )
            handle = create(
                target._handle,
                int(isinstance(target, NativeQwenPipelineModel)),
                C.byref(c),
                spans,
                len(spans),
                C.byref(layer),
                C.byref(fc),
                output_norm,
                embedding_norm,
                hidden_norm,
                rope.inverse_frequency.data_ptr(),
                *([2] if residual else []),
            )
        if not handle:
            backend._check(-1)
        self = cls()
        self._handle, self._native, self._backend = handle, native, backend
        self.target, self._mtp_weights = target, weights
        self._cfg, self._weights = copy.deepcopy(cfg), target._weights
        self._dtype, self._device = target.dtype, target.device
        self._projection_profile = target.projection_profile
        self._projection_info = target.projection_info
        self._projection_info["mtp_draft_projection_profile"] = (
            draft_projection_profile
            if residual or target.dtype == torch.float16
            else "bf16"
        )
        self._projection_info.update(
            mtp_draft_activation_quantization_passes=2 if residual else 0,
            mtp_draft_primary_projection_workspace_bytes=quant_bytes,
            mtp_draft_secondary_projection_workspace_bytes=0,
            mtp_draft_precision_scope=(
                "Original Q8 weights/FP16 scales and FP16 activations/KV; explicit two residual-A8 passes/FP32 activation scales and changed reduction arithmetic, not original FP16 parity. Shared vocabulary/head retain the target profile."
                if residual
                else "Original typed activation math; shared vocabulary/head retain the target profile."
            ),
        )
        self.mtp_device = device
        self._max_tokens, self._mode = target._max_tokens, "resident"
        self._tensor_parallel_device = target._tensor_parallel_device
        self._tensor_parallel_transport = target._tensor_parallel_transport
        self._mtp_digest, self._mtp_stamps = initial_digest, initial_stamps
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, native.venturi_qwen_destroy, handle)
        try:
            if native.venturi_qwen_projection_info is not None:
                values = (C.c_uint64 * 3)()
                backend._check(native.venturi_qwen_projection_info(handle, values))
                if tuple(values) != (2 if residual else 0, quant_bytes, 0):
                    raise RuntimeError(
                        "MTP draft profile/workspace accounting mismatch"
                    )
            self._metadata()
        except Exception:
            self.close()
            raise
        return self

    def _metadata(self):
        self.target._metadata()
        if (
            _digest(self._mtp_weights) != self._mtp_digest
            or _stamps(self._mtp_weights) != self._mtp_stamps
        ):
            raise RuntimeError("MTP metadata/source changed; recreate model")

    def create_batch(self, capacities):
        from .qwen_mtp_batch import NativeQwenMTPBatchSession

        return NativeQwenMTPBatchSession.create(self, capacities)

    def create_session(self, capacity):
        if (
            type(capacity) is not int
            or not 0 < capacity <= self._cfg["max_position_embeddings"]
        ):
            raise ValueError("MTP capacity exceeds context")
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("MTP model closed")
            self._metadata()
            handle = self._native.venturi_qwen_session(self._handle, capacity)
            if not handle:
                self._backend._check(-1)
            return NativeQwenMTPSession(self, handle, capacity)


def _base_mtp_stats(values, times):
    names = (
        "proposed_tokens",
        "accepted_tokens",
        "verification_calls",
        "replayed_input_tokens",
        "rollback_calls",
        "target_forward_calls",
        "target_input_tokens",
        "draft_forward_calls",
        "draft_input_tokens",
        "emitted_tokens",
        "forced_proposals",
        "draft_position",
        "mtp_model_device_bytes",
        "mtp_request_device_bytes",
        "primary_target_backup_bytes",
        "secondary_target_backup_bytes",
        "mtp_remote_embedding_bytes",
        "mtp_pinned_bridge_bytes",
        "mtp_device_index",
        "secondary_target_request_bytes",
    )
    result = dict(zip(names, map(int, values)))
    result.update(
        zip(
            (
                "draft_wall_seconds_including_warmup_catchup",
                "verification_wall_seconds",
                "rollback_replay_wall_seconds",
                "prefill_wall_seconds",
                "checkpoint_save_wall_seconds",
            ),
            times,
        )
    )
    result.update(
        draft_tokens_per_round=1,
        acceptance_rate=values[1] / values[0] if values[0] else None,
        accepted_by_depth=[int(values[1])],
        proposed_by_depth=[int(values[0])],
        sampling="target greedy; acceptance is matching proposals / proposed tokens",
        fresh_generation_only=True,
    )
    return result


class NativeQwenMTPSession(NativeQwenSession):
    def generate(
        self,
        prompt_token_ids,
        *,
        max_new_tokens,
        eos_token_ids=(),
        draft_tokens=1,
        recovery="replay",
    ):
        if getattr(self, "_target_graph_enabled", False) and (
            draft_tokens != 2 or recovery != "prefix"
        ):
            raise ValueError(
                "Target layer graphs require explicit MTP2 prefix recovery"
            )
        if type(draft_tokens) is not int or draft_tokens not in (1, 2):
            raise ValueError("MTP draft_tokens must be explicit 1 or 2")
        if recovery not in ("replay", "prefix") or (
            draft_tokens == 1 and recovery != "replay"
        ):
            raise ValueError("Prefix recovery requires explicit MTP2")
        if draft_tokens == 1:
            return super().generate(
                prompt_token_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
        if self.model._max_tokens < 3:
            raise ValueError("MTP2 requires at least three projection rows")
        fn = getattr(
            self.model._native,
            "venturi_qwen_generate2_prefix"
            if recovery == "prefix"
            else "venturi_qwen_generate2",
            None,
        )
        if fn is None:
            raise RuntimeError("MTP2 recovery export absent; rebuild explicitly")
        return self._generate_with(
            fn,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    def enable_target_layer_graph(self, enabled=True):
        """Explicit native scalar MTP2 prototype; no offload/PP/cohort fallback."""
        if type(enabled) is not bool:
            raise ValueError("Target layer graph enabled must be boolean")
        fn = getattr(self.model._native, "venturi_qwen_layer_graph", None)
        if fn is None:
            raise RuntimeError("Target layer graph export absent; rebuild explicitly")
        with self._lock:
            self._check()
            self.model._backend._check(fn(self._handle, int(enabled)))
            self._target_graph_enabled = enabled

    def target_layer_graph_info(self):
        with self._lock:
            self._check()
            return self._target_layer_graph_info_unchecked()

    def _target_layer_graph_info_unchecked(self):
        fn = getattr(self.model._native, "venturi_qwen_layer_graph_info", None)
        if fn is None:
            raise RuntimeError(
                "Target layer graph info export absent; rebuild explicitly"
            )
        values = (C.c_uint64 * 11)()
        self.model._backend._check(fn(self._handle, values))
        keys = (
            "enabled",
            "ready",
            "captures",
            "replays",
            "fallback_calls",
            "eager_positioned_warm_calls",
            "nodes",
            "device_position_bytes",
            "pinned_position_bytes",
            "last_position",
            "value_workspace_bytes",
        )
        return dict(
            zip(keys, map(int, values)),
            scope="Request-lifetime counters; target layers only, fixed T3/prefix2; embedding/head/selection and other widths remain eager",
        )

    def _collect_verification_trace(self, enabled=True):
        """Untimed diagnostic only: retain complete FP16 head bytes per forward."""
        if type(enabled) is not bool:
            raise ValueError("Verification trace enabled must be boolean")
        fn = getattr(self.model._native, "venturi_qwen_verification_trace", None)
        if fn is None:
            raise RuntimeError("Verification trace export absent; rebuild explicitly")
        with self._lock:
            self._check()
            self.model._backend._check(fn(self._handle, int(enabled)))

    def _verification_trace(self):
        fn = getattr(self.model._native, "venturi_qwen_verification_data", None)
        if fn is None:
            raise RuntimeError(
                "Verification trace data export absent; rebuild explicitly"
            )
        with self._lock:
            self._check()
            meta = (C.c_uint64 * 2)()
            self.model._backend._check(fn(self._handle, None, 0, None, 0, meta))
            bits, widths = (C.c_ubyte * meta[0])(), (C.c_int * meta[1])()
            self.model._backend._check(
                fn(self._handle, bits, meta[0], widths, meta[1], meta)
            )
            return dict(widths=list(widths), fp16_bytes=bytes(bits))

    def prepare_prefix(self):
        """Explicit allocation outside generation timing; request state stays fresh."""
        fn = getattr(self.model._native, "venturi_qwen_prepare_prefix", None)
        if fn is None:
            raise RuntimeError("MTP2 prefix export absent; rebuild explicitly")
        with self._lock:
            self._check()
            self.model._backend._check(fn(self._handle))

    def stats(self):
        with self._lock:
            self._check()
            values, times = (C.c_uint64 * 20)(), (C.c_double * 5)()
            self.model._backend._check(
                self.model._native.venturi_qwen_stats(self._handle, values, times)
            )
            result = _base_mtp_stats(values, times)
            fn = getattr(self.model._native, "venturi_qwen_chain_stats", None)
            if fn is not None:
                chain = (C.c_uint64 * 13)()
                self.model._backend._check(fn(self._handle, chain))
                if chain[0] == 2:
                    result.update(
                        draft_tokens_per_round=2,
                        proposed_by_depth=list(chain[1:3]),
                        verified_by_depth=list(chain[3:5]),
                        accepted_by_depth=list(chain[5:7]),
                        draft_rewind_calls=int(chain[7]),
                        verified_draft_catchup_inputs=int(chain[8]),
                        verification_calls_by_input_width={
                            "2": int(chain[9]),
                            "3": int(chain[10]),
                        },
                        forced_by_depth=list(chain[11:13]),
                        unused_suffix_proposals=int(values[0] - chain[3] - chain[4]),
                        conditional_acceptance_by_depth=[
                            chain[5 + j] / chain[3 + j] if chain[3 + j] else None
                            for j in range(2)
                        ],
                        sampling="target greedy; overall acceptance counts accepted prefix tokens / all drafted proposals; conditional depth acceptance uses verified proposals only",
                    )
            fn = getattr(self.model._native, "venturi_qwen_prefix_info", None)
            if fn is not None:
                extra = (C.c_uint64 * 3)()
                self.model._backend._check(fn(self._handle, extra))
                if extra[0] or extra[1] or result["draft_tokens_per_round"] == 2:
                    result.update(
                        prefix_trace_extra_primary_bytes=int(extra[0]),
                        prefix_trace_extra_secondary_bytes=int(extra[1]),
                        target_recovery="prefix-trace"
                        if extra[2]
                        else "checkpoint-replay",
                    )
            if self.model.projection_info.get("head_gdn_enabled"):
                memory = (C.c_uint64 * 3)()
                self.model._backend._check(
                    self.model._native.venturi_qwen_head_gdn_memory(
                        self._handle, memory
                    )
                )
                result.update(
                    peer_target_request_device_bytes=int(memory[0]),
                    peer_target_backup_device_bytes=int(memory[1]),
                    peer_target_prefix_extra_device_bytes=int(memory[2]),
                    physical_primary_target_backup_device_bytes=result[
                        "primary_target_backup_bytes"
                    ]
                    - int(memory[1]),
                    backup_memory_scope="Legacy primary_target_backup_bytes/prefix_trace_extra_primary_bytes are first-target-stage totals across both TP ranks; peer fields split these totals, not additional duplicated backups.",
                )
            if getattr(self, "_target_graph_enabled", False):
                result["target_layer_graph"] = self._target_layer_graph_info_unchecked()
            return result

    def info(self):
        with self._lock:
            result = super().info()
            result.update(
                mtp=self.stats(),
                mtp_memory_scope="Base fields describe shared target only; named MTP, backup and prefix-trace fields are additional. Count model bytes once, request bytes per session.",
            )
            return result

    def forward(self, *args, **kwargs):
        raise ValueError(
            "MTP production supports fresh/reset generate, not arbitrary forward; use draft_forward for diagnostics"
        )

    def draft_forward(self, token_ids, hidden):
        """Diagnostic CPU copies only; inputs are post-final-norm target states."""
        _ids(token_ids, self.model._cfg["vocab_size"])
        shape = (len(token_ids), self.model._cfg["hidden_size"])
        if (
            not isinstance(hidden, torch.Tensor)
            or tuple(hidden.shape) != shape
            or hidden.device.type != "cpu"
            or hidden.dtype != self.model.dtype
            or not hidden.is_contiguous()
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError(
                "MTP diagnostic requires finite contiguous CPU hidden states in execution dtype"
            )
        if len(token_ids) > self.model._max_tokens:
            raise ValueError("MTP diagnostic exceeds max_tokens")
        with self._lock:
            self._check()
            ids = (C.c_int64 * len(token_ids))(*token_ids)
            logits = torch.empty(
                self.model._cfg["vocab_size"], dtype=torch.float32, device="cpu"
            )
            output = torch.empty(shape, dtype=self.model.dtype, device="cpu")
            self.model._backend._check(
                self.model._native.venturi_qwen_draft(
                    self._handle,
                    ids,
                    len(ids),
                    hidden.data_ptr(),
                    logits.data_ptr(),
                    output.data_ptr(),
                )
            )
            return dict(hidden=output, logits=logits)

    def draft_snapshot(self):
        with self._lock:
            self._check()
            pos, cfg = self.stats()["draft_position"], self.model._cfg
            tensors = []
            for kind in (2, 3):
                value = torch.empty(
                    (pos, cfg["num_key_value_heads"], cfg["head_dim"]),
                    dtype=self.model.dtype,
                    device="cpu",
                )
                self.model._backend._check(
                    self.model._native.venturi_qwen_draft_snapshot(
                        self._handle,
                        kind,
                        value.data_ptr(),
                        value.numel() * value.element_size(),
                    )
                )
                tensors.append(value)
            return dict(position=pos, keys=tensors[0], values=tensors[1])

    def _force_proposals(self, ids):
        """Explicit correctness-test hook, never enabled by the inference CLI."""
        _ids(ids, self.model._cfg["vocab_size"])
        with self._lock:
            self._check()
            values = (C.c_int64 * len(ids))(*ids)
            self.model._backend._check(
                self.model._native.venturi_qwen_debug_proposals(
                    self._handle, values, len(values)
                )
            )
