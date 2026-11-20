"""Owned C++/CUDA Qwen forward/cache/greedy loop, with explicit weight residency.

Python preflights checkpoint metadata and prepares small constants at load time.
Native code owns copies, mapped Q8 planes, buffers, state, cursor and generation.
Streaming and resident modes never substitute for one another. A model currently
serializes calls on its private stream; this is not continuous/paged batching or
whole-model CUDA graphs. Diagnostic forward returns owned CPU FP32 logits.
"""

import copy
import ctypes as C
import hashlib
import json
import threading
import weakref

import torch

from ..ninfer import NInferArtifact, row_geometry
from ..ops import CudaBackend, DenseWeight
from ..ops.vocabulary import _ids
from .gdn_frontend import GDNFrontendWeights
from .qwen_weights import QwenQ8Weights
from .qwen_rope import QwenTextRoPE
from .qwen_model import _memory_guard, qwen_resident_bytes
from .decoder_layer import _outside_capture


def _head_gdn_workspace(cfg, rows):
    heads = cfg["linear_num_value_heads"] // 2
    channels = (
        cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        + heads * cfg["linear_value_head_dim"]
    )
    values = heads * cfg["linear_value_head_dim"]
    return rows * ((channels * 2 + values * 3 + heads * 2) * 2 + heads * 8)


class Config(C.Structure):
    _fields_ = (
        [
            (k, C.c_int)
            for k in (
                "device",
                "dtype",
                "layers",
                "hidden",
                "intermediate",
                "vocab",
                "hq",
                "hkv",
                "dim",
                "rotary",
                "gqk",
                "gvh",
                "gkd",
                "gvd",
                "convolution",
                "max_tokens",
                "max_position",
                "resident",
            )
        ]
        + [(k, C.c_float) for k in ("epsilon", "gdn_scale", "attention_scale")]
        + [
            ("second_device", C.c_int),
            ("peer_transport", C.c_int),
            ("reserve", C.c_uint64),
        ]
    )


class Span(C.Structure):
    _fields_ = [("path", C.c_char_p)] + [
        (k, C.c_uint64) for k in ("logical", "file_offset", "bytes", "file_bytes")
    ]


class Matrix(C.Structure):
    _fields_ = [
        ("offset", C.c_uint64),
        ("scales", C.c_uint64),
        ("n", C.c_int),
        ("k", C.c_int),
    ]


class Layer(C.Structure):
    _fields_ = (
        [(k, C.c_int) for k in ("kind", "conv_fp32", "gamma_fp32", "ab_fp32")]
        + [("matrix", Matrix * 7)]
        + [
            (k, C.c_void_p)
            for k in (
                "norm",
                "post",
                "conv",
                "rate",
                "bias",
                "gamma",
                "a",
                "b",
                "qgamma",
                "kgamma",
            )
        ]
    )


def _bind(backend):
    n = backend._native
    ptr, integer, u64 = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "venturi_qwen_sizes": ([C.POINTER(u64)], integer),
        "venturi_qwen_create": (
            [
                C.POINTER(Config),
                C.POINTER(Span),
                integer,
                C.POINTER(Layer),
                C.POINTER(Matrix),
                C.POINTER(Matrix),
                ptr,
                ptr,
            ],
            ptr,
        ),
        "venturi_qwen_destroy": ([ptr], None),
        "venturi_qwen_session": ([ptr, integer], ptr),
        "venturi_qwen_session_destroy": ([ptr], None),
        "venturi_qwen_forward": (
            [ptr, ptr, integer, ptr, u64, integer, ptr, u64],
            integer,
        ),
        "venturi_qwen_generate": (
            [ptr, ptr, integer, integer, ptr, integer, ptr, ptr],
            integer,
        ),
        "venturi_qwen_info": ([ptr, C.POINTER(u64)], integer),
        "venturi_qwen_timings": ([ptr, C.POINTER(C.c_double)], integer),
        "venturi_qwen_parallel_info": ([ptr, C.POINTER(u64)], integer),
        "venturi_qwen_reset": ([ptr], integer),
        "venturi_qwen_snapshot": ([ptr, integer, integer, ptr, u64], integer),
        "venturi_qwen_debug_failure": ([ptr, integer, integer], integer),
    }
    fn = getattr(n, "venturi_qwen_head_gdn_session_info", None)
    if fn is not None:
        fn.argtypes, fn.restype = [ptr, C.POINTER(u64)], integer
    for name, (args, result) in signatures.items():
        fn = getattr(n, name)
        fn.argtypes, fn.restype = args, result
    stage = n.venturi_qwen_stage_create
    stage.argtypes = signatures["venturi_qwen_create"][0] + [integer, integer]
    stage.restype = ptr
    sizes = (u64 * 4)()
    backend._check(n.venturi_qwen_sizes(sizes))
    if list(sizes) != [C.sizeof(t) for t in (Config, Span, Matrix, Layer)]:
        raise RuntimeError("Native Qwen descriptor ABI mismatch; rebuild explicitly")
    return n


def _digest(weights):
    return hashlib.sha256(
        json.dumps([weights.config, weights.artifact.manifest], sort_keys=True).encode()
    ).digest()


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
    ):
        if not isinstance(weights, QwenQ8Weights) or not isinstance(
            backend, CudaBackend
        ):
            raise TypeError(
                "Native Qwen requires preflighted Q8 weights and explicit CudaBackend"
            )
        if weight_mode not in ("streaming", "resident"):
            raise ValueError("Select explicit streaming or resident weight_mode")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError(
                "Native Qwen max_tokens must be 1..1024; longer prompts are chunked"
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Native Qwen supports explicit FP16/BF16 only")
        head_profile = projection_profile == "q8-residual-column-mlp-head-gdn"
        column_profile = projection_profile in (
            "q8-residual-column-mlp",
            "q8-residual-column-mlp-head-gdn",
        )
        if projection_profile not in (
            "fp16",
            "q8-residual",
            "q8-residual-column-mlp",
            "q8-residual-column-mlp-head-gdn",
        ):
            raise ValueError(
                "Projection profile must be explicit fp16/q8-residual/q8-residual-column-mlp/q8-residual-column-mlp-head-gdn"
            )
        if column_profile and (
            tensor_parallel_device is None or tensor_parallel_transport != "host"
        ):
            raise ValueError("Column MLP requires explicit two-device host TP")
        if projection_profile != "fp16" and (
            dtype != torch.float16
            or weight_mode != "resident"
            or _stage_range is not None
        ):
            raise ValueError(
                "q8-residual requires a complete resident FP16 target, not PP"
            )
        profiled_create = None
        if projection_profile != "fp16":
            try:
                profiled_create = getattr(
                    backend._native,
                    "venturi_qwen_create_head_gdn"
                    if head_profile
                    else "venturi_qwen_create_column_mlp"
                    if column_profile
                    else "venturi_qwen_create_profiled",
                )
            except AttributeError as e:
                raise RuntimeError(
                    "Explicit projection profile exports absent; rebuild blocks"
                ) from e
        target = backend._device(dtype, device)
        if tensor_parallel_transport not in ("peer", "host"):
            raise ValueError("Tensor parallel transport must be explicit peer or host")
        if tensor_parallel_device is None and tensor_parallel_transport != "peer":
            raise ValueError(
                "Host transport requires an explicit tensor parallel device"
            )
        peer = None
        if tensor_parallel_device is not None:
            if weight_mode != "resident":
                raise ValueError("Tensor parallel requires explicit resident weights")
            peer = backend._device(dtype, tensor_parallel_device)
            if peer == target:
                raise ValueError("Tensor parallel peer must be a different CUDA device")
            if tensor_parallel_transport == "peer" and not (
                torch.cuda.can_device_access_peer(target.index, peer.index)
                and torch.cuda.can_device_access_peer(peer.index, target.index)
            ):
                raise ValueError("Bidirectional CUDA peer access is required")
        if column_profile and torch.cuda.get_device_capability(target) != (7, 0):
            raise ValueError("Column MLP requires an explicit SM70 target")
        _outside_capture(target)
        owned = QwenQ8Weights.from_ninfer(NInferArtifact(weights.artifact.path))
        cfg = copy.deepcopy(owned.config)
        if _stage_range is not None:
            if (
                not isinstance(_stage_range, tuple)
                or len(_stage_range) != 2
                or any(type(i) is not int for i in _stage_range)
                or not 0
                <= _stage_range[0]
                < _stage_range[1]
                <= cfg["num_hidden_layers"]
                or weight_mode != "resident"
                or peer is not None
            ):
                raise ValueError(
                    "Internal pipeline stage requires a valid resident non-TP range"
                )
        if column_profile and cfg["intermediate_size"] % 64:
            raise ValueError("Column MLP requires group-aligned equal input halves")
        if head_profile and (
            cfg["linear_num_key_heads"] % 2
            or cfg["linear_num_key_heads"] < 2
            or cfg["linear_num_value_heads"] > 128
            or cfg["linear_value_head_dim"] % 64
            or not 1 <= cfg["linear_value_head_dim"] <= 256
            or cfg["hidden_size"] % 32
            or cfg["hidden_size"] > 32768
            or not 2 <= cfg["linear_conv_kernel_dim"] <= 8
        ):
            raise ValueError(
                "Head GDN requires aligned semantic heads and convolution geometry"
            )
        head_workspace = _head_gdn_workspace(cfg, max_tokens) if head_profile else 0
        rope = QwenTextRoPE.from_config(cfg)
        q8_bytes = []
        for spec in owned.plan.bindings:
            if spec.format == "q8_g32_fp16":
                n, k = spec.shape
                padded = (k + 127) // 128 * 128
                q8_bytes.append(n * (padded + padded // 32 * 2))
        small = qwen_resident_bytes(owned, dtype=dtype) - sum(q8_bytes)
        ep = (cfg["hidden_size"] + 127) // 128 * 128
        peak = max(*q8_bytes, max_tokens * (ep + ep // 32 * 2))
        max_projection_k = max(
            s.shape[1] for s in owned.plan.bindings if s.format == "q8_g32_fp16"
        )
        quant_bytes = (
            max_tokens * ((max_projection_k + 31) // 32) * 36 * 2
            if projection_profile != "fp16"
            else 0
        )
        # Temporary loader constants + native owned copies coexist at creation.
        required = (
            (sum(q8_bytes) if weight_mode == "resident" else peak)
            + 2 * small
            + 4 * 2**20
            + rope.rotary_dim * 2
        )
        if _stage_range is not None:
            # A conservative bound for small loader/native constants remains
            # shared with the full adapter. Large Q8 payloads are stage-local.
            selected = {}
            first, last = _stage_range
            for spec in owned.plan.bindings:
                if spec.format != "q8_g32_fp16":
                    continue
                role = spec.binding
                include = (
                    role == "text/token_embedding"
                    and first == 0
                    or role == "text/output_head"
                    and last == cfg["num_hidden_layers"]
                    or role.startswith("text/layers/")
                    and first <= int(role.split("/")[2]) < last
                )
                if include:
                    obj = owned._object(role)
                    n, k = spec.shape
                    padded = (k + 127) // 128 * 128
                    selected[obj["offset"]] = n * (padded + padded // 32 * 2)
            required = (
                sum(selected.values()) + 2 * small + 4 * 2**20 + rope.rotary_dim * 2
            )
        if peer is not None:
            primary = secondary = max_k = max_rows = 0
            for spec in owned.plan.bindings:
                if spec.format != "q8_g32_fp16":
                    continue
                n, k = spec.shape
                padded = (k + 127) // 128 * 128
                half = 1 + (n - 1) // 2
                embedding = spec.binding == "text/token_embedding"
                if column_profile and (
                    spec.binding.endswith("/mlp/down")
                    or head_profile
                    and spec.binding.endswith("/gdn/output")
                ):
                    local_pad = ((k // 2 + 127) // 128) * 128
                    encoded = n * (local_pad + local_pad // 32 * 2)
                    primary += encoded
                    secondary += encoded
                else:
                    primary += (n if embedding else half) * (padded + padded // 32 * 2)
                    secondary += (
                        n - half if not embedding or owned.plan.tied_embeddings else 0
                    ) * (padded + padded // 32 * 2)
                max_k, max_rows = max(max_k, k), max(max_rows, half)
            column_partial = (
                max_tokens * cfg["hidden_size"] * 4 if column_profile else 0
            )
            column_peer_extra = (
                column_partial + max_tokens * cfg["intermediate_size"] * 3
                if column_profile
                else 0
            )
            if column_profile:
                max_rows = max(max_rows, cfg["intermediate_size"])
            required = (
                primary
                + column_partial * 2
                + head_workspace
                + 2 * small
                + 4 * 2**20
                + rope.rotary_dim * 2
                + max_tokens * max_rows * 2
            )
            _memory_guard(
                peer,
                secondary
                + max_tokens * (max_k + max_rows) * 2
                + quant_bytes
                + column_peer_extra
                + head_workspace
                + (2 * small if head_profile else 0),
                reserve_bytes,
                None,
            )
        _memory_guard(target, required + quant_bytes, reserve_bytes, None)
        native = _bind(backend)
        if profiled_create is not None:
            profiled_create.argtypes = native.venturi_qwen_create.argtypes + (
                [] if column_profile else [C.c_int]
            )
            profiled_create.restype = C.c_void_p
        initial_digest = _digest(owned)
        initial_stamps = tuple(
            (
                f.path.stat().st_dev,
                f.path.stat().st_ino,
                f.path.stat().st_size,
                f.path.stat().st_mtime_ns,
                f.path.stat().st_ctime_ns,
            )
            for f in owned.artifact.files
        )
        keep = []

        def norm(role):
            r = backend.upload_zero_centered_rmsnorm(
                owned.read_float(role),
                dtype=dtype,
                device=target,
                epsilon=cfg["rms_norm_eps"],
            )
            keep.append(r)
            return r.storage_tensors[0].data_ptr()

        def matrix(role):
            obj = owned._object(role)
            if obj["format"] != "q8_g32_fp16":
                raise ValueError("Native matrix descriptor must be original-derived Q8")
            *_, offset, _ = row_geometry(obj["shape"], obj["format"])
            return Matrix(obj["offset"], obj["offset"] + offset, *obj["shape"])

        layers = (Layer * cfg["num_hidden_layers"])()
        for i, kind in enumerate(cfg["layer_types"]):
            layer_desc = layers[i]
            prefix = f"text/layers/{i}/"
            layer_desc.post = norm(prefix + "post_attention_norm")
            if kind == "linear_attention":
                f = GDNFrontendWeights.from_ninfer(owned.artifact, layer=i).upload(
                    backend, dtype=dtype, device=target, epsilon=cfg["rms_norm_eps"]
                )
                keep.append(f)
                layer_desc.kind, layer_desc.norm = (
                    0,
                    f.input_norm.storage_tensors[0].data_ptr(),
                )
                layer_desc.conv, layer_desc.rate, layer_desc.bias = (
                    f.conv._weight.data_ptr(),
                    f.gates._rate.data_ptr(),
                    f.gates._bias.data_ptr(),
                )
                layer_desc.gamma = f.gated_norm._weight.data_ptr()
                layer_desc.conv_fp32 = int(f.conv._weight.dtype == torch.float32)
                layer_desc.gamma_fp32 = int(f.gated_norm._weight.dtype == torch.float32)
                layer_desc.ab_fp32 = int(dtype == torch.float16)
                for name in ("a", "b"):
                    value = owned.read_float(prefix + "gdn/" + name)
                    r = (
                        backend.upload_fp32_linear(
                            DenseWeight(value.float()), dtype=dtype, device=target
                        )
                        if layer_desc.ab_fp32
                        else backend.upload_linear(
                            DenseWeight(value), dtype=dtype, device=target
                        )
                    )
                    keep.append(r)
                    setattr(layer_desc, name, r.storage_tensors[0].data_ptr())
                for j, name in enumerate(("qkv", "z", "output")):
                    layer_desc.matrix[j] = matrix(prefix + "gdn/" + name)
            else:
                layer_desc.kind, layer_desc.norm = 1, norm(prefix + "input_norm")
                qk = backend.upload_qwen_qk_norm(
                    owned.read_float(prefix + "attention/q_norm"),
                    owned.read_float(prefix + "attention/k_norm"),
                    dtype=dtype,
                    device=target,
                    epsilon=cfg["rms_norm_eps"],
                )
                keep.append(qk)
                layer_desc.qgamma, layer_desc.kgamma = (
                    qk.query_gamma.data_ptr(),
                    qk.key_gamma.data_ptr(),
                )
                for j, name in enumerate(("q", "k", "v", "o")):
                    layer_desc.matrix[j] = matrix(prefix + "attention/" + name)
            for j, name in enumerate(("gate", "up", "down"), 4):
                layer_desc.matrix[j] = matrix(prefix + "mlp/" + name)
        final_norm = norm("text/output_norm")
        c = Config(
            target.index,
            int(dtype == torch.bfloat16),
            cfg["num_hidden_layers"],
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
            max_tokens,
            cfg["max_position_embeddings"],
            int(weight_mode == "resident"),
            cfg["rms_norm_eps"],
            cfg["linear_key_head_dim"] ** -0.5,
            cfg["head_dim"] ** -0.5,
            -1 if peer is None else peer.index,
            int(tensor_parallel_transport == "host"),
            reserve_bytes,
        )
        spans = (Span * len(owned.artifact.files))(
            *(
                Span(
                    str(f.path.resolve()).encode(),
                    f.logical_start,
                    f.payload_start,
                    f.payload_bytes,
                    f.path.stat().st_size,
                )
                for f in owned.artifact.files
            )
        )
        embedding, head = matrix("text/token_embedding"), matrix("text/output_head")
        torch.cuda.current_stream(target).synchronize()
        create = (
            native.venturi_qwen_stage_create
            if _stage_range is not None
            else native.venturi_qwen_create
        )
        if profiled_create is not None:
            create = profiled_create
        handle = create(
            C.byref(c),
            spans,
            len(spans),
            layers,
            C.byref(embedding),
            C.byref(head),
            final_norm,
            rope.inverse_frequency.data_ptr(),
            *(
                (2,)
                if profiled_create is not None and not column_profile
                else (_stage_range or ())
            ),
        )
        if not handle:
            backend._check(-1)
        self = cls()
        self._handle, self._native, self._backend = handle, native, backend
        self._weights, self._cfg, self._dtype, self._device = owned, cfg, dtype, target
        self._tensor_parallel_device = peer
        self._projection_profile = (
            "bf16"
            if projection_profile == "fp16" and dtype == torch.bfloat16
            else projection_profile
        )
        self._projection_info = dict(
            projection_profile=self._projection_profile,
            activation_quantization_passes=2 if projection_profile != "fp16" else 0,
            primary_projection_workspace_bytes=quant_bytes,
            peer_projection_workspace_bytes=quant_bytes if peer is not None else 0,
            projection_precision_scope=f"Original Q8 weights/scales; {dtype} storage/KV, FP32 GDN. q8-residual uses explicit group32 activation high/residual Q8 passes with FP32 scales and changed reduction order; not original FP16 arithmetic.",
        )
        if column_profile:
            self._projection_info["projection_precision_scope"] += (
                " Column MLP keeps partial outputs/host transport FP32, then sums before one FP16 cast; changed FP32 association versus full-K down, not prior-profile byte parity."
            )
            query = getattr(native, "venturi_q8_swiglu_quant_enabled", None)
            if query is not None:
                query.argtypes, query.restype = [], C.c_int
                enabled = bool(query())
                self._projection_info["swiglu_quant_fusion"] = dict(
                    compiled_opt_in=enabled,
                    eligible_intermediate_per_rank=8704,
                    eligible_token_range=[1, 32],
                    model_has_eligible_mlp=enabled and cfg["intermediate_size"] // 2 == 8704,
                    precision="Original F16 rounding and both residual-A8 passes preserved",
                )
            query = getattr(native, "venturi_q8_rms_quant_enabled", None)
            if query is not None:
                query.argtypes, query.restype = [], C.c_int
                enabled = bool(query())
                self._projection_info["rms_quant_fusion"] = dict(
                    compiled_opt_in=enabled,
                    model_has_eligible_norm=enabled and cfg["hidden_size"] == 5120,
                    eligible_token_range=[1, 32],
                    location="post-attention norm and primary MLP input packing",
                    peer_input="Original materialized FP16 normalized values; peer packing unchanged",
                )
        if profiled_create is not None:
            try:
                query = native.venturi_qwen_projection_info
                query.argtypes, query.restype = (
                    [C.c_void_p, C.POINTER(C.c_uint64)],
                    C.c_int,
                )
                values = (C.c_uint64 * 3)()
                backend._check(query(handle, values))
                if list(values) != [
                    2,
                    quant_bytes,
                    quant_bytes if peer is not None else 0,
                ]:
                    raise RuntimeError(
                        "Native projection workspace/profile accounting mismatch"
                    )
                if column_profile:
                    query = native.venturi_qwen_column_mlp_info
                    query.argtypes, query.restype = (
                        [C.c_void_p, C.POINTER(C.c_uint64)],
                        C.c_int,
                    )
                    values = (C.c_uint64 * 8)()
                    backend._check(query(handle, values))
                    partial = max_tokens * cfg["hidden_size"] * 4
                    unique_down = len(
                        {
                            owned._object(f"text/layers/{i}/mlp/down")["offset"]
                            for i in range(cfg["num_hidden_layers"])
                        }
                    )
                    expected = [
                        1,
                        partial,
                        partial,
                        partial,
                        max_tokens * cfg["intermediate_size"] * 2,
                        max_tokens * cfg["intermediate_size"],
                        partial,
                        unique_down,
                    ]
                    if list(values) != expected:
                        raise RuntimeError(
                            "Native column MLP memory/layout accounting mismatch"
                        )
                    self._projection_info.update(
                        column_mlp_enabled=True,
                        column_mlp_buffers=dict(
                            zip(
                                [
                                    "enabled",
                                    "primary_local_fp32_bytes",
                                    "primary_remote_fp32_bytes",
                                    "peer_partial_fp32_bytes",
                                    "peer_gate_up_bytes",
                                    "peer_activation_bytes",
                                    "pinned_fp32_partial_bytes",
                                    "column_down_matrices",
                                ],
                                map(int, values),
                            )
                        ),
                        mlp_down_weight_layout="bit-preserving input-column halves, original group32 scales",
                        partial_and_host_transport_dtype="float32",
                    )
                if head_profile:
                    query = native.venturi_qwen_head_gdn_info
                    query.argtypes, query.restype = (
                        [C.c_void_p, C.POINTER(C.c_uint64)],
                        C.c_int,
                    )
                    values = (C.c_uint64 * 5)()
                    backend._check(query(handle, values))
                    gdn_count = cfg["layer_types"].count("linear_attention")
                    if list(values) != [
                        1,
                        head_workspace,
                        head_workspace,
                        gdn_count,
                        gdn_count,
                    ]:
                        raise RuntimeError(
                            "Native head GDN workspace/layout accounting mismatch"
                        )
                    self._projection_info.update(
                        head_gdn_enabled=True,
                        head_gdn_primary_workspace_bytes=head_workspace,
                        head_gdn_peer_workspace_bytes=head_workspace,
                        gdn_weight_layout="bit-preserved semantic Q/K/V head rows and group32 output-column halves",
                        gdn_state_layout="private packed FP32 value-head halves, split convolution histories and MTP prefixes on both devices; canonical joining diagnostics only",
                    )
                    self._projection_info["projection_precision_scope"] += (
                        " Head-local GDN output also uses FP32 column partial/host transport/sum before one FP16 cast; changed FP32 association, not prior-profile byte parity."
                    )
            except Exception:
                native.venturi_qwen_destroy(handle)
                raise
        self._tensor_parallel_transport = (
            tensor_parallel_transport if peer is not None else None
        )
        self._max_tokens, self._mode, self._digest, self._stamps = (
            max_tokens,
            weight_mode,
            initial_digest,
            initial_stamps,
        )
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, native.venturi_qwen_destroy, handle)
        try:
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

    def _metadata(self):
        if _digest(self._weights) != self._digest:
            raise RuntimeError("Native Qwen metadata changed; recreate model")
        stamps = tuple(
            (
                f.path.stat().st_dev,
                f.path.stat().st_ino,
                f.path.stat().st_size,
                f.path.stat().st_mtime_ns,
                f.path.stat().st_ctime_ns,
            )
            for f in self._weights.artifact.files
        )
        if stamps != self._stamps:
            raise RuntimeError("Native Qwen source changed; recreate and verify model")
        _outside_capture(self.device)

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
        from .qwen_batch import NativeQwenBatchSession

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
