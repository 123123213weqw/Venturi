"""Native descriptor assembly: constants upload, layer table, ABI structs.

Uploads the small FP constants through the backend, builds the ctypes Layer
table, the Config and the Span array exactly as the native constructor
expects. The returned Descriptors object must outlive the native create call
(it keeps uploaded tensors alive); the native side copies everything into its
own ownership before create returns.
"""

import ctypes as C
from dataclasses import dataclass

import torch

from ...ninfer import row_geometry
from ...ops import DenseWeight
from ..gdn_frontend import GDNFrontendWeights
from ... import codecs
from .abi import Config, Layer, Matrix, Span, _bind, _digest


@dataclass
class Descriptors:
    native: object
    layers: object
    final_norm: int
    rope: object
    embedding: object
    head: object
    spans: object
    config: object
    keep: list
    initial_digest: object
    initial_stamps: tuple


def build_descriptors(spec, owned, cfg, budget):
    backend, target, dtype = spec.backend, spec.target, spec.dtype
    max_tokens = spec.max_tokens
    peer = spec.tensor_parallel_device
    native = _bind(backend)
    profiled_create = spec.profiled_create
    if profiled_create is not None:
        profiled_create.argtypes = native.venturi_qwen_create.argtypes + (
            [C.c_int] if spec.experimental_flags or not spec.column_profile else []
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
        upload = backend.upload_rmsnorm if cfg.get("venturi_gguf_q8_0") else backend.upload_zero_centered_rmsnorm
        r = upload(
            owned.read_float(role),
            dtype=dtype,
            device=target,
            epsilon=cfg["rms_norm_eps"],
        )
        keep.append(r)
        return r.storage_tensors[0].data_ptr()

    def matrix(role):
        obj = owned._object(role)
        codec = codecs.codec_for(obj["format"])
        if codec.quantized:
            allowed = codecs.EXECUTION_QUALIFIED.get(
                torch.cuda.get_device_capability(spec.target), set()
            )
            if obj["format"] not in allowed:
                raise ValueError(
                    f"Native matrix descriptor requires an execution-qualified "
                    f"codec on this architecture, got {obj['format']!r}; "
                    f"qualified here: {sorted(allowed)}"
                )
        bits, group, groups, high_offset, scales_offset, _ = row_geometry(obj["shape"], obj["format"])
        high_abs = obj["offset"] + high_offset if bits > 4 else 0
        return Matrix(obj["offset"], high_abs, obj["offset"] + scales_offset, *obj["shape"])

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
    matrix_format = next(
        s.format
        for s in owned.plan.bindings
        if s.binding == "text/token_embedding"
    )
    matrix_codec = codecs.codec_for(matrix_format)
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
        budget.rope.rotary_dim,
        cfg["linear_num_key_heads"],
        cfg["linear_num_value_heads"],
        cfg["linear_key_head_dim"],
        cfg["linear_value_head_dim"],
        cfg["linear_conv_kernel_dim"],
        max_tokens,
        cfg["max_position_embeddings"],
        int(spec.weight_mode == "resident"),
        matrix_codec.bits,
        matrix_codec.group,
        int(spec.dequant_resident),
        cfg["rms_norm_eps"],
        cfg["linear_key_head_dim"] ** -0.5,
        cfg["head_dim"] ** -0.5,
        -1 if peer is None else peer.index,
        int(spec.tensor_parallel_transport == "host"),
        spec.reserve_bytes,
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
    return Descriptors(
        native=native,
        layers=layers,
        final_norm=final_norm,
        rope=budget.rope,
        embedding=embedding,
        head=head,
        spans=spans,
        config=c,
        keep=keep,
        initial_digest=initial_digest,
        initial_stamps=initial_stamps,
    )
