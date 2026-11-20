"""Strict Qwen3.5 text source roles; no weight reads or architecture guessing.

Original HF names are kept separate from logical runtime bindings. Norm vectors
remain raw zero-centered parameters; convolution alone is transposed from
[channels,1,kernel] to chronological [kernel,channels]. Q/gate rows stay head-
interleaved, and Q/K/V or gate/up projections are not silently fused/reordered.
"""

from dataclasses import dataclass
import copy
import math

from ..checkpoint import Checkpoint, CheckpointError
from ..ninfer import row_geometry


@dataclass(frozen=True)
class QwenSourceBinding:
    source: str
    binding: str
    source_shape: tuple[int, ...]
    shape: tuple[int, ...]
    format: str
    transform: str = "identity"

    @property
    def nbytes(self):
        if self.format == "q8_g32_fp16":
            return row_geometry(self.shape, self.format)[-1]
        return math.prod(self.shape) * (4 if self.format == "fp32" else 2)

    @property
    def source_dtypes(self):
        return ("BF16", "F32") if self.format == "fp32" else ("BF16",)


@dataclass(frozen=True)
class QwenSourcePlan:
    config: dict
    bindings: tuple[QwenSourceBinding, ...]
    prefix: str
    tied_embeddings: bool

    @classmethod
    def from_config(cls, raw):
        if not isinstance(raw, dict) or raw.get("quantization_config") is not None:
            raise ValueError("Expected original floating Qwen3.5 source config")
        if raw.get("model_type") == "qwen3_5":
            cfg, prefix = raw.get("text_config"), "model.language_model"
        elif raw.get("model_type") == "qwen3_5_text":
            cfg, prefix = raw, "model"
        else:
            raise ValueError(
                "Unsupported source model_type; expected Qwen3.5 text or conditional model"
            )
        if (
            not isinstance(cfg, dict)
            or cfg.get("model_type") != "qwen3_5_text"
            or cfg.get("quantization_config") is not None
        ):
            raise ValueError("Expected unquantized qwen3_5_text config")
        cfg = copy.deepcopy(cfg)
        keys = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "vocab_size",
            "max_position_embeddings",
        )
        if any(type(cfg.get(k)) is not int or not 0 < cfg[k] < 2**31 for k in keys):
            raise ValueError("Qwen dimensions must be positive int32 values")
        h, i, layers, hq, hkv, d, hk, hv, dk, dv, kernel, vocab, _ = (
            cfg[k] for k in keys
        )
        if hq % hkv or hv % hk or dk not in (64, 128, 256) or kernel > 32:
            raise ValueError("Unsupported Qwen attention/GDN geometry")
        if (
            cfg.get("dtype") != "bfloat16"
            or cfg.get("hidden_act") != "silu"
            or cfg.get("attention_bias") is not False
            or cfg.get("attn_output_gate") is not True
            or cfg.get("mlp_only_layers", []) != []
        ):
            raise ValueError(
                "Expected original BF16 bias-free gated-attention/SwiGLU Qwen"
            )
        epsilon = cfg.get("rms_norm_eps")
        if (
            type(epsilon) not in (int, float)
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("Invalid Qwen norm epsilon")
        kinds = cfg.get("layer_types")
        if (
            not isinstance(kinds, list)
            or len(kinds) != layers
            or any(k not in ("linear_attention", "full_attention") for k in kinds)
        ):
            raise ValueError("Qwen requires an explicit supported type for every layer")
        tied = raw.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
        if (
            type(tied) is not bool
            or "tie_word_embeddings" in cfg
            and cfg["tie_word_embeddings"] != tied
        ):
            raise ValueError("Inconsistent tied-embedding config")
        if cfg.get("mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("Qwen GDN recurrent state must be FP32")
        cfg["tie_word_embeddings"] = tied
        specs = []

        def add(
            source,
            target,
            shape,
            format="q8_g32_fp16",
            *,
            transform="identity",
            source_shape=None,
        ):
            if any(
                type(d) is not int or not 0 < d < 2**31
                for d in (*shape, *(source_shape or shape))
            ):
                raise ValueError(
                    "Derived Qwen tensor dimensions must fit positive int32"
                )
            specs.append(
                QwenSourceBinding(
                    source,
                    target,
                    tuple(source_shape or shape),
                    tuple(shape),
                    format,
                    transform,
                )
            )

        add(prefix + ".embed_tokens.weight", "text/token_embedding", (vocab, h))
        if not tied:
            add("lm_head.weight", "text/output_head", (vocab, h))
        add(prefix + ".norm.weight", "text/output_norm", (h,), "bf16")
        for layer, kind in enumerate(kinds):
            src, dst = f"{prefix}.layers.{layer}.", f"text/layers/{layer}/"
            add(src + "input_layernorm.weight", dst + "input_norm", (h,), "bf16")
            add(
                src + "post_attention_layernorm.weight",
                dst + "post_attention_norm",
                (h,),
                "bf16",
            )
            for name, shape in (("gate", (i, h)), ("up", (i, h)), ("down", (h, i))):
                add(src + f"mlp.{name}_proj.weight", dst + f"mlp/{name}", shape)
            if kind == "linear_attention":
                channels, values = 2 * hk * dk + hv * dv, hv * dv
                for name, target, shape in (
                    ("in_proj_qkv", "qkv", (channels, h)),
                    ("in_proj_z", "z", (values, h)),
                    ("out_proj", "output", (h, values)),
                ):
                    add(
                        src + f"linear_attn.{name}.weight", dst + f"gdn/{target}", shape
                    )
                # Preserve small sensitive a/b projections as BF16, not Q8.
                for name in ("a", "b"):
                    add(
                        src + f"linear_attn.in_proj_{name}.weight",
                        dst + f"gdn/{name}",
                        (hv, h),
                        "bf16",
                    )
                add(
                    src + "linear_attn.conv1d.weight",
                    dst + "gdn/convolution",
                    (kernel, channels),
                    "bf16",
                    transform="conv-kc",
                    source_shape=(channels, 1, kernel),
                )
                add(src + "linear_attn.A_log", dst + "gdn/a_log", (hv,), "fp32")
                add(src + "linear_attn.dt_bias", dst + "gdn/dt_bias", (hv,), "fp32")
                add(src + "linear_attn.norm.weight", dst + "gdn/norm", (dv,), "bf16")
            else:
                for name, shape in (
                    ("q", (2 * hq * d, h)),
                    ("k", (hkv * d, h)),
                    ("v", (hkv * d, h)),
                    ("o", (h, hq * d)),
                ):
                    add(
                        src + f"self_attn.{name}_proj.weight",
                        dst + f"attention/{name}",
                        shape,
                    )
                for name in ("q", "k"):
                    add(
                        src + f"self_attn.{name}_norm.weight",
                        dst + f"attention/{name}_norm",
                        (d,),
                        "bf16",
                    )
        return cls(cfg, tuple(specs), prefix, tied)

    def ignored_names(self, names):
        expected = {s.source for s in self.bindings}
        names = set(names)
        missing = expected - names
        ignored = names - expected
        if missing:
            raise CheckpointError(f"Missing Qwen source tensors: {sorted(missing)[:5]}")
        allowed = (
            ("model.visual.", "mtp.") if self.prefix == "model.language_model" else ()
        )
        unknown = [name for name in ignored if not name.startswith(allowed)]
        if unknown:
            raise CheckpointError(
                f"Unexpected source tensors (not silently ignored): {sorted(unknown)[:5]}"
            )
        return tuple(sorted(ignored))

    def validate_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("Expected inspected Checkpoint")
        if QwenSourcePlan.from_config(dict(checkpoint.config)) != self:
            raise CheckpointError("Source plan does not match checkpoint config")
        ignored = self.ignored_names(checkpoint.tensors)
        for spec in self.bindings:
            stored = checkpoint.tensors[spec.source]
            if (
                stored.shape != spec.source_shape
                or stored.dtype not in spec.source_dtypes
            ):
                raise CheckpointError(
                    f"Qwen source shape/dtype mismatch: {spec.source}"
                )
        return ignored
