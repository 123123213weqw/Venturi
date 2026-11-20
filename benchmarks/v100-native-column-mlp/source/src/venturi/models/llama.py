"""Strict classic Llama checkpoint adapter over Venturi's independent operators.

This is dense BF16 inference (FP64 reference for diagnostics), not a universal
HF adapter. Fused RMSNorm/SiLU arithmetic has fewer BF16 rounding boundaries than
Transformers; model logits are not promised bit-identical. No RoPE scaling,
Q/K norm, biases, quantized checkpoint interpretation, or hybrid/GDN fallback.
"""

from dataclasses import dataclass
import math
import threading

import torch

from ..loader import LoadedCheckpoint
from ..ops.blocks import _epsilon
from ..ops.weights import DenseWeight
from .decoder_layer import (
    AttentionDimensions,
    DecoderLayerWeights,
    _outside_capture,
    _owner_stream,
    _shape,
    _stream,
    _unique_bytes,
)
from .gated_mlp import GatedMLPWeights
from .gpt_neox import validate_token_ids


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]

    @property
    def dimensions(self):
        return AttentionDimensions(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.hidden_size // self.num_attention_heads,
        )

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") != "llama":
            raise ValueError(
                "Requires model_type='llama'; hybrid Qwen/GDN is not Llama"
            )
        sizes = {}
        for key in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        ):
            value = raw.get(key)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"Llama {key} must be a positive int32")
            sizes[key] = value
        sizes["num_key_value_heads"] = raw.get(
            "num_key_value_heads", sizes["num_attention_heads"]
        )
        _shape(sizes["num_key_value_heads"], 1)
        if sizes["hidden_size"] % sizes["num_attention_heads"]:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (
            raw.get("head_dim", sizes["hidden_size"] // sizes["num_attention_heads"])
            != sizes["hidden_size"] // sizes["num_attention_heads"]
        ):
            raise ValueError("Independent head_dim is not supported by classic Llama")
        for key in (
            "quantization_config",
            "rope_scaling",
            "rope_parameters",
            "sliding_window",
            "layer_types",
        ):
            if raw.get(key) is not None:
                raise ValueError(f"Llama {key} is unsupported; no silent fallback")
        for key in ("attention_bias", "mlp_bias", "qk_norm", "use_qk_norm"):
            if raw.get(key, False) is not False:
                raise ValueError(f"Llama {key} must be False")
        if raw.get("hidden_act", "silu") != "silu" or raw.get("pretraining_tp", 1) != 1:
            raise ValueError("Llama requires SiLU and pretraining_tp=1")
        if raw.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("Llama requires full-head RoPE")
        tied = raw.get("tie_word_embeddings", False)
        if type(tied) is not bool:
            raise ValueError("tie_word_embeddings must be boolean")
        theta = raw.get("rope_theta", 10000.0)
        if (
            type(theta) not in (int, float)
            or not math.isfinite(theta)
            or not torch.finfo(torch.float32).tiny
            <= theta
            <= torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "rope_theta must be finite, positive and representable in FP32"
            )
        bos, eos = raw.get("bos_token_id"), raw.get("eos_token_id")
        eos = () if eos is None else (eos,) if type(eos) is int else eos
        validate_token_ids(eos, sizes["vocab_size"], allow_empty=True)
        if bos is not None:
            validate_token_ids([bos], sizes["vocab_size"])
        config = cls(
            **sizes,
            rms_norm_eps=_epsilon(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(theta),
            tie_word_embeddings=tied,
            bos_token_id=bos,
            eos_token_ids=tuple(eos),
        )
        config.dimensions  # Validate native head dimensions and grouping now.
        return config


@dataclass(frozen=True)
class LlamaWeights:
    """Named, validated checkpoint views. No transpose, cast, or GPU allocation."""

    config: LlamaConfig
    embedding: DenseWeight
    norm: torch.Tensor
    head: DenseWeight
    layers: tuple[DecoderLayerWeights, ...]
    bound_names: tuple[str, ...]

    @classmethod
    def bind(cls, loaded):
        if not isinstance(loaded, LoadedCheckpoint):
            raise TypeError("Llama binding requires LoadedCheckpoint")
        c, weights = LlamaConfig.from_dict(loaded.config), loaded.weights
        h, i, v, d = c.hidden_size, c.intermediate_size, c.vocab_size, c.dimensions
        expected = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,)}
        if not c.tie_word_embeddings or "lm_head.weight" in weights:
            expected["lm_head.weight"] = (v, h)
        shapes = {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (d.query_width, h),
            "self_attn.k_proj.weight": (d.kv_width, h),
            "self_attn.v_proj.weight": (d.kv_width, h),
            "self_attn.o_proj.weight": (h, d.query_width),
            "mlp.gate_proj.weight": (i, h),
            "mlp.up_proj.weight": (i, h),
            "mlp.down_proj.weight": (h, i),
        }
        for n in range(c.num_hidden_layers):
            expected.update(
                {f"model.layers.{n}.{k}": shape for k, shape in shapes.items()}
            )
        missing, unexpected = (
            set(expected) - weights.keys(),
            weights.keys() - set(expected),
        )
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint bindings: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        dtype = weights["model.embed_tokens.weight"].dtype
        if dtype not in (torch.bfloat16, torch.float64):
            raise ValueError(
                "Llama adapter requires BF16 weights, or explicit FP64 reference fixtures"
            )
        device = weights["model.embed_tokens.weight"].device
        for name, shape in expected.items():
            x = weights[name]
            if (
                x.shape != shape
                or x.dtype != dtype
                or x.device != device
                or not x.is_contiguous()
            ):
                raise ValueError(
                    f"Checkpoint tensor {name} requires {shape}, {dtype}, {device}, contiguous"
                )
            if x.requires_grad or not bool(torch.isfinite(x).all()):
                raise ValueError(
                    f"Checkpoint tensor {name} must be finite and inference-only"
                )
        embed = DenseWeight(weights["model.embed_tokens.weight"])
        if c.tie_word_embeddings:
            if "lm_head.weight" in weights and not torch.equal(
                weights["lm_head.weight"], embed.data
            ):
                raise ValueError("Tied lm_head.weight differs from embedding weight")
            head = embed
        else:
            head = DenseWeight(weights["lm_head.weight"])
        layers = []
        for n in range(c.num_hidden_layers):
            prefix = f"model.layers.{n}."

            def w(name):
                return DenseWeight(weights[prefix + name + ".weight"])

            layers.append(
                DecoderLayerWeights(
                    weights[prefix + "input_layernorm.weight"],
                    w("self_attn.q_proj"),
                    w("self_attn.k_proj"),
                    w("self_attn.v_proj"),
                    w("self_attn.o_proj"),
                    GatedMLPWeights(
                        weights[prefix + "post_attention_layernorm.weight"],
                        w("mlp.gate_proj"),
                        w("mlp.up_proj"),
                        w("mlp.down_proj"),
                    ),
                )
            )
        return cls(
            c,
            embed,
            weights["model.norm.weight"],
            head,
            tuple(layers),
            tuple(sorted(expected)),
        )

    def upload(self, backend, *, device="cuda:0"):
        return LlamaModel(self, backend, device=device)


class LlamaModel:
    """Resident weights only; each session owns its KV, plans and execution lane."""

    def __init__(self, weights, backend, *, device="cuda:0"):
        if not isinstance(weights, LlamaWeights):
            raise TypeError("Bind the checkpoint with LlamaWeights.bind first")
        self.config = c = weights.config
        self.dtype = weights.embedding.data.dtype
        self.device = backend._device(self.dtype, device)
        if self.dtype == torch.float64 and backend.name != "torch-reference":
            raise ValueError("FP64 Llama weights are reference-only")
        self.backend = backend
        self.bound_names = weights.bound_names
        self.embedding = backend.upload_linear(
            weights.embedding, dtype=self.dtype, device=self.device
        )
        self.head = (
            self.embedding
            if c.tie_word_embeddings
            else backend.upload_linear(
                weights.head, dtype=self.dtype, device=self.device
            )
        )
        self.norm = backend.upload_rmsnorm(
            weights.norm, dtype=self.dtype, device=self.device, epsilon=c.rms_norm_eps
        )
        self.layers = tuple(
            w.upload(
                backend,
                dimensions=c.dimensions,
                dtype=self.dtype,
                device=self.device,
                epsilon=c.rms_norm_eps,
            )
            for w in weights.layers
        )

    @property
    def storage_tensors(self):
        return tuple(
            t
            for p in (self.embedding, self.head, self.norm, *self.layers)
            for t in p.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return _unique_bytes(self.storage_tensors)

    def create_session(self, capacity, *, max_tokens=128):
        return LlamaSession(self, capacity, max_tokens=max_tokens)


class LlamaSession:
    """Eager, single-stream, chunked prefill/decode; not a scheduler or graph path.

    One activation arena is reused sequentially across all layers. Shape plans
    reuse model weights and are cached within max_tokens (default 128). Returned
    logits are owned copies, never overwritten by the next forward. Call close()
    after use to synchronize this lane and release plan handles deterministically.
    """

    def __init__(self, model, capacity, *, max_tokens=128):
        _shape(capacity, max_tokens)
        if capacity > model.config.max_position_embeddings:
            raise ValueError("Session capacity exceeds model context")
        self.model, self.capacity = model, capacity
        self.max_tokens = min(max_tokens, capacity)
        self.position, self.failed, self._closed = 0, False, False
        self._stream = _stream(model.device)
        _outside_capture(model.device)
        self._lock = threading.Lock()
        self.caches = tuple(
            layer.empty_cache(capacity=capacity) for layer in model.layers
        )
        self._plans = {}
        first = self._prepare(self.max_tokens)[0]
        self.workspace = first.empty_workspace()
        self._hidden = tuple(
            torch.empty(
                (self.max_tokens, model.config.hidden_size),
                dtype=model.dtype,
                device=model.device,
            )
            for _ in range(2)
        )
        # Position tables are setup, not a numerical fallback for decoder kernels.
        d = model.config.dimensions.head_dim
        inv = model.config.rope_theta ** (
            -torch.arange(0, d, 2, dtype=torch.float32, device=model.device) / d
        )
        phase = (
            torch.arange(capacity, dtype=torch.float32, device=model.device)[:, None]
            * inv[None, :]
        )
        if not bool(torch.isfinite(phase).all()):
            raise ValueError("RoPE phases exceed the finite FP32 table contract")
        self._cos, self._sin = phase.cos().contiguous(), phase.sin().contiguous()
        if model.device.type == "cuda":
            torch.cuda.current_stream(model.device).synchronize()

    @property
    def cache_bytes(self):
        return sum(c.nbytes for c in self.caches)

    def _check(self):
        if self._closed:
            raise RuntimeError("Llama session is closed")
        _outside_capture(self.model.device)
        _owner_stream(self.model.device, self._stream)

    def _prepare(self, tokens):
        if tokens not in self._plans:
            self._plans[tokens] = tuple(
                layer.prepare(tokens=tokens) for layer in self.model.layers
            )
        return self._plans[tokens]

    def reset(self):
        with self._lock:
            self._check()
            for cache in self.caches:
                cache.reset()
            self.position, self.failed = 0, False

    @torch.inference_mode()
    def forward(self, token_ids, *, all_logits=False):
        validate_token_ids(token_ids, self.model.config.vocab_size)
        if type(all_logits) is not bool:
            raise ValueError("all_logits must be boolean")
        with self._lock:
            self._check()
            if self.failed:
                raise RuntimeError("Llama session failed; reset required")
            if self.position + len(token_ids) > self.capacity:
                raise ValueError("Input exceeds session capacity")
            results = []
            try:
                for start in range(0, len(token_ids), self.max_tokens):
                    ids = token_ids[start : start + self.max_tokens]
                    count, p = len(ids), self.position
                    plans = self._prepare(count)
                    x, out = (b[:count] for b in self._hidden)
                    embedding = self.model.embedding.prepare_embedding(ids)
                    embedding.run(out=x)
                    cos, sin = self._cos[p : p + count], self._sin[p : p + count]
                    for plan, cache in zip(plans, self.caches):
                        plan.at(position=p).run(
                            x, cos, sin, out=out, workspace=self.workspace, cache=cache
                        )
                        x, out = out, x
                    # Skip all but the final chunk's output head for last-logit use.
                    if all_logits or start + count == len(token_ids):
                        self.model.norm.prepare(tokens=count).run(x, out=out)
                        projected = out if all_logits else out[-1:]
                        head = self.model.head.prepare(tokens=projected.shape[0])
                        logits = torch.empty(
                            (projected.shape[0], self.model.config.vocab_size),
                            dtype=self.model.dtype,
                            device=self.model.device,
                        )
                        try:
                            head.run(
                                projected,
                                out=logits,
                                workspace=self.workspace.linear
                                if head.workspace_bytes
                                else None,
                            )
                            # Plans/ID buffers must outlive queued native work. This
                            # eager milestone synchronizes before releasing them.
                            if self.model.device.type == "cuda":
                                torch.cuda.current_stream(
                                    self.model.device
                                ).synchronize()
                        finally:
                            close = getattr(head, "close", None)
                            if close is not None:
                                close()
                        results.append(
                            logits
                            if self.model.dtype == torch.float64
                            else logits.float()
                        )
                    elif self.model.device.type == "cuda":
                        torch.cuda.current_stream(self.model.device).synchronize()
                    self.position += count
            except Exception:
                self.failed = True
                raise
            return torch.cat(results, dim=0) if all_logits else results[-1][0]

    def close(self):
        with self._lock:
            self._check()
            if self.model.device.type == "cuda":
                torch.cuda.current_stream(self.model.device).synchronize()
            for plans in self._plans.values():
                for plan in plans:
                    plan.close()
            self._closed = True
