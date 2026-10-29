"""Classic GPT-NeoX inference using Venturi-bound weights and PyTorch operators.

No Transformers model, downloaded Python, or model.generate is used here. This
first adapter intentionally supports one explicit mathematical architecture.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from ..loader import LoadedCheckpoint
from ..text.tokens import validate_token_ids


def _positive_int(raw, name):
    value = raw.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"GPT-NeoX {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class GPTNeoXConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: float
    rotary_emb_base: float
    rotary_dim: int
    use_parallel_residual: bool
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]):
        if raw.get("model_type") != "gpt_neox":
            raise ValueError(
                "This adapter requires model_type='gpt_neox'; no architecture fallback"
            )
        sizes = {
            key: _positive_int(raw, key)
            for key in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "max_position_embeddings",
            )
        }
        if sizes["hidden_size"] % sizes["num_attention_heads"]:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        for key in ("quantization_config", "rope_scaling", "rope_parameters"):
            if raw.get(key) is not None:
                raise ValueError(
                    f"GPT-NeoX {key} is not supported by this dense classic adapter"
                )
        for modern, classic, default in (
            ("partial_rotary_factor", "rotary_pct", 0.25),
            ("rope_theta", "rotary_emb_base", 10000.0),
        ):
            if modern in raw and (
                type(raw[modern]) not in (int, float)
                or raw[modern] != raw.get(classic, default)
            ):
                raise ValueError(
                    f"{modern} conflicts with the classic {classic} contract"
                )
        if raw.get("hidden_act", "gelu") != "gelu":
            raise ValueError(
                "Only exact GELU is supported, not approximate GELU variants"
            )
        if raw.get("tie_word_embeddings", False) is not False:
            raise ValueError("Tied GPT-NeoX embeddings are not implemented")
        for key in ("attention_bias", "mlp_bias"):
            if raw.get(key, True) is not True:
                raise ValueError(f"GPT-NeoX requires {key}=True")
        parallel = raw.get("use_parallel_residual", True)
        if type(parallel) is not bool:
            raise ValueError("use_parallel_residual must be boolean")
        numbers = {}
        for key, default in (
            ("layer_norm_eps", 1e-5),
            ("rotary_emb_base", 10000.0),
            ("rotary_pct", 0.25),
        ):
            value = raw.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            numbers[key] = float(value)
        if numbers["layer_norm_eps"] <= 0 or numbers["rotary_emb_base"] <= 0:
            raise ValueError("LayerNorm epsilon and RoPE base must be positive")
        if not 0 <= numbers["rotary_pct"] <= 1:
            raise ValueError("rotary_pct must be in [0,1]")
        rotary_dim = int(
            sizes["hidden_size"] // sizes["num_attention_heads"] * numbers["rotary_pct"]
        )
        if rotary_dim % 2:
            raise ValueError("GPT-NeoX rotary dimension must be even")
        bos = raw.get("bos_token_id")
        eos = raw.get("eos_token_id")
        if eos is None:
            eos = ()
        elif type(eos) is int:
            eos = (eos,)
        elif isinstance(eos, (list, tuple)):
            eos = tuple(eos)
        else:
            raise ValueError("eos_token_id must be an integer or list of integers")
        if bos is not None:
            validate_token_ids([bos], sizes["vocab_size"])
        validate_token_ids(eos, sizes["vocab_size"], allow_empty=True)
        return cls(
            **sizes,
            layer_norm_eps=numbers["layer_norm_eps"],
            rotary_emb_base=numbers["rotary_emb_base"],
            rotary_dim=rotary_dim,
            use_parallel_residual=parallel,
            bos_token_id=bos,
            eos_token_ids=tuple(eos),
        )



def _expected_shapes(config):
    h, m, v = config.hidden_size, config.intermediate_size, config.vocab_size
    result = {
        "gpt_neox.embed_in.weight": (v, h),
        "embed_out.weight": (v, h),
        "gpt_neox.final_layer_norm.weight": (h,),
        "gpt_neox.final_layer_norm.bias": (h,),
    }
    for i in range(config.num_hidden_layers):
        prefix = f"gpt_neox.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            result[prefix + name + ".weight"] = (h,)
            result[prefix + name + ".bias"] = (h,)
        for name, shape in (
            ("attention.query_key_value", (3 * h, h)),
            ("attention.dense", (h, h)),
            ("mlp.dense_h_to_4h", (m, h)),
            ("mlp.dense_4h_to_h", (h, m)),
        ):
            result[prefix + name + ".weight"] = shape
            result[prefix + name + ".bias"] = (shape[0],)
    return result


class GPTNeoXModel:
    """Read-only shared weights and RoPE tables; mutable KV state belongs to sessions.

    Learned tensors are bound without transposition or dtype conversion. Caller
    must not mutate the original LoadedCheckpoint while a model is using it.
    """

    def __init__(self, checkpoint: LoadedCheckpoint):
        self.config = GPTNeoXConfig.from_dict(checkpoint.config)
        shapes = _expected_shapes(self.config)
        missing = shapes.keys() - checkpoint.weights.keys()
        if missing:
            raise ValueError(f"Missing GPT-NeoX parameters: {sorted(missing)}")
        embedding = checkpoint.weights["gpt_neox.embed_in.weight"]
        self.device, self.dtype = embedding.device, embedding.dtype
        if self.device.type not in ("cpu", "cuda") or self.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Dense GPT-NeoX requires CPU/CUDA floating-point weights")
        for name, shape in shapes.items():
            tensor = checkpoint.weights[name]
            if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(
                    f"GPT-NeoX parameter {name} requires contiguous shape {shape}"
                )
            if tensor.device != self.device or tensor.dtype != self.dtype:
                raise ValueError(
                    f"GPT-NeoX parameter {name} has inconsistent dtype/device"
                )
        # Legacy serialization buffers are not learned parameters. Recompute
        # RoPE from config in FP32 (FP64 for the numerical test path), rather
        # than using historical FP16-rounded inv_freq buffers.
        legacy = {}
        for i in range(self.config.num_hidden_layers):
            prefix = f"gpt_neox.layers.{i}.attention."
            legacy[prefix + "bias"] = (
                1,
                1,
                self.config.max_position_embeddings,
                self.config.max_position_embeddings,
            )
            legacy[prefix + "masked_bias"] = ()
            legacy[prefix + "rotary_emb.inv_freq"] = (self.config.rotary_dim // 2,)
        for name in checkpoint.weights.keys() - shapes.keys():
            if (
                name not in legacy
                or tuple(checkpoint.weights[name].shape) != legacy[name]
            ):
                raise ValueError(f"Unsupported GPT-NeoX tensor {name}")
        self._checkpoint = checkpoint
        self._weights = checkpoint.weights
        self._prefixes = tuple(
            f"gpt_neox.layers.{i}." for i in range(self.config.num_hidden_layers)
        )
        with torch.inference_mode():
            calculation_dtype = (
                torch.float64 if self.dtype == torch.float64 else torch.float32
            )
            indices = torch.arange(
                0,
                self.config.rotary_dim,
                2,
                device=self.device,
                dtype=calculation_dtype,
            )
            inverse = 1.0 / (
                self.config.rotary_emb_base
                ** (indices / max(self.config.rotary_dim, 1))
            )
            positions = torch.arange(
                self.config.max_position_embeddings,
                device=self.device,
                dtype=calculation_dtype,
            )
            angles = torch.outer(positions, inverse)
            angles = torch.cat((angles, angles), dim=-1)
            self._cos = angles.cos().to(self.dtype)
            self._sin = angles.sin().to(self.dtype)
        self._ready = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._ready = torch.cuda.Event()
                self._ready.record(torch.cuda.current_stream(self.device))

    def create_session(self, capacity: int) -> "GPTNeoXSession":
        return GPTNeoXSession(self, capacity)

    def create_cuda_graph_session(
        self, capacity: int, *, decode_start: int, decode_steps: int,
        prefill_lengths: Sequence[int] = (),
    ):
        from .cuda_graph import GPTNeoXCUDAGraphSession

        return GPTNeoXCUDAGraphSession(
            self, capacity, decode_start=decode_start, decode_steps=decode_steps,
            prefill_lengths=prefill_lengths,
        )

    def _linear(self, x, prefix):
        return F.linear(
            x, self._weights[prefix + ".weight"], self._weights[prefix + ".bias"]
        )

    def _norm(self, x, prefix):
        return F.layer_norm(
            x,
            (self.config.hidden_size,),
            self._weights[prefix + ".weight"],
            self._weights[prefix + ".bias"],
            self.config.layer_norm_eps,
        )


class GPTNeoXSession:
    """One unpadded sequence, fixed-capacity KV cache, single owning CUDA stream.

    Different sessions share model weights but never KV storage. This eager
    correctness runner allocates intermediates; it is not a graph-captured or
    allocation-free full-model implementation.
    """

    def __init__(self, model: GPTNeoXModel, capacity: int):
        if (
            type(capacity) is not int
            or not 0 < capacity <= model.config.max_position_embeddings
        ):
            raise ValueError("Session capacity must fit max_position_embeddings")
        self.model, self.capacity = model, capacity
        self._position, self._failed = 0, False
        self._stream = None
        if model.device.type == "cuda":
            stream = torch.cuda.current_stream(model.device)
            stream.wait_event(model._ready)
            self._stream = stream.cuda_stream
        shape = (
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            capacity,
            model.config.head_dim,
        )
        with torch.inference_mode():
            self._keys = torch.empty(shape, dtype=model.dtype, device=model.device)
            self._values = torch.empty_like(self._keys)

    @property
    def position(self):
        return self._position

    @property
    def cache_bytes(self):
        return 2 * self._keys.numel() * self._keys.element_size()

    def _check_stream(self):
        if (
            self._stream is not None
            and torch.cuda.current_stream(self.model.device).cuda_stream != self._stream
        ):
            raise ValueError("A session must execute/reset on its owning CUDA stream")

    def reset(self):
        self._check_stream()
        self._position, self._failed = 0, False

    def forward(self, token_ids: Sequence[int], *, all_logits: bool = False):
        validate_token_ids(token_ids, self.model.config.vocab_size)
        if type(all_logits) is not bool:
            raise ValueError("all_logits must be boolean")
        self._check_stream()
        if self._failed:
            raise RuntimeError("Session failed during execution; reset it before reuse")
        if self.position + len(token_ids) > self.capacity:
            raise ValueError("Input would exceed the session KV capacity")
        try:
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.model.device.type, enabled=False),
            ):
                return self._forward(token_ids, all_logits)
        except Exception:
            self._failed = True
            raise

    def _forward(self, token_ids, all_logits):
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.model.device)
        return self._forward_tensor(ids, all_logits)

    def _forward_tensor(self, ids, all_logits):
        """Internal tensor-input path; callers validate IDs before execution."""
        model, config = self.model, self.model.config
        length = ids.numel()
        start, end = self.position, self.position + length
        hidden = F.embedding(ids, model._weights["gpt_neox.embed_in.weight"])
        cosine = model._cos[start:end].unsqueeze(0).unsqueeze(0)
        sine = model._sin[start:end].unsqueeze(0).unsqueeze(0)
        # SDPA's rectangular is_causal=True mask is upper-left aligned. Cached
        # chunks require absolute positions instead, not that default mask.
        mask = None
        if start > 0 and length > 1:
            queries = torch.arange(start, end, device=model.device)
            keys = torch.arange(end, device=model.device)
            mask = keys.unsqueeze(0) <= queries.unsqueeze(1)
        for layer, prefix in enumerate(model._prefixes):
            original = hidden
            normalized = model._norm(hidden, prefix + "input_layernorm")
            qkv = model._linear(normalized, prefix + "attention.query_key_value")
            qkv = qkv.view(
                length, config.num_attention_heads, 3 * config.head_dim
            )
            q, k, v = qkv.permute(1, 0, 2).unsqueeze(0).chunk(3, dim=-1)
            if config.rotary_dim:

                def rotate(tensor):
                    part = tensor[..., : config.rotary_dim]
                    first, second = part.chunk(2, dim=-1)
                    shifted = torch.cat((-second, first), dim=-1)
                    return torch.cat(
                        (
                            part * cosine + shifted * sine,
                            tensor[..., config.rotary_dim :],
                        ),
                        dim=-1,
                    )

                q, k = rotate(q), rotate(k)
            self._keys[layer, :, start:end].copy_(k[0])
            self._values[layer, :, start:end].copy_(v[0])
            attention = F.scaled_dot_product_attention(
                q,
                self._keys[layer, :, :end].unsqueeze(0),
                self._values[layer, :, :end].unsqueeze(0),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=start == 0 and length > 1,
            )
            attention = (
                attention[0].transpose(0, 1).reshape(length, config.hidden_size)
            )
            attention = model._linear(attention, prefix + "attention.dense")
            if config.use_parallel_residual:
                mlp_input = original
            else:
                hidden = attention + original
                mlp_input = hidden
            mlp = model._linear(
                model._norm(mlp_input, prefix + "post_attention_layernorm"),
                prefix + "mlp.dense_h_to_4h",
            )
            mlp = model._linear(
                F.gelu(mlp, approximate="none"), prefix + "mlp.dense_4h_to_h"
            )
            hidden = (
                (mlp + attention) + original
                if config.use_parallel_residual
                else mlp + hidden
            )
        # Final normalization is token-local. Next-token generation needs only
        # the last row; retain every row for scoring/training-style callers.
        hidden = model._norm(
            hidden if all_logits else hidden[-1], "gpt_neox.final_layer_norm"
        )
        logits = F.linear(hidden, model._weights["embed_out.weight"])
        self._position = end
        return logits
