"""Native-operator Qwen full-attention decoder layer with dense request KV.

Zero-centered Q/K norm, packed query/output gate, partial text RoPE, causal GQA,
output/residual and post-norm MLP. Dense cache lifecycle/capture is shared with
other adapters; this is not paged KV, batching or a whole-model scheduler.
"""

from dataclasses import dataclass, field
import threading

import torch

from ..ops.blocks import _shape, _tensor, _no_overlap
from .decoder_layer import (
    AttentionDimensions,
    DenseKVCacheOwner,
    DenseAttentionWorkspace,
    LayerKVCache,
    _outside_capture,
    _stream,
    _owner_stream,
    _run_dense_step,
    _capture_dense_step,
)
from .gated_mlp import ResidentGatedMLP, GatedMLPWorkspace
from .qwen_weights import QwenQ8Weights
from .qwen_rope import QwenTextRoPE


def _layout(plan):
    r, d, t = plan.resident, plan.resident.dimensions, plan.tokens
    widths = dict(
        normalized=r.hidden_size,
        qg=2 * d.query_width,
        key=d.kv_width,
        query=d.query_width,
        attended=d.query_width,
        projected=r.hidden_size,
        residual=r.hidden_size,
        mlp_norm=r.hidden_size,
        mlp_gate=plan.mlp.intermediate_size,
        mlp_up=plan.mlp.intermediate_size,
    )
    size, spans = 0, {}
    alignment = 256 // (torch.finfo(r.dtype).bits // 8)
    for name, width in widths.items():
        spans[name] = (size, width)
        size += (t * width + alignment - 1) // alignment * alignment
    return spans, size


@dataclass(frozen=True)
class ResidentQwenAttentionLayer(DenseKVCacheOwner):
    _backend: object
    layer_index: int
    dimensions: AttentionDimensions
    rope: QwenTextRoPE
    norm: object
    query_gate: object
    key: object
    value: object
    output: object
    qk_norm: object
    mlp: ResidentGatedMLP

    @classmethod
    def from_q8(cls, weights, backend, *, layer, dtype=torch.bfloat16, device="cuda:0"):
        if not isinstance(weights, QwenQ8Weights):
            raise TypeError("Expected complete preflighted QwenQ8Weights")
        weights.layer_bindings(layer)
        cfg = weights.config
        if cfg["layer_types"][layer] != "full_attention":
            raise ValueError("Requested layer is not full attention")
        if dtype not in (torch.float16, torch.bfloat16) and not (
            backend.capabilities.reference and dtype == torch.float64
        ):
            raise ValueError(
                "Qwen attention layer supports FP16/BF16, plus explicit FP64 reference mode"
            )
        target = backend.validate_device(dtype, device)
        _outside_capture(target)
        rope = QwenTextRoPE.from_config(cfg)
        dimensions = AttentionDimensions(
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            rope.rotary_dim,
            "split_half",
        )
        prefix = f"text/layers/{layer}/"

        def norm(role):
            return backend.upload_zero_centered_rmsnorm(
                weights.read_float(prefix + role),
                dtype=dtype,
                device=target,
                epsilon=cfg["rms_norm_eps"],
            )

        def upload(role):
            return backend.upload_linear(
                weights.read_linear(prefix + role), dtype=dtype, device=target
            )

        input_norm = norm("input_norm")
        q, k, v, o = (upload("attention/" + name) for name in ("q", "k", "v", "o"))
        qk = backend.upload_qwen_qk_norm(
            weights.read_float(prefix + "attention/q_norm"),
            weights.read_float(prefix + "attention/k_norm"),
            dtype=dtype,
            device=target,
            epsilon=cfg["rms_norm_eps"],
        )
        mlp_norm = norm("post_attention_norm")
        gate, up, down = (upload("mlp/" + name) for name in ("gate", "up", "down"))
        return cls(
            backend,
            layer,
            dimensions,
            rope,
            input_norm,
            q,
            k,
            v,
            o,
            qk,
            ResidentGatedMLP(backend, mlp_norm, gate, up, down),
        )

    @property
    def dtype(self):
        return self.query_gate.dtype

    @property
    def device(self):
        return self.query_gate.device

    @property
    def hidden_size(self):
        return self.query_gate.in_features

    @property
    def storage_tensors(self):
        return tuple(
            t
            for op in (
                self.norm,
                self.query_gate,
                self.key,
                self.value,
                self.output,
                self.qk_norm,
                self.mlp,
            )
            for t in op.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def empty_cache(self, *, capacity):
        if type(capacity) is not int or not 0 < capacity <= self.rope.maximum_position:
            raise ValueError("Cache capacity exceeds context")
        return LayerKVCache(self, capacity)

    def prepare(self, *, tokens):
        _shape(tokens, self.hidden_size)
        _outside_capture(self.device)
        d = self.dimensions
        return PreparedQwenAttentionLayer(
            self,
            self.norm.prepare(tokens=tokens),
            *(
                p.prepare(tokens=tokens)
                for p in (self.query_gate, self.key, self.value, self.output)
            ),
            self.qk_norm.prepare(
                tokens=tokens,
                query_heads=d.query_heads,
                kv_heads=d.kv_heads,
                rotary_dim=d.rotary_dim,
            ),
            self._backend.prepare_qwen_output_gate(
                tokens=tokens,
                heads=d.query_heads,
                head_dim=d.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            self.mlp.prepare(tokens=tokens),
            self._backend.prepare_add(
                tokens=tokens,
                features=self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
        )


@dataclass(frozen=True)
class QwenAttentionWorkspace(DenseAttentionWorkspace):
    activation: torch.Tensor
    linear: torch.Tensor | None
    buffers: dict
    query_heads: torch.Tensor
    key_heads: torch.Tensor
    attended_heads: torch.Tensor
    mlp: GatedMLPWorkspace
    projection_scratch: tuple
    _stream: int | None
    _lock: object = field(
        default_factory=threading.Lock, init=False, compare=False, repr=False
    )

    @property
    def nbytes(self):
        return self.activation.numel() * self.activation.element_size() + (
            self.linear.numel() if self.linear is not None else 0
        )


@dataclass(frozen=True)
class PreparedQwenAttentionLayer:
    resident: ResidentQwenAttentionLayer
    norm: object
    query_gate: object
    key: object
    value: object
    output: object
    qk_norm: object
    gate: object
    mlp: object
    add: object
    _closed: bool = field(default=False, init=False)

    @property
    def tokens(self):
        return self.query_gate.tokens

    @property
    def projections(self):
        return (self.query_gate, self.key, self.value, self.output)

    @property
    def storage_tensors(self):
        return self.resident.storage_tensors

    @property
    def linear_workspace_bytes(self):
        return max(
            self.mlp.linear_workspace_bytes,
            *(p.workspace_bytes for p in self.projections),
        )

    def _check_open(self):
        if self._closed or any(
            not p._finalizer.alive
            for p in (*self.projections, self.mlp.gate, self.mlp.up, self.mlp.down)
            if hasattr(p, "_finalizer")
        ):
            raise RuntimeError("Qwen attention plan has been closed")

    def close(self):
        for p in (*self.projections, self.mlp):
            close = getattr(p, "close", None)
            if close is not None:
                close()
        object.__setattr__(self, "_closed", True)

    def empty_workspace(self):
        self._check_open()
        r, d, t = self.resident, self.resident.dimensions, self.tokens
        _outside_capture(r.device)
        spans, size = _layout(self)
        activation = torch.empty(size, dtype=r.dtype, device=r.device)
        buffers = {
            name: activation[start : start + t * width].view(t, width)
            for name, (start, width) in spans.items()
        }
        linear = (
            torch.empty(self.linear_workspace_bytes, dtype=torch.uint8, device=r.device)
            if self.linear_workspace_bytes
            else None
        )

        def scratch(size):
            return linear[:size] if size else None

        return QwenAttentionWorkspace(
            activation,
            linear,
            buffers,
            buffers["query"].view(t, d.query_heads, d.head_dim),
            buffers["key"].view(t, d.kv_heads, d.head_dim),
            buffers["attended"].view(t, d.query_heads, d.head_dim),
            GatedMLPWorkspace(
                buffers["mlp_norm"],
                buffers["mlp_gate"],
                buffers["mlp_up"],
                scratch(self.mlp.linear_workspace_bytes),
            ),
            tuple(scratch(p.workspace_bytes) for p in self.projections),
            _stream(r.device),
        )

    def at(self, *, position, cache):
        self._check_open()
        r, d = self.resident, self.resident.dimensions
        _outside_capture(r.device)
        if not isinstance(cache, LayerKVCache) or cache._owner is not r:
            raise ValueError("Cache belongs to different resident weights")
        if (
            type(position) is not int
            or position < 0
            or position + self.tokens > min(cache.capacity, r.rope.maximum_position)
        ):
            raise ValueError("Step exceeds KV cache capacity")
        end = position + self.tokens
        return QwenAttentionStep(
            self,
            position,
            cache,
            r._backend.prepare_attention(
                tokens=self.tokens,
                kv_tokens=end,
                query_heads=d.query_heads,
                kv_heads=d.kv_heads,
                head_dim=d.head_dim,
                dtype=r.dtype,
                device=r.device,
            ),
            cache._keys[position:end],
            cache._values[position:end].view(self.tokens, d.kv_width),
            cache._keys[:end],
            cache._values[:end],
        )


@dataclass(frozen=True)
class QwenAttentionStep:
    layer: PreparedQwenAttentionLayer
    position: int
    cache: LayerKVCache
    attention: object
    key_tail: torch.Tensor
    value_tail: torch.Tensor
    key_prefix: torch.Tensor
    value_prefix: torch.Tensor

    def _preflight(self, x, cos, sin, out, workspace, cache):
        p, r, w = self.layer, self.layer.resident, workspace
        p._check_open()
        d = r.dimensions
        t = p.tokens
        if cache is not self.cache or cache._owner is not r:
            raise ValueError("Cache belongs to different step/weights")
        _owner_stream(r.device, cache._stream)
        if cache.failed:
            raise RuntimeError("Cache failed; reset required")
        if cache.length != self.position:
            raise ValueError("Cache visible length does not match position")
        end = self.position + t
        if end > cache.capacity:
            raise ValueError("Step exceeds KV capacity")
        if not isinstance(w, QwenAttentionWorkspace):
            raise TypeError("Expected QwenAttentionWorkspace")
        _owner_stream(r.device, w._stream)
        for name, tensor, shape, dtype in (
            ("x", x, (t, r.hidden_size), r.dtype),
            ("out", out, (t, r.hidden_size), r.dtype),
            ("cos", cos, (t, d.rotary_dim // 2), torch.float32),
            ("sin", sin, (t, d.rotary_dim // 2), torch.float32),
        ):
            _tensor(tensor, shape, dtype, r.device, name)
        spans, size = _layout(p)
        _tensor(w.activation, (size,), r.dtype, r.device, "activation arena")
        if not isinstance(w.buffers, dict) or w.buffers.keys() != spans.keys():
            raise ValueError("Workspace buffer directory mismatch")
        for name, (start, width) in spans.items():
            buf = w.buffers[name]
            _tensor(buf, (t, width), r.dtype, r.device, name)
            if (
                buf.data_ptr()
                != w.activation.data_ptr() + start * w.activation.element_size()
            ):
                raise ValueError("Workspace buffer address changed")
        for name, view, flat, shape in (
            (
                "query heads",
                w.query_heads,
                w.buffers["query"],
                (t, d.query_heads, d.head_dim),
            ),
            ("key heads", w.key_heads, w.buffers["key"], (t, d.kv_heads, d.head_dim)),
            (
                "attended heads",
                w.attended_heads,
                w.buffers["attended"],
                (t, d.query_heads, d.head_dim),
            ),
        ):
            _tensor(view, shape, r.dtype, r.device, name)
            if view.data_ptr() != flat.data_ptr():
                raise ValueError("Workspace head view address changed")
        if p.linear_workspace_bytes:
            _tensor(
                w.linear,
                (p.linear_workspace_bytes,),
                torch.uint8,
                r.device,
                "linear scratch",
            )
            if w.linear.data_ptr() % 256:
                raise ValueError("Linear scratch must be aligned")
        elif w.linear is not None:
            raise ValueError("No linear scratch required")
        if len(w.projection_scratch) != 4:
            raise ValueError("Projection scratch metadata mismatch")
        for op, scratch in zip(p.projections, w.projection_scratch):
            if op.workspace_bytes:
                _tensor(
                    scratch,
                    (op.workspace_bytes,),
                    torch.uint8,
                    r.device,
                    "projection scratch",
                )
                if scratch.data_ptr() != w.linear.data_ptr():
                    raise ValueError("Scratch arena prefix mismatch")
            elif scratch is not None:
                raise ValueError("Workspace-free projection received scratch")
        if not isinstance(w.mlp, GatedMLPWorkspace):
            raise TypeError("Expected MLP workspace")
        p.mlp.validate(w.buffers["residual"], out=out, workspace=w.mlp)
        for actual, name in (
            (w.mlp.hidden, "mlp_norm"),
            (w.mlp.gate, "mlp_gate"),
            (w.mlp.up, "mlp_up"),
        ):
            if actual.data_ptr() != w.buffers[name].data_ptr():
                raise ValueError("MLP view address mismatch")
        if (
            p.mlp.linear_workspace_bytes
            and w.mlp.linear.data_ptr() != w.linear.data_ptr()
        ):
            raise ValueError("MLP scratch arena prefix mismatch")
        for name, buf in (("keys", cache._keys), ("values", cache._values)):
            _tensor(
                buf, (cache.capacity, d.kv_heads, d.head_dim), r.dtype, r.device, name
            )
        for name, view, base, start, shape in (
            (
                "key tail",
                self.key_tail,
                cache._keys,
                self.position,
                (t, d.kv_heads, d.head_dim),
            ),
            (
                "value tail",
                self.value_tail,
                cache._values,
                self.position,
                (t, d.kv_width),
            ),
            (
                "key prefix",
                self.key_prefix,
                cache._keys,
                0,
                (end, d.kv_heads, d.head_dim),
            ),
            (
                "value prefix",
                self.value_prefix,
                cache._values,
                0,
                (end, d.kv_heads, d.head_dim),
            ),
        ):
            _tensor(view, shape, r.dtype, r.device, name)
            if (
                view.data_ptr()
                != base.data_ptr() + start * d.kv_width * base.element_size()
            ):
                raise ValueError("Bound cache view address changed")
        p.qk_norm.validate(
            w.buffers["qg"],
            w.key_heads,
            cos,
            sin,
            query_out=w.query_heads,
            key_out=self.key_tail,
        )
        p.gate.validate(w.attended_heads, w.buffers["qg"], out=w.attended_heads)
        readonly = (x, cos, sin) + r.storage_tensors
        writable = (out, w.activation, *cache.storage_tensors) + (
            (w.linear,) if w.linear is not None else ()
        )
        for i, buf in enumerate(writable):
            _no_overlap(buf, readonly + writable[:i])
        return end

    def _execute(self, x, cos, sin, out, workspace, cache):
        p, w = self.layer, workspace
        b = w.buffers
        p.norm.run(x, out=b["normalized"])
        for op, target, scratch in zip(
            p.projections[:3],
            (b["qg"], b["key"], self.value_tail),
            w.projection_scratch[:3],
        ):
            op.run(b["normalized"], out=target, workspace=scratch)
        p.qk_norm.run(
            b["qg"],
            w.key_heads,
            cos,
            sin,
            query_out=w.query_heads,
            key_out=self.key_tail,
        )
        self.attention.run(
            w.query_heads, self.key_prefix, self.value_prefix, out=w.attended_heads
        )
        p.gate.run(w.attended_heads, b["qg"], out=w.attended_heads)
        p.output.run(
            b["attended"], out=b["projected"], workspace=w.projection_scratch[3]
        )
        p.add.run(x, b["projected"], out=b["residual"])
        p.mlp.run(b["residual"], out=out, workspace=w.mlp)

    def run(self, x, cos, sin, *, out, workspace):
        return _run_dense_step(
            self, x, cos, sin, out=out, workspace=workspace, cache=self.cache
        )

    def capture(self, x, cos, sin, *, out, workspace, warmup=3):
        return _capture_dense_step(
            self,
            x,
            cos,
            sin,
            out=out,
            workspace=workspace,
            cache=self.cache,
            warmup=warmup,
        )
