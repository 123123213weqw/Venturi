"""Bias-free pre-RMSNorm causal GQA + SwiGLU decoder layer, not a model adapter.

Weights upload once; token-shape plans own execution metadata/handles only.
Position steps share those handles. K/V projections write directly to a dense
request cache; attention and MLP reuse an explicit per-lane activation arena.
No checkpoint interpretation, GDN, Q/K norm, attention gate, bias or sampling.
"""

from dataclasses import dataclass, field
import threading

import torch

from ..ops.blocks import _no_overlap, _shape, _tensor
from ..ops.weights import DenseWeight, Q4RowSplitWeight
from .gated_mlp import GatedMLPWeights, GatedMLPWorkspace


def _outside_capture(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Use explicit capture/replay; preparation/state mutation must be outside CUDA graph capture"
                )


def _stream(device):
    return (
        torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    )


def _owner_stream(device, stream):
    if _stream(device) != stream:
        raise RuntimeError("Cache/workspace execution must use its owning CUDA stream")


def _unique_bytes(tensors):
    return sum(
        t.numel() * t.element_size()
        for t in {(t.device, t.data_ptr()): t for t in tensors}.values()
    )


@dataclass(frozen=True)
class AttentionDimensions:
    query_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int | None = None
    rope_layout: str = "split_half"

    def __post_init__(self):
        _shape(self.query_heads, self.kv_heads)
        if type(self.head_dim) is not int or self.head_dim not in (64, 128, 256):
            raise ValueError("head_dim must be 64, 128 or 256")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        r = self.head_dim if self.rotary_dim is None else self.rotary_dim
        if type(r) is not int or not 0 < r <= self.head_dim or r % 2:
            raise ValueError("rotary_dim must be positive, even and at most head_dim")
        if self.rope_layout not in ("split_half", "interleaved"):
            raise ValueError("rope_layout must be split_half or interleaved")
        _shape(self.query_width, self.kv_width)
        object.__setattr__(self, "rotary_dim", r)

    @property
    def query_width(self):
        return self.query_heads * self.head_dim

    @property
    def kv_width(self):
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class DecoderLayerWeights:
    attention_norm: torch.Tensor
    query: DenseWeight | Q4RowSplitWeight
    key: DenseWeight | Q4RowSplitWeight
    value: DenseWeight | Q4RowSplitWeight
    output: DenseWeight | Q4RowSplitWeight
    mlp: GatedMLPWeights

    def __post_init__(self):
        if not isinstance(self.mlp, GatedMLPWeights):
            raise TypeError("mlp must be GatedMLPWeights")
        h = self.mlp.gate.in_features
        for weight in (self.query, self.key, self.value, self.output):
            if not isinstance(weight, (DenseWeight, Q4RowSplitWeight)):
                raise TypeError(
                    "Decoder projections require explicit dense or Q4 descriptors"
                )
            weight.validate()
        if any(w.in_features != h for w in (self.query, self.key, self.value)):
            raise ValueError("Q/K/V input widths must equal MLP hidden size")
        if self.key.out_features != self.value.out_features:
            raise ValueError("K/V output widths must match")
        if (self.output.in_features, self.output.out_features) != (
            self.query.out_features,
            h,
        ):
            raise ValueError("Output projection must map query width to hidden size")
        if (
            not isinstance(self.attention_norm, torch.Tensor)
            or self.attention_norm.shape != (h,)
            or not self.attention_norm.is_floating_point()
        ):
            raise ValueError("attention_norm must be a floating-point [hidden] vector")

    def upload(
        self,
        backend,
        *,
        dimensions,
        dtype=torch.bfloat16,
        device="cuda:0",
        epsilon=1e-6,
    ):
        if not isinstance(dimensions, AttentionDimensions):
            raise TypeError("Explicit AttentionDimensions are required")
        if (
            self.query.out_features != dimensions.query_width
            or self.key.out_features != dimensions.kv_width
        ):
            raise ValueError("Projection widths do not match AttentionDimensions")
        if dtype != torch.bfloat16 and not (
            backend.name == "torch-reference" and dtype == torch.float64
        ):
            raise ValueError(
                "Decoder currently supports BF16, plus explicit FP64 reference mode"
            )
        return ResidentDecoderLayer(
            backend,
            dimensions,
            backend.upload_rmsnorm(
                self.attention_norm, dtype=dtype, device=device, epsilon=epsilon
            ),
            backend.upload_linear(self.query, dtype=dtype, device=device),
            backend.upload_linear(self.key, dtype=dtype, device=device),
            backend.upload_linear(self.value, dtype=dtype, device=device),
            backend.upload_linear(self.output, dtype=dtype, device=device),
            self.mlp.upload(backend, dtype=dtype, device=device, epsilon=epsilon),
        )


@dataclass(frozen=True)
class ResidentDecoderLayer:
    _backend: object
    dimensions: AttentionDimensions
    norm: object
    query: object
    key: object
    value: object
    output: object
    mlp: object

    @property
    def dtype(self):
        return self.query.dtype

    @property
    def device(self):
        return self.query.device

    @property
    def hidden_size(self):
        return self.query.in_features

    @property
    def storage_tensors(self):
        return tuple(
            t
            for p in (
                self.norm,
                self.query,
                self.key,
                self.value,
                self.output,
                self.mlp,
            )
            for t in p.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return _unique_bytes(self.storage_tensors)

    def empty_cache(self, *, capacity):
        """One layer of one request; no allocation per append, no implicit prefix sharing."""
        _shape(capacity, 1)
        self._backend._device(self.dtype, self.device)
        return LayerKVCache(self, capacity)

    def prepare(self, *, tokens):
        _shape(tokens, self.hidden_size)
        d, backend = self.dimensions, self._backend
        cfg = dict(
            tokens=tokens,
            head_dim=d.head_dim,
            rotary_dim=d.rotary_dim,
            layout=d.rope_layout,
            dtype=self.dtype,
            device=self.device,
        )
        rope_q = backend.prepare_rope(heads=d.query_heads, **cfg)
        rope_k = backend.prepare_rope(heads=d.kv_heads, **cfg)
        return PreparedDecoderLayer(
            self,
            self.norm.prepare(tokens=tokens),
            self.query.prepare(tokens=tokens),
            self.key.prepare(tokens=tokens),
            self.value.prepare(tokens=tokens),
            self.output.prepare(tokens=tokens),
            self.mlp.prepare(tokens=tokens),
            rope_q,
            rope_k,
            backend.prepare_add(
                tokens=tokens,
                features=self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
        )


class LayerKVCache:
    """Dense BF16 (or FP64 reference) K/V with host-committed visible length.

    Bound to resident weights and one CUDA stream. reset/truncate change only
    visibility; stale tail slots are never read. Successful submission commits
    length, not GPU completion. Keep all owners alive until queued work finishes.
    Runtime submission failures poison the cache; reset discards the sequence.
    """

    def __init__(self, resident, capacity):
        if not isinstance(resident, ResidentDecoderLayer):
            raise TypeError("Cache requires ResidentDecoderLayer")
        _shape(capacity, 1)
        _outside_capture(resident.device)
        self._owner = resident
        self.capacity = capacity
        self.device, self.dtype = resident.device, resident.dtype
        d = resident.dimensions
        self._keys = torch.empty(
            (capacity, d.kv_heads, d.head_dim), dtype=self.dtype, device=self.device
        )
        self._values = torch.empty_like(self._keys)
        self._length, self._failed = 0, False
        self._stream = _stream(self.device)
        self._lock = threading.Lock()

    @property
    def length(self):
        return self._length

    @property
    def failed(self):
        return self._failed

    @property
    def storage_tensors(self):
        return (self._keys, self._values)

    @property
    def nbytes(self):
        return _unique_bytes(self.storage_tensors)

    @property
    def keys(self):
        """Borrowed read-only-by-contract visible prefix, valid on the owner stream."""
        _owner_stream(self.device, self._stream)
        return self._keys[: self.length]

    @property
    def values(self):
        _owner_stream(self.device, self._stream)
        return self._values[: self.length]

    def reset(self):
        _outside_capture(self.device)
        _owner_stream(self.device, self._stream)
        with self._lock:
            self._length, self._failed = 0, False

    def truncate(self, length):
        """Discard a visible suffix, not a fork/copy-on-write cache implementation."""
        _outside_capture(self.device)
        _owner_stream(self.device, self._stream)
        with self._lock:
            if self.failed:
                raise RuntimeError("Cache failed; reset required")
            if type(length) is not int or not 0 <= length <= self.length:
                raise ValueError("truncate length must be within the visible prefix")
            self._length = length


def _arena_layout(max_tokens, hidden, slot, dtype):
    alignment = 256 // (torch.finfo(dtype).bits // 8)

    def align(n):
        return (n + alignment - 1) // alignment * alignment

    offsets = [0]
    for width in (hidden, hidden, slot, slot):
        offsets.append(offsets[-1] + align(max_tokens * width))
    return tuple(offsets)


@dataclass(frozen=True)
class DecoderWorkspace:
    """One lane's arena; share across sequential compatible layers/shapes only."""

    activation: torch.Tensor
    linear: torch.Tensor | None
    max_tokens: int
    hidden_size: int
    slot_features: int
    _stream: int | None
    _lock: object = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @property
    def nbytes(self):
        return _unique_bytes(
            (self.activation,) + ((self.linear,) if self.linear is not None else ())
        )

    def _views(self, plan):
        t, h = plan.tokens, plan.resident.hidden_size
        i, q = plan.mlp.intermediate_size, plan.resident.dimensions.query_width
        offsets = _arena_layout(
            self.max_tokens, self.hidden_size, self.slot_features, self.activation.dtype
        )

        def view(slot, width):
            return self.activation[offsets[slot] : offsets[slot] + t * width].view(
                t, width
            )

        hidden, residual = view(0, h), view(1, h)
        return (
            hidden,
            residual,
            view(2, q),
            view(3, q),
            GatedMLPWorkspace(
                hidden,
                view(2, i),
                view(3, i),
                self.linear[: plan.mlp.linear_workspace_bytes]
                if plan.mlp.linear_workspace_bytes
                else None,
            ),
        )


@dataclass(frozen=True)
class PreparedDecoderLayer:
    resident: ResidentDecoderLayer
    norm: object
    query: object
    key: object
    value: object
    output: object
    mlp: object
    rope_q: object
    rope_k: object
    add: object
    _closed: bool = field(default=False, init=False)

    @property
    def tokens(self):
        return self.query.tokens

    @property
    def storage_tensors(self):
        return self.resident.storage_tensors

    @property
    def linear_workspace_bytes(self):
        return max(
            self.mlp.linear_workspace_bytes,
            *(
                p.workspace_bytes
                for p in (self.query, self.key, self.value, self.output)
            ),
        )

    def _check_open(self):
        if self._closed or any(
            not p._finalizer.alive
            for p in (
                self.query,
                self.key,
                self.value,
                self.output,
                self.mlp.gate,
                self.mlp.up,
                self.mlp.down,
            )
            if hasattr(p, "_finalizer")
        ):
            raise RuntimeError("Decoder execution plan has been closed")

    def at(self, *, position):
        """Bind position-only attention metadata; never create/copy Linear weights."""
        self._check_open()
        if type(position) is not int or position < 0:
            raise ValueError("position must be a nonnegative integer")
        d = self.resident.dimensions
        attention = self.resident._backend.prepare_attention(
            tokens=self.tokens,
            kv_tokens=position + self.tokens,
            query_heads=d.query_heads,
            kv_heads=d.kv_heads,
            head_dim=d.head_dim,
            dtype=self.resident.dtype,
            device=self.resident.device,
        )
        return DecoderStep(self, position, attention)

    def empty_workspace(self, *, max_tokens=None):
        self._check_open()
        r = self.resident
        _outside_capture(r.device)
        max_tokens = self.tokens if max_tokens is None else max_tokens
        _shape(max_tokens, r.hidden_size)
        if max_tokens < self.tokens:
            raise ValueError("Workspace must fit this plan's tokens")
        slot = max(r.dimensions.query_width, self.mlp.intermediate_size)
        layout = _arena_layout(max_tokens, r.hidden_size, slot, r.dtype)
        return DecoderWorkspace(
            torch.empty(layout[-1], dtype=r.dtype, device=r.device),
            torch.empty(self.linear_workspace_bytes, dtype=torch.uint8, device=r.device)
            if self.linear_workspace_bytes
            else None,
            max_tokens,
            r.hidden_size,
            slot,
            _stream(r.device),
        )

    def close(self):
        """Call only after queued work/replays finish. Other shapes stay valid."""
        for plan in (self.query, self.key, self.value, self.output, self.mlp):
            close = getattr(plan, "close", None)
            if close is not None:
                close()
        object.__setattr__(self, "_closed", True)


@dataclass(frozen=True)
class DecoderStep:
    layer: PreparedDecoderLayer
    position: int
    attention: object

    def _preflight(self, x, cos, sin, out, workspace, cache):
        p, r = self.layer, self.layer.resident
        p._check_open()
        if not isinstance(cache, LayerKVCache) or cache._owner is not r:
            raise ValueError("Cache must belong to these resident layer weights")
        _owner_stream(r.device, cache._stream)
        if cache.failed:
            raise RuntimeError("Cache failed; reset required")
        if cache.length != self.position:
            raise ValueError("Cache visible length does not match step position")
        end = self.position + p.tokens
        if end > cache.capacity:
            raise ValueError("Step exceeds KV cache capacity")
        d = r.dimensions
        for name, tensor in (("x", x), ("out", out)):
            _tensor(tensor, (p.tokens, r.hidden_size), r.dtype, r.device, name)
        for name, tensor in (("cos", cos), ("sin", sin)):
            _tensor(
                tensor, (p.tokens, d.rotary_dim // 2), torch.float32, r.device, name
            )
        if not isinstance(workspace, DecoderWorkspace):
            raise TypeError("An explicit DecoderWorkspace is required")
        if (
            type(workspace.max_tokens) is not int
            or workspace.max_tokens < p.tokens
            or workspace.hidden_size != r.hidden_size
            or workspace.slot_features < max(d.query_width, p.mlp.intermediate_size)
        ):
            raise ValueError("Workspace extents do not fit the decoder layer")
        _shape(workspace.max_tokens, workspace.slot_features)
        _owner_stream(r.device, workspace._stream)
        layout = _arena_layout(
            workspace.max_tokens,
            workspace.hidden_size,
            workspace.slot_features,
            r.dtype,
        )
        _tensor(
            workspace.activation,
            (layout[-1],),
            r.dtype,
            r.device,
            "workspace.activation",
        )
        if r.device.type == "cuda" and workspace.activation.data_ptr() % 256:
            raise ValueError("Activation arena requires 256-byte alignment")
        buffers = [out, workspace.activation]
        if workspace.linear is None:
            if p.linear_workspace_bytes:
                raise ValueError("Explicit linear workspace is required")
        else:
            if not isinstance(workspace.linear, torch.Tensor):
                raise TypeError("workspace.linear must be a tensor")
            if (
                workspace.linear.ndim != 1
                or workspace.linear.numel() < p.linear_workspace_bytes
            ):
                raise ValueError("Linear workspace is too small")
            _tensor(
                workspace.linear,
                (workspace.linear.numel(),),
                torch.uint8,
                r.device,
                "workspace.linear",
            )
            if workspace.linear.data_ptr() % 256:
                raise ValueError("cuBLAS workspace requires 256-byte alignment")
            buffers.append(workspace.linear)
        for name, tensor in (("keys", cache._keys), ("values", cache._values)):
            _tensor(
                tensor,
                (cache.capacity, d.kv_heads, d.head_dim),
                r.dtype,
                r.device,
                "cache." + name,
            )
            buffers.append(tensor)
        readonly = (x, cos, sin) + r.storage_tensors
        for index, buffer in enumerate(buffers):
            _no_overlap(buffer, readonly + tuple(buffers[:index]))
        return end

    def _execute(self, x, cos, sin, out, workspace, cache):
        p, d = self.layer, self.layer.resident.dimensions
        hidden, residual, q, attended, mlp_ws = workspace._views(p)
        end = self.position + p.tokens
        key, value = (
            cache._keys[self.position : end],
            cache._values[self.position : end],
        )
        p.norm.run(x, out=hidden)
        for linear, target in (
            (p.query, q),
            (p.key, key.view(p.tokens, d.kv_width)),
            (p.value, value.view(p.tokens, d.kv_width)),
        ):
            linear.run(
                hidden,
                out=target,
                workspace=workspace.linear[: linear.workspace_bytes]
                if linear.workspace_bytes
                else None,
            )
        # Linear writes directly into uncommitted tail slots; no append copy.
        p.rope_q.run(
            q.view(p.tokens, d.query_heads, d.head_dim),
            cos,
            sin,
            out=q.view(p.tokens, d.query_heads, d.head_dim),
        )
        p.rope_k.run(key, cos, sin, out=key)
        self.attention.run(
            q.view(p.tokens, d.query_heads, d.head_dim),
            cache._keys[:end],
            cache._values[:end],
            out=attended.view(p.tokens, d.query_heads, d.head_dim),
        )
        p.output.run(
            attended,
            out=hidden,
            workspace=workspace.linear[: p.output.workspace_bytes]
            if p.output.workspace_bytes
            else None,
        )
        p.add.run(x, hidden, out=residual)
        p.mlp.run(residual, out=out, workspace=mlp_ws)

    def run(self, x, cos, sin, *, out, workspace, cache):
        _outside_capture(self.layer.resident.device)
        if not isinstance(cache, LayerKVCache):
            raise TypeError("An explicit LayerKVCache is required")
        if not isinstance(workspace, DecoderWorkspace):
            raise TypeError("An explicit DecoderWorkspace is required")
        with workspace._lock, cache._lock:
            end = self._preflight(x, cos, sin, out, workspace, cache)
            try:
                self._execute(x, cos, sin, out, workspace, cache)
            except Exception:
                cache._failed = True
                raise
            cache._length = end
        return out

    def capture(self, x, cos, sin, *, out, workspace, cache, warmup=3):
        """Capture one fixed position, with no visible-length update during capture.

        Prefix must already be valid. Warmup/capture overwrite only uncommitted
        tail slots. Replay uses CapturedDecoderStep.run(), never a raw graph.
        """
        device = self.layer.resident.device
        if device.type != "cuda":
            raise ValueError("Decoder graph capture requires CUDA")
        _outside_capture(device)
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        if not isinstance(cache, LayerKVCache):
            raise TypeError("An explicit LayerKVCache is required")
        if not isinstance(workspace, DecoderWorkspace):
            raise TypeError("An explicit DecoderWorkspace is required")
        with workspace._lock, cache._lock:
            self._preflight(x, cos, sin, out, workspace, cache)
            owner = torch.cuda.current_stream(device)
            preparation = torch.cuda.Stream(device=device)
            preparation.wait_stream(owner)
            try:
                # PyTorch requires capture on a non-default stream. Cache/arena
                # ownership stays with the caller; bridge both dependencies while
                # the cache is exclusively locked, and finish setup before return.
                with torch.cuda.device(device), torch.cuda.stream(preparation):
                    for _ in range(warmup):
                        self._execute(x, cos, sin, out, workspace, cache)
                    preparation.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=preparation):
                        self._execute(x, cos, sin, out, workspace, cache)
                owner.wait_stream(preparation)
                owner.synchronize()
            except Exception:
                cache._failed = True
                preparation.synchronize()
                raise
        return CapturedDecoderStep(self, graph, x, cos, sin, out, workspace, cache)


def _signature(tensors):
    return tuple((t.data_ptr(), tuple(t.shape), t.dtype, t.device) for t in tensors)


@dataclass(frozen=True)
class CapturedDecoderStep:
    """Fixed-buffer, fixed-position graph owner; replay commits cache visibility."""

    step: DecoderStep
    _graph: object
    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    out: torch.Tensor
    workspace: DecoderWorkspace
    cache: LayerKVCache

    def __post_init__(self):
        object.__setattr__(self, "_addresses", _signature(self._tensors()))

    def _tensors(self):
        return (
            (self.x, self.cos, self.sin, self.out, self.workspace.activation)
            + ((self.workspace.linear,) if self.workspace.linear is not None else ())
            + self.cache.storage_tensors
            + self.step.layer.storage_tensors
        )

    def run(self):
        _outside_capture(self.step.layer.resident.device)
        with self.workspace._lock, self.cache._lock:
            end = self.step._preflight(
                self.x, self.cos, self.sin, self.out, self.workspace, self.cache
            )
            if _signature(self._tensors()) != self._addresses:
                raise ValueError("Captured tensor storage changed; recapture required")
            try:
                self._graph.replay()
            except Exception:
                self.cache._failed = True
                raise
            self.cache._length = end
        return self.out
