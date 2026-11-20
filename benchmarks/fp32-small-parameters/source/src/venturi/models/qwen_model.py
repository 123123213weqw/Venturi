"""Explicit Q8 hybrid text model and conservative synchronous single-request lane.

Numerical execution uses the selected backend's operators. Native mode never
calls a Transformers model or dequantizes a complete matrix. This eager Python
orchestrator is a correctness milestone, not a batched/paged native runtime or
an optimized graph/sampling engine. FP64/Torch is explicit diagnostic mode only.
"""

from dataclasses import dataclass
import copy
import math
import threading

import torch

from ..ops.blocks import _shape, _no_overlap
from ..ops.vocabulary import _ids
from .decoder_layer import _outside_capture, _owner_stream, _stream, _unique_bytes
from .qwen_attention_layer import ResidentQwenAttentionLayer
from .qwen_gdn_layer import ResidentQwenGDNLayer
from .qwen_rope import QwenTextRoPE
from .qwen_weights import QwenQ8Weights


def qwen_resident_bytes(weights, *, reference=False, dtype=torch.bfloat16, layer=None):
    """Tensor payload budget, not allocator/context/cache/workspace peak memory.

    Native Q8 remains encoded. Reference matrices are explicitly expanded to
    FP32/FP64 (without padded columns); norm gamma is effective FP32/FP64.
    Tied embeddings are counted once. An optional layer selects only its roles.
    Metadata-only: no payload reads/uploads.
    """
    if not isinstance(weights, QwenQ8Weights):
        raise TypeError("Expected complete preflighted QwenQ8Weights")
    specs = weights.plan.bindings
    if layer is not None:
        names = set(weights.layer_bindings(layer))
        specs = tuple(s for s in specs if s.binding in names)
    total = 0
    work_bytes = 8 if dtype == torch.float64 else 4
    for spec in specs:
        if spec.format == "q8_g32_fp16":
            # File planes are separately aligned; the gap between code/scales
            # is not uploaded. Include K padding, but not storage-file padding.
            n, k = spec.shape
            padded = (k + 127) // 128 * 128
            total += (
                n * k * work_bytes if reference else n * (padded + padded // 32 * 2)
            )
        elif spec.binding.endswith(
            ("input_norm", "post_attention_norm", "output_norm", "/q_norm", "/k_norm")
        ):
            total += math.prod(spec.shape) * work_bytes
        elif dtype == torch.float16 and spec.binding.endswith(
            ("/gdn/a", "/gdn/b", "/gdn/convolution")
        ):
            total += math.prod(spec.shape) * 4
        elif reference and dtype == torch.float64:
            total += math.prod(spec.shape) * 8
        else:
            total += spec.nbytes
    return total


def _memory_guard(device, required, reserve, budget):
    if type(reserve) is not int or reserve < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("memory_budget_bytes must be a positive integer")
    available = budget
    if device.type == "cuda":
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info(device)
        available = free if available is None else min(free, available)
    if available is not None and required + reserve > available:
        raise MemoryError(
            f"Qwen needs at least {required:,} tensor bytes plus {reserve:,} reserved "
            f"bytes; available budget is {available:,}. No implicit offload, precision "
            "downgrade or external-backend fallback is allowed."
        )


@dataclass(frozen=True)
class ResidentQwenModel:
    backend: object
    config: dict
    embedding: object
    head: object
    norm: object
    layers: tuple
    rope: QwenTextRoPE

    @classmethod
    def from_q8(
        cls,
        weights,
        backend,
        *,
        dtype=torch.bfloat16,
        device="cuda:0",
        reserve_bytes=512 * 2**20,
        memory_budget_bytes=None,
    ):
        if not isinstance(weights, QwenQ8Weights):
            raise TypeError("Expected complete preflighted QwenQ8Weights")
        if dtype not in (torch.float16, torch.bfloat16) and not (
            backend.name == "torch-reference" and dtype == torch.float64
        ):
            raise ValueError(
                "Qwen model supports FP16/BF16, plus explicit FP64 reference"
            )
        # Reject unsupported RoPE/attention geometry and insufficient memory
        # before reading a single payload or allocating resident weights.
        cfg = copy.deepcopy(weights.config)
        if "full_attention" in cfg["layer_types"] and cfg["head_dim"] not in (
            64,
            128,
            256,
        ):
            raise ValueError("Native Qwen attention head_dim must be 64, 128 or 256")
        rope = QwenTextRoPE.from_config(cfg)
        target = backend._device(dtype, device)
        _outside_capture(target)
        _memory_guard(
            target,
            qwen_resident_bytes(
                weights, reference=backend.name == "torch-reference", dtype=dtype
            ),
            reserve_bytes,
            memory_budget_bytes,
        )

        def upload(binding):
            return backend.upload_linear(
                weights.read_linear(binding), dtype=dtype, device=target
            )

        embedding = upload("text/token_embedding")
        head = embedding if weights.plan.tied_embeddings else upload("text/output_head")
        norm = backend.upload_zero_centered_rmsnorm(
            weights.read_float("text/output_norm"),
            dtype=dtype,
            device=target,
            epsilon=cfg["rms_norm_eps"],
        )
        # Each layer reads/uploads one encoded matrix at a time. No CPU dense
        # model and no second whole-model list of descriptors is retained.
        layers = tuple(
            (
                ResidentQwenGDNLayer
                if kind == "linear_attention"
                else ResidentQwenAttentionLayer
            ).from_q8(weights, backend, layer=i, dtype=dtype, device=target)
            for i, kind in enumerate(cfg["layer_types"])
        )
        return cls(backend, copy.deepcopy(cfg), embedding, head, norm, layers, rope)

    @property
    def dtype(self):
        return self.embedding.dtype

    @property
    def device(self):
        return self.embedding.device

    @property
    def storage_tensors(self):
        return tuple(
            t
            for op in (self.embedding, self.head, self.norm, *self.layers)
            for t in op.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return _unique_bytes(self.storage_tensors)

    def create_session(self, capacity, *, max_tokens=128, reserve_bytes=64 * 2**20):
        return QwenSession(
            self, capacity, max_tokens=max_tokens, reserve_bytes=reserve_bytes
        )


def _close(plans):
    for plan in plans:
        close = getattr(plan, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class _ShapeLane:
    layers: tuple
    workspaces: tuple
    hidden: tuple
    norm: object
    heads: tuple
    head_scratch: torch.Tensor | None

    def close(self):
        _close((*self.layers, *self.heads))

    @property
    def nbytes(self):
        return (
            sum(w.nbytes for w in {id(w): w for w in self.workspaces}.values())
            + _unique_bytes(self.hidden)
            + (self.head_scratch.numel() if self.head_scratch is not None else 0)
        )


class QwenSession:
    """One request, one owning stream, synchronized commit after whole forward.

    Conv history/recurrent state and dense KV are private to this session. One
    workspace per layer KIND is reused sequentially, not per layer. Only decode
    and max-prefill shape lanes are retained; other tail shapes are temporary.
    Errors after execution starts poison the whole request: reset is required,
    not truncate/rollback (GDN state cannot be truncated). Returned logits own
    their storage. No concurrent execution, whole-model graphs or sampling yet.
    """

    def __init__(self, model, capacity, *, max_tokens=128, reserve_bytes=64 * 2**20):
        if not isinstance(model, ResidentQwenModel):
            raise TypeError("Expected ResidentQwenModel")
        _shape(capacity, max_tokens)
        if capacity > model.config["max_position_embeddings"]:
            raise ValueError("Session capacity exceeds model context")
        _outside_capture(model.device)
        c, element = model.config, torch.finfo(model.dtype).bits // 8
        state_element = 8 if model.dtype == torch.float64 else 4
        kinds = c["layer_types"]
        channels = 2 * c["linear_num_key_heads"] * c["linear_key_head_dim"] + (
            c["linear_num_value_heads"] * c["linear_value_head_dim"]
        )
        request_bytes = kinds.count("full_attention") * (
            capacity * c["num_key_value_heads"] * c["head_dim"] * 2 * element
        ) + kinds.count("linear_attention") * (
            (c["linear_conv_kernel_dim"] - 1) * channels * element
            + c["linear_num_value_heads"]
            * c["linear_value_head_dim"]
            * c["linear_key_head_dim"]
            * state_element
        )
        # This is a lower-bound request-state guard. Workspace/logits/allocator
        # overhead additionally consume reserve and can still raise a normal OOM.
        _memory_guard(model.device, request_bytes, reserve_bytes, None)
        self.model, self.capacity = model, capacity
        self.max_tokens = min(capacity, max_tokens)
        self.position, self.failed, self._closed = 0, False, False
        self._stream, self._lock = _stream(model.device), threading.Lock()
        self._states = tuple(
            layer.zero_state() if isinstance(layer, ResidentQwenGDNLayer) else None
            for layer in model.layers
        )
        self._caches = tuple(
            layer.empty_cache(capacity=capacity)
            if isinstance(layer, ResidentQwenAttentionLayer)
            else None
            for layer in model.layers
        )
        self._lanes = {}
        self._sync()

    @property
    def cache_bytes(self):
        return sum(s.nbytes for s in (*self._states, *self._caches) if s is not None)

    @property
    def workspace_bytes(self):
        return sum(lane.nbytes for lane in self._lanes.values())

    def _sync(self):
        if self.model.device.type == "cuda":
            torch.cuda.current_stream(self.model.device).synchronize()

    def _check(self):
        if self._closed:
            raise RuntimeError("Qwen session is closed")
        _outside_capture(self.model.device)
        _owner_stream(self.model.device, self._stream)

    def _make_lane(self, tokens):
        plans, heads = [], []
        try:
            workspaces, shared = [], {}
            for layer in self.model.layers:
                p = layer.prepare(tokens=tokens)
                plans.append(p)
                kind = type(layer)
                if kind not in shared:
                    shared[kind] = p.empty_workspace()
                workspaces.append(shared[kind])
            m = self.model
            hidden = tuple(
                torch.empty(
                    tokens, m.config["hidden_size"], dtype=m.dtype, device=m.device
                )
                for _ in range(2)
            )
            heads.append(m.head.prepare(tokens=1))
            if tokens != 1:
                heads.append(m.head.prepare(tokens=tokens))
            scratch_bytes = max(p.workspace_bytes for p in heads)
            scratch = (
                torch.empty(scratch_bytes, dtype=torch.uint8, device=m.device)
                if scratch_bytes
                else None
            )
            return _ShapeLane(
                tuple(plans),
                tuple(workspaces),
                hidden,
                m.norm.prepare(tokens=tokens),
                tuple(heads),
                scratch,
            )
        except Exception:
            self._sync()
            _close((*plans, *heads))
            raise

    def _lane(self, tokens):
        if tokens not in self._lanes:
            lane = self._make_lane(tokens)
            if tokens in (1, self.max_tokens):
                self._lanes[tokens] = lane
            return lane
        return self._lanes[tokens]

    def reset(self):
        with self._lock:
            self._check()
            # Surface any prior asynchronous failure before reusing mutable
            # storage. A device-level CUDA failure can require a new process.
            self._sync()
            for state in self._states:
                if state is not None:
                    state.history.zero_()
                    state.recurrent.zero_()
            for cache in self._caches:
                if cache is not None:
                    cache.reset()
            self._sync()
            self.position, self.failed = 0, False

    @torch.inference_mode()
    def forward(self, token_ids, *, all_logits=False):
        _ids(token_ids, self.model.config["vocab_size"])
        if type(all_logits) is not bool:
            raise ValueError("all_logits must be boolean")
        with self._lock:
            self._check()
            if self.failed:
                raise RuntimeError("Qwen session failed; reset required")
            if self.position + len(token_ids) > self.capacity:
                raise ValueError("Input exceeds session capacity")
            committed = self.position
            results = []
            active = None
            started = False
            try:
                for start in range(0, len(token_ids), self.max_tokens):
                    ids = token_ids[start : start + self.max_tokens]
                    t, position = len(ids), committed + start
                    active = self._lane(t)
                    x, out = active.hidden
                    cos, sin = self.model.rope.tables(
                        self.model.backend,
                        position=position,
                        tokens=t,
                        dtype=self.model.dtype,
                        device=self.model.device,
                    )
                    embedding = self.model.embedding.prepare_embedding(ids)
                    steps = tuple(
                        p.at(position=position, cache=cache)
                        if cache is not None
                        else None
                        for p, cache in zip(active.layers, self._caches)
                    )
                    # Preflight ALL layers before any numerical/state update for
                    # this chunk. The session lock owns all shared workspaces.
                    for i, (p, w, state, cache, step) in enumerate(
                        zip(
                            active.layers,
                            active.workspaces,
                            self._states,
                            self._caches,
                            steps,
                        )
                    ):
                        source, target = (
                            active.hidden[i % 2],
                            active.hidden[(i + 1) % 2],
                        )
                        if state is not None:
                            p.validate(
                                source,
                                out=target,
                                history=state.history,
                                state=state.recurrent,
                                history_out=state.history,
                                state_out=state.recurrent,
                                workspace=w,
                            )
                        else:
                            step._preflight(source, cos, sin, target, w, cache)
                    for state in self._states:
                        if state is not None:
                            _no_overlap(state.history, self.model.storage_tensors)
                            _no_overlap(state.recurrent, self.model.storage_tensors)
                    started = True
                    embedding.run(out=x)
                    for p, w, state, cache, step in zip(
                        active.layers,
                        active.workspaces,
                        self._states,
                        self._caches,
                        steps,
                    ):
                        if state is not None:
                            p.run(
                                x,
                                out=out,
                                history=state.history,
                                state=state.recurrent,
                                history_out=state.history,
                                state_out=state.recurrent,
                                workspace=w,
                            )
                        else:
                            step.run(x, cos, sin, out=out, workspace=w)
                        x, out = out, x
                    if all_logits or start + t == len(token_ids):
                        active.norm.run(x, out=out)
                        projected = out if all_logits else out[-1:]
                        head = active.heads[-1] if all_logits else active.heads[0]
                        logits = torch.empty(
                            projected.shape[0],
                            self.model.config["vocab_size"],
                            dtype=self.model.dtype,
                            device=self.model.device,
                        )
                        head.run(projected, out=logits, workspace=active.head_scratch)
                        results.append(
                            logits
                            if self.model.dtype == torch.float64
                            else logits.float()
                        )
                    # IDs, tables and temporary shape handles must outlive work;
                    # commit only after all kernels in this chunk completed.
                    self._sync()
                    if t not in self._lanes:
                        active.close()
                    active = None
                result = torch.cat(results, dim=0) if all_logits else results[-1][0]
                self._sync()
            except Exception:
                if started:
                    self.failed = True
                    for cache in self._caches:
                        if cache is not None:
                            cache._failed = True
                raise
            finally:
                if active is not None and all(
                    active is not lane for lane in self._lanes.values()
                ):
                    self._sync()
                    active.close()
            self.position = committed + len(token_ids)
            return result

    def close(self):
        with self._lock:
            self._check()
            self._sync()
            for lane in self._lanes.values():
                lane.close()
            self._lanes.clear()
            self._closed = True
