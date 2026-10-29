"""Explicit one-phase-at-a-time Q8 streaming for full-model correctness work.

Not an implicit resident fallback, production scheduler, or throughput win. Load
and release vocabulary/one layer/head in separate synchronized phases; never
retain a complete GPU model. Native numerical execution uses the owned operators.
Python handles loading, explicit KV copies, diagnostic traces and synchronization.
A Torch CPU reference profile is deliberately allocating and independent of the
native numerical kernels, not a hidden execution fallback.
"""

from dataclasses import dataclass, field
import copy
import hashlib
import json
import threading

import torch

from ..ninfer import NInferArtifact
from ..ops.blocks import _shape
from ..ops.vocabulary import _ids
from .decoder_layer import _outside_capture, _owner_stream, _stream
from .qwen_model import _memory_guard, qwen_resident_bytes, qwen_request_bytes
from .qwen_weights import QwenQ8Weights
from .qwen_rope import QwenTextRoPE
from .qwen_gdn_layer import ResidentQwenGDNLayer, GDNLayerState
from .qwen_attention_layer import ResidentQwenAttentionLayer


def _stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class _StagedKV:
    keys: torch.Tensor
    values: torch.Tensor

    @property
    def nbytes(self):
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()


@dataclass
class _StreamingLane:
    hidden: tuple
    workspaces: dict = field(default_factory=dict)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.hidden) + sum(
            w.nbytes for w in self.workspaces.values()
        )


class StreamingQwenModel:
    """Metadata and small final norm only; weight phases serialize per model.

    CPU encoded reads are bounded by the largest selected matrix. Native mode
    never expands a quantized matrix. Torch reference mode explicitly does so.
    No graphs, batching or sampler is supplied by this correctness adapter.
    Source/config are immutable by contract and checked throughout execution.
    """

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
        return cls(
            weights,
            backend,
            dtype=dtype,
            device=device,
            reserve_bytes=reserve_bytes,
            memory_budget_bytes=memory_budget_bytes,
        )

    def __init__(
        self, weights, backend, *, dtype, device, reserve_bytes, memory_budget_bytes
    ):
        if not isinstance(weights, QwenQ8Weights):
            raise TypeError("Expected complete preflighted QwenQ8Weights")
        if dtype not in (torch.float16, torch.bfloat16) and not (
            backend.capabilities.reference and dtype == torch.float64
        ):
            raise ValueError(
                "Streaming supports explicit FP16/BF16, plus FP64 reference"
            )
        # Reopen validated metadata so mutating the caller's manifest cannot
        # change which objects are streamed on a later request.
        self._weights = QwenQ8Weights.from_ninfer(NInferArtifact(weights.artifact.path))
        self.backend, self.dtype = backend, dtype
        self.config = copy.deepcopy(self._weights.config)
        self.rope = QwenTextRoPE.from_config(self.config)
        if "full_attention" in self.config["layer_types"] and self.config[
            "head_dim"
        ] not in (64, 128, 256):
            raise ValueError("Native Qwen attention head_dim must be 64, 128 or 256")
        self.device = backend.validate_device(dtype, device)
        self.reserve_bytes, self.memory_budget_bytes = (
            reserve_bytes,
            memory_budget_bytes,
        )
        self._files = tuple(f.path for f in self._weights.artifact.files)
        self._stamps = tuple(_stamp(f) for f in self._files)
        self._metadata_digest = _digest_config(self._weights.artifact.manifest)
        self._config_digest = _digest_config(self.config)
        self._weight_lock = threading.RLock()
        # Reference mode can use substantially more CPU conversion scratch
        # than this resident-tensor budget. Callers must budget it explicitly.
        # Preflight the largest phase before any payload read/device upload.
        self.peak_phase_weight_bytes = max(
            self._layer_bytes(i) for i in range(self.config["num_hidden_layers"])
        )
        self.peak_phase_weight_bytes = max(
            self.peak_phase_weight_bytes,
            *(
                self._role_bytes(r)
                for r in ("text/token_embedding", "text/output_head")
            ),
        )
        _memory_guard(
            self.device,
            self.peak_phase_weight_bytes,
            reserve_bytes,
            memory_budget_bytes,
        )
        self.norm = backend.upload_zero_centered_rmsnorm(
            self._weights.read_float("text/output_norm"),
            dtype=dtype,
            device=self.device,
            epsilon=self.config["rms_norm_eps"],
        )
        self.check_source()

    @property
    def persistent_weight_bytes(self):
        return self.norm.storage_nbytes

    def _role_bytes(self, role):
        obj = self._weights._object(role)
        n, k = obj["shape"]
        if self.backend.capabilities.reference:
            return n * k * (8 if self.dtype == torch.float64 else 4)
        padded = (k + 127) // 128 * 128
        return n * (padded + padded // 32 * 2)

    def _layer_bytes(self, i):
        return qwen_resident_bytes(
            self._weights,
            reference=self.backend.capabilities.reference,
            dtype=self.dtype,
            layer=i,
        )

    def check_source(self):
        if tuple(_stamp(f) for f in self._files) != self._stamps:
            raise RuntimeError(
                "Streaming artifact changed; create a new model after verification"
            )
        if (
            _digest_config(self.config) != self._config_digest
            or _digest_config(self._weights.config) != self._config_digest
            or _digest_config(self._weights.artifact.manifest) != self._metadata_digest
        ):
            raise RuntimeError("Streaming config/metadata changed; create a new model")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _before_phase(self, required):
        self.check_source()
        _outside_capture(self.device)
        self._sync()
        # Release only unused blocks in this process's caching allocator. Live
        # request buffers remain owned. This slow correctness mode makes no
        # allocator-efficiency or concurrent-throughput claim.
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        _memory_guard(
            self.device, required, self.reserve_bytes, self.memory_budget_bytes
        )

    def create_session(self, capacity, *, max_tokens=128, observer=None):
        return StreamingQwenSession(
            self, capacity, max_tokens=max_tokens, observer=observer
        )


class StreamingQwenSession:
    """Single-stream request state with explicit transient KV prefix staging.

    Neutral request KV does not retain resident weights. Each layer gets a new
    normally owned cache; prefix/tail copies never rebind or weaken cache-owner
    checks. Only public position commits after the ENTIRE forward succeeds.
    Failure after any state update poisons the request; reset is required.
    """

    def __init__(self, model, capacity, *, max_tokens, observer):
        _shape(capacity, max_tokens)
        if not isinstance(model, StreamingQwenModel):
            raise TypeError("Expected StreamingQwenModel")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable")
        _outside_capture(model.device)
        request = qwen_request_bytes(model.config, capacity, dtype=model.dtype)
        _memory_guard(
            model.device,
            request + model.peak_phase_weight_bytes,
            model.reserve_bytes,
            model.memory_budget_bytes,
        )
        self.model, self.capacity = model, capacity
        self.max_tokens = min(capacity, max_tokens)
        self.position, self.failed, self._closed = 0, False, False
        self._stream, self._lock = _stream(model.device), threading.Lock()
        self._observer = observer
        self._lanes = {}
        c = model.config
        opt = dict(dtype=model.dtype, device=model.device)
        state_dtype = torch.float64 if model.dtype == torch.float64 else torch.float32
        channels = (
            2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
            + c["linear_num_value_heads"] * c["linear_value_head_dim"]
        )
        self._states = tuple(
            GDNLayerState(
                torch.zeros(c["linear_conv_kernel_dim"] - 1, channels, **opt),
                torch.zeros(
                    c["linear_num_value_heads"],
                    c["linear_value_head_dim"],
                    c["linear_key_head_dim"],
                    dtype=state_dtype,
                    device=model.device,
                ),
            )
            if kind == "linear_attention"
            else None
            for kind in c["layer_types"]
        )
        self._kv = tuple(
            _StagedKV(
                torch.empty(capacity, c["num_key_value_heads"], c["head_dim"], **opt),
                torch.empty(capacity, c["num_key_value_heads"], c["head_dim"], **opt),
            )
            if kind == "full_attention"
            else None
            for kind in c["layer_types"]
        )
        self._sync()

    @property
    def cache_bytes(self):
        return sum(s.nbytes for s in (*self._states, *self._kv) if s is not None)

    @property
    def workspace_bytes(self):
        return sum(lane.nbytes for lane in self._lanes.values())

    def _sync(self):
        self.model._sync()

    def _check(self):
        if self._closed:
            raise RuntimeError("Streaming Qwen session is closed")
        _outside_capture(self.model.device)
        _owner_stream(self.model.device, self._stream)

    def _lane(self, tokens):
        if tokens in self._lanes:
            return self._lanes[tokens]
        lane = _StreamingLane(
            tuple(
                torch.empty(
                    tokens,
                    self.model.config["hidden_size"],
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
                for _ in range(2)
            )
        )
        if tokens in (1, self.max_tokens):
            self._lanes[tokens] = lane
        return lane

    def _notify(self, phase, position, tokens, layer=None):
        if self._observer is not None:
            self._observer(
                dict(phase=phase, position=position, tokens=tokens, layer=layer)
            )

    def _embedding(self, ids, out):
        m = self.model
        with m._weight_lock:
            m._before_phase(m._role_bytes("text/token_embedding"))
            resident = m.backend.upload_linear(
                m._weights.read_linear("text/token_embedding"),
                dtype=m.dtype,
                device=m.device,
            )
            lookup = resident.prepare_embedding(ids)
            try:
                lookup.run(out=out)
                self._sync()
            finally:
                self._sync()
        # Locals/IDs/weight planes die only after this phase completed.

    def _layer(self, i, position, x, out, lane, cos, sin):
        m = self.model
        kind = m.config["layer_types"][i]
        with m._weight_lock:
            m._before_phase(m._layer_bytes(i))
            cls = (
                ResidentQwenGDNLayer
                if kind == "linear_attention"
                else ResidentQwenAttentionLayer
            )
            resident = cls.from_q8(
                m._weights, m.backend, layer=i, dtype=m.dtype, device=m.device
            )
            plan = resident.prepare(tokens=x.shape[0])
            try:
                if kind not in lane.workspaces:
                    lane.workspaces[kind] = plan.empty_workspace()
                workspace = lane.workspaces[kind]
                state = self._states[i]
                if state is not None:
                    plan.run(
                        x,
                        out=out,
                        history=state.history,
                        state=state.recurrent,
                        history_out=state.history,
                        state_out=state.recurrent,
                        workspace=workspace,
                    )
                else:
                    kv = self._kv[i]
                    end = position + x.shape[0]
                    cache = resident.empty_cache(capacity=end)
                    cache.load_prefix(kv.keys[:position], kv.values[:position])
                    step = plan.at(position=position, cache=cache)
                    step.run(x, cos, sin, out=out, workspace=workspace)
                    kv.keys[position:end].copy_(cache._keys[position:end])
                    kv.values[position:end].copy_(cache._values[position:end])
                self._sync()
            finally:
                self._sync()
                plan.close()
        # No plan, cache owner or resident is retained by the lane/request.

    def _head(self, x, out, *, all_logits):
        m = self.model
        with m._weight_lock:
            m._before_phase(m._role_bytes("text/output_head"))
            m.norm.prepare(tokens=x.shape[0]).run(x, out=out)
            selected = out if all_logits else out[-1:]
            resident = m.backend.upload_linear(
                m._weights.read_linear("text/output_head"),
                dtype=m.dtype,
                device=m.device,
            )
            plan = resident.prepare(tokens=selected.shape[0])
            logits = torch.empty(
                selected.shape[0],
                m.config["vocab_size"],
                dtype=m.dtype,
                device=m.device,
            )
            scratch = (
                torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=m.device)
                if plan.workspace_bytes
                else None
            )
            try:
                plan.run(selected, out=logits, workspace=scratch)
                self._sync()
                return logits if m.dtype == torch.float64 else logits.float()
            finally:
                self._sync()
                close = getattr(plan, "close", None)
                if close is not None:
                    close()

    @torch.inference_mode()
    def forward(self, token_ids, *, all_logits=False, trace=None):
        _ids(token_ids, self.model.config["vocab_size"])
        if type(all_logits) is not bool:
            raise ValueError("all_logits must be boolean")
        if trace is not None and not isinstance(trace, list):
            raise TypeError("trace must be an explicit list or None")
        with self._lock:
            self._check()
            if self.failed:
                raise RuntimeError("Streaming Qwen session failed; reset required")
            self.model.check_source()
            if self.position + len(token_ids) > self.capacity:
                raise ValueError("Input exceeds session capacity")
            start_position = self.position
            started = False
            results = []
            try:
                for start in range(0, len(token_ids), self.max_tokens):
                    ids = token_ids[start : start + self.max_tokens]
                    position = start_position + start
                    t = len(ids)
                    lane = self._lane(t)
                    x, out = lane.hidden
                    cos, sin = self.model.rope.tables(
                        self.model.backend,
                        position=position,
                        tokens=t,
                        dtype=self.model.dtype,
                        device=self.model.device,
                    )
                    started = True
                    self._embedding(ids, x)
                    self._notify("embedding", position, t)
                    for i in range(self.model.config["num_hidden_layers"]):
                        self._layer(i, position, x, out, lane, cos, sin)
                        x, out = out, x
                        if trace is not None:
                            trace.append(
                                dict(
                                    position=position,
                                    layer=i,
                                    hidden=x.detach().to("cpu", copy=True),
                                )
                            )
                        self._notify("layer", position, t, i)
                    if all_logits or start + t == len(token_ids):
                        results.append(self._head(x, out, all_logits=all_logits))
                        self._notify("head", position, t)
                result = torch.cat(results) if all_logits else results[-1][0]
                self._sync()
                self.model.check_source()
            except Exception:
                if started:
                    self.failed = True
                raise
            self.position = start_position + len(token_ids)
            return result

    def reset(self):
        with self._lock:
            self._check()
            self._sync()
            for s in self._states:
                if s is not None:
                    s.history.zero_()
                    s.recurrent.zero_()
            self._sync()
            self.position, self.failed = 0, False
            # Neutral KV tails are ignored until overwritten; position is the
            # only visibility cursor. No GDN suffix truncation is provided.

    def close(self):
        with self._lock:
            self._check()
            self._sync()
            self._lanes.clear()
            self._states = ()
            self._kv = ()
            self._closed = True
