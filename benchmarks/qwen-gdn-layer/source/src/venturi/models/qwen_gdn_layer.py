"""Complete Qwen GDN decoder layer using owned native operators and explicit state.

Input norm -> QKV/z/a/b -> conv/gates/recurrence/gated norm -> output projection
-> residual -> post-attention norm/SwiGLU MLP/residual. This is one sequence, not
batching, a hybrid model/session, or trained-checkpoint quality qualification.
"""

from dataclasses import dataclass, field

import torch

from ..ops.blocks import _no_overlap, _shape, _tensor
from ..ops.weights import DenseWeight
from .gated_mlp import GatedMLPWorkspace, ResidentGatedMLP
from .gdn_frontend import GDNFrontendWeights, GDNMixerWorkspace
from .qwen_weights import QwenQ8Weights


def _outside_capture(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Allocate/prepare outside CUDA graph capture")


@dataclass(frozen=True)
class GDNLayerState:
    """Caller-owned chronological raw-input history and recurrent matrix.

    There is deliberately no hidden sequence cursor/commit. Reset both tensors
    to zero for a new request, serialize updates, and discard mutable state after
    a failed asynchronous execution. A scheduler must manage request ownership.
    """

    history: torch.Tensor
    recurrent: torch.Tensor

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.history, self.recurrent))


@dataclass(frozen=True)
class ResidentQwenGDNLayer:
    backend: object
    layer_index: int
    frontend: object
    qkv: object
    z: object
    a: object
    b: object
    output: object
    mlp: ResidentGatedMLP

    @classmethod
    def from_q8(cls, weights, backend, *, layer, dtype=torch.bfloat16, device="cuda:0"):
        """Upload one encoded projection at a time; never retain a dense model."""
        if not isinstance(weights, QwenQ8Weights):
            raise TypeError("Expected complete preflighted QwenQ8Weights")
        weights.layer_bindings(layer)
        cfg = weights.config
        if cfg["layer_types"][layer] != "linear_attention":
            raise ValueError("Requested layer is not a GDN layer")
        if dtype != torch.bfloat16 and not (
            backend.name == "torch-reference" and dtype == torch.float64
        ):
            raise ValueError(
                "GDN layer currently supports BF16, plus explicit FP64 reference mode"
            )
        target = backend._device(dtype, device)
        _outside_capture(target)
        prefix = f"text/layers/{layer}/"
        frontend = GDNFrontendWeights.from_ninfer(weights.artifact, layer=layer).upload(
            backend, dtype=dtype, device=target, epsilon=cfg["rms_norm_eps"]
        )

        def upload(role):
            descriptor = weights.read_linear(prefix + role)
            # An explicit diagnostic activation-precision conversion, not native
            # Q8->dense or a hidden arithmetic fallback.
            if isinstance(descriptor, DenseWeight) and descriptor.data.dtype != dtype:
                descriptor = DenseWeight(descriptor.data.to(dtype))
            return backend.upload_linear(descriptor, dtype=dtype, device=target)

        qkv, z, a, b, output = (
            upload("gdn/" + name) for name in ("qkv", "z", "a", "b", "output")
        )
        mlp_norm = backend.upload_zero_centered_rmsnorm(
            weights.read_float(prefix + "post_attention_norm"),
            dtype=dtype,
            device=target,
            epsilon=cfg["rms_norm_eps"],
        )
        gate, up, down = (upload("mlp/" + name) for name in ("gate", "up", "down"))
        return cls(
            backend,
            layer,
            frontend,
            qkv,
            z,
            a,
            b,
            output,
            ResidentGatedMLP(backend, mlp_norm, gate, up, down),
        )

    @property
    def dtype(self):
        return self.qkv.dtype

    @property
    def device(self):
        return self.qkv.device

    @property
    def hidden_size(self):
        return self.qkv.in_features

    @property
    def storage_tensors(self):
        return tuple(
            t
            for op in (
                self.frontend,
                self.qkv,
                self.z,
                self.a,
                self.b,
                self.output,
                self.mlp,
            )
            for t in op.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def zero_state(self):
        """Explicit per-request allocation/reset helper, outside timing/capture."""
        _outside_capture(self.device)
        r = self.frontend
        work = torch.float64 if self.dtype == torch.float64 else torch.float32
        return GDNLayerState(
            torch.zeros(
                r.conv._weight.shape[0] - 1,
                r.conv._weight.shape[1],
                dtype=self.dtype,
                device=self.device,
            ),
            torch.zeros(
                r.value_heads, r.value_dim, r.key_dim, dtype=work, device=self.device
            ),
        )

    def prepare(self, *, tokens):
        _shape(tokens, self.hidden_size)
        _outside_capture(self.device)
        return PreparedQwenGDNLayer(
            self,
            self.frontend.input_norm.prepare(tokens=tokens),
            *(
                op.prepare(tokens=tokens)
                for op in (self.qkv, self.z, self.a, self.b, self.output)
            ),
            self.frontend.prepare_mixer_core(tokens=tokens),
            self.mlp.prepare(tokens=tokens),
            self.backend.prepare_add(
                tokens=tokens,
                features=self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
        )


@dataclass(frozen=True)
class QwenGDNWorkspace:
    normalized: torch.Tensor
    qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    mixed: torch.Tensor
    projected: torch.Tensor
    residual: torch.Tensor
    mixer: GDNMixerWorkspace
    mlp: GatedMLPWorkspace
    linear: torch.Tensor | None
    # Metadata-only views created before capture, never during forward.
    z_heads: torch.Tensor
    mixed_heads: torch.Tensor
    projection_scratch: tuple

    @property
    def tensors(self):
        return (
            (
                self.normalized,
                self.qkv,
                self.z,
                self.a,
                self.b,
                self.mixed,
                self.projected,
                self.residual,
            )
            + self.mixer.tensors
            + (self.mlp.hidden, self.mlp.gate, self.mlp.up)
            + ((self.linear,) if self.linear is not None else ())
        )

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)


@dataclass(frozen=True)
class PreparedQwenGDNLayer:
    resident: ResidentQwenGDNLayer
    norm: object
    qkv: object
    z: object
    a: object
    b: object
    output: object
    mixer: object
    mlp: object
    add: object
    _closed: bool = field(default=False, init=False)

    @property
    def tokens(self):
        return self.qkv.tokens

    @property
    def projections(self):
        return (self.qkv, self.z, self.a, self.b, self.output)

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
            raise RuntimeError("GDN layer execution plan has been closed")

    def close(self):
        """Only after queued work/replays finish; shared resident weights survive."""
        for p in (*self.projections, self.mlp):
            close = getattr(p, "close", None)
            if close is not None:
                close()
        object.__setattr__(self, "_closed", True)

    def empty_workspace(self):
        self._check_open()
        r, t, f = self.resident, self.tokens, self.resident.frontend
        _outside_capture(r.device)
        opt = dict(dtype=r.dtype, device=r.device)

        def empty(width):
            return torch.empty(t, width, **opt)

        linear = (
            torch.empty(self.linear_workspace_bytes, dtype=torch.uint8, device=r.device)
            if self.linear_workspace_bytes
            else None
        )

        def scratch(size):
            return linear[:size] if size else None

        mlp = GatedMLPWorkspace(
            empty(r.hidden_size),
            empty(r.mlp.gate.out_features),
            empty(r.mlp.gate.out_features),
            scratch(self.mlp.linear_workspace_bytes),
        )
        z, mixed = (
            empty(f.value_heads * f.value_dim),
            empty(f.value_heads * f.value_dim),
        )
        return QwenGDNWorkspace(
            empty(r.hidden_size),
            empty(self.qkv.out_features),
            z,
            empty(f.value_heads),
            empty(f.value_heads),
            mixed,
            empty(r.hidden_size),
            empty(r.hidden_size),
            self.mixer.empty_workspace(),
            mlp,
            linear,
            z.view(t, f.value_heads, f.value_dim),
            mixed.view(t, f.value_heads, f.value_dim),
            tuple(scratch(p.workspace_bytes) for p in self.projections),
        )

    def validate(self, x, *, out, history, state, history_out, state_out, workspace):
        """Preflight the ENTIRE layer before a projection or state-changing launch."""
        self._check_open()
        if (
            not isinstance(workspace, QwenGDNWorkspace)
            or not isinstance(workspace.mixer, GDNMixerWorkspace)
            or not isinstance(workspace.mlp, GatedMLPWorkspace)
        ):
            raise TypeError("Expected complete explicit QwenGDNWorkspace")
        r, w, t, f = self.resident, workspace, self.tokens, self.resident.frontend
        work = torch.float64 if r.dtype == torch.float64 else torch.float32
        for name, value, shape, dtype in (
            ("x", x, (t, r.hidden_size), r.dtype),
            ("out", out, (t, r.hidden_size), r.dtype),
            (
                "history",
                history,
                (f.conv._weight.shape[0] - 1, self.qkv.out_features),
                r.dtype,
            ),
            (
                "history_out",
                history_out,
                (f.conv._weight.shape[0] - 1, self.qkv.out_features),
                r.dtype,
            ),
            ("state", state, (f.value_heads, f.value_dim, f.key_dim), work),
            ("state_out", state_out, (f.value_heads, f.value_dim, f.key_dim), work),
            ("normalized", w.normalized, (t, r.hidden_size), r.dtype),
            ("qkv", w.qkv, (t, self.qkv.out_features), r.dtype),
            ("z", w.z, (t, self.z.out_features), r.dtype),
            ("a", w.a, (t, f.value_heads), r.dtype),
            ("b", w.b, (t, f.value_heads), r.dtype),
            ("mixed", w.mixed, (t, self.output.in_features), r.dtype),
            ("projected", w.projected, (t, r.hidden_size), r.dtype),
            ("residual", w.residual, (t, r.hidden_size), r.dtype),
            ("z_heads", w.z_heads, (t, f.value_heads, f.value_dim), r.dtype),
            ("mixed_heads", w.mixed_heads, (t, f.value_heads, f.value_dim), r.dtype),
        ):
            _tensor(value, shape, dtype, r.device, name)
        for flat, heads in ((w.z, w.z_heads), (w.mixed, w.mixed_heads)):
            if flat.data_ptr() != heads.data_ptr():
                raise ValueError("GDN head views must alias their exact flat buffers")
        if self.linear_workspace_bytes:
            _tensor(
                w.linear,
                (self.linear_workspace_bytes,),
                torch.uint8,
                r.device,
                "linear scratch",
            )
            if w.linear.data_ptr() % 256:
                raise ValueError("Linear workspace requires 256-byte alignment")
        elif w.linear is not None:
            raise ValueError("No linear workspace required")
        if (
            not isinstance(w.projection_scratch, tuple)
            or len(w.projection_scratch) != 5
        ):
            raise ValueError("Projection scratch metadata must match five projections")
        for p, scratch in zip(self.projections, w.projection_scratch):
            if p.workspace_bytes:
                _tensor(
                    scratch,
                    (p.workspace_bytes,),
                    torch.uint8,
                    r.device,
                    "projection scratch",
                )
                if scratch.data_ptr() != w.linear.data_ptr():
                    raise ValueError(
                        "Projection scratch must use the shared arena prefix"
                    )
            elif scratch is not None:
                raise ValueError("Workspace-free projection cannot receive scratch")
        self.mlp.validate(w.residual, out=out, workspace=w.mlp)
        if (
            self.mlp.linear_workspace_bytes
            and w.mlp.linear.data_ptr() != w.linear.data_ptr()
        ):
            raise ValueError("MLP scratch must use the shared arena prefix")
        self.mixer.validate(
            w.qkv,
            w.a,
            w.b,
            w.z_heads,
            history=history,
            state=state,
            out=w.mixed_heads,
            history_out=history_out,
            state_out=state_out,
            workspace=w.mixer,
        )
        readonly = (x, history, state) + r.storage_tensors
        writable = (out, history_out, state_out) + w.tensors
        for i, value in enumerate(writable):
            paired = history if i == 1 else state if i == 2 else None
            skip = 1 if i == 1 else 2 if i == 2 else -1
            _no_overlap(value, tuple(v for j, v in enumerate(readonly) if j != skip))
            _no_overlap(value, writable[:i] + writable[i + 1 :])
            if paired is not None:
                _no_overlap(value, (paired,), allow_inplace=True)

    def run(self, x, *, out, history, state, history_out, state_out, workspace):
        """Native numerical kernels only; retain buffers/plan through completion.

        After an enqueue/asynchronous failure, discard mutable state/history.
        This raw operation performs no request cursor/commit, rollback or reset.
        """
        self.validate(
            x,
            out=out,
            history=history,
            state=state,
            history_out=history_out,
            state_out=state_out,
            workspace=workspace,
        )
        w = workspace
        self.norm.run(x, out=w.normalized)
        for p, target, scratch in zip(
            self.projections[:4], (w.qkv, w.z, w.a, w.b), w.projection_scratch[:4]
        ):
            p.run(w.normalized, out=target, workspace=scratch)
        self.mixer.run(
            w.qkv,
            w.a,
            w.b,
            w.z_heads,
            history=history,
            state=state,
            out=w.mixed_heads,
            history_out=history_out,
            state_out=state_out,
            workspace=w.mixer,
        )
        self.output.run(w.mixed, out=w.projected, workspace=w.projection_scratch[4])
        self.add.run(x, w.projected, out=w.residual)
        self.mlp.run(w.residual, out=out, workspace=w.mlp)
        return out
