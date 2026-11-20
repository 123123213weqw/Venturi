"""Strict small-parameter bindings for the Qwen GDN frontend.

No projections, hidden-state reorder, cache scheduler or full-model fallback.
The same prepared native operators are reusable by other architecture adapters.
"""

from dataclasses import dataclass
import math

import torch

from ..ninfer import NInferArtifact
from ..ops.blocks import _no_overlap, _shape, _tensor


@dataclass(frozen=True)
class GDNFrontendWeights:
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    gated_norm: torch.Tensor
    input_norm: torch.Tensor
    qk_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    source_names: tuple[str, ...]

    @classmethod
    def from_ninfer(cls, artifact, *, layer):
        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        cfg = artifact.manifest["components"]["text"]["config"]
        if cfg.get("model_type") != "qwen3_5_text":
            raise ValueError("GDN frontend binding currently requires qwen3_5_text")
        kinds = cfg.get("layer_types", [])
        if (
            type(layer) is not int
            or not 0 <= layer < len(kinds)
            or kinds[layer] != "linear_attention"
        ):
            raise ValueError("Requested layer must be an explicitly declared GDN layer")
        sizes = {}
        for key in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "hidden_size",
        ):
            value = cfg.get(key)
            _shape(value, 1)
            sizes[key] = value
        hq, hv, dk, dv, kernel, hidden = (
            sizes[k]
            for k in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "hidden_size",
            )
        )
        if hv % hq or dk not in (64, 128, 256) or kernel > 32:
            raise ValueError("Unsupported GDN geometry")
        channels = 2 * hq * dk + hv * dv
        _shape(1, channels)
        prefix = f"text/layers/{layer}/"
        wanted = (
            ("gdn/convolution", (kernel, channels), "bf16", torch.bfloat16),
            ("gdn/a_log", (hv,), "fp32", torch.float32),
            ("gdn/dt_bias", (hv,), "fp32", torch.float32),
            ("gdn/norm", (dv,), ("bf16", "fp32"), None),
            ("input_norm", (hidden,), "bf16", torch.bfloat16),
        )
        specs = []
        # Preflight every binding before reading payload bytes.
        for name, shape, format, dtype in wanted:
            binding = artifact.manifest["bindings"].get(prefix + name)
            if not isinstance(binding, dict) or binding.keys() != {"object"}:
                raise ValueError(
                    f"Missing/direct GDN constant binding required: {prefix + name}"
                )
            obj = artifact.objects[binding["object"]]
            if (
                obj["shape"] != list(shape)
                or obj.get("format")
                not in ((format,) if isinstance(format, str) else format)
                or obj.get("layout") != "contiguous_le_v1"
                or obj["bytes"]
                != math.prod(shape) * (4 if obj.get("format") == "fp32" else 2)
            ):
                raise ValueError(f"GDN constant shape/format mismatch: {prefix + name}")
            if obj["bytes"] > 8 * 2**20:
                raise ValueError("GDN constant exceeds bounded read size")
            specs.append(
                (
                    obj["id"],
                    shape,
                    dtype
                    or (torch.float32 if obj["format"] == "fp32" else torch.bfloat16),
                )
            )
        weights = []
        for name, shape, dtype in specs:
            raw = bytearray(artifact.read_object(name, max_bytes=8 * 2**20))
            value = torch.frombuffer(raw, dtype=dtype).reshape(shape)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"GDN constant must be finite: {name}")
            weights.append(value)
        return cls(
            *weights, hq, hv, dk, dv, tuple(prefix + name for name, *_ in wanted)
        )

    def upload(self, backend, *, dtype=torch.bfloat16, device="cuda:0", epsilon=1e-6):
        """Explicit activation precision conversion, preserving FP32 gate constants."""
        return ResidentGDNFrontend(
            backend.upload_causal_conv(self.conv.to(dtype), dtype=dtype, device=device),
            backend.upload_gdn_gates(
                self.a_log, self.dt_bias, dtype=dtype, device=device
            ),
            backend.upload_gated_rmsnorm(
                self.gated_norm
                if self.gated_norm.dtype == torch.float32 and dtype != torch.float64
                else self.gated_norm.to(dtype),
                dtype=dtype,
                device=device,
                epsilon=epsilon,
            ),
            backend.upload_zero_centered_rmsnorm(
                self.input_norm, dtype=dtype, device=device, epsilon=epsilon
            ),
            self.qk_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
        )


@dataclass(frozen=True)
class ResidentGDNFrontend:
    """Resident constants only. State, projections and scheduling remain explicit."""

    conv: object
    gates: object
    gated_norm: object
    input_norm: object
    qk_heads: int
    value_heads: int
    key_dim: int
    value_dim: int

    def prepare_mixer_core(self, *, tokens, normalize_qk=True):
        """Bind projected-input mixer math, not projections/residual or a request scheduler."""
        dtype, device = self.conv._weight.dtype, self.conv._weight.device
        if any(
            op.backend is not self.conv.backend for op in (self.gates, self.gated_norm)
        ):
            raise ValueError("GDN mixer constants must share one explicit backend")
        if self.gates.dtype != dtype or self.gated_norm.dtype != dtype:
            raise ValueError(
                "GDN mixer gate dtype must match convolution activation dtype"
            )
        work = torch.float64 if dtype == torch.float64 else torch.float32
        for name, value, shape, expected_dtype in (
            ("gate rate", self.gates._rate, (self.value_heads,), work),
            ("gate bias", self.gates._bias, (self.value_heads,), work),
            (
                "gated norm",
                self.gated_norm._weight,
                (self.value_dim,),
                self.gated_norm._weight.dtype,
            ),
        ):
            _tensor(value, shape, expected_dtype, device, name)
        recurrence = self.conv.backend.prepare_gdn(
            tokens=tokens,
            qk_heads=self.qk_heads,
            value_heads=self.value_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            normalize_qk=normalize_qk,
            dtype=dtype,
            device=device,
        )
        if self.conv._weight.shape[1] != recurrence.packed_features:
            raise ValueError("Convolution channels must match packed Q/K/V geometry")
        if self.gates._rate.shape != (
            self.value_heads,
        ) or self.gated_norm._weight.shape != (self.value_dim,):
            raise ValueError("GDN gate/norm constants must match recurrence geometry")
        return PreparedGDNMixerCore(
            self,
            self.conv.prepare(tokens=tokens),
            self.gates.prepare(tokens=tokens),
            recurrence,
            self.gated_norm.prepare(tokens=tokens, heads=self.value_heads),
        )

    @property
    def storage_tensors(self):
        return tuple(
            t
            for op in (self.conv, self.gates, self.gated_norm, self.input_norm)
            for t in op.storage_tensors
        )

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)


@dataclass(frozen=True)
class GDNMixerWorkspace:
    """Explicit per-execution scratch; no packed-Q/K/V unpack buffers are needed."""

    convolved: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    recurrent: torch.Tensor

    @property
    def tensors(self):
        return (self.convolved, self.log_decay, self.beta, self.recurrent)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)


@dataclass(frozen=True)
class PreparedGDNMixerCore:
    """Convolution -> gates -> packed recurrence -> gated norm of projected inputs.

    QKV/a/b/z projection, input norm, output projection and residual are NOT here.
    State/history ownership, reset and scheduling remain caller responsibilities.
    Each execution lane needs its own scratch and writable state. Retain plans
    and all operands until queued work/graph replays finish. Discard mutable state
    after an asynchronous error; there is no hidden cursor, rollback or fallback.
    """

    resident: ResidentGDNFrontend
    conv: object
    gates: object
    recurrence: object
    gated_norm: object

    def empty_workspace(self):
        """Allocate once before timing/capture; run() itself never allocates."""
        r = self.recurrence
        if r.device.type == "cuda":
            with torch.cuda.device(r.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Workspace allocation must run outside CUDA graph capture"
                    )
        work = torch.float64 if r.dtype == torch.float64 else torch.float32
        return GDNMixerWorkspace(
            torch.empty(r.tokens, r.packed_features, dtype=r.dtype, device=r.device),
            torch.empty(r.tokens, r.value_heads, dtype=work, device=r.device),
            torch.empty(r.tokens, r.value_heads, dtype=work, device=r.device),
            torch.empty(
                r.tokens, r.value_heads, r.value_dim, dtype=r.dtype, device=r.device
            ),
        )

    def validate(
        self, qkv, a, b, z, *, history, state, out, history_out, state_out, workspace
    ):
        """Validate all buffers/aliases before any state mutation or kernel launch.

        history stores chronological raw projected QKV inputs, not convolution
        outputs. Exact history/history_out and state/state_out aliasing is allowed;
        all other writable overlap is rejected, including scratch and constants.
        Inputs/gates must meet the individual operator's finite/stability contract.
        """
        if not isinstance(workspace, GDNMixerWorkspace):
            raise TypeError("Expected explicit GDNMixerWorkspace")
        r = self.recurrence
        work = torch.float64 if r.dtype == torch.float64 else torch.float32
        activation_shape = (r.tokens, r.value_heads, r.value_dim)
        history_shape = (self.resident.conv._weight.shape[0] - 1, r.packed_features)
        state_shape = (r.value_heads, r.value_dim, r.key_dim)
        for name, tensor, shape, dtype in (
            ("qkv", qkv, (r.tokens, r.packed_features), r.dtype),
            ("a", a, (r.tokens, r.value_heads), r.dtype),
            ("b", b, (r.tokens, r.value_heads), r.dtype),
            ("z", z, activation_shape, r.dtype),
            ("history", history, history_shape, r.dtype),
            ("history_out", history_out, history_shape, r.dtype),
            ("state", state, state_shape, work),
            ("state_out", state_out, state_shape, work),
            ("out", out, activation_shape, r.dtype),
            ("convolved", workspace.convolved, (r.tokens, r.packed_features), r.dtype),
            ("log_decay", workspace.log_decay, (r.tokens, r.value_heads), work),
            ("beta", workspace.beta, (r.tokens, r.value_heads), work),
            ("recurrent", workspace.recurrent, activation_shape, r.dtype),
        ):
            _tensor(tensor, shape, dtype, r.device, name)
        readonly = (qkv, a, b, z, history, state) + self.resident.storage_tensors
        writable = (out, history_out, state_out) + workspace.tensors
        for i, target in enumerate(writable):
            # Only the two explicitly paired state buffers may be in-place.
            paired = history if i == 1 else state if i == 2 else None
            skipped_index = 4 if i == 1 else 5 if i == 2 else -1
            _no_overlap(
                target, tuple(t for j, t in enumerate(readonly) if j != skipped_index)
            )
            _no_overlap(target, writable[:i] + writable[i + 1 :])
            if paired is not None:
                _no_overlap(target, (paired,), allow_inplace=True)

    def run(
        self, qkv, a, b, z, *, history, state, out, history_out, state_out, workspace
    ):
        self.validate(
            qkv,
            a,
            b,
            z,
            history=history,
            state=state,
            out=out,
            history_out=history_out,
            state_out=state_out,
            workspace=workspace,
        )
        self.conv.run(
            qkv, history=history, out=workspace.convolved, history_out=history_out
        )
        self.gates.run(a, b, log_decay=workspace.log_decay, beta=workspace.beta)
        self.recurrence.run_packed(
            workspace.convolved,
            workspace.log_decay,
            workspace.beta,
            state=state,
            out=workspace.recurrent,
            state_out=state_out,
        )
        self.gated_norm.run(workspace.recurrent, z, out=out)
        return out
