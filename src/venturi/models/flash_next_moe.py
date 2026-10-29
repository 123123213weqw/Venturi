"""Original floating Qwen4-Exp MoE reference and bounded expert row bindings.

No expert pruning, quantized codec, cache/offload scheduler, EP/TP dispatch or
native kernel is implied. The slow reference fetches only selected experts and
preserves source rounding/accumulation order before future kernel work.
"""

from dataclasses import dataclass

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F

from ..dtypes import torch_dtype
from ..flash_next import _positive


@dataclass(frozen=True)
class MoEGeometry:
    hidden_size: int
    num_experts: int
    top_k: int
    intermediate_size: int
    shared_intermediate_size: int
    normalize_topk: bool = True

    def __post_init__(self):
        for key in ("hidden_size", "num_experts", "top_k", "intermediate_size", "shared_intermediate_size"):
            _positive(getattr(self, key), key)
        if self.top_k > self.num_experts or type(self.normalize_topk) is not bool:
            raise ValueError("Invalid MoE selection/normalization geometry")

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "qwen4_exp":
            raise ValueError("Explicit qwen4_exp configuration required")
        c = raw.get("text_config")
        if not isinstance(c, dict) or c.get("model_type") != "qwen4_exp_text":
            raise ValueError("Explicit qwen4_exp_text configuration required")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("Only the qualified SiLU MoE activation is supported")
        # Pinned upstream config default, not bool(c.get(...)) which turns an
        # omitted official field into False. Explicit None is invalid.
        return cls(c.get("hidden_size"), c.get("num_experts"), c.get("num_experts_per_tok"),
                   c.get("moe_intermediate_size"), c.get("shared_expert_intermediate_size"),
                   c.get("norm_topk_prob", True))


class FlashNextTopKRouter(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.weight = nn.Parameter(torch.zeros(geometry.num_experts, geometry.hidden_size))

    def forward(self, hidden_states):
        g = self.geometry
        if hidden_states.ndim < 1 or hidden_states.shape[-1] != g.hidden_size:
            raise ValueError("Router input hidden size mismatch")
        if hidden_states.dtype != self.weight.dtype or hidden_states.device != self.weight.device:
            raise ValueError("Router input and weights must share dtype/device")
        x = hidden_states.reshape(-1, g.hidden_size)
        logits = F.linear(x, self.weight)
        probabilities = F.softmax(logits, dtype=torch.float32, dim=-1)
        values, indices = torch.topk(probabilities, g.top_k, dim=-1)
        if g.normalize_topk:
            values = values / values.sum(-1, keepdim=True)
        return logits, values.to(logits.dtype), indices


class TensorExpertStore:
    """Explicit floating fixture/sample bank. Missing experts are errors, not zeros.

    Tensors are caller-owned, read-only by contract. A sampled bank does not
    constitute a complete checkpoint and must never bypass actual routing.
    """

    def __init__(self, geometry, experts):
        self.geometry, self.experts = geometry, dict(experts)
        for expert, pair in self.experts.items():
            if type(expert) is not int or not 0 <= expert < geometry.num_experts or len(pair) != 2:
                raise ValueError("Invalid explicit expert bank")
            gate, down = pair
            if gate.shape != (2 * geometry.intermediate_size, geometry.hidden_size) or (
                down.shape != (geometry.hidden_size, geometry.intermediate_size)
            ) or gate.dtype != down.dtype or gate.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError("Original floating expert geometry/dtype required")

    def read(self, expert, *, device, dtype):
        if type(expert) is not int or expert not in self.experts:
            raise ValueError(f"Selected expert {expert} is absent; cannot prune/substitute")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Explicit floating execution dtype required")
        return tuple(t.to(device=device, dtype=dtype) for t in self.experts[expert])


class FusedFloatingExpertStore:
    """CPU safetensors slices from original fused3D expert tensors, one at a time.

    Metadata-only construction. Reads never get_tensor/materialize the full bank.
    No caching or asynchronous GPU dispatch; explicit dtype/device conversion is
    a reference operation, not a claimed efficient expert-offload implementation.
    """

    def __init__(self, checkpoint, geometry, prefix, *, max_expert_bytes=64 * 2**20):
        if type(max_expert_bytes) is not int or not 0 < max_expert_bytes <= 256 * 2**20:
            raise ValueError("Explicit bounded expert read size required")
        self.geometry = geometry
        g = geometry
        self.specs = []
        for suffix, shape in (("gate_up_proj", (g.num_experts, 2 * g.intermediate_size, g.hidden_size)),
                              ("down_proj", (g.num_experts, g.hidden_size, g.intermediate_size))):
            spec = checkpoint.tensors.get(f"{prefix}.{suffix}")
            if spec is None or spec.shape != shape or spec.dtype not in ("F32", "F16", "BF16"):
                raise ValueError("Original fused floating expert tensor required; FP8 needs its own codec")
            self.specs.append(spec)
        if len({s.dtype for s in self.specs}) != 1:
            raise ValueError("Expert matrices must share a dtype")
        if sum(s.nbytes // g.num_experts for s in self.specs) > max_expert_bytes:
            raise ValueError("Selected expert exceeds the bounded read budget")

    def read(self, expert, *, device, dtype):
        if type(expert) is not int or not 0 <= expert < self.geometry.num_experts:
            raise ValueError("Selected expert outside the bank")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Explicit floating execution dtype required")
        output = []
        for spec in self.specs:
            with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                view = reader.get_slice(spec.name)
                if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                    raise ValueError("Expert backing metadata changed")
                # Own only this expert's bounded allocation after closing mmap.
                value = view[expert:expert + 1][0].clone()
            output.append(value.to(device=device, dtype=dtype))
        return tuple(output)


def routed_experts(hidden_states, indices, scores, store):
    g = store.geometry
    if hidden_states.ndim != 2 or hidden_states.shape[-1] != g.hidden_size or (
        indices.shape != (hidden_states.shape[0], g.top_k) or scores.shape != indices.shape
    ):
        raise ValueError("Routed expert input/selection geometry mismatch")
    if indices.dtype != torch.int64 or scores.dtype != hidden_states.dtype or any(
        t.device != hidden_states.device for t in (indices, scores)
    ):
        raise ValueError("Routed indices/scores dtype/device mismatch")
    if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= g.num_experts):
        raise ValueError("Router must select real expert ids, not padding or missing ids")
    output = torch.zeros_like(hidden_states)
    # Same ascending expert order and K-major token order as the pinned oracle.
    # GPU .tolist()/per-expert control flow is intentionally a correctness baseline.
    for expert in torch.unique(indices, sorted=True).tolist():
        positions, tokens = torch.where((indices == expert).T)
        gate_up, down = store.read(expert, device=hidden_states.device, dtype=hidden_states.dtype)
        gate, up = F.linear(hidden_states[tokens], gate_up).chunk(2, dim=-1)
        current = F.linear(F.silu(gate) * up, down)
        current = current * scores[tokens, positions, None]
        output.index_add_(0, tokens, current.to(output.dtype))
    return output


class FlashNextMoE(nn.Module):
    """Stateless floating route + selected experts + sigmoid shared expert gate."""

    def __init__(self, geometry, expert_store):
        super().__init__()
        if expert_store.geometry != geometry:
            raise ValueError("Expert store/route geometry mismatch")
        self.geometry, self.expert_store = geometry, expert_store
        # Explicit setup-only experimental dispatch; the default reference path
        # and its rounding/order remain unchanged when this is None.
        self.expert_dispatch = None
        self.gate = FlashNextTopKRouter(geometry)
        self.shared_expert = nn.ModuleDict({
            "gate_proj": nn.Linear(geometry.hidden_size, geometry.shared_intermediate_size, bias=False),
            "up_proj": nn.Linear(geometry.hidden_size, geometry.shared_intermediate_size, bias=False),
            "down_proj": nn.Linear(geometry.shared_intermediate_size, geometry.hidden_size, bias=False),
        })
        self.shared_expert_gate = nn.Linear(geometry.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        g = self.geometry
        if hidden_states.ndim < 1 or hidden_states.shape[-1] != g.hidden_size:
            raise ValueError("MoE input hidden size mismatch")
        x = hidden_states.reshape(-1, g.hidden_size)
        # Validate before any matrix call, including the shared path.
        _, scores, selected = self.gate(x)
        shared = self.shared_expert["down_proj"](
            F.silu(self.shared_expert["gate_proj"](x)) * self.shared_expert["up_proj"](x))
        expert = (routed_experts(x, selected, scores, self.expert_store) if self.expert_dispatch is None
                  else self.expert_dispatch(x, selected, scores))
        shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
        return (expert + shared).reshape(hidden_states.shape)

    @classmethod
    def from_checkpoint(cls, checkpoint, geometry, prefix, *, max_expert_bytes=64 * 2**20,
                        max_core_bytes=128 * 2**20):
        store = FusedFloatingExpertStore(checkpoint, geometry, prefix + ".experts",
                                        max_expert_bytes=max_expert_bytes)
        specs = {}
        g = geometry
        shapes = {"gate.weight": (g.num_experts, g.hidden_size),
                  "shared_expert_gate.weight": (1, g.hidden_size),
                  "shared_expert.gate_proj.weight": (g.shared_intermediate_size, g.hidden_size),
                  "shared_expert.up_proj.weight": (g.shared_intermediate_size, g.hidden_size),
                  "shared_expert.down_proj.weight": (g.hidden_size, g.shared_intermediate_size)}
        for key, shape in shapes.items():
            spec = checkpoint.tensors.get(f"{prefix}.{key}")
            if spec is None or spec.shape != shape or spec.dtype != store.specs[0].dtype:
                raise ValueError(f"Original floating MoE core tensor required: {key}")
            specs[key] = spec
        if type(max_core_bytes) is not int or max_core_bytes <= 0 or sum(
            s.nbytes for s in specs.values()
        ) > max_core_bytes:
            raise ValueError("MoE router/shared weights exceed the explicit core budget")
        result = cls(geometry, store).to(dtype=torch_dtype(store.specs[0].dtype))
        with torch.no_grad():
            parameters = dict(result.named_parameters())
            for key, spec in specs.items():
                with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(spec.name)
                    if tuple(value.shape) != spec.shape or value.dtype != parameters[key].dtype:
                        raise ValueError("MoE core backing metadata changed")
                    parameters[key].copy_(value)
        return result.eval()
