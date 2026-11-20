"""Universal MoE geometry: expert bank shape shared by every store and layer.

Model families construct this from their own config keys; nothing here reads
configs or knows a model name.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertGeometry:
    hidden_size: int
    num_experts: int
    top_k: int
    intermediate_size: int
    shared_intermediate_size: int = 0
    renormalize: bool = True

    def __post_init__(self):
        for name in ("hidden_size", "num_experts", "top_k", "intermediate_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.shared_intermediate_size) is not int or self.shared_intermediate_size < 0:
            raise ValueError("shared_intermediate_size must be a nonnegative integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed the expert count")

    def expert_bytes(self, element_size: int) -> int:
        """gate_up (2*inter, hidden) + down (hidden, inter) bytes per expert."""
        return (2 * self.intermediate_size * self.hidden_size
                + self.hidden_size * self.intermediate_size) * element_size
