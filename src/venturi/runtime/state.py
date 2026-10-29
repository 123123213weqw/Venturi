"""Framework-free request-state identity and explicit transport contracts.

An exported state may be a leased opaque device view, not serialized bytes. Its
producer must stay alive and unmodified until transfer returns. No format reader,
backend selection, precision conversion or implicit CPU fallback lives here.
"""
from dataclasses import dataclass, field
import math
from typing import Protocol


@dataclass(frozen=True)
class StateIdentity:
    checkpoint: str
    layout: str
    precision: str
    arithmetic: str

    def __post_init__(self):
        if any(type(v) is not str or not v for v in
               (self.checkpoint, self.layout, self.precision, self.arithmetic)):
            raise ValueError("Explicit checkpoint/layout/precision/arithmetic identity required")


@dataclass(frozen=True)
class StateComponent:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.dtype) is not str
                or not self.dtype or type(self.nbytes) is not int or self.nbytes < 0
                or type(self.shape) is not tuple
                or any(type(n) is not int or n < 0 for n in self.shape)):
            raise ValueError("Invalid request-state component descriptor")


@dataclass(frozen=True)
class RequestState:
    identity: StateIdentity
    position: int
    components: tuple[StateComponent, ...]
    payload: object = field(repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.identity, StateIdentity) or type(self.position) is not int
                or self.position <= 0 or type(self.components) is not tuple
                or any(not isinstance(c, StateComponent) for c in self.components)
                or len({c.name for c in self.components}) != len(self.components)):
            raise ValueError("Invalid request-state envelope")

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.components)


@dataclass(frozen=True)
class TransferReceipt:
    position: int
    payload_bytes: int
    wall_seconds: float
    transport: str
    staging_bytes: int = 0

    def __post_init__(self):
        if (type(self.position) is not int or self.position <= 0
                or type(self.payload_bytes) is not int or self.payload_bytes < 0
                or type(self.staging_bytes) is not int or self.staging_bytes < 0
                or type(self.wall_seconds) not in (int, float)
                or not math.isfinite(self.wall_seconds) or self.wall_seconds < 0
                or type(self.transport) is not str or not self.transport):
            raise ValueError("Invalid state-transfer receipt")


class StateTransport(Protocol):
    def transfer(self, state: RequestState, target, *, slot: int) -> TransferReceipt:
        """Validate, copy, synchronize and commit, or raise without an ACK."""
        ...


class PDTarget(Protocol):
    """Exclusively leased, prepared execution target, not a whole-model loader."""
    capacities: tuple[int, ...]
    max_token_rows: int
    vocab_size: int
    state_identity: StateIdentity

    def reset(self, *, slots): ...
    def step(self, prompts, *, slots, sample_slots, decode_slots=()): ...
    def export_state(self, slot: int, *, position: int) -> RequestState: ...
