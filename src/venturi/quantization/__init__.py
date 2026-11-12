"""Quantization registry and format system (vLLM-style).

This package is the engine-wide source of truth for weight formats:
storage layout, byte accounting, execution qualification per architecture,
and storage verification. Model adapters consume it without knowing
which codec is active.

New code should import from here:
    from venturi.quantization import Codec, codec_for, EXECUTION_QUALIFIED
"""

from ..codecs import CODECS, DENSE_ITEMSIZE, Codec, codec_for, dequant_row_reference
from ..storage_verify import verify_artifact, write_evidence

# Registry: format name → Codec instance (mirrors vLLM's QUANTIZATION_METHODS).
QUANTIZATION_METHODS: dict[str, Codec] = CODECS

# Per-architecture execution qualification.
EXECUTION_QUALIFIED: dict[tuple[int, int], set[str]] = {
    (7, 0): {"q8_g32_fp16"},
    (8, 9): {"q8_g32_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"},
    (8, 6): {"q8_g32_fp16"},
}


def get_quant_method(name: str) -> Codec:
    """Look up a quantization method by name. Raises ValueError if unknown."""
    return codec_for(name)


def list_quant_methods() -> list[str]:
    """List all registered quantization method names."""
    return sorted(CODECS)


__all__ = [
    "CODECS", "EXECUTION_QUALIFIED", "QUANTIZATION_METHODS",
    "Codec", "codec_for", "dequant_row_reference",
    "get_quant_method", "list_quant_methods",
    "verify_artifact", "write_evidence",
]
