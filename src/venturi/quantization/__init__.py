"""Quantization registry."""
from .codecs import CODECS, Codec, codec_for
from .storage_verify import verify_artifact, write_evidence
from .base import QuantizationConfig

EXECUTION_QUALIFIED = {
    (7, 0): {"q8_g32_fp16"},
    (8, 9): {"q8_g32_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"},
    (8, 6): {"q8_g32_fp16"},
}

def get_quant_method(name): return codec_for(name)
def list_quant_methods(): return sorted(CODECS)

__all__ = ["CODECS", "EXECUTION_QUALIFIED", "Codec", "codec_for",
           "QuantizationConfig", "get_quant_method", "list_quant_methods",
           "verify_artifact", "write_evidence"]
