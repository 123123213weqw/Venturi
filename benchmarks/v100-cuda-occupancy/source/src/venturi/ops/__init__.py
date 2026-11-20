"""Optional architecture-independent execution backends."""

from .ada import AdaBackend, PreparedLinear, ResidentQ4Weight
from .attention import PreparedCausalAttention, PreparedRoPE
from .embedding import PreparedEmbedding
from .gdn import PreparedGDN
from .blocks import CudaBackend, ResidentLinear, ResidentRMSNorm, TorchBackend
from .state_space import (
    ResidentCausalConv,
    PreparedCausalConv,
    ResidentGDNGates,
    PreparedGDNGates,
    ResidentGatedRMSNorm,
    PreparedGatedRMSNorm,
)
from .vocabulary import NInferVocabulary, Q8VocabularyWeights, ResidentQ8Vocabulary
from .qwen_attention import (
    ResidentQwenQKNorm,
    PreparedQwenQKNormRoPE,
    PreparedQwenOutputGate,
)
from .weights import DenseWeight, Q4RowSplitWeight, Q8RowSplitWeight
from .sampling import PreparedGreedy, prepare_greedy

__all__ = [
    "PreparedGreedy",
    "prepare_greedy",
    "ResidentQwenQKNorm",
    "PreparedQwenQKNormRoPE",
    "PreparedQwenOutputGate",
    "ResidentCausalConv",
    "PreparedCausalConv",
    "ResidentGDNGates",
    "PreparedGDNGates",
    "ResidentGatedRMSNorm",
    "PreparedGatedRMSNorm",
    "NInferVocabulary",
    "Q8VocabularyWeights",
    "ResidentQ8Vocabulary",
    "PreparedEmbedding",
    "PreparedGDN",
    "AdaBackend",
    "PreparedLinear",
    "Q4RowSplitWeight",
    "Q8RowSplitWeight",
    "DenseWeight",
    "CudaBackend",
    "TorchBackend",
    "ResidentLinear",
    "ResidentRMSNorm",
    "ResidentQ4Weight",
    "PreparedRoPE",
    "PreparedCausalAttention",
]
