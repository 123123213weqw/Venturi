"""Optional architecture-independent execution backends."""

from .ada import AdaBackend, PreparedLinear, ResidentQ4Weight
from .attention import PreparedCausalAttention, PreparedRoPE
from .embedding import PreparedEmbedding
from .gdn import PreparedGDN
from .blocks import CudaBackend, ResidentLinear, ResidentRMSNorm, TorchBackend
from .vocabulary import NInferVocabulary, Q8VocabularyWeights, ResidentQ8Vocabulary
from .weights import DenseWeight, Q4RowSplitWeight

__all__ = [
    "NInferVocabulary",
    "Q8VocabularyWeights",
    "ResidentQ8Vocabulary",
    "PreparedEmbedding",
    "PreparedGDN",
    "AdaBackend",
    "PreparedLinear",
    "Q4RowSplitWeight",
    "DenseWeight",
    "CudaBackend",
    "TorchBackend",
    "ResidentLinear",
    "ResidentRMSNorm",
    "ResidentQ4Weight",
    "PreparedRoPE",
    "PreparedCausalAttention",
]
