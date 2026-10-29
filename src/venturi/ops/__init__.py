"""Public ops API, resolved only when a concrete export is requested.

Importing the namespace does not load unrelated model families, tensor frameworks
or native adapters. Existing public names retain their implementation identities.
"""

from importlib import import_module

_EXPORTS = {
    "Backend": ".backend",
    "BackendCapabilities": ".backend",
    "LinearPlan": ".backend",
    "LinearWeight": ".backend",
    "OperatorBackend": ".backend",
    "OperatorSupport": ".backend",
    "UnsupportedOperationError": ".backend",
    "create_backend": ".backend",
    "PreparedQ8DP4A": ".q8_dp4a",
    "prepare_q8_dp4a": ".q8_dp4a",
    "PreparedQ8WMMA": ".q8_wmma",
    "prepare_q8_wmma": ".q8_wmma",
    "PreparedGreedy": ".sampling",
    "prepare_greedy": ".sampling",
    "ResidentQwenQKNorm": ".qwen_attention",
    "PreparedQwenQKNormRoPE": ".qwen_attention",
    "PreparedQwenOutputGate": ".qwen_attention",
    "ResidentCausalConv": ".state_space",
    "PreparedCausalConv": ".state_space",
    "ResidentGDNGates": ".state_space",
    "PreparedGDNGates": ".state_space",
    "ResidentGatedRMSNorm": ".state_space",
    "PreparedGatedRMSNorm": ".state_space",
    "NInferVocabulary": ".vocabulary",
    "Q8VocabularyWeights": ".vocabulary",
    "ResidentQ8Vocabulary": ".vocabulary",
    "PreparedEmbedding": ".embedding",
    "PreparedGDN": ".gdn",
    "AdaBackend": ".ada",
    "PreparedLinear": ".ada",
    "Q4RowSplitWeight": ".weights",
    "Q8RowSplitWeight": ".weights",
    "DenseWeight": ".weights",
    "CudaBackend": ".blocks",
    "TorchBackend": ".blocks",
    "ResidentLinear": ".blocks",
    "ResidentRMSNorm": ".blocks",
    "ResidentQ4Weight": ".ada",
    "PreparedRoPE": ".attention",
    "PreparedCausalAttention": ".attention",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
