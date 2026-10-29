"""Public models API, resolved only when a concrete export is requested.

Importing the namespace does not load unrelated model families, tensor frameworks
or native adapters. Existing public names retain their implementation identities.
"""

from importlib import import_module

_EXPORTS = {
    "AdaptiveQwenMTPBatch": ".qwen_adaptive_mtp",
    "NativeQwenBatchSession": ".qwen_batch",
    "NativeQwenMTPBatchSession": ".qwen_mtp_batch",
    "QwenMTPPlan": ".qwen_mtp",
    "QwenMTPWeights": ".qwen_mtp",
    "NativeQwenMTPModel": ".qwen_mtp_native",
    "NativeQwenMTPSession": ".qwen_mtp_native",
    "NativeQwenModel": ".qwen_native",
    "NativeQwenPipelineModel": ".qwen_pipeline",
    "NativeQwenPipelineSession": ".qwen_pipeline",
    "NativeQwenSession": ".qwen_native",
    "StreamingQwenModel": ".qwen_streaming",
    "StreamingQwenSession": ".qwen_streaming",
    "qwen_request_bytes": ".qwen_model",
    "ResidentQwenModel": ".qwen_model",
    "QwenSession": ".qwen_model",
    "qwen_resident_bytes": ".qwen_model",
    "ResidentQwenAttentionLayer": ".qwen_attention_layer",
    "PreparedQwenAttentionLayer": ".qwen_attention_layer",
    "QwenAttentionWorkspace": ".qwen_attention_layer",
    "QwenAttentionStep": ".qwen_attention_layer",
    "QwenTextRoPE": ".qwen_rope",
    "GDNLayerState": ".qwen_gdn_layer",
    "PreparedQwenGDNLayer": ".qwen_gdn_layer",
    "QwenGDNWorkspace": ".qwen_gdn_layer",
    "ResidentQwenGDNLayer": ".qwen_gdn_layer",
    "QwenSourceBinding": ".qwen_source",
    "QwenSourcePlan": ".qwen_source",
    "QwenQ8Weights": ".qwen_weights",
    "GDNFrontendWeights": ".gdn_frontend",
    "GDNMixerWorkspace": ".gdn_frontend",
    "PreparedGDNMixerCore": ".gdn_frontend",
    "ResidentGDNFrontend": ".gdn_frontend",
    "LlamaConfig": ".llama",
    "LlamaWeights": ".llama",
    "LlamaModel": ".llama",
    "LlamaSession": ".llama",
    "GPTNeoXConfig": ".gpt_neox",
    "GPTNeoXModel": ".gpt_neox",
    "GPTNeoXSession": ".gpt_neox",
    "GatedMLPWeights": ".gated_mlp",
    "GatedMLPWorkspace": ".gated_mlp",
    "PreparedGatedMLP": ".gated_mlp",
    "ResidentGatedMLP": ".gated_mlp",
    "AttentionDimensions": ".decoder_layer",
    "DecoderLayerWeights": ".decoder_layer",
    "ResidentDecoderLayer": ".decoder_layer",
    "PreparedDecoderLayer": ".decoder_layer",
    "DecoderStep": ".decoder_layer",
    "CapturedDecoderStep": ".decoder_layer",
    "DecoderWorkspace": ".decoder_layer",
    "LayerKVCache": ".decoder_layer",
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
