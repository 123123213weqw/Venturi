"""Public models API, resolved only when a concrete export is requested.

Importing the namespace does not load unrelated model families, tensor frameworks
or native adapters. Existing public names retain their implementation identities.
"""

from importlib import import_module

_EXPORTS = {
    "AdaptiveQwenMTPBatch": "venturi.models.qwen.qwen_adaptive_mtp",
    "NativeQwenBatchSession": "venturi.models.qwen.qwen_batch",
    "NativeQwenMTPBatchSession": "venturi.models.qwen.qwen_mtp_batch",
    "QwenMTPPlan": "venturi.models.qwen.qwen_mtp",
    "QwenMTPWeights": "venturi.models.qwen.qwen_mtp",
    "NativeQwenMTPModel": "venturi.models.qwen.qwen_mtp_native",
    "NativeQwenMTPSession": "venturi.models.qwen.qwen_mtp_native",
    "NativeQwenModel": "venturi.models.qwen.qwen_native",
    "NativeQwenPipelineModel": "venturi.models.qwen.qwen_pipeline",
    "NativeQwenPipelineSession": "venturi.models.qwen.qwen_pipeline",
    "NativeQwenSession": "venturi.models.qwen.qwen_native",
    "StreamingQwenModel": "venturi.models.qwen.qwen_streaming",
    "StreamingQwenSession": "venturi.models.qwen.qwen_streaming",
    "qwen_request_bytes": "venturi.models.qwen.qwen_model",
    "ResidentQwenModel": "venturi.models.qwen.qwen_model",
    "QwenSession": "venturi.models.qwen.qwen_model",
    "qwen_resident_bytes": "venturi.models.qwen.qwen_model",
    "ResidentQwenAttentionLayer": "venturi.models.qwen.qwen_attention_layer",
    "PreparedQwenAttentionLayer": "venturi.models.qwen.qwen_attention_layer",
    "QwenAttentionWorkspace": "venturi.models.qwen.qwen_attention_layer",
    "QwenAttentionStep": "venturi.models.qwen.qwen_attention_layer",
    "QwenTextRoPE": "venturi.models.qwen.qwen_rope",
    "GDNLayerState": "venturi.models.qwen.qwen_gdn_layer",
    "PreparedQwenGDNLayer": "venturi.models.qwen.qwen_gdn_layer",
    "QwenGDNWorkspace": "venturi.models.qwen.qwen_gdn_layer",
    "ResidentQwenGDNLayer": "venturi.models.qwen.qwen_gdn_layer",
    "QwenSourceBinding": "venturi.models.qwen.qwen_source",
    "QwenSourcePlan": "venturi.models.qwen.qwen_source",
    "QwenRuntime": "venturi.models.qwen.native_runtime",
    "QwenWeights": "venturi.models.qwen.qwen_weights",
    "GDNFrontendWeights": "venturi.models.qwen.gdn_frontend",
    "GDNMixerWorkspace": "venturi.models.qwen.gdn_frontend",
    "PreparedGDNMixerCore": "venturi.models.qwen.gdn_frontend",
    "ResidentGDNFrontend": "venturi.models.qwen.gdn_frontend",
    "LlamaConfig": ".llama",
    "LlamaWeights": ".llama",
    "LlamaModel": ".llama",
    "LlamaSession": ".llama",
    "GPTNeoXConfig": "venturi.models.neox.gpt_neox",
    "GPTNeoXModel": "venturi.models.neox.gpt_neox",
    "GPTNeoXSession": "venturi.models.neox.gpt_neox",
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
