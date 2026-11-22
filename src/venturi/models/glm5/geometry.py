"""GLM-5.3-Flash text geometry parsed from GGUF metadata (glm5-next arch).

Strict, fail-closed parsing of the llama.cpp-converted metadata keys measured
on the real Unsloth UD quantization. Layer composition: 46 blocks where the
last `nextn_predict_layers` are MTP draft blocks; blocks 0..2 dense MLP; 11
sparse-MLA blocks (identified by their fused-MLA tensor set); the rest KDA
linear-attention blocks (ssm_* tensors); MoE on every sparse block.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Glm5TextGeometry:
    hidden_size: int
    num_layers: int          # main blocks (block_count - nextn)
    nextn_layers: int        # MTP draft blocks at the tail
    vocab_size: int
    head_count: int
    kv_lora_rank: int
    q_lora_rank: int
    key_length_mla: int
    value_length_mla: int
    indexer_heads: int
    indexer_key_length: int
    indexer_top_k: int
    indexer_kpool: int
    kda_head_dim: int
    kda_heads: int
    ssm_conv_kernel: int
    gate_lower_bound: float
    expert_count: int
    expert_used_count: int
    expert_intermediate: int
    expert_shared_count: int
    expert_shared_intermediate: int
    expert_weights_scale: float
    dense_feed_forward: int
    leading_dense_blocks: int
    hc_streams: int
    hc_sinkhorn_iterations: int
    rms_eps: float
    hc_eps: float
    context_length: int
    eos_token_id: int
    bos_token_id: int

    def attention_kind(self, layer: int) -> str:
        """'sparse_mla' | 'kda' — derived from which tensors the block carries."""
        return "sparse_mla" if layer in self.sparse_mla_layers else "kda"

    def mlp_kind(self, layer: int) -> str:
        return "dense" if layer < self.leading_dense_blocks else "sparse"

    def __post_init__(self):
        if self.nextn_layers < 0 or self.num_layers <= 0:
            raise ValueError("GLM5 layer counts must be positive with nonnegative MTP tail")
        sparse = tuple(range(self.leading_dense_blocks, self.num_layers, 4))
        object.__setattr__(self, "sparse_mla_layers", sparse)


_T = "glm5-next."


def _positive(metadata, key):
    value = metadata.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"GLM5 metadata key missing or nonpositive: {key}")
    return value


def _scalar(metadata, key, *, cls):
    value = metadata.get(key)
    if type(value) is not cls:
        raise ValueError(f"GLM5 metadata key missing or wrong type: {key}")
    return value


def glm5_text_geometry_from_metadata(metadata) -> Glm5TextGeometry:
    """Parse the llama.cpp glm5-next metadata block, fail-closed."""
    if metadata.get("general.architecture") != "glm5-next":
        raise ValueError("Not a glm5-next GGUF model")
    blocks = _positive(metadata, _T + "block_count")
    nextn = metadata.get(_T + "nextn_predict_layers")
    if type(nextn) is not int or not 0 <= nextn < blocks:
        raise ValueError("GLM5 nextn_predict_layers must be an int within the block count")
    heads = _positive(metadata, _T + "attention.head_count")
    kda_head_dim = _positive(metadata, _T + "kda.head_dim")
    geometry = Glm5TextGeometry(
        hidden_size=_positive(metadata, _T + "embedding_length"),
        num_layers=blocks - nextn,
        nextn_layers=nextn,
        vocab_size=_positive(metadata, _T + "vocab_size"),
        head_count=heads,
        kv_lora_rank=_positive(metadata, _T + "attention.kv_lora_rank"),
        q_lora_rank=_positive(metadata, _T + "attention.q_lora_rank"),
        key_length_mla=_positive(metadata, _T + "attention.key_length_mla"),
        value_length_mla=_positive(metadata, _T + "attention.value_length_mla"),
        indexer_heads=_positive(metadata, _T + "attention.indexer.head_count"),
        indexer_key_length=_positive(metadata, _T + "attention.indexer.key_length"),
        indexer_top_k=_positive(metadata, _T + "attention.indexer.top_k"),
        indexer_kpool=_positive(metadata, _T + "attention.indexer.kpool"),
        kda_head_dim=kda_head_dim,
        kda_heads=heads,
        ssm_conv_kernel=_positive(metadata, _T + "ssm.conv_kernel"),
        gate_lower_bound=_scalar(metadata, _T + "kda.gate_lower_bound", cls=float),
        expert_count=_positive(metadata, _T + "expert_count"),
        expert_used_count=_positive(metadata, _T + "expert_used_count"),
        expert_intermediate=_positive(metadata, _T + "expert_feed_forward_length"),
        expert_shared_count=_positive(metadata, _T + "expert_shared_count"),
        expert_shared_intermediate=_positive(metadata, _T + "expert_shared_feed_forward_length"),
        expert_weights_scale=_scalar(metadata, _T + "expert_weights_scale", cls=float),
        dense_feed_forward=_positive(metadata, _T + "feed_forward_length"),
        leading_dense_blocks=_positive(metadata, _T + "leading_dense_block_count"),
        hc_streams=_positive(metadata, _T + "hyper_connection.count"),
        hc_sinkhorn_iterations=_positive(metadata, _T + "hyper_connection.sinkhorn_iterations"),
        rms_eps=_scalar(metadata, _T + "attention.layer_norm_rms_epsilon", cls=float),
        hc_eps=_scalar(metadata, _T + "hyper_connection.epsilon", cls=float),
        context_length=_positive(metadata, _T + "context_length"),
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id", -1),
        bos_token_id=metadata.get("tokenizer.ggml.bos_token_id", -1),
    )
    if _scalar(metadata, _T + "expert_gating_func", cls=int) != 2:
        raise ValueError("GLM5 MoE gating function must be sigmoid (id 2)")
    if not _scalar(metadata, _T + "expert_weights_norm", cls=bool):
        raise ValueError("GLM5 expert_weights_norm must be true for noaux_tc routing")
    if geometry.expert_used_count > geometry.expert_count:
        raise ValueError("GLM5 experts_per_tok exceeds the expert count")
    return geometry


# Tensor-name templates measured on the real GGUF; (template, present_when).
MLA_TENSORS = (
    "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
    "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
    "attn_v_b.weight", "attn_output.weight",
    "indexer.attn_k.weight", "indexer.attn_q_b.weight",
    "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
    "indexer_compressor_ape.weight", "indexer_compressor_gate.weight",
)
KDA_TENSORS = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ssm_a", "ssm_beta.weight", "ssm_conv1d_k.weight", "ssm_conv1d_q.weight",
    "ssm_conv1d_v.weight", "ssm_dt.bias", "ssm_f_a.weight", "ssm_f_b.weight",
    "ssm_g_a.weight", "ssm_g_b.weight", "ssm_norm.weight",
)
PER_LAYER_SHARED = (
    "attn_norm.weight", "ffn_norm.weight",
    "hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
    "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
)
SPARSE_ONLY = ("ffn_gate_inp.weight", "exp_probs_b.bias")
SPARSE_MLP_TENSORS = (
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
)
GLOBAL_TENSORS = ("token_embd.weight", "output.weight", "output_norm.weight")


DENSE_MLP_TENSORS = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
NEXTN_TENSORS = ("nextn.eh_proj.weight", "nextn.enorm.weight",
                 "nextn.hnorm.weight", "nextn.shared_head_norm.weight")


def layer_tensor_names(geometry: Glm5TextGeometry, layer: int):
    """Expected tensor names (suffixes) for one block, by its kinds.

    The trailing nextn (MTP draft) blocks carry the sparse-MLA attention set,
    sparse MoE and their own nextn.* projections — but no hc_* mixing and no
    KDA ssm tensors. Measured on the real weights.
    """
    is_nextn = layer >= geometry.num_layers
    if is_nextn:
        names = list(MLA_TENSORS) + ["attn_norm.weight", "ffn_norm.weight"]
        names.extend(SPARSE_MLP_TENSORS)
        names.extend(SPARSE_ONLY)
        names.extend(NEXTN_TENSORS)
        return tuple(f"blk.{layer}.{name}" for name in names)
    attention = MLA_TENSORS if geometry.attention_kind(layer) == "sparse_mla" else KDA_TENSORS
    names = list(attention) + list(PER_LAYER_SHARED)
    if geometry.mlp_kind(layer) == "sparse":
        names.extend(SPARSE_MLP_TENSORS)
        names.extend(SPARSE_ONLY)
    else:
        names.extend(DENSE_MLP_TENSORS)
    return tuple(f"blk.{layer}.{name}" for name in names)
