"""Residency budget planning: encoded byte accounting and memory admission.

Computes the Q8/small/scratch budget (including pipeline stage pruning and
tensor-parallel primary/secondary splits) and runs the explicit memory guards.
No allocation happens here; the guards fail closed with no implicit fallback.
"""

from dataclasses import dataclass

from ..qwen_model import _memory_guard, qwen_resident_bytes
from ..qwen_rope import QwenTextRoPE
from ...codecs import codec_for


def _head_gdn_workspace(cfg, rows):
    heads = cfg["linear_num_value_heads"] // 2
    channels = (
        cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        + heads * cfg["linear_value_head_dim"]
    )
    values = heads * cfg["linear_value_head_dim"]
    return rows * ((channels * 2 + values * 3 + heads * 2) * 2 + heads * 8)


@dataclass(frozen=True)
class Budget:
    head_workspace: int
    quant_bytes: int
    rope: object


def plan_budget(spec, owned, cfg):
    max_tokens = spec.max_tokens
    dtype = spec.dtype
    peer = spec.tensor_parallel_device
    target = spec.target
    head_workspace = _head_gdn_workspace(cfg, max_tokens) if spec.head_profile else 0
    rope = QwenTextRoPE.from_config(cfg)
    q8_bytes = []
    for binding in owned.plan.bindings:
        codec = codec_for(binding.format)
        if codec.quantized:
            n, k = binding.shape
            q8_bytes.append(codec.upload_bytes(n, k))
    small = qwen_resident_bytes(owned, dtype=dtype) - sum(q8_bytes)
    ep = (cfg["hidden_size"] + 127) // 128 * 128
    peak = max(*q8_bytes, max_tokens * (ep + ep // 32 * 2))
    max_projection_k = max(
        s.shape[1]
        for s in owned.plan.bindings
        if codec_for(s.format).quantized
    )
    quant_bytes = (
        max_tokens * ((max_projection_k + 31) // 32) * 36 * 2
        if spec.projection_profile != "fp16"
        else 0
    )
    # Temporary loader constants + native owned copies coexist at creation.
    required = (
        (sum(q8_bytes) if spec.weight_mode == "resident" else peak)
        + 2 * small
        + 4 * 2**20
        + rope.rotary_dim * 2
    )
    if spec.stage_range is not None:
        # A conservative bound for small loader/native constants remains
        # shared with the full adapter. Large Q8 payloads are stage-local.
        selected = {}
        first, last = spec.stage_range
        for binding in owned.plan.bindings:
            if not codec_for(binding.format).quantized:
                continue
            role = binding.binding
            include = (
                role == "text/token_embedding"
                and first == 0
                or role == "text/output_head"
                and last == cfg["num_hidden_layers"]
                or role.startswith("text/layers/")
                and first <= int(role.split("/")[2]) < last
            )
            if include:
                obj = owned._object(role)
                n, k = binding.shape
                selected[obj["offset"]] = codec_for(binding.format).upload_bytes(n, k)
        required = (
            sum(selected.values()) + 2 * small + 4 * 2**20 + rope.rotary_dim * 2
        )
    if peer is not None:
        primary = secondary = max_k = max_rows = 0
        for binding in owned.plan.bindings:
            codec = codec_for(binding.format)
            if not codec.quantized:
                continue
            n, k = binding.shape
            padded = (k + 127) // 128 * 128
            codes = padded * codec.bits // 8
            scales = (padded // codec.group) * 2
            half = 1 + (n - 1) // 2
            embedding = binding.binding == "text/token_embedding"
            if spec.column_profile and (
                binding.binding.endswith("/mlp/down")
                or spec.head_profile
                and binding.binding.endswith("/gdn/output")
            ):
                local_pad = ((k // 2 + 127) // 128) * 128
                local_codes = local_pad * codec.bits // 8
                local_scales = (local_pad // codec.group) * 2
                encoded = n * (local_codes + local_scales)
                primary += encoded
                secondary += encoded
            else:
                primary += (n if embedding else half) * (codes + scales)
                secondary += (
                    n - half if not embedding or owned.plan.tied_embeddings else 0
                ) * (codes + scales)
            max_k, max_rows = max(max_k, k), max(max_rows, half)
        column_partial = (
            max_tokens * cfg["hidden_size"] * 4 if spec.column_profile else 0
        )
        column_peer_extra = (
            column_partial + max_tokens * cfg["intermediate_size"] * 3
            if spec.column_profile
            else 0
        )
        if spec.column_profile:
            max_rows = max(max_rows, cfg["intermediate_size"])
        required = (
            primary
            + column_partial * 2
            + head_workspace
            + 2 * small
            + 4 * 2**20
            + rope.rotary_dim * 2
            + max_tokens * max_rows * 2
        )
        _memory_guard(
            peer,
            secondary
            + max_tokens * (max_k + max_rows) * 2
            + quant_bytes
            + column_peer_extra
            + head_workspace
            + (2 * small if spec.head_profile else 0),
            spec.reserve_bytes,
            None,
        )
    _memory_guard(target, required + quant_bytes, spec.reserve_bytes, None)
    return Budget(head_workspace=head_workspace, quant_bytes=quant_bytes, rope=rope)
