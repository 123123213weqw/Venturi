"""Source preparation: re-open the validated artifact and check config geometry.

The artifact is re-opened from its own path (never trusting the caller's
in-memory copy), then stage-range and profile geometry constraints are
enforced against the freshly copied config, in the historical order.
"""

import copy

from ...ninfer import NInferArtifact
from ..decoder_layer import _outside_capture
from ..qwen_weights import QwenQ8Weights


def prepare_source(spec):
    _outside_capture(spec.target)
    # Re-open from the artifact's own path with the caller's declared matrix
    # codec, never trusting the in-memory copy. The execution gate in the spec
    # has already restricted this to q8 in practice.
    matrix_format = next(
        s.format
        for s in spec.weights.plan.bindings
        if s.binding == "text/token_embedding"
    )
    owned = QwenQ8Weights.from_ninfer(
        NInferArtifact(spec.weights.artifact.path), matrix_format=matrix_format
    )
    cfg = copy.deepcopy(owned.config)
    if spec.stage_range is not None:
        if (
            not isinstance(spec.stage_range, tuple)
            or len(spec.stage_range) != 2
            or any(type(i) is not int for i in spec.stage_range)
            or not 0
            <= spec.stage_range[0]
            < spec.stage_range[1]
            <= cfg["num_hidden_layers"]
            or spec.weight_mode != "resident"
            or spec.tensor_parallel_device is not None
        ):
            raise ValueError(
                "Internal pipeline stage requires a valid resident non-TP range"
            )
    if spec.column_profile and cfg["intermediate_size"] % 64:
        raise ValueError("Column MLP requires group-aligned equal input halves")
    if spec.head_profile and (
        cfg["linear_num_key_heads"] % 2
        or cfg["linear_num_key_heads"] < 2
        or cfg["linear_num_value_heads"] > 128
        or cfg["linear_value_head_dim"] % 64
        or not 1 <= cfg["linear_value_head_dim"] <= 256
        or cfg["hidden_size"] % 32
        or cfg["hidden_size"] > 32768
        or not 2 <= cfg["linear_conv_kernel_dim"] <= 8
    ):
        raise ValueError(
            "Head GDN requires aligned semantic heads and convolution geometry"
        )
    return owned, cfg
