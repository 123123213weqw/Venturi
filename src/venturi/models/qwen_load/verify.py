"""Post-create accounting reconciliation against the native runtime.

Builds the projection-info summary and verifies the native workspace/profile,
column-MLP and head-GDN accounting vectors. Any mismatch destroys the handle
and re-raises; nothing partial is left behind.
"""

import ctypes as C

import torch


def normalized_profile(spec):
    return (
        "bf16"
        if spec.projection_profile == "fp16" and spec.dtype == torch.bfloat16
        else spec.projection_profile
    )


def projection_info(spec, cfg, quant_bytes, native):
    dtype = spec.dtype
    info = dict(
        projection_profile=normalized_profile(spec),
        activation_quantization_passes=2 if spec.projection_profile != "fp16" else 0,
        primary_projection_workspace_bytes=quant_bytes,
        peer_projection_workspace_bytes=(
            quant_bytes if spec.tensor_parallel_device is not None else 0
        ),
        projection_precision_scope=f"Original Q8 weights/scales; {dtype} storage/KV, FP32 GDN. q8-residual uses explicit group32 activation high/residual Q8 passes with FP32 scales and changed reduction order; not original FP16 arithmetic.",
    )
    if spec.column_profile:
        info[
            "projection_precision_scope"
        ] += " Column MLP keeps partial outputs/host transport FP32, then sums before one FP16 cast; changed FP32 association versus full-K down, not prior-profile byte parity."
        query = getattr(spec.backend._native, "venturi_q8_swiglu_quant_enabled", None)
        if query is not None:
            query.argtypes, query.restype = [], C.c_int
            enabled = bool(query())
            info["swiglu_quant_fusion"] = dict(
                compiled_opt_in=enabled,
                eligible_intermediate_per_rank=8704,
                eligible_token_range=[1, 32],
                model_has_eligible_mlp=enabled
                and cfg["intermediate_size"] // 2 == 8704,
                precision="Original F16 rounding and both residual-A8 passes preserved",
            )
        query = getattr(spec.backend._native, "venturi_q8_rms_quant_enabled", None)
        if query is not None:
            query.argtypes, query.restype = [], C.c_int
            enabled = bool(query())
            info["rms_quant_fusion"] = dict(
                compiled_opt_in=enabled,
                model_has_eligible_norm=enabled and cfg["hidden_size"] == 5120,
                eligible_token_range=[1, 32],
                location="post-attention norm and primary MLP input packing",
                peer_input="Original materialized FP16 normalized values; peer packing unchanged",
            )
    return info


def reconcile(spec, runtime, handle, cfg, owned, budget, info):
    native = runtime._native
    max_tokens = spec.max_tokens
    quant_bytes = budget.quant_bytes
    head_workspace = budget.head_workspace
    if spec.profiled_create is None:
        return  # fp16-profile models have no profiled native accounting to check
    try:
        query = native.venturi_qwen_projection_info
        query.argtypes, query.restype = (
            [C.c_void_p, C.POINTER(C.c_uint64)],
            C.c_int,
        )
        values = (C.c_uint64 * 3)()
        runtime._check(query(handle, values))
        if list(values) != [
            2,
            quant_bytes,
            quant_bytes if spec.tensor_parallel_device is not None else 0,
        ]:
            raise RuntimeError(
                "Native projection workspace/profile accounting mismatch"
            )
        if spec.column_profile:
            query = native.venturi_qwen_column_mlp_info
            query.argtypes, query.restype = (
                [C.c_void_p, C.POINTER(C.c_uint64)],
                C.c_int,
            )
            values = (C.c_uint64 * 8)()
            runtime._check(query(handle, values))
            partial = max_tokens * cfg["hidden_size"] * 4
            unique_down = len(
                {
                    owned.plane_offset(f"text/layers/{i}/mlp/down")
                    for i in range(cfg["num_hidden_layers"])
                }
            )
            expected = [
                1,
                partial,
                partial,
                partial,
                max_tokens * cfg["intermediate_size"] * 2,
                max_tokens * cfg["intermediate_size"],
                partial,
                unique_down,
            ]
            if list(values) != expected:
                raise RuntimeError(
                    "Native column MLP memory/layout accounting mismatch"
                )
            info.update(
                column_mlp_enabled=True,
                column_mlp_buffers=dict(
                    zip(
                        [
                            "enabled",
                            "primary_local_fp32_bytes",
                            "primary_remote_fp32_bytes",
                            "peer_partial_fp32_bytes",
                            "peer_gate_up_bytes",
                            "peer_activation_bytes",
                            "pinned_fp32_partial_bytes",
                            "column_down_matrices",
                        ],
                        map(int, values),
                    )
                ),
                mlp_down_weight_layout="bit-preserving input-column halves, original group32 scales",
                partial_and_host_transport_dtype="float32",
            )
        if spec.head_profile:
            query = native.venturi_qwen_head_gdn_info
            query.argtypes, query.restype = (
                [C.c_void_p, C.POINTER(C.c_uint64)],
                C.c_int,
            )
            values = (C.c_uint64 * 5)()
            runtime._check(query(handle, values))
            gdn_count = cfg["layer_types"].count("linear_attention")
            if list(values) != [
                1,
                head_workspace,
                head_workspace,
                gdn_count,
                gdn_count,
            ]:
                raise RuntimeError(
                    "Native head GDN workspace/layout accounting mismatch"
                )
            info.update(
                head_gdn_enabled=True,
                head_gdn_primary_workspace_bytes=head_workspace,
                head_gdn_peer_workspace_bytes=head_workspace,
                gdn_weight_layout="bit-preserved semantic Q/K/V head rows and group32 output-column halves",
                gdn_state_layout="private packed FP32 value-head halves, split convolution histories and MTP prefixes on both devices; canonical joining diagnostics only",
            )
            info[
                "projection_precision_scope"
            ] += " Head-local GDN output also uses FP32 column partial/host transport/sum before one FP16 cast; changed FP32 association, not prior-profile byte parity."
    except Exception:
        native.venturi_qwen_destroy(handle)
        raise
