"""Pure load-time numerical/capability validation for canonical row QuantLinear.

Codec storage admission, kernel capability, and model execution qualification
are deliberately independent. Unsupported policy combinations never fall back.
"""

import torch
from .codecs import (
    codec_for,
    EXECUTION_QUALIFIED,
    ACTIVATION_INT8_QUALIFIED,
    FUSED_LINEAR_QUALIFIED,
)


def validate_linear_contracts(spec, weights):
    arch = tuple(torch.cuda.get_device_capability(spec.target))
    if arch not in ((7, 0), (8, 9)):
        raise ValueError("Unsupported QuantLinear architecture")
    row = []
    for binding in weights.plan.bindings:
        codec = codec_for(binding.format)
        if not codec.quantized:
            continue
        if binding.format not in EXECUTION_QUALIFIED.get(arch, ()):
            raise ValueError(
                "Matrix codec is not execution-qualified on this architecture: "
                + binding.format
            )
        if (
            spec.projection_profile == "int8"
            and binding.format not in ACTIVATION_INT8_QUALIFIED.get(arch, ())
        ):
            raise ValueError(
                "One-pass int8 model execution qualification pending: " + binding.format
            )
        if (
            spec.linear_implementation == "fused"
            and binding.format not in FUSED_LINEAR_QUALIFIED.get(arch, ())
        ):
            raise ValueError(
                "Fused model execution qualification pending: " + binding.format
            )
        if codec.bits != 8:
            row.append(binding)
            if (
                spec.dtype != torch.float16
                or spec.weight_mode != "resident"
                or spec.tensor_parallel_device is not None
                or spec.stage_range is not None
            ):
                raise ValueError(
                    "row-codec matrices require resident single-device FP16 without TP/PP"
                )
            if spec.projection_profile not in ("fp16", "int8"):
                raise ValueError("Residual activation policy requires Q8 matrices")
        if spec.projection_profile == "int8" and not (
            codec.bits == 8 or codec.bits == 5 and arch == (7, 0)
        ):
            raise ValueError(
                "Explicit int8 activation policy has no qualified kernel for "
                + binding.format
            )
    if spec.linear_implementation == "dequant" and not row:
        raise ValueError("dequant reference requires Q4/Q5/Q6 matrices")
    if spec.linear_implementation == "dequant" and (
        spec.dtype != torch.float16
        or spec.tensor_parallel_device is not None
        or spec.stage_range is not None
        or spec.weight_mode != "resident"
    ):
        raise ValueError(
            "dequant reference requires resident single-device FP16 without TP/PP"
        )
    if spec.dequant_resident and any(
        codec_for(b.format).quantized and codec_for(b.format).bits == 8
        for b in weights.plan.bindings
    ):
        raise ValueError("FP16 shadow requires only Q4/Q5/Q6 matrices")


def describe_linear_contracts(weights, profile):
    """Load-time diagnostics, reusable by other model adapters; no device IO."""
    return {
        fmt: {
            "layout": codec_for(fmt).layout,
            "activation_policy": profile,
            "weight_decode": (
                "fp16-rounded-q-times-scale"
                if codec_for(fmt).bits in (5, 6) and profile == "fp16"
                else "fp32-scaled-integer-products"
            ),
            "accumulation": "fp32; implementation-dependent association",
        }
        for fmt in sorted(
            {b.format for b in weights.plan.bindings if codec_for(b.format).quantized}
        )
    }


def validate_reported_plan(fmt, profile, architecture, values):
    """Bind qualification to the ACTUAL native family, not the selector string."""
    codec = codec_for(fmt)
    passes = 0 if profile == "fp16" else 1 if profile == "int8" else 2
    if len(values) != 6 or tuple(values[:3]) != (codec.bits, codec.group, passes):
        raise RuntimeError("Native QuantLinear numerical/codec contract mismatch")
    family = int(values[3])
    if family == 0 and fmt not in FUSED_LINEAR_QUALIFIED.get(tuple(architecture), ()):
        raise ValueError(
            "Native backend selected an unqualified direct/fused family: " + fmt
        )
    if family == 1 and (codec.bits not in (4, 5, 6) or passes != 0):
        raise RuntimeError("Native dense row-codec plan contract mismatch")
    if family == 2 and passes == 0 or family not in (0, 1, 2):
        raise RuntimeError("Native QuantLinear family/activation-policy mismatch")
    return (
        "resident-fp16-shadow"
        if values[5]
        else ("direct", "chunked-dense", "activation-int8")[family]
    )
