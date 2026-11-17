"""Pure load-time numerical/capability validation for canonical row QuantLinear.

Codec storage admission, kernel capability, and model execution qualification
are deliberately independent. Unsupported policy combinations never fall back.
"""

import torch
from .codecs import codec_for, EXECUTION_QUALIFIED, ACTIVATION_INT8_QUALIFIED


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
                if codec_for(fmt).bits in (4, 5, 6) and profile == "fp16"
                else "fp32-scaled-integer-products"
            ),
            "accumulation": "fp32; implementation-dependent association",
        }
        for fmt in sorted(
            {b.format for b in weights.plan.bindings if codec_for(b.format).quantized}
        )
    }
