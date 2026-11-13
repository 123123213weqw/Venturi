"""Source-relative safetensors export verification, independent of the writer."""

from collections import Counter
import json
from pathlib import Path
import struct
import torch
from safetensors import safe_open
from venturi.container.checkpoint import inspect_checkpoint
from .qwen_source import QwenSourcePlan
from .qwen_weights import QwenWeights
from .qwen_mtp import QwenMTPPlan, QwenMTPWeights
from .qwen_export import _digest, _source_evidence, _stamp


def verify_qwen_q8_export(
    source, artifact, report, *, component="text", source_manifest=None
):
    if component not in ("text", "mtp"):
        raise ValueError("Component must be text or mtp")
    source = Path(source).resolve()
    path = Path(artifact).resolve()
    published = json.loads(Path(report).read_text())
    before = _stamp(path)
    checkpoint = inspect_checkpoint(source)
    plan = (QwenMTPPlan if component == "mtp" else QwenSourcePlan).from_config(
        dict(checkpoint.config)
    )
    ignored = plan.validate_checkpoint(checkpoint)
    evidence, stamps = _source_evidence(
        checkpoint,
        manifest=(
            Path(source_manifest) if source_manifest else source / "venturi-source.json"
        ),
        allow_unverified=False,
    )
    if (
        Path(published["output"]).resolve() != path
        or published["file_bytes"] != path.stat().st_size
    ):
        raise ValueError("Published path/size mismatch")
    digest = _digest(path)
    if digest != published["sha256"]:
        raise ValueError("Published SHA256 mismatch")
    weights = (QwenMTPWeights if component == "mtp" else QwenWeights).open(path)
    if (
        weights.config != plan.config
        or weights.provenance != evidence
        or published.get("source") != evidence
    ):
        raise ValueError("Export config/source provenance mismatch")
    if published["role_count"] != len(plan.bindings) or published[
        "tensor_count"
    ] != len(weights.source.list_tensors()):
        raise ValueError("Export tensor count mismatch")
    floats = rows = 0
    max_error = 0.0
    for spec in plan.bindings:
        with safe_open(
            str(checkpoint.tensors[spec.source].file), framework="pt", device="cpu"
        ) as reader:
            if spec.format == "q8_g32_fp16":
                sliced = reader.get_slice(spec.source)
                n, k = spec.shape
                padded = (k + 127) // 128 * 128
                groups = padded // 32
                for row in sorted({0, n // 2, n - 1}):
                    values = sliced[row : row + 1].double().flatten().tolist()
                    expected = []
                    for begin in range(0, k, 32):
                        group = values[begin : begin + 32]
                        scale = torch.tensor(
                            max(map(abs, group)) / 127, dtype=torch.float16
                        ).item()
                        expected.extend(
                            (
                                min(127, max(-127, round(x / scale))) * scale
                                if scale
                                else 0.0
                            )
                            for x in group
                        )
                    codes = weights.source.read_bytes(
                        spec.source + ".codes", row * padded, padded
                    )
                    scales = weights.source.read_bytes(
                        spec.source + ".scales", row * groups * 2, groups * 2
                    )
                    actual = []
                    for i in range(k):
                        code = codes[i] if codes[i] < 128 else codes[i] - 256
                        scale = struct.unpack_from("<e", scales, i // 32 * 2)[0]
                        if code == -128:
                            raise ValueError("Forbidden Q8 code -128")
                        actual.append(code * scale)
                    if actual != expected:
                        raise ValueError(
                            "Q8 sample differs from original source: " + spec.binding
                        )
                    max_error = max(
                        max_error, max(abs(a - b) for a, b in zip(actual, values))
                    )
                    rows += 1
            else:
                expected = reader.get_tensor(spec.source)
                if spec.transform == "conv-kc":
                    expected = expected.squeeze(1).transpose(0, 1)
                expected = expected.to(
                    torch.float32 if spec.format == "fp32" else torch.bfloat16
                )
                if not torch.equal(weights.read_float(spec.binding), expected):
                    raise ValueError("Floating payload differs from original source")
                floats += 1
    if _stamp(path) != before or any(
        _stamp(source / name) != stamp for name, stamp in stamps.items()
    ):
        raise ValueError("Source/export changed during verification")
    return dict(
        scope="Source-relative storage verification, not model quality",
        artifact_sha256=digest,
        file_bytes=path.stat().st_size,
        text_tensors=len(plan.bindings) if component == "text" else 0,
        component=component,
        component_tensors=len(plan.bindings),
        excluded_tensors=len(ignored),
        full_floating_tensors_equal=floats,
        independently_checked_q8_rows=rows,
        max_sampled_abs_weight_quantization_error=max_error,
        target_formats=dict(Counter(s.format for s in plan.bindings)),
        source=evidence,
    )
