"""Independent post-publication storage checks; never a model-quality verifier."""

from collections import Counter
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint import inspect_checkpoint
from .models.qwen_source import QwenSourcePlan
from .models.qwen_weights import QwenQ8Weights
from .ninfer import NInferArtifact
from .qwen_export import _digest, _source_evidence, _stamp


def verify_qwen_q8_export(source, artifact, report):
    source = Path(source).expanduser().resolve()
    artifact = Path(artifact).expanduser().resolve()
    published = json.loads(Path(report).read_text())
    before = _stamp(artifact)
    checkpoint = inspect_checkpoint(source)
    plan = QwenSourcePlan.from_config(dict(checkpoint.config))
    ignored = plan.validate_checkpoint(checkpoint)
    evidence, stamps = _source_evidence(
        checkpoint, manifest=source / "venturi-source.json", allow_unverified=False
    )
    if (
        Path(published["output"]).resolve() != artifact
        or published["file_bytes"] != artifact.stat().st_size
    ):
        raise ValueError("Artifact does not match the published export report")
    digest = _digest(artifact)
    if digest != published["sha256"]:
        raise ValueError("Published artifact SHA256 mismatch")
    container = NInferArtifact(artifact)
    if container.manifest["components"]["text"]["config"] != plan.config:
        raise ValueError("Artifact config differs from original source")
    if (
        container.manifest.get("provenance", {}).get("source") != evidence
        or published.get("source") != evidence
    ):
        raise ValueError(
            "Artifact/report source provenance differs from verified source"
        )
    QwenQ8Weights.from_ninfer(container)
    if published["artifact_id"] != container.artifact_id or published[
        "tensor_count"
    ] != len(plan.bindings):
        raise ValueError("Artifact identity/count mismatch")
    floats, rows, max_error = 0, 0, 0.0
    for spec in plan.bindings:
        obj = container.objects[container.manifest["bindings"][spec.binding]["object"]]
        with safe_open(
            str(checkpoint.tensors[spec.source].file), framework="pt", device="cpu"
        ) as reader:
            if spec.format == "q8_g32_fp16":
                sliced = reader.get_slice(spec.source)
                n, k = spec.shape
                for row in sorted({0, n // 2, n - 1}):
                    raw = sliced[row : row + 1].double().flatten().tolist()
                    expected = []
                    # Independent scalar nearest-even quantization, using the
                    # stored FP16 divisor, not the production Torch quantizer.
                    for start in range(0, k, 32):
                        group = raw[start : start + 32]
                        scale = torch.tensor(
                            max(map(abs, group)) / 127, dtype=torch.float16
                        ).item()
                        expected.extend(
                            min(127, max(-127, round(x / scale))) * scale
                            if scale
                            else 0
                            for x in group
                        )
                    actual = container.decode_row(obj["id"], row)
                    if actual != expected:
                        raise ValueError(
                            f"Q8 sample differs from original source: {spec.binding} row {row}"
                        )
                    max_error = max(
                        max_error, max(abs(x - y) for x, y in zip(actual, raw))
                    )
                    rows += 1
            else:
                value = reader.get_tensor(spec.source)
                if spec.transform == "conv-kc":
                    value = value.squeeze(1).transpose(0, 1)
                dtype = torch.float32 if spec.format == "fp32" else torch.bfloat16
                actual = torch.frombuffer(
                    bytearray(container.read_object(obj["id"], max_bytes=8 * 2**20)),
                    dtype=dtype,
                ).reshape(spec.shape)
                if not torch.equal(actual, value.to(dtype)):
                    raise ValueError(
                        f"Floating payload differs from original source: {spec.binding}"
                    )
                floats += 1
    if _stamp(artifact) != before or any(
        _stamp(source / name) != stamp for name, stamp in stamps.items()
    ):
        raise ValueError("Source/artifact changed during verification")
    return dict(
        scope="Full source/artifact hashes, all floating tensors, selected Q8 rows per matrix; NOT full forward/generation/quality qualification",
        artifact_id=container.artifact_id,
        artifact_sha256=digest,
        file_bytes=artifact.stat().st_size,
        text_tensors=len(plan.bindings),
        excluded_tensors=len(ignored),
        full_floating_tensors_equal=floats,
        independently_checked_q8_rows=rows,
        max_sampled_abs_weight_quantization_error=max_error,
        target_formats=dict(Counter(s.format for s in plan.bindings)),
        source=evidence,
    )
