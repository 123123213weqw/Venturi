"""CPU-only, bounded-row original HF -> separately identified Q8 text artifact.

No GPU allocation, remote code, Q4 re-quantization or model forward. Each code/scale plane is a real safetensors tensor; no private object directory.
"""

import ctypes
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

from venturi.container.checkpoint import CheckpointError, _read_json, inspect_checkpoint
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.container.safetensors import SafetensorsWriter
from venturi.ops.weights import Q8RowSplitWeight

QUANTIZATION = {
    "scheme": "q8_g32_fp16",
    "codes": "signed [-127,127]; nearest-even clip(round(x/stored_scale))",
    "scale": "FP16(max(abs(group))/127); underflowed groups become zero",
    "group_size": 32,
    "k_padding": 128,
    "small_float_policy": "Preserve BF16 a/b projections, convolution and decoder/QK norms; preserve or exactly widen BF16/F32 gated norm, A_log and dt_bias to FP32",
}


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 2**20), b""):
            h.update(data)
    return h.hexdigest()


def _stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _source_evidence(checkpoint, *, manifest, allow_unverified):
    root = checkpoint.directory
    required = {p.name for p in checkpoint.files} | {"config.json"}
    if (root / "model.safetensors.index.json").exists():
        required.add("model.safetensors.index.json")
    record = _read_json(manifest) if manifest.exists() else None
    if record is None and not allow_unverified:
        raise CheckpointError(
            "Verified venturi-source.json required; unverified local sources need explicit opt-in"
        )
    expected = {}
    if record is not None:
        if record.get("status") != "complete":
            raise CheckpointError(
                "Source acquisition is not complete/verified; never export a partial source"
            )
        files = record.get("files")
        if not isinstance(files, list):
            raise CheckpointError("Source verification file records required")
        for f in files:
            if not isinstance(f, dict) or not isinstance(f.get("name"), str):
                raise CheckpointError("Malformed verified source file record")
            name = f["name"]
            if (
                name in expected
                or name in ("", ".", "..")
                or Path(name).name != name
                or "\\" in name
            ):
                raise CheckpointError(
                    "Source verification paths must be unique direct filenames"
                )
            expected[name] = f
        if required - expected.keys():
            raise CheckpointError(
                "Source verification does not cover every checkpoint file/config/index"
            )
    names = sorted(expected if record is not None else required)
    stamps, hashes = {}, {}
    for name in names:
        path = root / name
        before = _stamp(path)
        f = expected.get(name)
        if f and (type(f.get("bytes")) is not int or before[2] != f["bytes"]):
            raise CheckpointError(f"Source file size changed: {name}")
        digest = _digest(path)
        if _stamp(path) != before:
            raise CheckpointError(f"Source file changed during verification: {name}")
        if f and (
            f.get("actual_sha256") != digest
            or f.get("lfs_sha256") not in (None, digest)
        ):
            raise CheckpointError(f"Source file hash mismatch: {name}")
        stamps[name], hashes[name] = before, digest
    if _read_json(root / "config.json") != dict(checkpoint.config):
        raise CheckpointError("Source config changed since inspection")
    return (
        dict(
            kind=(
                "verified-acquisition"
                if record is not None
                else "explicitly-unverified-local-floating-source"
            ),
            repo=record.get("repo") if record else None,
            revision=record.get("revision") if record else None,
            endpoint=record.get("source_endpoint") if record else None,
            files_sha256=hashes,
        ),
        stamps,
    )


def _export_qwen_q8(
    source,
    output,
    *,
    rows_per_chunk=256,
    source_manifest=None,
    allow_unverified_source=False,
    mtp=False,
):
    """Publish one complete new artifact atomically; existing outputs are never replaced.

    Full source metadata/provenance is preflighted before conversion. Only one
    bounded source row chunk and its quantized planes are materialized at a time;
    all output weights are streamed to disk, not accumulated in CPU/GPU memory.
    The original source and any existing mixed-Q4 artifact remain untouched.
    """
    if type(rows_per_chunk) is not int or not 0 < rows_per_chunk <= 4096:
        raise ValueError("rows_per_chunk must be an integer in 1..4096")
    if type(allow_unverified_source) is not bool:
        raise ValueError("allow_unverified_source must be boolean")
    started = perf_counter()
    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to replace an existing output")
    if not target.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    checkpoint = inspect_checkpoint(source)
    if mtp:
        from venturi.models.qwen.qwen_mtp import QwenMTPPlan

        plan = QwenMTPPlan.from_config(dict(checkpoint.config))
    else:
        plan = QwenSourcePlan.from_config(dict(checkpoint.config))
    ignored = plan.validate_checkpoint(checkpoint)
    for spec in plan.bindings:
        if spec.format != "q8_g32_fp16" and spec.nbytes > 8 * 2**20:
            raise ValueError("Floating parameter exceeds the explicit 8 MiB read bound")
    manifest_path = (
        Path(source_manifest).expanduser()
        if source_manifest
        else checkpoint.directory / "venturi-source.json"
    )
    evidence, stamps = _source_evidence(
        checkpoint, manifest=manifest_path, allow_unverified=allow_unverified_source
    )
    tensors = {}
    for spec in plan.bindings:
        if spec.format == "q8_g32_fp16":
            n, k = spec.shape
            padded = (k + 127) // 128 * 128
            tensors[spec.source + ".codes"] = ("I8", (n, padded))
            tensors[spec.source + ".scales"] = ("F16", (n, padded // 32))
        else:
            tensors[spec.source] = (
                {"bf16": "BF16", "fp32": "F32"}[spec.format],
                spec.shape,
            )
    component = "mtp" if mtp else "text"
    metadata = {
        "venturi.config": json.dumps(dict(checkpoint.config), sort_keys=True),
        "venturi.component": component,
        "venturi.matrix_format": "q8_g32_fp16",
        "venturi.source": json.dumps(evidence, sort_keys=True),
    }

    def raw(value):
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("Export requires contiguous CPU tensors")
        size = value.numel() * value.element_size()
        return memoryview((ctypes.c_ubyte * size).from_address(value.data_ptr())).cast(
            "B"
        )

    with SafetensorsWriter(target, tensors, metadata) as writer:
        for spec in plan.bindings:
            with safe_open(
                str(checkpoint.tensors[spec.source].file), framework="pt", device="cpu"
            ) as reader:
                sliced = reader.get_slice(spec.source)
                if (
                    tuple(sliced.get_shape()) != spec.source_shape
                    or sliced.get_dtype() not in spec.source_dtypes
                ):
                    raise CheckpointError("Source header changed: " + spec.source)
                if spec.format == "q8_g32_fp16":
                    n, k = spec.shape
                    padded = (k + 127) // 128 * 128
                    for begin in range(0, n, rows_per_chunk):
                        end = min(n, begin + rows_per_chunk)
                        dense = sliced[begin:end].contiguous()
                        weight = Q8RowSplitWeight.from_dense(
                            dense, rows_per_chunk=rows_per_chunk
                        )
                        writer.write_slice(
                            spec.source + ".codes", raw(weight.codes), begin * padded
                        )
                        writer.write_slice(
                            spec.source + ".scales",
                            raw(weight.scales),
                            begin * (padded // 32) * 2,
                        )
                else:
                    value = reader.get_tensor(spec.source)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite source parameter: " + spec.source)
                    if spec.transform == "conv-kc":
                        value = value.squeeze(1).transpose(0, 1).contiguous()
                    value = value.to(
                        torch.float32 if spec.format == "fp32" else torch.bfloat16
                    ).contiguous()
                    writer.write_slice(spec.source, raw(value))
        writer.finish()
        for name, stamp in stamps.items():
            if _stamp(checkpoint.directory / name) != stamp:
                raise CheckpointError("Source file changed during export: " + name)
        from .qwen_weights import QwenWeights
        from .qwen_mtp import QwenMTPWeights

        admitted = (QwenMTPWeights if mtp else QwenWeights).open(writer.temporary)
        digest = _digest(writer.temporary)
        writer.publish()
    return dict(
        output=str(target),
        sha256=digest,
        file_bytes=target.stat().st_size,
        tensor_count=len(tensors),
        role_count=len(plan.bindings),
        tensor_payload_bytes=admitted.source.total_payload_bytes(),
        ignored_source_tensors=len(ignored),
        source=evidence,
        rows_per_chunk=rows_per_chunk,
        seconds=perf_counter() - started,
        component=component,
        scope="Standard safetensors storage export; not trained-model quality qualification",
    )


def export_qwen_q8(
    source,
    output,
    *,
    rows_per_chunk=256,
    source_manifest=None,
    allow_unverified_source=False,
):
    """Export Qwen text tensors and Q8 planes directly to safetensors."""
    return _export_qwen_q8(
        source,
        output,
        rows_per_chunk=rows_per_chunk,
        source_manifest=source_manifest,
        allow_unverified_source=allow_unverified_source,
    )


def export_qwen_mtp_q8(
    source,
    output,
    *,
    rows_per_chunk=256,
    source_manifest=None,
    allow_unverified_source=False,
):
    """Export only the original 15 MTP tensors; vocabulary remains externally shared."""
    return _export_qwen_q8(
        source,
        output,
        rows_per_chunk=rows_per_chunk,
        source_manifest=source_manifest,
        allow_unverified_source=allow_unverified_source,
        mtp=True,
    )
