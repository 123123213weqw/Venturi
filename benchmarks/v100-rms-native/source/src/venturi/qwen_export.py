"""CPU-only, bounded-row original HF -> separately identified Q8 text artifact.

No GPU allocation, remote code, Q4 re-quantization or model forward. The v3
container is reused as a storage format, not as an external execution engine.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from time import perf_counter
import uuid

import torch
from safetensors import safe_open

from .checkpoint import CheckpointError, _read_json, inspect_checkpoint
from .models.qwen_source import QwenSourcePlan
from .ninfer import MAGIC, NInferArtifact, _align, row_geometry
from .ops.weights import Q8RowSplitWeight


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
    return dict(
        kind="verified-acquisition"
        if record is not None
        else "explicitly-unverified-local-floating-source",
        repo=record.get("repo") if record else None,
        revision=record.get("revision") if record else None,
        endpoint=record.get("source_endpoint") if record else None,
        files_sha256=hashes,
    ), stamps


def _write_tensor(stream, tensor):
    """Write a contiguous CPU tensor's bytes without a NumPy/full-size copy."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Export writes require contiguous CPU storage")
    size = tensor.numel() * tensor.element_size()
    view = memoryview((ctypes.c_ubyte * size).from_address(tensor.data_ptr())).cast("B")
    # FileIO normally writes all bytes, but tolerate legitimate short writes.
    while view:
        count = stream.write(view)
        if not count:
            raise OSError("Short artifact payload write")
        view = view[count:]


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
        from .models.qwen_mtp import QwenMTPPlan

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
    objects, bindings, offset = [], {}, 0
    for spec in plan.bindings:
        offset = _align(offset)
        objects.append(
            dict(
                id=spec.source,
                kind="tensor",
                shape=list(spec.shape),
                format=spec.format,
                layout="row_split_k128_v1"
                if spec.format == "q8_g32_fp16"
                else "contiguous_le_v1",
                offset=offset,
                bytes=spec.nbytes,
            )
        )
        bindings[spec.binding] = {"object": spec.source}
        offset += spec.nbytes
    if plan.tied_embeddings:
        bindings["text/output_head"] = dict(bindings["text/token_embedding"])
    directory = dict(
        components={
            "text": {"config": plan.config},
            **({"mtp": {"config": plan.config, "target": "text"}} if mtp else {}),
        },
        objects=objects,
        bindings=bindings,
        uses=[],
        files=[{"path": None, "payload_bytes": offset}],
        provenance=dict(
            converter="venturi-qwen-mtp-q8-v1" if mtp else "venturi-qwen-q8-v1",
            source=evidence,
            quantization=QUANTIZATION,
            excluded_tensors=list(ignored),
            excluded_components=["vision", "target_text_payload"]
            if mtp
            else ["vision", "mtp"],
            transforms={
                s.binding: s.transform
                for s in plan.bindings
                if s.transform != "identity"
            },
            norm_policy="Raw zero-centered parameters; runtime must apply 1+weight except GDN gated norm",
            q_layout="Per-head [query D | gate D]; never all-query followed by all-gate",
            source_bindings={s.binding: s.source for s in plan.bindings},
        ),
    )
    encoded = json.dumps(
        directory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    payload_start = _align(32 + len(encoded), 4096)
    total = payload_start + offset
    if shutil.disk_usage(target.parent).free < total + 256 * 2**20:
        raise OSError(
            "Insufficient disk space for a separate complete Q8 artifact and reserve"
        )
    artifact_id = uuid.uuid4().bytes
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".venturi-q8-", suffix=".partial", dir=target.parent, delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.truncate(total)  # New sparse zero-filled padding, never old bytes.
            stream.write(
                MAGIC + struct.pack("<Q", len(encoded)) + artifact_id + encoded
            )
            for spec, obj in zip(plan.bindings, objects):
                source_file = checkpoint.tensors[spec.source].file
                with safe_open(
                    str(source_file), framework="pt", device="cpu"
                ) as reader:
                    sliced = reader.get_slice(spec.source)
                    if (
                        tuple(sliced.get_shape()) != spec.source_shape
                        or sliced.get_dtype() not in spec.source_dtypes
                    ):
                        raise CheckpointError(f"Source header changed: {spec.source}")
                    base = payload_start + obj["offset"]
                    if spec.format == "q8_g32_fp16":
                        n, k = spec.shape
                        padded = (k + 127) // 128 * 128
                        scales_offset = row_geometry(spec.shape, spec.format)[4]
                        for begin in range(0, n, rows_per_chunk):
                            end = min(n, begin + rows_per_chunk)
                            dense = sliced[begin:end].contiguous()
                            weight = Q8RowSplitWeight.from_dense(
                                dense, rows_per_chunk=rows_per_chunk
                            )
                            stream.seek(base + begin * padded)
                            _write_tensor(stream, weight.codes)
                            stream.seek(
                                base + scales_offset + begin * (padded // 32) * 2
                            )
                            _write_tensor(stream, weight.scales)
                            del dense, weight
                    else:
                        value = reader.get_tensor(spec.source)
                        if not bool(torch.isfinite(value).all()):
                            raise ValueError(
                                f"Nonfinite source parameter: {spec.source}"
                            )
                        if spec.transform == "conv-kc":
                            value = value.squeeze(1).transpose(0, 1).contiguous()
                        value = value.to(
                            torch.float32 if spec.format == "fp32" else torch.bfloat16
                        ).contiguous()
                        stream.seek(base)
                        _write_tensor(stream, value)
                        del value
            stream.flush()
            os.fsync(stream.fileno())
        for name, stamp in stamps.items():
            if _stamp(checkpoint.directory / name) != stamp:
                raise CheckpointError(f"Source file changed during export: {name}")
        artifact = NInferArtifact(temp)
        if len(artifact.objects) != len(plan.bindings):
            raise RuntimeError("Export artifact directory validation failed")
        digest = _digest(temp)
        os.link(temp, target)  # Atomic no-clobber publication on the same filesystem.
        return dict(
            output=str(target),
            artifact_id=artifact_id.hex(),
            sha256=digest,
            file_bytes=total,
            tensor_payload_bytes=sum(s.nbytes for s in plan.bindings),
            tensor_count=len(plan.bindings),
            ignored_source_tensors=len(ignored),
            source=evidence,
            rows_per_chunk=rows_per_chunk,
            seconds=perf_counter() - started,
            scope="Storage conversion only; not full trained-model quality or native generation qualification",
        )
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def export_qwen_q8(
    source,
    output,
    *,
    rows_per_chunk=256,
    source_manifest=None,
    allow_unverified_source=False,
):
    """Export the target text payload without changing its existing format."""
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
