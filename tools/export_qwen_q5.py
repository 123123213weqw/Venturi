"""Explicit CPU-only original Qwen HF source to a separate Q5 text artifact."""
import argparse
import json
from pathlib import Path

from venturi.models.qwen.qwen_export import _digest, _source_evidence, _stamp
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.container.checkpoint import CheckpointError, inspect_checkpoint
from venturi.container.safetensors import SafetensorsWriter
from venturi.ops.weights import Q5RowSplitWeight

import ctypes

import torch
from safetensors import safe_open


def export_qwen_q5(
    source,
    output,
    *,
    rows_per_chunk=256,
    source_manifest=None,
    allow_unverified_source=False,
):
    """Publish one complete new q5_g64_fp16 artifact atomically.

    Mirrors export_qwen_q8: full source provenance preflight, one bounded row
    chunk materialized at a time, streamed planes (codes/high/scales), never
    replacing an existing output. Storage export only, not a quality claim.
    """
    if type(rows_per_chunk) is not int or not 0 < rows_per_chunk <= 4096:
        raise ValueError("rows_per_chunk must be an integer in 1..4096")
    if type(allow_unverified_source) is not bool:
        raise ValueError("allow_unverified_source must be boolean")
    from time import perf_counter

    started = perf_counter()
    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to replace an existing output")
    if not target.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    checkpoint = inspect_checkpoint(source)
    plan = QwenSourcePlan.from_config(dict(checkpoint.config), matrix_format="q5_g64_fp16")
    ignored = plan.validate_checkpoint(checkpoint, matrix_format="q5_g64_fp16")
    for spec in plan.bindings:
        if spec.format != "q5_g64_fp16" and spec.nbytes > 8 * 2**20:
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
        if spec.format == "q5_g64_fp16":
            n, k = spec.shape
            padded = (k + 127) // 128 * 128
            tensors[spec.source + ".codes"] = ("U8", (n, padded // 2))
            tensors[spec.source + ".high"] = ("U8", (n, padded // 8))
            tensors[spec.source + ".scales"] = ("F16", (n, padded // 64))
        else:
            tensors[spec.source] = (
                {"bf16": "BF16", "fp32": "F32"}[spec.format],
                spec.shape,
            )
    metadata = {
        "venturi.config": json.dumps(dict(checkpoint.config), sort_keys=True),
        "venturi.component": "text",
        "venturi.matrix_format": "q5_g64_fp16",
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
                if spec.format == "q5_g64_fp16":
                    n, k = spec.shape
                    padded = (k + 127) // 128 * 128
                    for begin in range(0, n, rows_per_chunk):
                        end = min(n, begin + rows_per_chunk)
                        dense = sliced[begin:end].contiguous()
                        weight = Q5RowSplitWeight.from_dense(
                            dense, rows_per_chunk=rows_per_chunk
                        )
                        writer.write_slice(
                            spec.source + ".codes", raw(weight.codes), begin * (padded // 2)
                        )
                        writer.write_slice(
                            spec.source + ".high", raw(weight.high), begin * (padded // 8)
                        )
                        writer.write_slice(
                            spec.source + ".scales",
                            raw(weight.scales),
                            begin * (padded // 64) * 2,
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
        from venturi.models.qwen.qwen_weights import QwenWeights

        admitted = QwenWeights.open(writer.temporary)
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
        component="text",
        scope="q5_g64_fp16 storage export; not trained-model quality qualification",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--source-manifest")
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument("--allow-unverified-source", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    report = export_qwen_q5(
        args.source,
        args.output,
        rows_per_chunk=args.rows_per_chunk,
        source_manifest=args.source_manifest,
        allow_unverified_source=args.allow_unverified_source,
    )
    if args.report:
        Path(args.report).expanduser().absolute().write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:800])


if __name__ == "__main__":
    main()
