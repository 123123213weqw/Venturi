"""Explicit Linux CUDA build; never invoked implicitly by execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def _build_backend(
    output_dir: str | Path,
    *,
    nvcc: str | None,
    component: str,
    architecture="sm89",
    q8_token_tile=None,
    q8_row_tile=None,
    q8_decode_mode=0,
    q8_row_pair=False,
    q8_swiglu_quant=False,
    q8_rms_quant=False,
) -> Path:
    """Build the optional C ABI library and retain compiler output/provenance."""
    import sys

    if architecture not in ("sm70", "sm89") or (
        component == "ada" and architecture != "sm89"
    ):
        raise ValueError("Blocks support explicit sm70/sm89; Ada Q4 supports sm89 only")
    # Only SM70 has a qualified tiled full-model performance profile. Preserve
    # the original SM89 dispatch until it has its own matched qualification.
    if q8_token_tile is None:
        q8_token_tile = 4 if architecture == "sm70" and component == "blocks" else 1
    if q8_row_tile is None:
        q8_row_tile = 4 if architecture == "sm70" and component == "blocks" else 1
    if any(
        type(t) is not int or t not in (1, 2, 4, 8)
        for t in (q8_token_tile, q8_row_tile)
    ):
        raise ValueError("Q8 token/row tiles must be explicit 1, 2, 4 or 8")
    if type(q8_decode_mode) is not int or q8_decode_mode not in (0, 1, 2):
        raise ValueError("Q8 decode mode must be explicit 0, 1 or 2")
    if type(q8_row_pair) is not bool or (
        q8_row_pair and (component != "blocks" or architecture != "sm70")
    ):
        raise ValueError("Q8 row-pair is an explicit boolean SM70 blocks experiment")
    if type(q8_swiglu_quant) is not bool or (
        q8_swiglu_quant and (component != "blocks" or architecture != "sm70")
    ):
        raise ValueError("Q8 SwiGLU quantization is an explicit boolean SM70 blocks experiment")
    if type(q8_rms_quant) is not bool or (
        q8_rms_quant and (component != "blocks" or architecture != "sm70")
    ):
        raise ValueError("Q8 RMS quantization is an explicit boolean SM70 blocks experiment")
    if sys.platform != "linux":
        raise RuntimeError(
            "Build Venturi CUDA operators on a Linux CUDA machine, not the local Mac"
        )
    compiler = nvcc or shutil.which("nvcc")
    if not compiler:
        home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        compiler = str(home / "bin/nvcc")
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    ).stdout
    source = Path(__file__).parent / "csrc"
    selected = (
        [
            source / "blocks.cu",
            source / "qwen_runtime.cuh",
            source / "qwen_head_gdn.cuh",
            source / "qwen_pipeline.cuh",
            source / "qwen_mtp.cuh",
            source / "qwen_batch.cuh",
            source / "q8_dp4a.cuh",
            source / "q8_column_tp.cuh",
            source / "q8_wmma.cuh",
            source / "q8_shared.cuh",
            source / "q8_fused.cuh",
            source / "q8_prefetch.cuh",
            source / "q8_projection_dispatch.cuh",
            source / "q8_producer.cuh",
            source / "q8_rms_producer.cuh",
        ]
        if component == "blocks"
        else [
            p
            for p in source.rglob("*")
            if p.is_file()
            and p.name
            not in (
                "blocks.cu",
                "qwen_runtime.cuh",
                "qwen_head_gdn.cuh",
                "qwen_pipeline.cuh",
                "qwen_mtp.cuh",
                "qwen_batch.cuh",
                "q8_dp4a.cuh",
                "q8_column_tp.cuh",
                "q8_wmma.cuh",
                "q8_shared.cuh",
                "q8_fused.cuh",
                "q8_prefetch.cuh",
                "q8_projection_dispatch.cuh",
                "q8_producer.cuh",
                "q8_rms_producer.cuh",
            )
        ]
    )
    files = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(selected)
        if p.is_file()
    }
    flags = [
        "-O3",
        "-std=c++20",
        "-lineinfo",
        "-shared",
        "--cudart=static",
        "-Xcompiler=-fPIC",
        f"-gencode=arch=compute_{architecture[2:]},code=sm_{architecture[2:]}",
        f"-gencode=arch=compute_{architecture[2:]},code=compute_{architecture[2:]}",
        *(
            [
                "-DVENTURI_BLOCKS_ARCH=" + architecture[2:],
                f"-DVENTURI_Q8_TOKEN_TILE={q8_token_tile}",
                f"-DVENTURI_Q8_ROW_TILE={q8_row_tile}",
                f"-DVENTURI_Q8_DECODE_MODE={q8_decode_mode}",
                f"-DVENTURI_Q8_ROW_PAIR={int(q8_row_pair)}",
                f"-DVENTURI_Q8_SWIGLU_QUANT={int(q8_swiglu_quant)}",
                f"-DVENTURI_Q8_RMS_QUANT={int(q8_rms_quant)}",
            ]
            if component == "blocks"
            else []
        ),
        *(["-I" + str(source / "third_party/ninfer")] if component == "ada" else []),
    ]
    key = hashlib.sha256(
        json.dumps([component, files, version, flags], sort_keys=True).encode()
    ).hexdigest()[:16]
    output = Path(output_dir).expanduser().resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    library = output / f"libventuri_{component}.so"
    import fcntl

    with (output / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = output / "build.json"
        if library.exists() and manifest_path.exists():
            try:
                record = json.loads(manifest_path.read_text())
                valid = (
                    record["source_sha256"] == files
                    and (
                        component != "blocks"
                        or record.get("architecture") == architecture
                    )
                    and record["library_sha256"]
                    == hashlib.sha256(library.read_bytes()).hexdigest()
                )
                if valid:
                    return library
            except (OSError, ValueError, KeyError, TypeError):
                pass  # A missing/corrupted cache is rebuilt, not silently loaded.
        # Serialize same-key builds and never publish a partially written library.
        with tempfile.TemporaryDirectory(prefix="build-", dir=output) as staging:
            target = Path(staging) / library.name
            command = [
                compiler,
                *flags,
                str(
                    source / ("blocks.cu" if component == "blocks" else "q4_linear.cu")
                ),
                *(["-lcublas"] if component == "blocks" else []),
                "-o",
                str(target),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            (output / "build.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"CUDA compilation failed; see {output / 'build.log'}"
                )
            record = {
                "architecture": architecture,
                "source_sha256": files,
                "nvcc": version,
                "command": command,
                "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            manifest = Path(staging) / "build.json"
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(target, library)
            os.replace(manifest, output / "build.json")
    return library


def build_cuda_backend(output_dir: str | Path, *, nvcc: str | None = None) -> Path:
    """Build the existing attributed Q4 operator library explicitly."""
    return _build_backend(output_dir, nvcc=nvcc, component="ada")


def build_blocks_backend(
    output_dir: str | Path,
    *,
    nvcc: str | None = None,
    architecture="sm89",
    q8_token_tile=None,
    q8_row_tile=None,
    q8_decode_mode=0,
    q8_row_pair=False,
    q8_swiglu_quant=False,
    q8_rms_quant=False,
) -> Path:
    """Build Venturi-owned blocks and cuBLAS bridge, without an external engine."""
    return _build_backend(
        output_dir,
        nvcc=nvcc,
        component="blocks",
        architecture=architecture,
        q8_token_tile=q8_token_tile,
        q8_row_tile=q8_row_tile,
        q8_decode_mode=q8_decode_mode,
        q8_row_pair=q8_row_pair,
        q8_swiglu_quant=q8_swiglu_quant,
        q8_rms_quant=q8_rms_quant,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--component", choices=("ada", "blocks"), default="ada")
    parser.add_argument("--architecture", choices=("sm70", "sm89"), default="sm89")
    parser.add_argument("--q8-token-tile", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--q8-row-tile", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--q8-decode-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--q8-row-pair", action="store_true", help="Experimental SM70 native projection dispatch; default off")
    parser.add_argument("--q8-swiglu-quant", action="store_true", help="Experimental SM70 native SwiGLU/two-pass activation quantization fusion; default off")
    parser.add_argument("--q8-rms-quant", action="store_true", help="Experimental SM70 native post-attention RMS/two-pass activation quantization fusion; default off")
    args = parser.parse_args()
    if args.component == "ada" and args.architecture != "sm89":
        parser.error("Ada Q4 supports sm89 only")
    if args.q8_row_pair and (args.component != "blocks" or args.architecture != "sm70"):
        parser.error("--q8-row-pair requires --component blocks --architecture sm70")
    if args.q8_swiglu_quant and (args.component != "blocks" or args.architecture != "sm70"):
        parser.error("--q8-swiglu-quant requires --component blocks --architecture sm70")
    if args.q8_rms_quant and (args.component != "blocks" or args.architecture != "sm70"):
        parser.error("--q8-rms-quant requires --component blocks --architecture sm70")
    build = build_blocks_backend if args.component == "blocks" else build_cuda_backend
    print(
        build(
            args.output_dir,
            nvcc=args.nvcc,
            **(
                dict(
                    architecture=args.architecture,
                    q8_token_tile=args.q8_token_tile,
                    q8_row_tile=args.q8_row_tile,
                    q8_decode_mode=args.q8_decode_mode,
                    q8_row_pair=args.q8_row_pair,
                    q8_swiglu_quant=args.q8_swiglu_quant,
                    q8_rms_quant=args.q8_rms_quant,
                )
                if args.component == "blocks"
                else {}
            ),
        )
    )


if __name__ == "__main__":
    main()
