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
        [source / "blocks.cu", source / "qwen_runtime.cuh"]
        if component == "blocks"
        else [
            p
            for p in source.rglob("*")
            if p.is_file() and p.name not in ("blocks.cu", "qwen_runtime.cuh")
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
) -> Path:
    """Build Venturi-owned blocks and cuBLAS bridge, without an external engine."""
    return _build_backend(
        output_dir,
        nvcc=nvcc,
        component="blocks",
        architecture=architecture,
        q8_token_tile=q8_token_tile,
        q8_row_tile=q8_row_tile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--component", choices=("ada", "blocks"), default="ada")
    parser.add_argument("--architecture", choices=("sm70", "sm89"), default="sm89")
    parser.add_argument("--q8-token-tile", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--q8-row-tile", type=int, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    if args.component == "ada" and args.architecture != "sm89":
        parser.error("Ada Q4 supports sm89 only")
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
                )
                if args.component == "blocks"
                else {}
            ),
        )
    )


if __name__ == "__main__":
    main()
