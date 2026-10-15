"""Explicit Linux CUDA build; never invoked implicitly by execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def build_cuda_backend(output_dir: str | Path, *, nvcc: str | None = None) -> Path:
    """Build the optional C ABI library and retain compiler output/provenance."""
    import sys

    if sys.platform != "linux":
        raise RuntimeError(
            "Build Venturi Ada on a Linux CUDA machine, not the local Mac"
        )
    compiler = nvcc or shutil.which("nvcc")
    if not compiler:
        home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        compiler = str(home / "bin/nvcc")
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    ).stdout
    source = Path(__file__).parent / "csrc"
    files = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    flags = [
        "-O3",
        "-std=c++20",
        "-lineinfo",
        "-shared",
        "--cudart=static",
        "-Xcompiler=-fPIC",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
        "-I" + str(source / "third_party/ninfer"),
    ]
    key = hashlib.sha256(
        json.dumps([files, version, flags], sort_keys=True).encode()
    ).hexdigest()[:16]
    output = Path(output_dir).expanduser().resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    library = output / "libventuri_ada.so"
    import fcntl

    with (output / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = output / "build.json"
        if library.exists() and manifest_path.exists():
            try:
                record = json.loads(manifest_path.read_text())
                valid = (
                    record["source_sha256"] == files
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
                str(source / "q4_linear.cu"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nvcc")
    args = parser.parse_args()
    print(build_cuda_backend(args.output_dir, nvcc=args.nvcc))


if __name__ == "__main__":
    main()
