"""Explicit Qwen runtime build, linked to one concrete independently built ops library."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from venturi.ops.build import _compile_library


def build_qwen_runtime(output_dir, *, operators, nvcc=None):
    if sys.platform != "linux":
        raise RuntimeError("Build the Qwen runtime on the remote Linux CUDA host")
    operators = Path(operators).expanduser().resolve(strict=True)
    record = json.loads(operators.with_name("build.json").read_text())
    digest = hashlib.sha256(operators.read_bytes()).hexdigest()
    if record["library_sha256"] != digest or not record["soname"].startswith(
        "libventuri_ops_"
    ):
        raise ValueError("An intact explicit ops build is required")
    architecture = record["architecture"]
    if architecture not in ("sm70", "sm89"):
        raise ValueError("Unsupported Qwen target architecture")
    compiler = nvcc or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("An explicit CUDA compiler is required")
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    ).stdout
    root = Path(__file__).resolve().parents[2]
    entry = Path(__file__).parent / "csrc/qwen.cu"
    # Link by concrete filename; its unique SONAME prevents cross-build binding.
    flags = [
        "-O3",
        "-std=c++20",
        "-lineinfo",
        "-shared",
        "--cudart=static",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-fvisibility=hidden",
        "-Xlinker=-z",
        "-Xlinker=defs",
        f"-DVENTURI_BLOCKS_ARCH={architecture[2:]}",
        f"-gencode=arch=compute_{architecture[2:]},code=sm_{architecture[2:]}",
        f"-gencode=arch=compute_{architecture[2:]},code=compute_{architecture[2:]}",
        str(operators),
        "-lcuda",
        "-lcublas",
        "-Xlinker=-rpath",
        "-Xlinker=$ORIGIN",
    ]
    library = _compile_library(
        output_dir,
        "qwen",
        entry,
        root,
        compiler,
        version,
        flags,
        architecture=architecture,
        dependencies={"ops_sha256": digest},
    )
    # The resulting pair is relocatable; never depend on a global library search path.
    dependency = library.parent / record["soname"]
    if (
        not dependency.exists()
        or hashlib.sha256(dependency.read_bytes()).hexdigest() != digest
    ):
        with tempfile.TemporaryDirectory(
            prefix=".ops-pair-", dir=library.parent
        ) as staging:
            temporary = Path(staging) / record["soname"]
            shutil.copyfile(operators, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Operator library changed during pairing")
            temporary.replace(dependency)
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ops-library", required=True)
    parser.add_argument("--nvcc")
    args = parser.parse_args()
    print(
        build_qwen_runtime(args.output_dir, operators=args.ops_library, nvcc=args.nvcc)
    )


if __name__ == "__main__":
    main()
