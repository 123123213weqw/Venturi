"""Explicit REMOTE-only standalone SM70 CUDA operator build; no Rust/Torch JIT."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['linear','tiled'], default='linear')
    a = p.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Compile this CUDA operator only on an authorized remote Linux GPU host')
    output = Path(a.output).resolve()
    if output.exists():
        raise RuntimeError('Refuse to overwrite a compiled experiment library')
    source = Path(__file__).resolve().parents[1]/'src/venturi/ops/csrc'/('flash_next_fp8_decode_tiled.cu' if a.variant == 'tiled' else 'flash_next_fp8_decode.cu')
    nvcc = shutil.which('nvcc')
    if nvcc is None and Path('/usr/local/cuda/bin/nvcc').is_file():
        nvcc = '/usr/local/cuda/bin/nvcc'
    if nvcc is None:
        raise RuntimeError('Remote nvcc not found; no local fallback')
    toolkit = Path(nvcc).resolve().parent.parent
    command = [nvcc, '-Xptxas=-v', '-O3', '--shared', '-Xcompiler=-fPIC', '--ftz=false', '--fmad=false',
        '-gencode=arch=compute_70,code=sm_70', '-Xlinker=-rpath', '-Xlinker='+str(toolkit/'lib64'),
        str(source), '-o', str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    print(json.dumps(dict(command=command, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        nvcc=subprocess.check_output([nvcc, '--version'], text=True)), indent=2))


if __name__ == '__main__':
    main()
