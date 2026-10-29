"""Explicit REMOTE-only CUDA build; no Rust, Torch JIT or import-time build."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Build only on an authorized remote Linux GPU host')
    output = Path(a.output).resolve()
    if output.exists():
        raise ValueError('Refuse to overwrite prior build')
    source = Path(__file__).resolve().parents[1]/'src/venturi/ops/csrc/flash_next_q4.cu'
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    toolkit = Path(nvcc).resolve().parent.parent
    command = [nvcc,'-O3','-Xptxas=-v','--shared','-Xcompiler=-fPIC','--ftz=false','--fmad=false',
               '-gencode=arch=compute_70,code=sm_70','-Xlinker=-rpath','-Xlinker='+str(toolkit/'lib64'),str(source),'-o',str(output)]
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,check=True)
    print(json.dumps(dict(command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         nvcc=subprocess.check_output([nvcc,'--version'],text=True)),indent=2))


if __name__ == '__main__':
    main()
