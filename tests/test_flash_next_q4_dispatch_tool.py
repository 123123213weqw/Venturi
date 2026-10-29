"""Read-only CPU CLI contract, no GPU or performance claim."""
import os
from pathlib import Path
import subprocess
import sys

def test_dispatch_benchmark_help_no_gpu_or_outputs(tmp_path):
 root=Path(__file__).resolve().parents[1]
 p=subprocess.run([sys.executable,str(root/'tools/benchmark_flash_next_q4_dispatch.py'),'--help'],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=45)
 assert p.returncode==0,p.stderr
 assert '--dispatch' in p.stdout and '{canonical,coalesced,graph}' in p.stdout
 assert '{cohort,profile}' in p.stdout
 assert not list(tmp_path.iterdir())
