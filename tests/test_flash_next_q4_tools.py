"""Experiment help is read-only: no checkpoint, compilation or CUDA admission."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('name',['build_flash_next_q4.py','qualify_flash_next_q4.py',
                               'benchmark_flash_next_q4_full.py','benchmark_flash_next_q4_fallback.py'])
def test_help_without_gpu_or_weight_files(name):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'tools'/name),'--help'],env=dict(os.environ,PYTHONPATH=str(root/'src')),
                          capture_output=True,text=True,check=True,timeout=30)
    assert '--output' in result.stdout
