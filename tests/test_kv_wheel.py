"""Installed native source dependencies must survive wheel packaging (CPU)."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_wheel_includes_unified_KV_headers_and_resolves_native_build_inputs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    project=tmp_path/'package';project.mkdir()
    for name in('pyproject.toml','README.md'):shutil.copyfile(root/name,project/name)
    shutil.copytree(root/'src',project/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-B','-c',
        'from setuptools.build_meta import build_wheel; build_wheel("dist")'],
        cwd=project,env=env,text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    installed=tmp_path/'installed'
    with zipfile.ZipFile(next((project/'dist').glob('*.whl')))as wheel:
        for name in('kv_pool.cuh','kv_prefix.cuh'):
            assert 'venturi/ops/csrc/kv/'+name in wheel.namelist()
        wheel.extractall(installed)
    # Load the wheel's actual dependency walker, not an editable local package.
    package=installed/'venturi'
    spec=importlib.util.spec_from_file_location('wheel_build',package/'ops/build.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    inputs=build._source_hashes(package/'models/qwen/csrc/qwen.cu',package)
    assert {'ops/csrc/kv/kv_pool.cuh','ops/csrc/kv/kv_prefix.cuh'}<=inputs.keys()
