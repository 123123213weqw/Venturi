"""Real pool/manager C++ control flow on POSIX shared maps; NEVER CUDA math.

These fast lifecycle regressions complement, not replace, GPU mechanism and
native-Qwen numerical tests. Streams/copies use synchronous host-only stubs.
"""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def host_pool(tmp_path_factory):
    compiler=shutil.which("c++")
    if not compiler:pytest.skip("Host C++ compiler required for lifecycle controls")
    output=tmp_path_factory.mktemp("kv-host")
    text=(ROOT/'src/venturi/models/qwen/csrc/qwen_runtime.cuh').read_text()
    def method(marker):
        start=text.index(marker);index=text.index('{',start)+1;depth=1;end=index
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        return text[start:end]
    (output/'session_methods.inc').write_text('\n'.join(method(m)for m in (
        '    void attach_prefix(std::uint64_t key,int tokens)',
        '    void publish_prefix(std::uint64_t key,int tokens)')))
    exe=output/'kv-host'
    subprocess.run([compiler,'-std=c++17','-O0','-I'+str(ROOT/'tests/kv_host'),
        '-I'+str(ROOT/'src'),'-I'+str(output),str(ROOT/'tests/kv_host/regressions.cpp'),'-o',str(exe)],check=True,timeout=90)
    return exe


@pytest.mark.parametrize('case',['reset','private_reset','entry_aba','leases','capacity64','cow_tail','cow_pressure',
    'rejection_ownership','snapshot_failure','validation','session_boundaries','donor_append'])
def test_unified_kv_lifecycle_host_controls(host_pool,case):
    subprocess.run([str(host_pool),case],check=True,timeout=15)
