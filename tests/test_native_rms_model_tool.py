"""CPU-only guards for the full-model qualification/timing evidence harness."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module')
def harness():
    spec=importlib.util.spec_from_file_location('native_rms_model_tool',Path(__file__).parents[1]/'tools/projection/benchmark_native_rms_model.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_state_signature_preserves_bytes_and_ignores_runtime_metadata(harness):
    x=torch.tensor([0.,-0.,1.],dtype=torch.float32)
    a=harness.signature(dict(position=3,gdn=[(x,None)],kv=[],compiled_flag=False))
    b=harness.signature(dict(position=3,gdn=[(x.clone(),None)],kv=[],compiled_flag=True))
    assert a==b and a['position']==3
    changed=x.clone();changed[1]=0.
    assert harness.signature(x)!=harness.signature(changed)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_matching_nonfinite_outputs_must_not_pass_qualification(harness,bad):
    with pytest.raises(AssertionError,match='Nonfinite'):
        harness.signature(dict(gdn=[torch.tensor([bad])],kv=[]))


def test_measured_output_token_denominator_and_cpu_time(harness,monkeypatch):
    wall=iter([10.,12.]);cpu=iter([3.,3.5])
    monkeypatch.setattr(harness,'perf_counter',lambda:next(wall));monkeypatch.setattr(harness,'process_time',lambda:next(cpu));monkeypatch.setattr(harness,'time',lambda:100.)
    value,r=harness.measured(lambda:'actual result',248)
    assert value=='actual result'
    assert r['started_unix']==100. and r['wall_seconds']==2. and r['process_cpu_seconds']==.5
    assert r['process_cpu_cores']==.25 and r['tps']==124. and r['output_tokens']==248
