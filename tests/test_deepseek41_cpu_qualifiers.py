"""CLI/admission checks without depending on downloaded research/reference files."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


TOOLS=Path(__file__).parents[1]/'tools'


def load(name):
    spec=importlib.util.spec_from_file_location(name,TOOLS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['qualify_deepseek41_tokenizer','qualify_gguf_iq4nl','qualify_gguf_projection_formats'])
def test_help_does_not_require_target_or_checkpoint(name):
    text=subprocess.check_output([sys.executable,str(TOOLS/(name+'.py')),'--help'],text=True)
    assert '--reference-source' in text and '--output' in text and 'CPU' in text


@pytest.mark.parametrize('name,method', [
    ('qualify_deepseek41_tokenizer','_source_function'),
    ('qualify_gguf_iq4nl','_reference'),
    ('qualify_gguf_projection_formats','references'),
])
def test_wrong_source_hash_is_rejected_before_parse_or_execution(name,method):
    module=load(name)
    with pytest.raises(ValueError,match='SHA256'):
        getattr(module,method)(b'raise AssertionError("must not execute unverified source")')


@pytest.mark.parametrize('name,extra', [
    ('qualify_deepseek41_tokenizer',['--assets-root','missing-assets']),
    ('qualify_gguf_iq4nl',[]),
    ('qualify_gguf_projection_formats',[]),
])
def test_invalid_source_never_creates_success_report(tmp_path,name,extra):
    source=tmp_path/'source.py';source.write_text('raise AssertionError("must not execute")')
    output=tmp_path/'result.json'
    result=subprocess.run([sys.executable,str(TOOLS/(name+'.py')),
        '--reference-source',str(source),'--output',str(output),*extra],capture_output=True,text=True)
    assert result.returncode != 0 and not output.exists()
    assert 'must not execute' not in result.stderr


def test_row_qualifier_help_and_hash_guard(monkeypatch):
    name='qualify_deepseek41_rows'
    text=subprocess.check_output([sys.executable,str(TOOLS/(name+'.py')),'--help'],text=True)
    assert '--hash-reference-source' in text and '--quant-reference-source' in text
    monkeypatch.syspath_prepend(str(TOOLS))
    module=load(name)
    with pytest.raises(ValueError,match='SHA256'):
        module._hash_reference(b'raise AssertionError("must not execute unverified source")')
    with pytest.raises(ValueError,match='SHA256'):
        module._engram_reference(b'raise AssertionError("must not execute unverified source")')
