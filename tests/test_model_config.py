from dataclasses import replace
import hashlib
import os
import subprocess
import sys

import pytest

from venturi.model_assets import ModelAsset
from venturi.model_config import read_verified_json_asset


def asset(root,raw=b'{"a":1}'):
    (root/'config.json').write_bytes(raw);rev='a'*40
    return ModelAsset('config.json',len(raw),hashlib.sha256(raw).hexdigest(),
        f'https://huggingface.co/o/m/resolve/{rev}/config.json','o/m',rev,'config.json','metadata')


def test_explicit_size_sha_and_private_json_object(tmp_path):
    a=asset(tmp_path,b'{"value":1e-20,"nested":{"text":"hello"}}')
    value=read_verified_json_asset(a,tmp_path)
    assert value['value']==1e-20
    value['nested']['text']='changed'
    assert read_verified_json_asset(a,tmp_path)['nested']['text']=='hello'
    with pytest.raises(ValueError,match='SHA256'):read_verified_json_asset(replace(a,sha256='0'*64),tmp_path)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',
    b'{"x":-Infinity}',b'{"x":1e9999}',b'[]',b'null',b'{"x":"\xff"}',b'not JSON'])
def test_full_sha_does_not_accept_ambiguous_or_invalid_json(tmp_path,raw):
    a=asset(tmp_path,raw)
    with pytest.raises(ValueError):read_verified_json_asset(a,tmp_path)


@pytest.mark.parametrize('budget',[0,-1,True,65*2**20,1])
def test_json_budget_validated_before_read(tmp_path,budget):
    a=asset(tmp_path)
    with pytest.raises(ValueError):read_verified_json_asset(a,tmp_path,max_bytes=budget)


def test_symlink_fifo_and_partial_are_refused(tmp_path):
    a=asset(tmp_path);p=tmp_path/a.filename;p.unlink();p.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError,match='Symlink'):read_verified_json_asset(a,tmp_path)
    p.unlink();os.mkfifo(p)
    with pytest.raises(ValueError,match='regular'):read_verified_json_asset(a,tmp_path)
    p.unlink();p.write_bytes(b'1')
    with pytest.raises(ValueError,match='exact size'):read_verified_json_asset(a,tmp_path)


def test_torch_free_json_import():
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.model_config;assert "torch" not in sys.modules'])
