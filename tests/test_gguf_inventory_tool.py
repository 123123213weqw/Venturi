import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from venturi.model_assets import ModelAsset,ModelBundle

TOOL=Path(__file__).parents[1]/'tools/inventory_gguf.py'
spec=importlib.util.spec_from_file_location('gguf_inventory_tool',TOOL)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


@pytest.fixture
def model(tmp_path):
    root=tmp_path/'model';root.mkdir()
    raw=b'GGUF'+struct.pack('<IQQ',3,1,0)+struct.pack('<Q',1)+b't'+struct.pack('<IQIQ',1,8,0,0)
    raw+=bytes(96-len(raw))
    (root/'body.gguf').write_bytes(raw)
    a=ModelAsset('body.gguf',len(raw),hashlib.sha256(raw).hexdigest(),
        f'https://huggingface.co/o/m/resolve/{"b"*40}/body.gguf','o/m','b'*40,'body.gguf','backbone')
    return root,ModelBundle('official/model','c'*40,'synthetic',(a,))


def test_header_is_not_full_hash_or_model_support(model):
    root,bundle=model
    report=mod.inventory_bundle(bundle,root)
    assert not report['all_full_asset_hashes_verified'] and not report['claims_model_support']
    assert report['model_tps'] is None
    assert report['files'][0]['on_disk_bytes_by_dtype']=={'F32':32}
    assert not report['files'][0]['header']['payload_sha256_verified']


def test_full_hash_requires_explicit_scan(model):
    root,bundle=model
    assert mod.inventory_bundle(bundle,root,verify_assets=True)['all_full_asset_hashes_verified']
    # Unchanged header, wrong values/bytes: header-only still cannot decide.
    p=root/'body.gguf';raw=p.read_bytes();p.write_bytes(raw[:-1]+b'1')
    assert not mod.inventory_bundle(bundle,root)['all_full_asset_hashes_verified']
    with pytest.raises(ValueError,match='SHA256'):mod.inventory_bundle(bundle,root,verify_assets=True)


def test_partial_and_missing_are_diagnostic_only(model):
    root,bundle=model;p=root/'body.gguf';part=p.with_name(p.name+'.venturi.part')
    part.write_bytes(p.read_bytes()[:57]);p.unlink()
    report=mod.inventory_bundle(bundle,root,allow_partial=True)
    assert not report['files'][0]['header']['file_length_complete']
    assert not report['all_full_asset_hashes_verified']
    part.unlink()
    assert mod.inventory_bundle(bundle,root,allow_partial=True)['files'][0]['status']=='missing'
    with pytest.raises(ValueError,match='cannot qualify'):
        mod.inventory_bundle(bundle,root,allow_partial=True,verify_assets=True)


def test_cli_source_identity_and_exclusive_output(model,tmp_path,capsys):
    root,bundle=model;manifest=tmp_path/'manifest.json';output=tmp_path/'inventory.json'
    manifest.write_text(json.dumps(dict(model=bundle.model,official_revision=bundle.official_revision,
        variant=bundle.variant,files=[vars(a) for a in bundle.files])))
    args=['--manifest',str(manifest),'--root',str(root),'--output',str(output)]
    mod.main(args)
    assert json.loads(output.read_text())['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='overwrite'):mod.main(args)


def test_help_and_import_are_torch_free():
    out=subprocess.check_output([sys.executable,str(TOOL),'--help'],text=True)
    assert '--allow-partial' in out and '--verify-assets' in out
    code=f'import runpy,sys;runpy.run_path({str(TOOL)!r},run_name="not_main");assert "torch" not in sys.modules'
    subprocess.check_call([sys.executable,'-c',code])



def test_broken_final_symlink_cannot_hide_behind_a_partial(model):
    root,bundle=model;p=root/'body.gguf';raw=p.read_bytes();p.unlink()
    p.symlink_to(root/'absent');(root/'body.gguf.venturi.part').write_bytes(raw)
    with pytest.raises(ValueError,match='Symlinked'):
        mod.inventory_bundle(bundle,root,allow_partial=True)
