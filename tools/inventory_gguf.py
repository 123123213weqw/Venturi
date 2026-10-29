"""Read-only GGUF bundle inventory; no model loading, GPU, download or TPS claim.

Default: header-only, payload hashes UNVERIFIED even for complete-length files.
--verify-assets explicitly performs potentially expensive full SHA scans.
--allow-partial is diagnostic only and cannot be combined with verification.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from venturi.gguf_inventory import inspect_gguf
from venturi.model_assets import ModelBundle


def inventory_bundle(bundle, root, *, allow_partial=False, verify_assets=False):
    if allow_partial and verify_assets:
        raise ValueError('Partial diagnostics cannot qualify full asset hashes')
    if not isinstance(bundle, ModelBundle):
        raise ValueError('Explicit immutable ModelBundle required')
    root=Path(root).resolve(strict=True)
    records=[]
    for asset in bundle.files:
        path=root/asset.filename
        if path.is_symlink():raise ValueError('Symlinked GGUF file refused')
        if not path.exists() and allow_partial:
            path=root/(asset.filename+'.venturi.part')
            if path.is_symlink():raise ValueError('Symlinked GGUF partial refused')
        if not path.exists() and allow_partial:
            records.append(dict(filename=asset.filename,role=asset.role,status='missing',
                full_asset_sha256_verified=False))
            continue
        # Full hashes are an independent explicit admission, never inferred
        # from header SHA, dimensions, total file length or a sparse file size.
        if verify_assets:asset.verify(root)
        v=inspect_gguf(path,expected_bytes=asset.bytes,allow_partial=allow_partial)
        dtype_bytes=Counter()
        for tensor in v.tensors:dtype_bytes[tensor.dtype]+=tensor.bytes
        records.append(dict(filename=asset.filename,role=asset.role,status='header inventoried',
            full_asset_sha256_verified=verify_assets,
            on_disk_bytes_by_dtype=dict(dtype_bytes),header=asdict(v)))
    return dict(scope='GGUF header/layout inventory, not runtime/resident-memory admission or model correctness',
        model=bundle.model,official_revision=bundle.official_revision,variant=bundle.variant,
        expected_disk_bytes=bundle.disk_bytes,
        all_full_asset_hashes_verified=verify_assets and all(r['full_asset_sha256_verified'] for r in records),
        claims_model_support=False,model_tps=None,files=records)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--allow-partial',action='store_true')
    group.add_argument('--verify-assets',action='store_true')
    a=p.parse_args(argv)
    output=Path(a.output)
    if output.exists():raise ValueError('Refuse to overwrite an existing inventory')
    raw=Path(a.manifest).read_bytes()
    bundle=ModelBundle.from_dict(json.loads(raw))
    result=inventory_bundle(bundle,a.root,allow_partial=a.allow_partial,verify_assets=a.verify_assets)
    result['manifest_sha256']=hashlib.sha256(raw).hexdigest()
    with output.open('x') as f:f.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(files=len(result['files']),full_hashes_verified=result['all_full_asset_hashes_verified'],
        claims_model_support=False,model_tps=None)))


if __name__=='__main__':main()
