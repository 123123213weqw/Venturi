"""Bounded explicit full-SHA JSON assets. No execution/discovery/download/GPU.

Returns a private parsed object, not a live filesystem lease. Config schema and
architecture support remain separate gates. Reject duplicate keys/nonfinite
numbers rather than silently choosing a meaning for ambiguous model metadata.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from venturi.model_assets import ModelAsset


def read_verified_json_asset(asset,root,*,max_bytes=2**20):
    if not isinstance(asset,ModelAsset):raise TypeError('Explicit ModelAsset required')
    if type(max_bytes) is not int or not 0<max_bytes<=64*2**20:
        raise ValueError('Explicit positive bounded JSON byte budget required')
    if asset.bytes>max_bytes:raise ValueError('JSON asset exceeds byte budget')
    path=Path(root).resolve(strict=True)/asset.filename
    if path.is_symlink():raise ValueError('Symlinked JSON asset refused')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb',buffering=0) as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size!=asset.bytes:
            raise ValueError('Full regular JSON asset with exact size required')
        raw=stream.read(asset.bytes+1);after=os.fstat(stream.fileno())
        fingerprint=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fingerprint(before)!=fingerprint(after):raise ValueError('JSON asset changed while reading')
    if len(raw)!=asset.bytes or hashlib.sha256(raw).hexdigest()!=asset.sha256:
        raise ValueError('Full JSON asset SHA256/size mismatch')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON object key')
            result[key]=value
        return result
    def constant(value):raise ValueError('Nonfinite JSON constant')
    def floating(value):
        result=float(value)
        if not math.isfinite(result):raise ValueError('Nonfinite JSON floating value')
        return result
    try:
        value=json.loads(raw,object_pairs_hook=unique,parse_constant=constant,parse_float=floating)
    except (UnicodeError,RecursionError) as e:
        raise ValueError('Invalid/deep JSON asset') from e
    if not isinstance(value,dict):raise ValueError('JSON asset must be an object')
    return value
