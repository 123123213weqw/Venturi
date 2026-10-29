"""Explicit full-SHA-verified integer GGUF metadata reads, not weight loading.

Verification necessarily scans the ENTIRE selected asset. Decode only named,
bounded I8/I16/I32/I64 tensors afterward. Never accepts partial/header-only or
complete-length-only evidence; no automatic cache trust, download, GPU or Torch.
Use only on healthy storage during offline preparation, not a decode hot path.
"""
from dataclasses import dataclass
import hashlib
from itertools import islice
import os
from pathlib import Path
import stat
import struct

from venturi.gguf_inventory import inspect_gguf, GgufArraySummary, GgufLimits, _HeaderReader
from venturi.model_assets import ModelAsset


_INTEGER_FORMATS={24:'b',25:'h',26:'i',27:'q'}
_HEADER_INTEGER_FORMATS={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',10:'Q',11:'q'}


@dataclass(frozen=True)
class IntegerMetadataTensor:
    name: str
    shape: tuple[int,...]
    type_id: int
    values: tuple[int,...]  # flat, outermost-first row-major order


@dataclass(frozen=True)
class HeaderIntegerArray:
    name: str
    element_type: int  # GGUF metadata type ID, NOT ggml tensor type ID
    values: tuple[int,...]


@dataclass(frozen=True)
class VerifiedIntegerMetadata:
    asset: ModelAsset
    header_sha256: str
    tensors: tuple[IntegerMetadataTensor,...]
    scalars: tuple[tuple[str,object],...] = ()
    header_arrays: tuple[HeaderIntegerArray,...] = ()

    def header_array(self,name):
        return next(a for a in self.header_arrays if a.name==name)

    def tensor(self,name):
        return next(t for t in self.tensors if t.name==name)


def _fingerprint(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _header_arrays(stream,inventory,names,max_elements):
    if not names:return ()
    stream.seek(0)
    reader=_HeaderReader(stream,inventory.header_bytes_read,GgufLimits())
    if reader.read(4)!=b'GGUF':raise ValueError('Verified header magic differs')
    version,nt,nkv=reader.unpack('IQQ')
    if version!=3:raise ValueError('Verified header version differs')
    wanted=set(names);arrays={};counted=0
    for _ in range(nkv):
        key=reader.string();typ,=reader.unpack('I')
        if key not in wanted:
            reader.value(typ);continue
        if typ!=9:raise ValueError('Requested header integer array is not an array')
        element_type,count=reader.unpack('IQ')
        counted+=count
        if element_type not in _HEADER_INTEGER_FORMATS or counted>max_elements:
            raise ValueError('Header integer array type/element budget differs')
        reader.array_items+=count
        if reader.array_items>reader.limits.array_items:raise ValueError('Header array budget exceeded')
        arrays[key]=HeaderIntegerArray(key,element_type,reader.unpack(_HEADER_INTEGER_FORMATS[element_type]*count))
    if set(arrays)!=wanted:raise ValueError('Required header integer array missing')
    return tuple(arrays[name] for name in names)


def read_verified_integer_metadata(asset,root,names,*,max_payload_bytes=8*2**20,max_elements=2_000_000,
                                   header_array_names=(),max_header_array_elements=4096):
    """Hash one immutable manifest asset, then decode selected integer tensors.

    The SHA scan and reads share an open descriptor. Its header must match the
    bounded inventory; before/after file fingerprints detect concurrent changes.
    Returned values are private tuples, not a live mmap/lease against writers.
    This is preparation integrity, not semantic/model/quantization qualification.
    """
    if not isinstance(asset,ModelAsset):raise TypeError('Explicit ModelAsset required')
    if isinstance(names,str):raise ValueError('Explicit tensor-name sequence required')
    names=tuple(islice(names,129))
    if (not 1<=len(names)<=128 or any(not isinstance(n,str) or not n for n in names)
            or len(set(names))!=len(names)):
        raise ValueError('Unique explicit integer tensor names required')
    if (type(max_payload_bytes) is not int or not 0<max_payload_bytes<=64*2**20
            or type(max_elements) is not int or not 0<max_elements<=2_000_000):
        raise ValueError('Bounded positive integer metadata budgets required')
    if isinstance(header_array_names,str):raise ValueError('Explicit header-array name sequence required')
    header_array_names=tuple(islice(header_array_names,129))
    if (len(header_array_names)>128 or any(not isinstance(n,str) or not n for n in header_array_names)
            or len(set(header_array_names))!=len(header_array_names)
            or type(max_header_array_elements) is not int or not 0<max_header_array_elements<=4096):
        raise ValueError('Bounded unique header array names/elements required')
    path=Path(root).resolve(strict=True)/asset.filename
    if path.is_symlink():raise ValueError('Symlinked GGUF asset refused')
    flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)
    with os.fdopen(os.open(path,flags),'rb',buffering=0) as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size!=asset.bytes:
            raise ValueError('Full regular asset with exact manifest size required')
        inventory=inspect_gguf(path,expected_bytes=asset.bytes)
        directory={t.name:t for t in inventory.tensors}
        if any(name not in directory for name in names):
            raise ValueError('Required integer metadata tensor missing')
        selected=[directory[name] for name in names]
        if any(t.type_id not in _INTEGER_FORMATS for t in selected):
            raise ValueError('Only I8/I16/I32/I64 metadata can be decoded')
        if (sum(t.bytes for t in selected)>max_payload_bytes
                or sum(t.bytes//struct.calcsize(_INTEGER_FORMATS[t.type_id]) for t in selected)>max_elements):
            raise ValueError('Integer metadata payload/element budget exceeded')
        if _fingerprint(before)!=_fingerprint(os.fstat(stream.fileno())):
            raise ValueError('Asset changed during metadata inventory')
        digest=hashlib.sha256();header_digest=hashlib.sha256();read=0
        while read<asset.bytes:
            data=stream.read(min(8*2**20,asset.bytes-read))
            if not data:raise ValueError('Asset truncated during full SHA scan')
            digest.update(data)
            if read<inventory.header_bytes_read:
                header_digest.update(data[:inventory.header_bytes_read-read])
            read+=len(data)
        if stream.read(1) or _fingerprint(before)!=_fingerprint(os.fstat(stream.fileno())):
            raise ValueError('Asset changed during full SHA scan')
        if digest.hexdigest()!=asset.sha256:
            raise ValueError('Full source asset SHA256 mismatch; metadata NOT decoded')
        if header_digest.hexdigest()!=inventory.header_sha256:
            raise ValueError('Inventory header differs from the verified open asset')
        arrays=_header_arrays(stream,inventory,header_array_names,max_header_array_elements)
        tensors=[]
        for tensor in selected:
            stream.seek(tensor.offset);raw=stream.read(tensor.bytes)
            if len(raw)!=tensor.bytes:raise ValueError('Metadata payload truncated after SHA check')
            values=tuple(v[0] for v in struct.iter_unpack('<'+_INTEGER_FORMATS[tensor.type_id],raw))
            tensors.append(IntegerMetadataTensor(tensor.name,tensor.shape,tensor.type_id,values))
        if _fingerprint(before)!=_fingerprint(os.fstat(stream.fileno())):
            raise ValueError('Asset changed during integer metadata decoding')
        return VerifiedIntegerMetadata(asset,inventory.header_sha256,tuple(tensors),
            tuple((k,v) for k,v in inventory.metadata.items() if not isinstance(v,GgufArraySummary)),arrays)
