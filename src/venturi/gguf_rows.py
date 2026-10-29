"""Explicit full-SHA IQ4_NL CPU row storage component, not a model backend.

Admission scans the ENTIRE source asset once, on healthy storage only. Keep the
verified descriptor open; lookup reads only addressed rows, never reopens a path
or materializes a whole embedding table. No downloads, mmap, GPU or disk cache.
This opt-in prototype is not connected to production/native generation.
"""
from dataclasses import dataclass
import hashlib
from itertools import islice
import math
import os
from pathlib import Path
import stat
import threading

from venturi.gguf_inventory import inspect_gguf
from venturi.gguf_quantization import decode_iq4_nl_rows
from venturi.model_assets import ModelAsset


@dataclass(frozen=True)
class RowReadStats:
    requested_rows: int
    unique_rows: int
    row_bytes: int
    payload_bytes: int
    read_calls: int  # actual pread calls, not pages or physical disk operations


def _fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class VerifiedIQ4NLRows:
    """One read-only source lease, explicit [rows,width] schemas and private output.

    lookup_many merges row IDs across caller-provided requests, deduplicates and
    coalesces adjacent rows, then scatters into independent float32 results. It
    does NOT merge token/history states, schedule requests, prefetch or promise
    resident RAM. No persistent row cache; OS page cache is not measured here.
    Caller-supplied trusted source identities are an admission prerequisite.
    """

    def __init__(self, asset, root, *, expected_shapes, required_metadata=None, max_values=2**20):
        if not isinstance(asset, ModelAsset):raise TypeError('Explicit ModelAsset required')
        if (not isinstance(expected_shapes, dict) or not 1 <= len(expected_shapes) <= 64
                or any(not isinstance(k,str) or not k for k in expected_shapes)):
            raise ValueError('Explicit bounded named IQ4_NL table schemas required')
        if type(max_values) is not int or not 0 < max_values <= 2**20:
            raise ValueError('Explicit bounded lookup output budget required')
        required_metadata = {} if required_metadata is None else required_metadata
        if (not isinstance(required_metadata,dict) or len(required_metadata)>64
                or any(not isinstance(k,str) or not k or type(v) not in (str,int,float,bool)
                       or (isinstance(v,float) and not math.isfinite(v)) for k,v in required_metadata.items())):
            raise ValueError('Bounded explicit scalar metadata constraints required')
        shapes = {}
        for name, shape in expected_shapes.items():
            if (not isinstance(shape,(tuple,list)) or len(shape)!=2
                    or any(type(x) is not int for x in shape)
                    or not 0 < shape[0] < 2**63
                    or not 32 <= shape[1] <= 2**16 or shape[1]%32):
                raise ValueError('Bounded [rows,width] integer IQ4_NL table schema required')
            shapes[name] = tuple(shape)
        if not hasattr(os,'pread'):raise RuntimeError('POSIX pread required; no fallback backend')
        self._stream = None
        self._lock = threading.RLock()
        self._last_read = None
        self._poisoned = False
        self._max_values = max_values
        path = Path(root).resolve(strict=True)/asset.filename
        if path.is_symlink():raise ValueError('Symlinked GGUF source refused')
        stream = os.fdopen(os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
                                  |getattr(os,'O_NONBLOCK',0)), 'rb', buffering=0)
        try:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != asset.bytes:
                raise ValueError('Full regular source with exact manifest size required')
            inventory = inspect_gguf(path,expected_bytes=asset.bytes)
            for name,value in required_metadata.items():
                if type(inventory.metadata.get(name)) is not type(value) or inventory.metadata[name]!=value:
                    raise ValueError('Source scalar identity differs: '+name)
            directory = {tensor.name:tensor for tensor in inventory.tensors}
            selected = {}
            for name,shape in shapes.items():
                tensor = directory.get(name)
                if tensor is None or tensor.type_id != 20 or tensor.shape != shape:
                    raise ValueError('IQ4_NL table name/dtype/shape differs: '+name)
                selected[name] = tensor
            fingerprint = _fingerprint(before)
            if fingerprint != _fingerprint(os.fstat(stream.fileno())):
                raise ValueError('Source changed during header inventory')
            whole = hashlib.sha256();header = hashlib.sha256();offset = 0
            while offset < asset.bytes:
                raw = stream.read(min(8*2**20,asset.bytes-offset))
                if not raw:raise ValueError('Source truncated during full SHA scan')
                whole.update(raw)
                if offset < inventory.header_bytes_read:
                    header.update(raw[:inventory.header_bytes_read-offset])
                offset += len(raw)
            if stream.read(1) or fingerprint != _fingerprint(os.fstat(stream.fileno())):
                raise ValueError('Source changed during full SHA scan')
            if whole.hexdigest() != asset.sha256:
                raise ValueError('Full source SHA256 mismatch; row store NOT admitted')
            if header.hexdigest() != inventory.header_sha256:
                raise ValueError('Inventory header differs from verified source descriptor')
            self.asset = asset
            self.header_sha256 = inventory.header_sha256
            self._fingerprint = fingerprint
            self._tables = selected
            self._stream = stream
        except BaseException:
            stream.close()
            raise

    @property
    def closed(self):
        with self._lock:return self._stream is None

    @property
    def last_read(self):
        with self._lock:return self._last_read

    def close(self):
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self):
        if self.closed:raise ValueError('Row store is closed')
        return self

    def __exit__(self,*exc):self.close()

    def _check_lease(self):
        if self._fingerprint != _fingerprint(os.fstat(self._stream.fileno())):
            raise ValueError('Verified row source changed; explicit re-admission required')

    def lookup(self, name, row_ids):
        return self.lookup_many(name,(row_ids,))[0]

    def lookup_many(self, name, requests):
        """Snapshot bounded CPU int64 IDs; scatter outputs own independent storage.

        Each request may have its own shape. Empty requests perform no payload
        reads. Validate all IDs/budgets before I/O. Changed/EIO/nonfinite payloads
        poison the lease, rather than silently retrying untrusted storage.
        """
        import torch
        with self._lock:
            self._last_read = None
            if self._stream is None:raise ValueError('Row store is closed')
            if self._poisoned:raise ValueError('Row source lease failed; explicit re-admission required')
            if not isinstance(name,str) or name not in self._tables:
                raise ValueError('Explicit admitted table name required')
            if isinstance(requests,torch.Tensor):raise ValueError('Explicit request sequence required')
            requests = tuple(islice(requests,65))
            if not 1 <= len(requests) <= 64:raise ValueError('Bounded nonempty request sequence required')
            table = self._tables[name];rows,width = table.shape
            total = 0
            for ids in requests:
                if (not isinstance(ids,torch.Tensor) or ids.device.type!='cpu'
                        or ids.dtype!=torch.int64 or ids.ndim==0):
                    raise ValueError('CPU int64 request row-ID tensors required')
                total += ids.numel()
                if total*width > self._max_values:raise ValueError('Row output value budget exceeded')
            shapes = [tuple(ids.shape) for ids in requests]
            snapshots = [ids.reshape(-1).tolist() for ids in requests]
            if any(not 0 <= row < rows for ids in snapshots for row in ids):
                raise ValueError('Row ID outside admitted table bounds')
            unique = sorted(set(row for ids in snapshots for row in ids))
            row_bytes = width//32*18;calls = 0;pieces = []
            try:
                self._check_lease()
                start = 0
                while start < len(unique):
                    end = start+1
                    while end < len(unique) and unique[end]==unique[end-1]+1:end+=1
                    offset = table.offset+unique[start]*row_bytes
                    remaining = (end-start)*row_bytes
                    while remaining:
                        raw = os.pread(self._stream.fileno(),remaining,offset);calls+=1
                        if not raw:raise ValueError('Short IQ4_NL row payload read')
                        pieces.append(raw);offset+=len(raw);remaining-=len(raw)
                    start = end
                decoded = (decode_iq4_nl_rows(b''.join(pieces),width,max_values=self._max_values)
                           if unique else torch.empty((0,width),dtype=torch.float32,device='cpu'))
                self._check_lease()
            except (OSError,ValueError):
                self._poisoned = True
                raise
            positions = {row:i for i,row in enumerate(unique)}
            outputs = tuple(decoded.index_select(0,torch.tensor([positions[row] for row in ids],
                dtype=torch.int64,device='cpu')).reshape(shape+(width,))
                for shape,ids in zip(shapes,snapshots))
            self._last_read = RowReadStats(total,len(unique),row_bytes,len(unique)*row_bytes,calls)
            return outputs
