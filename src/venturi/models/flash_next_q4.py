"""Experimental routed-expert W4A16, explicit RTN codec and lazy disk archive.

This is NOT AWQ/GPTQ, a BF16-derived checkpoint or a quality-qualified model.
Requantizing official FP8 must be labelled as such. Only routed experts change;
router/shared expert/PLE/state are untouched. No import-time compilation or
production/default binding. Packed caches do not retain expanded FP16 weights.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ..ops.weights import Q4RowSplitWeight


CODEC = 'rtn_symmetric7_g64_fp16_row_split_k128_v1'


def quantize_q4_rtn(weight, *, max_elements=16*2**20):
    """CPU weight-only RTN: FP16 scale first, ties-to-even, q in [-7,7].

    Each contiguous K64 group uses max(abs(w))/7. Scale is rounded to FP16
    BEFORE code rounding and floored at the smallest positive FP16 subnormal.
    Padding is zero. The unused -8 code is accepted by the general descriptor
    and decoder, but never produced here. Output has independent owned storage.
    """
    if not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or (
            weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or min(weight.shape) < 1):
        raise ValueError('Explicit nonempty CPU floating weight required')
    if type(max_elements) is not int or max_elements <= 0 or weight.numel() > max_elements:
        raise ValueError('Q4 quantization exceeds explicit element budget')
    value = weight.float()
    if not torch.isfinite(value).all():
        raise ValueError('Finite source weights required')
    rows, cols = weight.shape
    padded = (cols+127)//128*128
    blocks = F.pad(value, (0, padded-cols)).reshape(rows, padded//64, 64)
    scales = (blocks.abs().amax(-1)/7).clamp_min(2**-24).half()
    if not torch.isfinite(scales).all():
        raise ValueError('Source scale overflows FP16')
    codes = (blocks/scales.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    if not torch.isfinite((codes.float()*scales.float().unsqueeze(-1)).half()).all():
        raise ValueError('Quantized weight overflows FP16 execution')
    # Fully padded groups may have scale zero; their codes remain exactly zero.
    return Q4RowSplitWeight.from_codes(codes.reshape(rows, padded)[:, :cols], scales[:, :(cols+63)//64])


def decode_q4_fp16(weight):
    """Bounded diagnostic/fallback FP16 matrix, no retained decoded cache."""
    weight.validate()
    unsigned = torch.stack((weight.codes & 15, weight.codes >> 4), -1).reshape(
        weight.out_features, weight.padded_k).to(torch.int16)
    signed = torch.where(unsigned < 8, unsigned, unsigned-16)
    return (signed.float().reshape(weight.out_features, -1, 64)*
            weight.scales.float().unsqueeze(-1)).reshape(weight.out_features, -1)[:, :weight.in_features].half()


class LazyQ4ExpertArchive:
    """One source layer, bounded one-expert conversion, immutable owned files.

    Archive identity is explicit and source files must remain immutable. Initial
    creation refuses an existing directory; reuse verifies manifest/codec and
    every expert's SHA256 on its first read. A miss is quantization/setup work,
    never hidden as a steady-state TPS result. No complete-Q4-checkpoint claim.
    Serial setup/read only; use one instance per layer, do not share writers.
    """

    def __init__(self, source, directory, *, source_identity, create=False):
        self.source, self.geometry = source, source.geometry
        self.directory = Path(directory)
        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError('Explicit source identity/provenance required')
        identity = dict(codec=CODEC, source=source_identity, geometry=vars(self.geometry))
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.manifest = dict(identity, experts={})
            self._save_manifest()
        else:
            self.manifest = json.loads((self.directory/'manifest.json').read_text())
            if {k:self.manifest[k] for k in identity} != identity:
                raise ValueError('Q4 archive source/codec/geometry mismatch')
        self._verified = set()
        self._lock = threading.RLock()
        self.converted = self.reads = 0

    def _save_manifest(self):
        path = self.directory/'manifest.json'
        temporary = path.with_suffix('.json.part')
        temporary.write_text(json.dumps(self.manifest, sort_keys=True, indent=2)+'\n')
        temporary.replace(path)

    @property
    def stats(self):
        return dict(converted=self.converted, reads=self.reads, experts=len(self.manifest['experts']),
                    encoded_bytes=sum(x['encoded_bytes'] for x in self.manifest['experts'].values()),
                    retained_decoded_bytes=0)

    def packed(self, expert):
        if type(expert) is not int or not 0 <= expert < self.geometry.num_experts:
            raise ValueError('Selected real expert id required')
        with self._lock:
            key = str(expert)
            path = self.directory/f'expert-{expert:04d}.safetensors'
            if key not in self.manifest['experts']:
                if path.exists():
                    raise ValueError('Uncommitted Q4 payload exists; preserve it, do not overwrite')
                gate_up, down = self.source.read(expert, device='cpu', dtype=torch.float16)
                matrices = tuple(quantize_q4_rtn(w) for w in (gate_up, down))
                tensors = {name+'.'+part:getattr(w, part) for name,w in zip(('gate_up','down'), matrices)
                           for part in ('codes','scales')}
                temporary = path.with_suffix('.safetensors.part')
                if temporary.exists():
                    raise ValueError('Interrupted Q4 payload exists; preserve evidence')
                save_file(tensors, str(temporary), metadata={'codec':CODEC})
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                temporary.replace(path)
                self.manifest['experts'][key] = dict(sha256=digest, encoded_bytes=sum(w.nbytes for w in matrices),
                    file_bytes=path.stat().st_size)
                self._save_manifest()
                self.converted += 1
            if expert not in self._verified:
                entry = self.manifest['experts'][key]
                if path.stat().st_size != entry['file_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Q4 archive payload checksum mismatch')
                self._verified.add(expert)
            with safe_open(str(path), framework='pt', device='cpu') as reader:
                if reader.metadata() != {'codec':CODEC} or set(reader.keys()) != {
                        name+'.'+part for name in ('gate_up','down') for part in ('codes','scales')}:
                    raise ValueError('Q4 archive tensor contract mismatch')
                result = tuple(Q4RowSplitWeight(reader.get_tensor(name+'.codes'), reader.get_tensor(name+'.scales'), k)
                               for name,k in (('gate_up',self.geometry.hidden_size),('down',self.geometry.intermediate_size)))
                if tuple(w.out_features for w in result) != (2*self.geometry.intermediate_size, self.geometry.hidden_size):
                    raise ValueError('Q4 expert shape mismatch')
                if any(not torch.isfinite(w.scales).all() or (w.scales < 0).any() for w in result):
                    raise ValueError('Q4 finite nonnegative scales required')
            self.reads += 1
            return result

    def read(self, expert, *, device, dtype):
        if dtype != torch.float16:
            raise ValueError('This W4A16 diagnostic is explicitly FP16 only')
        return tuple(decode_q4_fp16(w).to(device) for w in self.packed(expert))


class PackedQ4Dispatch:
    """Opt-in GPU packed LRU; M1 fused-dequant GEMV, M>1 decoded scratch GEMM.

    Retains ONLY code/scale planes. Prefill dequantization has one live expert's
    FP16 scratch, not a full FP16 expert cache. GPU allocations/leases/scratch
    exceed the retained-byte budget. Python route synchronization remains.
    """

    def __init__(self, archive, native, *, device, max_resident_bytes):
        if type(max_resident_bytes) is not int or max_resident_bytes < 0:
            raise ValueError('Explicit packed resident budget required')
        self.archive, self.native, self.geometry = archive, native, archive.geometry
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Explicit CUDA W4A16 dispatch required')
        self.max_resident_bytes = max_resident_bytes
        self.entries, self.retained_bytes = OrderedDict(), 0
        self.hits = self.misses = 0

    @property
    def stats(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self.entries), retained_bytes=self.retained_bytes,
                    retained_decoded_bytes=0, max_resident_bytes=self.max_resident_bytes, archive=self.archive.stats)

    def clear(self):
        self.entries.clear()
        self.retained_bytes = 0

    def _get(self, expert):
        if expert in self.entries:
            self.hits += 1
            self.entries.move_to_end(expert)
            return self.entries[expert]
        self.misses += 1
        weights = self.archive.packed(expert)
        size = sum(w.nbytes for w in weights)
        while self.entries and self.retained_bytes+size > self.max_resident_bytes:
            _, old = self.entries.popitem(last=False)
            self.retained_bytes -= sum(w.weight.nbytes for w in old)
        prepared = tuple(self.native.prepare(w, device=self.device) for w in weights)
        if size <= self.max_resident_bytes:
            self.entries[expert] = prepared
            self.retained_bytes += size
        return prepared

    def __call__(self, x, indices, scores):
        return self._run(x,indices,scores,packed=True)

    def decoded_reference(self, x, indices, scores):
        """Same packed cache, transient decoded FP16 + Torch linear control."""
        return self._run(x,indices,scores,packed=False)

    def _run(self, x, indices, scores, *, packed):
        g = self.geometry
        if x.ndim != 2 or x.shape[1] != g.hidden_size or x.dtype != torch.float16 or x.device != self.device or (
                indices.shape != (len(x),g.top_k) or indices.dtype != torch.int64 or scores.shape != indices.shape or
                scores.dtype != x.dtype or any(t.device != x.device for t in (indices,scores))):
            raise ValueError('Explicit FP16 CUDA routed geometry required')
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= g.num_experts):
            raise ValueError('Selected real expert ids required')
        output = torch.zeros_like(x)
        for expert in torch.unique(indices, sorted=True).tolist():
            positions, tokens = torch.where((indices == expert).T)
            gu, down = self._get(expert)
            current = x[tokens].contiguous()
            def linear(values, weight):
                if packed and len(values) == 1:
                    return self.native.linear(values, weight)
                return F.linear(values, self.native.decode(weight))
            gate, up = linear(current,gu).chunk(2,-1)
            current = linear(F.silu(gate)*up,down)
            output.index_add_(0,tokens,current*scores[tokens,positions,None])
        return output
