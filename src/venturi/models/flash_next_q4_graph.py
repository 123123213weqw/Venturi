"""Opt-in SM70 canonical-Q4 dispatch experiments; never the rejected packed GEMV.

B1 route download once, one coalesced H2D per expert pair, optional shared routed
MoE CUDA graph. Same sorted expert reduction and decoded FP16 Torch linear calls.
Not qualified/default/TP/PP, concurrent serving or a complete-model speed claim.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import threading

import torch
from torch.nn import functional as F

from ..ops.flash_next_q4 import NativeQ4, PreparedQ4
from ..ops.weights import Q4RowSplitWeight
from .flash_next_q4 import PackedQ4Dispatch


@dataclass(frozen=True)
class PairLayout:
    hidden: int
    intermediate: int

    def __post_init__(self):
        if any(type(v) is not int or not 0 < v <= 32768 for v in (self.hidden,self.intermediate)):
            raise ValueError('Explicit bounded expert matrix geometry required')
        if self.nbytes > 64*2**20:raise ValueError('Expert pair exceeds64MiB encoded budget')

    @property
    def planes(self):
        values=[];offset=0
        for n,k in ((2*self.intermediate,self.hidden),(self.hidden,self.intermediate)):
            padded=(k+127)//128*128
            for dtype,shape in ((torch.uint8,(n,padded//2)),(torch.float16,(n,padded//64))):
                size=math.prod(shape)*(2 if dtype==torch.float16 else 1)
                values.append((offset,size,dtype,shape,k));offset+=size
        return tuple(values)

    @property
    def nbytes(self):return sum(v[1] for v in self.planes)

    def unpack(self,payload):
        if payload.dtype!=torch.uint8 or payload.ndim!=1 or not payload.is_contiguous() or payload.numel()!=self.nbytes:
            raise ValueError('Exact contiguous coalesced expert payload required')
        parts=[payload[o:o+s].view(dtype).reshape(shape) for o,s,dtype,shape,k in self.planes]
        return (Q4RowSplitWeight(parts[0],parts[1],self.hidden),
                Q4RowSplitWeight(parts[2],parts[3],self.intermediate))

    def pack_cpu(self,weights):
        if not isinstance(weights,(tuple,list)) or len(weights)!=2:raise ValueError('Two Q4 expert matrices required')
        for w,(n,k) in zip(weights,((2*self.intermediate,self.hidden),(self.hidden,self.intermediate))):
            if not isinstance(w,Q4RowSplitWeight):raise ValueError('Explicit Q4 descriptors required')
            w.validate()
            if w.codes.device.type!='cpu' or (w.out_features,w.in_features)!=(n,k):raise ValueError('CPU pair geometry mismatch')
            if not torch.isfinite(w.scales).all() or (w.scales<0).any():raise ValueError('Finite nonnegative Q4 scales required')
        result=torch.empty(self.nbytes,dtype=torch.uint8)
        for (o,s,dtype,shape,k),v in zip(self.planes,(weights[0].codes,weights[0].scales,weights[1].codes,weights[1].scales)):
            result[o:o+s].copy_(v.view(torch.uint8).reshape(-1))
        return result


@dataclass(frozen=True)
class PreparedPair:
    payload: torch.Tensor
    weights: tuple


class CoalescedQ4(NativeQ4):
    def prepare_pair(self,weights,layout,*,device):
        payload=layout.pack_cpu(weights)
        if any(w.out_features*w.in_features*2>self.max_output_bytes for w in weights):raise ValueError('Pair decode exceeds operator scratch budget')
        device=torch.device(device)
        if device.type!='cuda':raise ValueError('Explicit CUDA device required')
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability(device)!=(7,0):raise ValueError('Qualified SM70 only')
            copied=payload.to(device)  # One blocking H2D, no unqualified async host lifetime.
            ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device))
            return PreparedPair(copied,tuple(PreparedQ4(w,ready) for w in layout.unpack(copied)))

    def decode_into(self,prepared,out):
        w=prepared.weight
        if out.shape!=(w.out_features,w.in_features) or out.dtype!=torch.float16 or out.device!=w.codes.device or not out.is_contiguous():
            raise ValueError('Exact contiguous CUDA FP16 decode destination required')
        if out.numel()*out.element_size()>self.max_output_bytes:raise ValueError('Q4 scratch exceeds operator budget')
        with torch.cuda.device(w.codes.device):
            w,stream=self._lease(prepared)
            status=self._decode(w.codes.data_ptr(),w.scales.data_ptr(),out.data_ptr(),w.out_features,w.in_features,w.padded_k,stream.cuda_stream)
            if status:raise RuntimeError(f'Q4 decode launch failed: {status}')
        return out

    def _decode_workspace_into(self,weight,out):
        """Only permanently owned graph workspace planes, NOT external leases.

        Workspace construction synchronizes before capture, replay stages on the
        launch stream, and close waits for its last completion event. Waiting
        on prepare()'s uncaptured external event inside capture is invalid.
        """
        weight.validate()
        if weight.codes.device.type!='cuda' or out.shape!=(weight.out_features,weight.in_features) or out.dtype!=torch.float16 or out.device!=weight.codes.device or not out.is_contiguous():
            raise ValueError('Owned CUDA workspace and exact FP16 destination required')
        if out.numel()*out.element_size()>self.max_output_bytes:raise ValueError('Workspace decode exceeds operator budget')
        stream=torch.cuda.current_stream(weight.codes.device)
        status=self._decode(weight.codes.data_ptr(),weight.scales.data_ptr(),out.data_ptr(),weight.out_features,weight.in_features,weight.padded_k,stream.cuda_stream)
        if status:raise RuntimeError(f'Workspace Q4 decode launch failed: {status}')
        return out


def one_route(indices,num_experts):
    """One explicit D2H for B1; None preserves baseline duplicate-ID semantics."""
    if indices.ndim!=2 or len(indices)!=1 or indices.dtype!=torch.int64:raise ValueError('B1 int64 route required')
    ids=indices.detach().cpu().tolist()[0]
    if any(not 0<=e<num_experts for e in ids):raise ValueError('Selected expert outside bank')
    if len(set(ids))!=len(ids):return None
    return tuple(sorted((e,pos) for pos,e in enumerate(ids)))


class RoutedQ4Graph:
    """One SERIAL reusable workspace for all same-geometry MoEs on one GPU.

    Copies into owned fixed addresses, then canonical decode+Torch GEMMs. Does
    not graph host file I/O, cache misses, routing download or the full model.
    Output is cloned; a completion event orders reuse across caller streams.
    """
    def __init__(self,native,geometry,*,device,max_workspace_bytes=64*2**20):
        self.device=torch.device(device);self.native=native;self.geometry=geometry
        if self.device.type!='cuda' or torch.cuda.get_device_capability(self.device)!=(7,0):raise ValueError('SM70 workspace required')
        self.layout=PairLayout(geometry.hidden_size,geometry.intermediate_size);g=geometry
        self.base_bytes=g.top_k*self.layout.nbytes+6*g.hidden_size*g.intermediate_size+4*g.hidden_size+10*g.top_k+8
        if type(max_workspace_bytes) is not int or max_workspace_bytes<=0 or self.base_bytes>max_workspace_bytes:raise ValueError('Explicit bounded graph workspace required')
        self.closed=False;self.calls=0;self._lock=threading.Lock();self._done=None
        with torch.cuda.device(self.device),torch.inference_mode():
            self.bank=torch.zeros((g.top_k,self.layout.nbytes),dtype=torch.uint8,device=self.device)
            self.x=torch.zeros((1,g.hidden_size),dtype=torch.float16,device=self.device)
            self.scores=torch.zeros((1,g.top_k),dtype=torch.float16,device=self.device)
            self.indices=torch.arange(g.top_k,dtype=torch.int64,device=self.device).reshape(1,-1)
            self.output=torch.zeros_like(self.x);self.zero=torch.zeros(1,dtype=torch.int64,device=self.device)
            self.gu=torch.empty((2*g.intermediate_size,g.hidden_size),dtype=torch.float16,device=self.device)
            self.down=torch.empty((g.hidden_size,g.intermediate_size),dtype=torch.float16,device=self.device)
            self.weights=[self.layout.unpack(row) for row in self.bank]
            # Initialize the exact Torch/cuBLAS path outside capture. Zero data
            # prevents any NaN from uninitialized scale planes during warmup.
            for _ in range(2):self._body()
            torch.cuda.synchronize(self.device);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self._body()
            self._done=torch.cuda.Event();self._done.record(torch.cuda.current_stream(self.device))

    def _body(self):
        order=torch.argsort(self.indices[0]);scores=self.scores[0].index_select(0,order)
        self.output.zero_()
        for i,(gu,down) in enumerate(self.weights):
            self.native._decode_workspace_into(gu,self.gu)
            gate,up=F.linear(self.x,self.gu).chunk(2,-1)
            activated=F.silu(gate)*up
            self.native._decode_workspace_into(down,self.down)
            current=F.linear(activated,self.down)
            self.output.index_add_(0,self.zero,current*scores[i].reshape(1,1))

    def run(self,x,indices,scores,pairs):
        if self.closed:raise RuntimeError('Graph workspace is closed')
        if not self._lock.acquire(blocking=False):raise RuntimeError('Serial graph workspace cannot serve concurrent callers')
        try:
            g=self.geometry
            if x.shape!=(1,g.hidden_size) or x.dtype!=torch.float16 or x.device!=self.device or indices.shape!=(1,g.top_k) or indices.dtype!=torch.int64 or scores.shape!=indices.shape or scores.dtype!=x.dtype or any(v.device!=self.device for v in (indices,scores)):
                raise ValueError('Explicit single-token workspace input required')
            if len(pairs)!=g.top_k or any(p.payload.numel()!=self.layout.nbytes or p.payload.device!=self.device for p in pairs):raise ValueError('Every sorted expert pair required')
            with torch.cuda.device(self.device),torch.inference_mode():
                stream=torch.cuda.current_stream(self.device);stream.wait_event(self._done)
                self.x.copy_(x);self.indices.copy_(indices);self.scores.copy_(scores)
                for i,pair in enumerate(pairs):
                    for w in pair.weights:stream.wait_event(w.ready)
                    self.bank[i].copy_(pair.payload);pair.payload.record_stream(stream)
                self.graph.replay();result=self.output.clone();self._done.record(stream)
            self.calls+=1;return result
        finally:self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Cannot close an active graph workspace')
        try:
            if not self.closed:
                if self._done is not None:self._done.synchronize()
                self.closed=True
                for name in ('graph','bank','x','scores','indices','output','zero','gu','down','weights'):
                    if hasattr(self,name):delattr(self,name)
        finally:self._lock.release()


class CoalescedQ4Dispatch:
    def __init__(self,archive,native,*,device,max_resident_bytes,workspace=None):
        if type(max_resident_bytes) is not int or max_resident_bytes<0:raise ValueError('Explicit nonnegative packed budget required')
        self.archive,self.native,self.geometry=archive,native,archive.geometry;self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('Explicit CUDA dispatch required')
        g=self.geometry;self.layout=PairLayout(g.hidden_size,g.intermediate_size);self.max_resident_bytes=max_resident_bytes;self.workspace=workspace
        if workspace is not None and (workspace.device!=self.device or workspace.geometry!=g):raise ValueError('Graph geometry/device mismatch')
        self.entries=OrderedDict();self.retained_bytes=0;self.hits=self.misses=self.fast_calls=self.fallback_calls=0
        self.zero=torch.zeros(1,dtype=torch.int64,device=self.device)

    @property
    def stats(self):
        return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),retained_bytes=self.retained_bytes,retained_decoded_bytes=0,max_resident_bytes=self.max_resident_bytes,archive=self.archive.stats,fast_calls=self.fast_calls,fallback_calls=self.fallback_calls,graph=self.workspace is not None)

    def clear(self):self.entries.clear();self.retained_bytes=0

    def _get_pair(self,expert):
        if expert in self.entries:
            self.hits+=1;self.entries.move_to_end(expert);return self.entries[expert]
        self.misses+=1;weights=self.archive.packed(expert)
        while self.entries and self.retained_bytes+self.layout.nbytes>self.max_resident_bytes:
            _,old=self.entries.popitem(last=False);self.retained_bytes-=old.payload.numel()
        pair=self.native.prepare_pair(weights,self.layout,device=self.device)
        if self.layout.nbytes<=self.max_resident_bytes:self.entries[expert]=pair;self.retained_bytes+=self.layout.nbytes
        return pair

    def _get(self,expert):return self._get_pair(expert).weights

    def decoded_reference(self,x,indices,scores):
        g=self.geometry
        if x.ndim!=2 or x.shape[1]!=g.hidden_size or x.dtype!=torch.float16 or x.device!=self.device or indices.shape!=(len(x),g.top_k) or indices.dtype!=torch.int64 or scores.shape!=indices.shape or scores.dtype!=x.dtype or any(v.device!=x.device for v in (indices,scores)):
            raise ValueError('Explicit FP16 CUDA routed geometry required')
        route=one_route(indices,g.num_experts) if len(x)==1 else None
        if route is None:
            self.fallback_calls+=1;return PackedQ4Dispatch._run(self,x,indices,scores,packed=False)
        self.fast_calls+=1
        if self.workspace is not None:return self.workspace.run(x,indices,scores,[self._get_pair(e) for e,pos in route])
        out=torch.zeros_like(x)
        for expert,pos in route:
            gu,down=self._get(expert)
            gate,up=F.linear(x,self.native.decode(gu)).chunk(2,-1)
            current=F.linear(F.silu(gate)*up,self.native.decode(down))
            out.index_add_(0,self.zero,current*scores[:,pos:pos+1])
        return out
