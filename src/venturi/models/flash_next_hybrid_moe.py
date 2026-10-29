"""Explicit experimental fixed-hot GPU / parallel-cold CPU expert execution.

No automatic placement, pruning, new quantization, serving or CPU speed claim.
CPU FP32 BLAS uses weights FIRST rounded to the selected FP16 execution format,
with FP16 boundaries after each linear/SiLU/multiply. CPU/GPU reductions can
round differently: this path requires independent numerical qualification.
Default MoE/production behavior is untouched until the caller installs dispatch.
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from collections import OrderedDict
from contextlib import nullcontext
import threading
import time

import torch
from torch.nn import functional as F

from .flash_next_moe import MoEGeometry


def cpu_expert(source, expert, inputs):
    """Only CPU tensors in worker threads; never touches a CUDA stream/context."""
    if inputs.device.type != 'cpu' or inputs.dtype != torch.float16:
        raise ValueError('Explicit CPU FP16 activation boundary required')
    with torch.inference_mode():
        if hasattr(source, 'read_compute'):
            gate_up, down = source.read_compute(expert)
        else:
            gate_up, down = source.read(expert, device=torch.device('cpu'), dtype=torch.float16)
            gate_up, down = gate_up.float(), down.float()
        native_half = getattr(source, 'compute_mode', 'staged-fp32') == 'native-half'
        gate, up = (F.linear(inputs, gate_up) if native_half else
                    F.linear(inputs.float(), gate_up).to(torch.float16)).chunk(2, -1)
        # Separate half-precision boundaries match the reference operation graph,
        # not CPU half GEMM's much slower fallback on Cascade Lake.
        intermediate = F.silu(gate) * up
        result = F.linear(intermediate, down) if native_half else F.linear(intermediate.float(), down).to(torch.float16)
        if not torch.isfinite(result).all():
            raise ValueError('Cold CPU expert output is nonfinite/overflowed')
        return result


class CPUDecodedExpertCache:
    """Bounded immutable FP32 compute weights, FIRST rounded to source FP16.

    Different-key misses decode concurrently; same-key requests coalesce. Budget
    counts retained ownership only, NOT active leases, FP8/FP16 decode scratch,
    page cache or source files. No quantization change, pinning or full-bank load.
    """

    def __init__(self, source, *, max_bytes, compute_mode='staged-fp32'):
        if not isinstance(getattr(source, 'geometry', None), MoEGeometry) or type(max_bytes) is not int or max_bytes<0:
            raise ValueError('Qualified CPU expert source and explicit byte budget required')
        if compute_mode not in ('staged-fp32','native-half'):
            raise ValueError('Explicit qualified CPU compute mode required')
        self.source,self.geometry,self.max_bytes=source,source.geometry,max_bytes
        self.compute_mode=compute_mode
        self.lock,self.entries,self.inflight=threading.Lock(),OrderedDict(),{}
        self.retained=0
        self.counts=dict(hits=0,misses=0,coalesced=0,evictions=0,bypasses=0)

    @property
    def stats(self):
        with self.lock:
            return dict(self.counts,entries=len(self.entries),retained_bytes=self.retained,max_bytes=self.max_bytes,compute_mode=self.compute_mode)

    def read_compute(self, expert):
        if type(expert) is not int or not 0<=expert<self.geometry.num_experts:
            raise ValueError('Real CPU expert ID required')
        owner=False
        with self.lock:
            if expert in self.entries:
                self.counts['hits']+=1;self.entries.move_to_end(expert)
                return self.entries[expert]
            if expert in self.inflight:
                future=self.inflight[expert];self.counts['coalesced']+=1
            else:
                future=Future();self.inflight[expert]=future;self.counts['misses']+=1;owner=True
        if not owner:
            return future.result()
        try:
            with torch.inference_mode():
                raw=self.source.read(expert,device=torch.device('cpu'),dtype=torch.float16)
                shapes=((2*self.geometry.intermediate_size,self.geometry.hidden_size),
                        (self.geometry.hidden_size,self.geometry.intermediate_size))
                if len(raw)!=2 or any(w.device.type!='cpu' or w.dtype!=torch.float16 or tuple(w.shape)!=shape
                                      for w,shape in zip(raw,shapes)):
                    raise ValueError('CPU source FP16 weight boundary differs')
                dtype=torch.float16 if self.compute_mode=='native-half' else torch.float32
                values=tuple(w.to(dtype=dtype,copy=True).contiguous() for w in raw)
            size=sum(w.numel()*w.element_size() for w in values)
            with self.lock:
                if size<=self.max_bytes:
                    while self.retained+size>self.max_bytes:
                        _,old=self.entries.popitem(last=False);self.retained-=sum(w.numel()*w.element_size() for w in old);self.counts['evictions']+=1
                    self.entries[expert]=values;self.retained+=size
                else:
                    self.counts['bypasses']+=1
                future.set_result(values);del self.inflight[expert]
            return values
        except BaseException as error:
            with self.lock:
                future.set_exception(error);del self.inflight[expert]
            raise

    def clear(self):
        with self.lock:
            if self.inflight:
                raise RuntimeError('Do not clear CPU cache while expert jobs are active')
            self.entries.clear();self.retained=0


class ColdExpertPool:
    """One bounded pool shared by ALL layers, not workers-per-layer/per-expert."""

    def __init__(self, workers, *, trace=False):
        if type(workers) is not int or not 1 <= workers <= 8 or torch.get_num_threads() != 1:
            raise ValueError('1..8 CPU workers and explicit Torch intra-op threads1 required')
        self.workers = workers
        self.trace = bool(trace)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='venturi-cold-expert')
        self.lock, self.active, self.closed = threading.Lock(), 0, False
        self.stats = dict(tasks=0, peak_active_workers=0, thread_cpu_seconds=0., task_wall_seconds=0.)

    def _run(self, source, expert, inputs):
        wall, cpu = time.perf_counter(), time.thread_time()
        with self.lock:
            self.active += 1
            self.stats['peak_active_workers'] = max(self.stats['peak_active_workers'], self.active)
        try:
            # Optional pure NVTX annotation, only in separate profiler runs;
            # never submits CUDA work from a CPU worker.
            with torch.cuda.nvtx.range('venturi.hybrid.cold-expert') if self.trace else nullcontext():
                return cpu_expert(source, expert, inputs)
        finally:
            with self.lock:
                self.active -= 1
                self.stats['tasks'] += 1
                self.stats['thread_cpu_seconds'] += time.thread_time()-cpu
                self.stats['task_wall_seconds'] += time.perf_counter()-wall

    def submit(self, source, expert, inputs):
        if self.closed:
            raise RuntimeError('Cold expert pool has been closed')
        return self.executor.submit(self._run, source, expert, inputs)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)


class FixedHotBank:
    """Fixed immutable weights; cold reads NEVER evict a hot expert."""

    def __init__(self, source, expert_ids, *, device, dtype, max_bytes):
        if not isinstance(getattr(source, 'geometry', None), MoEGeometry):
            raise ValueError('Explicit source expert geometry required')
        ids = tuple(expert_ids)
        if len(set(ids)) != len(ids) or any(type(e) is not int or not 0 <= e < source.geometry.num_experts for e in ids):
            raise ValueError('Unique real hot expert IDs required')
        self.geometry, self.source = source.geometry, source
        self.device, self.dtype = torch.device(device), dtype
        if dtype != torch.float16 or self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Explicit FP16 execution device required')
        size = 3*self.geometry.hidden_size*self.geometry.intermediate_size*2
        if type(max_bytes) is not int or max_bytes < size*len(ids):
            raise ValueError('Explicit hot expert byte budget exceeded')
        self.weights = {}
        self.stats = dict(hits=0, misses=0)
        for expert in sorted(ids):
            values = source.read(expert, device=self.device, dtype=dtype)
            expected = ((2*self.geometry.intermediate_size, self.geometry.hidden_size),
                        (self.geometry.hidden_size, self.geometry.intermediate_size))
            if len(values) != 2 or any(tuple(w.shape) != shape or w.dtype != dtype or w.device != self.device
                                      for w, shape in zip(values, expected)):
                raise ValueError('Hot source returned incompatible matrices')
            self.weights[expert] = tuple(w.clone(memory_format=torch.contiguous_format) for w in values)
        self.retained_bytes = len(ids)*size

    def read(self, expert, *, device, dtype):
        if torch.device(device) != self.device or dtype != self.dtype:
            raise ValueError('Fixed expert bank execution boundary changed')
        self.stats['hits' if expert in self.weights else 'misses'] += 1
        return self.weights[expert] if expert in self.weights else self.source.read(expert, device=device, dtype=dtype)

    def clear(self):
        self.weights.clear()


class HybridExpertDispatch:
    """Submit cold jobs, launch hot GEMMs, pack results, reduce ORIGINAL order.

    A CPU device can exercise the scheduling/rounding contract in unit tests;
    that is not evidence of real CPU/GPU overlap. CUDA capture is unsupported.
    Inputs/weights immutable, synchronous request boundary; no detached jobs on
    error and no cross-request mutable activation/result buffers.
    """

    def __init__(self, hot_bank, cpu_source, pool, *, max_tokens=256):
        if not isinstance(hot_bank, FixedHotBank) or cpu_source.geometry != hot_bank.geometry or (
                not isinstance(pool, ColdExpertPool)) or type(max_tokens) is not int or not 1 <= max_tokens <= 256:
            raise ValueError('Explicit bounded qualified hot/cold expert binding required')
        self.hot, self.cpu_source, self.pool = hot_bank, cpu_source, pool
        self.geometry, self.max_tokens = hot_bank.geometry, max_tokens
        self.trace = False
        self.stats = dict(calls=0, hot_expert_groups=0, cold_expert_groups=0, hot_token_expert_pairs=0,
                          cold_token_expert_pairs=0, cold_activation_d2h_bytes=0, cold_output_h2d_bytes=0)

    @torch.inference_mode()
    def __call__(self, hidden_states, indices, scores):
        g = self.geometry
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != g.hidden_size or (
                hidden_states.shape[0] > self.max_tokens or indices.shape != (hidden_states.shape[0], g.top_k)
                or scores.shape != indices.shape or indices.dtype != torch.int64 or
                hidden_states.dtype != self.hot.dtype or scores.dtype != hidden_states.dtype or
                any(t.device != self.hot.device for t in (hidden_states, indices, scores))):
            raise ValueError('Hybrid routed shape/device/dtype/token budget mismatch')
        if hidden_states.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise ValueError('CPU expert dispatch cannot be CUDA graph captured')
        # One route transfer, not per-expert .item()/GPU where/nonzero barriers.
        host_ids = indices.detach().to('cpu')
        if host_ids.numel() and (host_ids.min().item() < 0 or host_ids.max().item() >= g.num_experts):
            raise ValueError('Real expert IDs required; never prune/substitute')
        groups = []
        for expert in torch.unique(host_ids, sorted=True).tolist():
            positions, tokens = torch.where((host_ids == expert).T)
            groups.append((expert, positions, tokens))
        output = torch.zeros_like(hidden_states)
        if not groups:
            return output
        has_cold = any(e not in self.hot.weights for e, _, _ in groups)
        host_x = hidden_states.detach().to('cpu') if has_cold else None
        if has_cold:
            self.stats['cold_activation_d2h_bytes'] += hidden_states.numel()*hidden_states.element_size()
        # Pack route indices once; per-expert slices preserve original K-major
        # token ordering, including multi-token prefill routes.
        token_ids = torch.cat([t for _, _, t in groups]).to(hidden_states.device)
        positions = torch.cat([p for _, p, _ in groups]).to(hidden_states.device)
        futures, values, ranges = {}, {}, {}
        offset = 0
        try:
            for expert, _, tokens in groups:
                ranges[expert] = slice(offset, offset+len(tokens))
                offset += len(tokens)
                if expert not in self.hot.weights:
                    futures[expert] = self.pool.submit(self.cpu_source, expert, host_x[tokens])
            with torch.inference_mode():
                for expert, _, _ in groups:
                    if expert not in self.hot.weights:
                        continue
                    rows = token_ids[ranges[expert]]
                    gate_up, down = self.hot.weights[expert]
                    with torch.cuda.nvtx.range('venturi.hybrid.hot-expert') if self.trace else nullcontext():
                        gate, up = F.linear(hidden_states[rows], gate_up).chunk(2, -1)
                        values[expert] = F.linear(F.silu(gate)*up, down)
                if futures:
                    cold_ids = sorted(futures)
                    packed = torch.cat([futures[e].result() for e in cold_ids], 0)
                    self.stats['cold_output_h2d_bytes'] += packed.numel()*packed.element_size()
                    packed = packed.to(hidden_states.device)
                    offset = 0
                    for expert in cold_ids:
                        count = ranges[expert].stop-ranges[expert].start
                        values[expert] = packed[offset:offset+count]
                        offset += count
                # Never reduce by completion order or drop a slow cold branch.
                for expert, _, tokens in groups:
                    section = ranges[expert]
                    rows = token_ids[section]
                    value = values[expert]*scores[rows, positions[section], None]
                    output.index_add_(0, rows, value.to(output.dtype))
                    kind = 'hot' if expert in self.hot.weights else 'cold'
                    self.stats[kind+'_expert_groups'] += 1
                    self.stats[kind+'_token_expert_pairs'] += len(tokens)
                self.stats['calls'] += 1
                return output
        finally:
            # Model/request exceptions must not leave CPU tasks running against
            # a closed source or mutate a later request's scheduler counters.
            for future in futures.values():
                if not future.done():
                    future.cancel()
            for future in futures.values():
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass


class RouteCounter:
    """UNscored calibration only; count tokens, not unique experts per batch."""

    def __init__(self, source):
        self.source, self.counts = source, [0]*source.geometry.num_experts

    def __call__(self, x, ids, scores):
        from .flash_next_moe import routed_experts
        counts = torch.bincount(ids.reshape(-1), minlength=len(self.counts)).cpu().tolist()
        self.counts = [a+b for a, b in zip(self.counts, counts)]
        return routed_experts(x, ids, scores, self.source)

    def top(self, count):
        if type(count) is not int or not 0 <= count <= len(self.counts):
            raise ValueError('Explicit bounded hot expert count required')
        return tuple(sorted(sorted(range(len(self.counts)), key=lambda e: (-self.counts[e], e))[:count]))


class HybridBinding:
    """Own the shared worker pool/hot banks and release them at setup boundaries."""

    def __init__(self, core, banks, dispatches, pool, mode):
        self.core, self.banks, self.dispatches, self.pool, self.mode = core, banks, dispatches, pool, mode

    def stats(self):
        return dict(mode=self.mode, cpu_pool=None if self.pool is None else dict(self.pool.stats),
            layers=[dict(layer=i, hot_ids=sorted(bank.weights), retained_bytes=bank.retained_bytes,
                         bank=dict(bank.stats), dispatch=None if dispatch is None else dict(dispatch.stats),
                         cpu_cache=None if dispatch is None else dispatch.cpu_source.stats)
                    for i,(bank,dispatch) in enumerate(zip(self.banks,self.dispatches))])

    def close(self):
        if self.pool is not None:
            self.pool.close()
        for layer,bank,dispatch in zip(self.core.layers,self.banks,self.dispatches):
            if layer.mlp.expert_dispatch is dispatch:
                layer.mlp.expert_dispatch = None
            if layer.mlp.expert_store is bank:
                layer.mlp.expert_store = bank.source
            bank.clear()
            if dispatch is not None:
                dispatch.cpu_source.clear()


def bind_fixed_expert_policy(core, checkpoint, *, mode, hot_ids_per_layer,
                             hot_bytes_per_layer, workers=4, library=None,cpu_cache_bytes_per_layer=0,
                             cpu_compute_mode='staged-fp32'):
    """Explicit eager setup: matched fixed-hot GPU-cold vs parallel CPU-cold.

    Caller must first close any PREVIOUS HybridBinding and all active requests.
    No engine/default switch or OOM fallback. CPU mode is only for fixture tests.
    Core execution dtype FP16; CPU experts staged-FP32 GEMMs with FP16 boundaries.
    """
    from .flash_next_expert_policy import bind_fp8_expert_policy
    from .flash_next_fp8 import BlockFP8ExpertStore
    if mode not in ('fixed-hot-gpu-cold', 'fixed-hot-cpu-cold'):
        raise ValueError('Explicit matched fixed-hot mode required')
    hot = tuple(tuple(ids) for ids in hot_ids_per_layer)
    if len(hot) != len(core.layers) or type(hot_bytes_per_layer) is not int or hot_bytes_per_layer < 0 or (
            type(cpu_cache_bytes_per_layer) is not int or not 0<=cpu_cache_bytes_per_layer<=2**30):
        raise ValueError('One explicit bounded hot set per layer required')
    parameter = core.embed_tokens.weight
    if parameter.dtype != torch.float16 or any(layer.mlp.expert_dispatch is not None for layer in core.layers):
        raise ValueError('FP16 core with no active experimental dispatch required')
    # Drop only retained optimization state before another eager setup. Default
    # source binding remains numerically valid if subsequent setup fails.
    for layer in core.layers:
        if hasattr(layer.mlp.expert_store, 'clear'):
            layer.mlp.expert_store.clear()
    bind_fp8_expert_policy(core,checkpoint,
        backend='cuda-byte-decode' if parameter.device.type=='cuda' else 'cpu-vectorized',
        retained_bytes_per_layer=(0,)*len(core.layers),library=library)
    pool = ColdExpertPool(workers) if mode=='fixed-hot-cpu-cold' else None
    banks,dispatches = [],[]
    try:
        for i,(layer,ids) in enumerate(zip(core.layers,hot)):
            source=layer.mlp.expert_store
            bank=FixedHotBank(source,ids,device=parameter.device,dtype=parameter.dtype,max_bytes=hot_bytes_per_layer)
            banks.append(bank)
            cpu=BlockFP8ExpertStore(checkpoint,source.geometry,
                f'model.language_model.layers.{i}.mlp.experts',decode_strategy='vectorized') if pool else None
            if cpu is not None:
                cpu=CPUDecodedExpertCache(cpu,max_bytes=cpu_cache_bytes_per_layer,compute_mode=cpu_compute_mode)
            dispatches.append(HybridExpertDispatch(bank,cpu,pool,max_tokens=min(core.max_sequence,256)) if pool else None)
        if parameter.device.type=='cuda':
            torch.cuda.synchronize(parameter.device)
        for layer,bank,dispatch in zip(core.layers,banks,dispatches):
            layer.mlp.expert_store,layer.mlp.expert_dispatch=bank,dispatch
        return HybridBinding(core,banks,dispatches,pool,mode)
    except BaseException:
        if pool is not None:
            pool.close()
        for bank in banks:
            bank.clear()
        raise
