"""Compact CPU semantic Engram hash state, not lookup or speculative decoding.

Consumes explicit checkpoint-bound token maps/moduli/multipliers. Does NOT
regenerate tokenizer normalization, primes or RNG parameters, load weights, or
run a native model. Scalar integer equations are independent of the reference's
vectorized cache. Each state owns a fixed batch of equal-length sequences.
"""
from dataclasses import dataclass
import copy
import math

import torch


_I64=2**63-1


def _integer(value, low, high):
    return type(value) is int and low<=value<=high


@dataclass(frozen=True)
class EngramHashSpec:
    token_map: tuple[int,...]
    compressed_vocab_size: int
    pad_token_id: int
    layer_ids: tuple[int,...]
    max_ngram_size: int
    moduli: tuple[tuple[tuple[int,...],...],...]
    multipliers: tuple[tuple[int,...],...]

    def __post_init__(self):
        object.__setattr__(self,'token_map',tuple(self.token_map))
        object.__setattr__(self,'layer_ids',tuple(self.layer_ids))
        object.__setattr__(self,'moduli',tuple(tuple(tuple(row) for row in layer) for layer in self.moduli))
        object.__setattr__(self,'multipliers',tuple(tuple(row) for row in self.multipliers))
        if (not 1<=len(self.token_map)<=2**20
                or not _integer(self.compressed_vocab_size,1,len(self.token_map))
                or not _integer(self.pad_token_id,0,len(self.token_map)-1)
                or not all(_integer(x,0,self.compressed_vocab_size-1) for x in self.token_map)
                or set(self.token_map)!=set(range(self.compressed_vocab_size))):
            raise ValueError('Explicit complete compressed token map and raw pad token required')
        if (not 1<=len(self.layer_ids)<=64 or len(set(self.layer_ids))!=len(self.layer_ids)
                or not all(_integer(x,0,2**31-1) for x in self.layer_ids)
                or not _integer(self.max_ngram_size,2,8)):
            raise ValueError('Unique layer IDs and bounded ngram size required')
        n=self.max_ngram_size
        if len(self.moduli)!=len(self.layer_ids) or len(self.multipliers)!=len(self.layer_ids):
            raise ValueError('Hash layers must match explicit layout')
        heads=None
        for layer,multipliers in zip(self.moduli,self.multipliers):
            if len(layer)!=n-1 or len(multipliers)!=n:
                raise ValueError('Hash lookback dimensions differ')
            for row in layer:
                if heads is None:heads=len(row)
                if not 1<=len(row)==heads<=64 or not all(_integer(p,2,_I64) for p in row):
                    raise ValueError('Matching positive bucket moduli/head counts required')
            if sum(p for row in layer for p in row)>_I64:
                raise ValueError('Engram table row offsets overflow int64')
            if not all(_integer(m,1,_I64) and m%2 and (self.compressed_vocab_size-1)*m<=_I64 for m in multipliers):
                raise ValueError('Odd hash multipliers must not overflow int64 products')
        # Moduli are provided, not proven prime or regenerated from a seed.
        # Full checkpoint/tokenizer identity is an independent admission gate.

    @property
    def offsets(self):
        result=[]
        for layer in self.moduli:
            offsets=[];cursor=0
            for row in layer:
                for modulus in row:offsets.append(cursor);cursor+=modulus
            result.append(tuple(offsets))
        return tuple(result)

    @property
    def table_rows(self):
        return tuple(sum(p for row in layer for p in row) for layer in self.moduli)


class EngramHashState:
    """CPU oracle with N-1 tokens of private carry per fixed sequence slot.

    Chunks must be contiguous, same fixed batch and within max_sequence_tokens.
    No uninitialized cache reads, arbitrary rewind or implicit slot reuse. Use
    independent/forked states for independent requests; this is not a scheduler.
    """
    DEAD=-1

    def __init__(self,spec,*,batch_size,max_sequence_tokens):
        if not isinstance(spec,EngramHashSpec):raise TypeError('Explicit EngramHashSpec required')
        if not _integer(batch_size,1,64) or not _integer(max_sequence_tokens,1,2**20):
            raise ValueError('Positive bounded fixed batch/context required')
        self.spec=spec;self.batch_size=batch_size;self.max_sequence_tokens=max_sequence_tokens
        self.reset()

    def reset(self):
        self.position=0;self._history=[[] for _ in range(self.batch_size)]

    def fork(self):
        child=EngramHashState(self.spec,batch_size=self.batch_size,max_sequence_tokens=self.max_sequence_tokens)
        child.position=self.position;child._history=copy.deepcopy(self._history)
        return child

    def __call__(self,input_ids,start_pos,token_mask=None):
        if (not isinstance(input_ids,torch.Tensor) or input_ids.device.type!='cpu'
                or input_ids.dtype!=torch.int64 or input_ids.ndim!=2
                or input_ids.shape[0]!=self.batch_size or not input_ids.shape[1]
                or bool(((input_ids<0)|(input_ids>=len(self.spec.token_map))).any())):
            raise ValueError('Valid fixed-batch nonempty CPU int64 token IDs required')
        if (not _integer(start_pos,0,self.max_sequence_tokens) or start_pos!=self.position
                or start_pos+input_ids.shape[1]>self.max_sequence_tokens):
            raise ValueError('Contiguous positions within explicit sequence capacity required')
        if token_mask is not None and (not isinstance(token_mask,torch.Tensor)
                or token_mask.device.type!='cpu' or token_mask.dtype!=torch.bool
                or token_mask.shape!=input_ids.shape):
            raise ValueError('Optional boolean CPU token mask must match IDs')
        spec=self.spec;pad=spec.token_map[spec.pad_token_id];n=spec.max_ngram_size
        offsets=spec.offsets;history=copy.deepcopy(self._history);result=[]
        ids=input_ids.tolist();mask=token_mask.tolist() if token_mask is not None else None
        for row,tokens in enumerate(ids):
            outputs=[]
            for column,token in enumerate(tokens):
                current=spec.token_map[token] if mask is None or mask[row][column] else self.DEAD
                history[row].append(current)
                blocked=False;lookback=[]
                for shift in range(n):
                    previous=history[row][-1-shift] if shift<len(history[row]) else self.DEAD
                    blocked=blocked or previous==self.DEAD
                    lookback.append(pad if blocked else previous)
                layers=[]
                for layer,multipliers in enumerate(spec.multipliers):
                    rolling=lookback[0]*multipliers[0];hashes=[];index=0
                    for shift in range(1,n):
                        rolling^=lookback[shift]*multipliers[shift]
                        for modulus in spec.moduli[layer][shift-1]:
                            hashes.append(rolling%modulus+offsets[layer][index]);index+=1
                    layers.append(hashes)
                outputs.append(layers)
                history[row]=history[row][-(n-1):]
            result.append(outputs)
        output=torch.tensor(result,dtype=torch.int64)
        self._history=history;self.position+=input_ids.shape[1]
        return output


def bind_engram_hash_metadata(metadata, *, layer_ids, max_ngram_size, n_heads,
                              compressed_vocab_size, pad_token_id, vocab_size, table_rows):
    """Bind IST-style integer tensors to an explicit architecture hash layout.

    `metadata` must come from the full-SHA reader, not a header inventory.
    The caller must independently authenticate architecture/tokenizer identity,
    layer order and table-row counts; this function does not certify them.
    No tokenizer normalization, prime or multiplier regeneration/fallback.
    """
    from venturi.gguf_metadata import VerifiedIntegerMetadata
    if not isinstance(metadata,VerifiedIntegerMetadata):
        raise TypeError('Full-source-verified integer metadata required')
    layer_ids=tuple(layer_ids);table_rows=tuple(table_rows)
    if (not 1<=len(layer_ids)<=64 or not _integer(max_ngram_size,2,8)
            or not _integer(n_heads,1,64) or not _integer(vocab_size,1,2**20)):
        raise ValueError('Explicit bounded architecture layout required')
    layers=len(layer_ids);cols=(max_ngram_size-1)*n_heads
    if len(table_rows)!=layers or not all(_integer(r,1,_I64) for r in table_rows):
        raise ValueError('Explicit integer lookup row counts required')
    schema={
        'engram_token_map.weight':(26,(vocab_size,)),
        'engram_primes.weight':(27,(layers,cols)),
        'engram_offsets.weight':(27,(layers,cols)),
        'engram_multipliers.weight':(27,(layers,max_ngram_size)),
    }
    tensors={}
    for name,(typ,shape) in schema.items():
        try:t=metadata.tensor(name)
        except StopIteration as e:raise ValueError('Missing required Engram integer tensor: '+name) from e
        if t.type_id!=typ or t.shape!=shape or len(t.values)!=math.prod(shape):
            raise ValueError('Engram integer dtype/shape binding differs: '+name)
        tensors[name]=t.values
    flat=tensors['engram_primes.weight']
    moduli=tuple(tuple(tuple(flat[layer*cols+order*n_heads+head] for head in range(n_heads))
        for order in range(max_ngram_size-1)) for layer in range(layers))
    flat_m=tensors['engram_multipliers.weight']
    multipliers=tuple(tuple(flat_m[layer*max_ngram_size:(layer+1)*max_ngram_size]) for layer in range(layers))
    spec=EngramHashSpec(tensors['engram_token_map.weight'],compressed_vocab_size,pad_token_id,
        layer_ids,max_ngram_size,moduli,multipliers)
    if table_rows!=spec.table_rows:
        raise ValueError('Architecture lookup row counts differ from stored bucket moduli')
    offsets=tuple(v for row in spec.offsets for v in row)
    if tensors['engram_offsets.weight']!=offsets:
        raise ValueError('Stored Engram offsets differ from cumulative bucket layout')
    return spec
