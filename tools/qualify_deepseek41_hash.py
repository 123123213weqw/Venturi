"""Hash-bound scalar CPU hash/carry qualification, not tokenizer or model support.

Runs only official NgramHashState.forward with explicit synthetic buffers. No
source module import, prime/RNG generation, tokenizer normalization, weight row
lookup, GPU or full model execution. Model tables still require their own SHA.
"""
import argparse
import ast
import copy
import hashlib
import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from venturi.models.engram_hash import EngramHashSpec,EngramHashState,bind_engram_hash_metadata
from venturi.gguf_metadata import read_verified_integer_metadata
from venturi.model_assets import ModelAsset


def bind_fixture(spec):
    """Generated tiny integer-only GGUF; no real weight/table payloads."""
    layers=len(spec.layer_ids);cols=(spec.max_ngram_size-1)*len(spec.moduli[0][0])
    tensors=[('engram_token_map.weight',26,(len(spec.token_map),),spec.token_map),
        ('engram_primes.weight',27,(cols,layers),tuple(v for layer in spec.moduli for row in layer for v in row)),
        ('engram_offsets.weight',27,(cols,layers),tuple(v for row in spec.offsets for v in row)),
        ('engram_multipliers.weight',27,(spec.max_ngram_size,layers),tuple(v for row in spec.multipliers for v in row))]
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),0);payload=bytearray()
    for name,typ,dims,values in tensors:
        label=name.encode();header+=struct.pack('<Q',len(label))+label+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',typ,len(payload))
        fmt='i' if typ==26 else 'q';payload.extend(struct.pack('<'+fmt*len(values),*values));payload.extend(bytes((-len(payload))%32))
    raw=header+bytes((-len(header))%32)+payload
    sha=hashlib.sha256(raw).hexdigest();revision='0'*40
    # Synthetic manifest URL is never fetched; it only exercises typed source
    # identity admission of the generated fixture, NOT official model assets.
    asset=ModelAsset('params.gguf',len(raw),sha,
        f'https://huggingface.co/venturi-fixtures/engram/resolve/{revision}/params.gguf',
        'venturi-fixtures/engram',revision,'params.gguf','metadata')
    with tempfile.TemporaryDirectory(prefix='venturi-hash-integer-fixture-') as directory:
        (Path(directory)/asset.filename).write_bytes(raw)
        metadata=read_verified_integer_metadata(asset,directory,[t[0] for t in tensors])
        bound=bind_engram_hash_metadata(metadata,layer_ids=spec.layer_ids,max_ngram_size=spec.max_ngram_size,
            n_heads=len(spec.moduli[0][0]),compressed_vocab_size=spec.compressed_vocab_size,
            pad_token_id=spec.pad_token_id,vocab_size=len(spec.token_map),table_rows=spec.table_rows)
    assert bound==spec
    return bound,dict(scope='Generated integer-only fixture, NOT real checkpoint',sha256=sha,bytes=len(raw))


def qualify(source, *, binding_fixtures=False):
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NgramHashState')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    fn.decorator_list=[];scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<hashed official Engram hash forward>','exec'),scope)
    records=[]
    for n,heads in ((3,2),(4,8)):
        moduli=tuple(tuple(tuple(16_000_001+layer*1000+order*100+head*2 for head in range(heads)) for order in range(n-1)) for layer in range(2))
        # Explicit synthetic moduli: primality/prime generation is NOT checked.
        for edge in (False,True):
            base=(2**63-1)//2 if edge else 101
            multipliers=tuple(tuple(base-layer*10-shift*2 for shift in range(n)) for layer in range(2))
            spec=EngramHashSpec((0,1,2,1,0,2),3,3,(1,14),n,moduli,multipliers)
            fixture=None
            if binding_fixtures:spec,fixture=bind_fixture(spec)
            for b in (1,2):
                ids=torch.tensor([[1,4,3,5,1,2,3,4,0]]*b,dtype=torch.int64)
                mixed=torch.ones_like(ids,dtype=torch.bool);mixed[:,2:4]=False;mixed[-1,0]=False
                for mask in (None,mixed,torch.zeros_like(ids,dtype=torch.bool)):
                    for chunks in ((9,),(1,)*9,(2,3,1,3)):
                        actual=EngramHashState(spec,batch_size=b,max_sequence_tokens=9)
                        fake=SimpleNamespace(layout=SimpleNamespace(max_ngram_size=n),DEAD=-1,pad_id=spec.token_map[spec.pad_token_id],
                            token_map=torch.tensor(spec.token_map),cache=torch.empty(b,9,dtype=torch.int64),
                            primes=torch.tensor(spec.moduli),multipliers=torch.tensor(spec.multipliers),offsets=torch.tensor(spec.offsets))
                        position=0;all_equal=True
                        for length in chunks:
                            tokens=ids[:,position:position+length];m=mask[:,position:position+length] if mask is not None else None
                            observed=actual(tokens,position,m);expected=scope['forward'](fake,tokens,position,m)
                            all_equal=all_equal and torch.equal(observed,expected)
                            position+=length
                        records.append(dict(ngram=n,heads=heads,batch=b,near_int64_limit=edge,
                            mask='none' if mask is None else mask.tolist(),chunks=chunks,
                            byte_exact=all_equal,position=actual.position,
                            carried_tokens=sum(len(h) for h in actual._history),binding_fixture=fixture))
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source',required=True);p.add_argument('--reference-sha256',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--binding-fixtures',action='store_true',help='Use full-SHA generated integer GGUF fixtures before hash/carry comparison, not real weights')
    a=p.parse_args(argv)
    raw=Path(a.reference_source).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=a.reference_sha256:raise ValueError('Official source SHA256 mismatch')
    cases=qualify(raw.decode(),binding_fixtures=a.binding_fixtures)
    result=dict(scope='Synthetic explicit-map CPU hash/chunk/carry semantics only; NOT tokenizer normalization, prime/RNG generation, checkpoint/hash-table lookup, GPU, model correctness or TPS',
        official_repository='deepseek-ai/DeepSeek-V4.1-Flash-1017',official_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',
        official_source_sha256=sha,binding_fixtures=a.binding_fixtures,torch_version=torch.__version__,all_byte_exact=all(r['byte_exact'] for r in cases),cases=cases)
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_byte_exact']:raise SystemExit(1)


if __name__=='__main__':main()
