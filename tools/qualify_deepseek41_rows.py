"""CPU generated-table hash -> addressed IQ4_NL rows qualification, NOT real weights.

Official hash-forward and pinned llama.cpp dequantization are independent
oracles. Only their selected SHA-verified functions execute. Table payloads and
hash parameters are synthetic; no download, real model/GPU or speed claim.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from venturi.gguf_rows import VerifiedIQ4NLRows
from venturi.model_assets import ModelAsset
from venturi.models.engram_hash import EngramHashSpec, EngramHashState
from venturi.models.engram_projection import reference_engram_from_rows
from venturi.gguf_quantization import decode_bf16_rows,decode_q5_k_rows
from qualify_deepseek41_tokenizer import SOURCE_SHA256 as HASH_SOURCE_SHA256
from qualify_gguf_iq4nl import _reference as iq4_reference, SOURCE_SHA256 as IQ4_SOURCE_SHA256
from qualify_gguf_projection_formats import references as projection_references


MODEL_SOURCE_SHA256='e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'


def _engram_reference(source):
    if hashlib.sha256(source).hexdigest()!=MODEL_SOURCE_SHA256:
        raise ValueError('Pinned official Engram model source SHA256 differs')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engram')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    fn.decorator_list=[];scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 '<SHA-pinned official Engram forward>','exec'),scope)
    return scope['forward']


def _projection_fixture(columns,quant_source):
    q5,bf16=projection_references(quant_source);width=columns*256;copies=4;hidden=8
    rng=np.random.default_rng(417162+columns)
    raw=b''.join(struct.pack('<ee',2**-12,2**-13)+rng.integers(0,256,size=172,dtype=np.uint8).tobytes()
                 for _ in range((copies+1)*hidden*width//256))
    actual_w=decode_q5_k_rows(raw,width)
    golden_w=torch.from_numpy(q5(np.frombuffer(raw,dtype=np.uint8).reshape(-1,176))
                              .reshape((copies+1)*hidden,width).copy())
    actual_qk=[];golden_qk=[];hashes=[hashlib.sha256(raw).hexdigest()]
    for offset in (0,7):
        words=[(0x3e00+((i+offset)%11)*0x40)|((i%3==0)*0x8000) for i in range(copies*hidden)]
        packed=struct.pack('<'+'H'*len(words),*words);hashes.append(hashlib.sha256(packed).hexdigest())
        actual_qk.append(decode_bf16_rows(packed,hidden))
        golden_qk.append(torch.from_numpy(bf16(np.frombuffer(packed,dtype=np.uint8).reshape(-1,2))
                                         .reshape(copies,hidden).copy()))
    exact=torch.equal(actual_w.view(torch.int32),golden_w.view(torch.int32))
    exact=exact and all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(actual_qk,golden_qk))
    return (actual_w,*actual_qk),(golden_w,*golden_qk),dict(scope='Generated packed Q5_K WKV/BF16 q+k only',
        projection_input=width,hidden=hidden,copies=copies,packed_weight_sha256=hashes,decoded_byte_exact=exact)


def _hash_reference(source):
    if hashlib.sha256(source).hexdigest()!=HASH_SOURCE_SHA256:
        raise ValueError('Pinned official hash source SHA256 differs')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NgramHashState')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    fn.decorator_list=[];scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 '<SHA-pinned official hash forward>','exec'),scope)
    return scope['forward']


def _table_fixture(root,spec,dequantize):
    width=256;header=b'GGUF'+struct.pack('<IQQ',3,len(spec.layer_ids),0)
    payload=bytearray();golden={};schemas={}
    scales=(1.,-2.,2**-24,0.,-0.,65504.,.125)
    for layer,rows in zip(spec.layer_ids,spec.table_rows):
        name=f'blk.{layer}.engram_embd.weight';offset=len(payload)
        raw=b''.join(struct.pack('<e',scales[(row+block)%len(scales)])
            +bytes((row*73+column*17+layer+block)&255 for column in range(16))
            for row in range(rows) for block in range(width//32))
        # Full decode of these TINY generated tables only, using the independent
        # selected primary method. Never full-decode a real 110 GB table.
        expected=dequantize(np.frombuffer(raw,dtype=np.uint8).reshape(-1,18)).reshape(rows,width)
        golden[name]=torch.from_numpy(expected.copy());schemas[name]=(rows,width)
        label=name.encode();header+=struct.pack('<Q',len(label))+label+struct.pack('<IQQIQ',2,width,rows,20,offset)
        payload.extend(raw);payload.extend(bytes((-len(payload))%32))
    data=header+bytes((-len(header))%32)+payload
    name='GENERATED-rows.gguf';(root/name).write_bytes(data);rev='0'*40;repo='venturi-fixtures/engram-rows'
    asset=ModelAsset(name,len(data),hashlib.sha256(data).hexdigest(),
        f'https://huggingface.co/{repo}/resolve/{rev}/{name}',repo,rev,name,'sparse_lookup')
    return asset,schemas,golden


def qualify(hash_source,quant_source,model_source=None):
    official_hash=_hash_reference(hash_source);dequantize=iq4_reference(quant_source)
    official_engram=_engram_reference(model_source) if model_source is not None else None
    stream_rng=torch.Generator().manual_seed(417163)
    records=[];fixtures=[]
    primes=(3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97)
    for n,heads in ((3,2),(4,8)):
        moduli=tuple(tuple(tuple(primes[order*heads+head]+layer*100 for head in range(heads))
                           for order in range(n-1)) for layer in range(2))
        # The second layer's supplied moduli are intentionally not certified
        # prime. Parameter generation is NOT under test; table-row sums are.
        for edge in (False,True):
            base=(2**63-1)//2 if edge else 101
            multipliers=tuple(tuple(base-layer*10-shift*2 for shift in range(n)) for layer in range(2))
            spec=EngramHashSpec((0,1,2,1,0,2),3,3,(1,14),n,moduli,multipliers)
            projection=None
            if official_engram is not None:
                weights,golden_weights,projection=_projection_fixture((n-1)*heads,quant_source)
            with tempfile.TemporaryDirectory(prefix='venturi-generated-row-qualification-') as directory:
                root=Path(directory);asset,schemas,golden=_table_fixture(root,spec,dequantize)
                fixtures.append(dict(ngram=n,heads=heads,near_int64_limit=edge,
                    bytes=asset.bytes,sha256=asset.sha256,rows=spec.table_rows,width=256,projection=projection))
                with VerifiedIQ4NLRows(asset,root,expected_shapes=schemas) as lookup:
                    for batch in (1,2):
                        ids=torch.tensor([[1,4,3,5,1,2,3,4,0]]*batch,dtype=torch.int64)
                        mask=torch.ones_like(ids,dtype=torch.bool);mask[:,2:4]=False;mask[-1,0]=False
                        for token_mask in (None,mask,torch.zeros_like(ids,dtype=torch.bool)):
                            for chunks in ((9,),(1,)*9,(2,3,1,3)):
                                actual=EngramHashState(spec,batch_size=batch,max_sequence_tokens=9)
                                reference=SimpleNamespace(layout=SimpleNamespace(max_ngram_size=n),DEAD=-1,
                                    pad_id=spec.token_map[spec.pad_token_id],token_map=torch.tensor(spec.token_map),
                                    cache=torch.empty(batch,9,dtype=torch.int64),primes=torch.tensor(spec.moduli),
                                    multipliers=torch.tensor(spec.multipliers),offsets=torch.tensor(spec.offsets))
                                position=0;hash_exact=True;row_exact=True;projection_exact=True
                                fetched=0;calls=0;requested=0
                                for length in chunks:
                                    tokens=ids[:,position:position+length]
                                    active=token_mask[:,position:position+length] if token_mask is not None else None
                                    hashes=actual(tokens,position,active)
                                    expected_hashes=official_hash(reference,tokens,position,active)
                                    hash_exact=hash_exact and torch.equal(hashes,expected_hashes)
                                    for index,layer in enumerate(spec.layer_ids):
                                        name=f'blk.{layer}.engram_embd.weight'
                                        addressed=hashes[:,:,index,:]
                                        a,b=lookup.lookup_many(name,(addressed,addressed.clone()))
                                        expected=golden[name][expected_hashes[:,:,index,:]]
                                        row_exact=row_exact and torch.equal(a.view(torch.int32),expected.view(torch.int32))
                                        row_exact=row_exact and torch.equal(b.view(torch.int32),expected.view(torch.int32))
                                        if official_engram is not None:
                                            stream=torch.randn(batch,length,4,8,generator=stream_rng)
                                            candidate=reference_engram_from_rows(stream,a,*weights,norm_epsilon=1e-20,token_mask=active)
                                            w,q,k=golden_weights
                                            fake=SimpleNamespace(hc_mult=4,dim=8,eps=1e-20,clamp_value=1e-6,
                                                q_weight=q,k_weight=k,embed=lambda row_ids:golden[name][row_ids],
                                                wkv=lambda vectors:torch.nn.functional.linear(vectors,w))
                                            reference_output=official_engram(fake,stream,expected_hashes[:,:,index,:],active)
                                            projection_exact=projection_exact and torch.equal(candidate.view(torch.int32),reference_output.view(torch.int32))
                                        stats=lookup.last_read
                                        fetched+=stats.payload_bytes;calls+=stats.read_calls
                                        requested+=stats.requested_rows*stats.row_bytes
                                    position+=length
                                records.append(dict(ngram=n,heads=heads,batch=batch,near_int64_limit=edge,
                                    mask='none' if token_mask is None else token_mask.tolist(),chunks=chunks,
                                    hash_byte_exact=hash_exact,rows_byte_exact=row_exact,
                                    projection_gate_byte_exact=projection_exact if official_engram is not None else None,
                                    packed_bytes_read=fetched,undeduplicated_requested_bytes=requested,
                                    pread_calls=calls,scope='Synthetic logical row-read counters, NOT physical I/O or TPS'))
    return dict(all_byte_exact=all(r['hash_byte_exact'] and r['rows_byte_exact']
                and r['projection_gate_byte_exact'] is not False for r in records)
                and all(f['projection'] is None or f['projection']['decoded_byte_exact'] for f in fixtures),
                fixtures=fixtures,cases=records)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hash-reference-source',required=True)
    p.add_argument('--quant-reference-source',required=True)
    p.add_argument('--model-reference-source',help='Opt in to generated Q5_K/BF16 -> F32 WKV/gate comparison; NOT native mixed GPU arithmetic')
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    hp=Path(args.hash_reference_source);qp=Path(args.quant_reference_source)
    if hp.stat().st_size!=8138 or qp.stat().st_size!=64946:
        raise ValueError('Pinned source byte sizes differ')
    model=None
    if args.model_reference_source:
        mp=Path(args.model_reference_source)
        if mp.stat().st_size!=61669:raise ValueError('Pinned official model source size differs')
        model=mp.read_bytes()
    report=qualify(hp.read_bytes(),qp.read_bytes(),model)
    scope=('Generated CPU hash -> full-SHA IQ4_NL rows -> decoded Q5_K/BF16 -> explicit F32 WKV/Engram gate. NOT real weights, mixed GPU rounding, native layer/model quality/GPU/TPS'
           if model is not None else 'Generated tiny IQ4_NL table/full-SHA admission -> explicit CPU hash/row lookup only. NOT actual checkpoint values, full Engram/wkv/gating/model quality/GPU/TPS')
    root=Path(__file__).resolve().parents[1]
    implementation={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        'tools/qualify_deepseek41_rows.py','tools/qualify_gguf_iq4nl.py','tools/qualify_gguf_projection_formats.py',
        'src/venturi/gguf_quantization.py','src/venturi/gguf_rows.py','src/venturi/models/engram_hash.py',
        'src/venturi/models/engram_projection.py','src/venturi/models/engram_residual.py')}
    result=dict(scope=scope,implementation_sha256=implementation,
        model_source_sha256=MODEL_SOURCE_SHA256 if model is not None else None,
        official_hash_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',hash_source_sha256=HASH_SOURCE_SHA256,
        quant_revision='3cf03257f219afbe7334045ff7c6a06ac68c627d',quant_source_sha256=IQ4_SOURCE_SHA256,
        torch_version=torch.__version__,numpy_version=np.__version__,**report)
    with Path(args.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','fixtures')},indent=2))
    print(f"{len(report['cases'])} cases")
    if not report['all_byte_exact']:raise SystemExit(1)


if __name__=='__main__':main()
