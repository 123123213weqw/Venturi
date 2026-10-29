"""Synthetic CPU Q5_K/BF16 decoding vs selected pinned primary methods; no model/TPS."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np

from venturi.gguf_quantization import decode_bf16_rows,decode_q5_k_rows


SOURCE_SHA256='2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b'
REVISION='3cf03257f219afbe7334045ff7c6a06ac68c627d'


def references(raw):
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Pinned projection-format reference SHA256 differs')
    tree=ast.parse(raw);scope={'np':np,'QK_K':256}
    def method(class_name,name):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
        fn.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
            '<SHA-pinned primary quantization method>','exec'),scope)
        return scope[name]
    scale_min=method('Q4_K','get_scale_min')
    scope['Q4_K']=SimpleNamespace(K_SCALE_SIZE=12,get_scale_min=scale_min)
    q5=method('Q5_K','dequantize_blocks');bf16=method('BF16','dequantize_blocks')
    return lambda blocks:q5(None,blocks),lambda blocks:bf16(None,blocks)


def _same(actual,expected):
    return bool(np.array_equal(actual.numpy().view(np.uint32),expected.view(np.uint32)))


def qualify(source):
    if sys.byteorder!='little':raise ValueError('Pinned NumPy reference requires little-endian host')
    q5,bf16=references(source);records=[];output_digest=hashlib.sha256()
    finite=[bits for bits in range(65536) if bits&0x7c00!=0x7c00]
    for start in range(0,len(finite),512):
        patterns=finite[start:start+512];pieces=[]
        for bits in patterns:
            body=bytes((bits*13+i*17)&255 for i in range(172))
            pieces.extend((struct.pack('<H',bits)+struct.pack('<e',.125)+body,
                           struct.pack('<e',-.25)+struct.pack('<H',bits)+body))
        raw=b''.join(pieces)
        expected=q5(np.frombuffer(raw,dtype=np.uint8).reshape(-1,176))
        actual=decode_q5_k_rows(raw,256)
        exact=_same(actual,expected);output_digest.update(actual.numpy().astype('<f4',copy=False).tobytes())
        records.append(dict(label='Q5_K-finite-F16-d-and-dmin-exhaustive',patterns=len(patterns),
                            first_bits=patterns[0],last_bits=patterns[-1],blocks=len(pieces),byte_exact=exact))
    rng=np.random.default_rng(417161)
    for width,rows in ((256,1),(512,3),(6144,40)):
        raw=b''.join(struct.pack('<ee',2**-12,2**-13)+rng.integers(0,256,size=172,dtype=np.uint8).tobytes()
                     for _ in range(rows*width//256))
        expected=q5(np.frombuffer(raw,dtype=np.uint8).reshape(-1,176)).reshape(rows,width)
        records.append(dict(label='Q5_K-bounded-row-geometry',width=width,rows=rows,
                            byte_exact=_same(decode_q5_k_rows(raw,width),expected)))
    bf_patterns=[bits for bits in range(65536) if bits&0x7f80!=0x7f80]
    raw=struct.pack('<'+'H'*len(bf_patterns),*bf_patterns)
    expected=bf16(np.frombuffer(raw,dtype=np.uint8).reshape(-1,2))
    records.append(dict(label='BF16-all-finite-bit-patterns',patterns=len(bf_patterns),
                        byte_exact=_same(decode_bf16_rows(raw,1),expected)))
    rejected_q5=0;rejected_bf16=0
    for bits in range(65536):
        if bits&0x7c00==0x7c00:
            for prefix in (struct.pack('<H',bits)+b'\0\0',b'\0\0'+struct.pack('<H',bits)):
                try:decode_q5_k_rows(prefix+bytes(172),256)
                except ValueError:rejected_q5+=1
        if bits&0x7f80==0x7f80:
            try:decode_bf16_rows(struct.pack('<H',bits),1)
            except ValueError:rejected_bf16+=1
    return dict(all_byte_exact=all(r['byte_exact'] for r in records),
                q5_finite_scale_patterns_per_slot=len(finite),q5_exhaustive_blocks=2*len(finite),
                q5_exhaustive_output_sha256=output_digest.hexdigest(),bf16_finite_patterns=len(bf_patterns),
                q5_nonfinite_slot_patterns_rejected=rejected_q5,bf16_nonfinite_patterns_rejected=rejected_bf16,
                all_nonfinite_rejected=rejected_q5==4096 and rejected_bf16==256,cases=records)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv);source=Path(args.reference_source)
    if source.stat().st_size!=64946:raise ValueError('Pinned reference source size differs')
    report=qualify(source.read_bytes())
    root=Path(__file__).resolve().parents[1]
    implementation={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('src/venturi/gguf_quantization.py','tools/qualify_gguf_projection_formats.py')}
    result=dict(scope='Bounded synthetic packed Q5_K/BF16 CPU format decoding only. NOT actual weights/quantization quality/projection kernel/full model/GPU/TPS',
        reference_repository='ggml-org/llama.cpp',reference_revision=REVISION,
        reference_source_sha256=SOURCE_SHA256,implementation_sha256=implementation,numpy_version=np.__version__,**report)
    with Path(args.output).open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    if not result['all_byte_exact'] or not result['all_nonfinite_rejected']:raise SystemExit(1)


if __name__=='__main__':main()
