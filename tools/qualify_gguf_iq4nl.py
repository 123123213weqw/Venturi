"""CPU IQ4_NL format oracle comparison, NOT real Engram/table/model or TPS.

Compare to only the selected dequantizer in the SHA-pinned llama.cpp Python
reference. No upstream module import, C/Rust build, target access or GPU work.
"""
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

from venturi.gguf_quantization import decode_iq4_nl_rows


REVISION = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
SOURCE_SHA256 = '2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b'


def _reference(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned IQ4_NL reference SHA256 differs')
    tree=ast.parse(raw)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IQ4_NL')
    lut=next(ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='kvalues' for t in n.targets))
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='dequantize_blocks'))
    fn.decorator_list=[];scope={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 '<SHA-pinned IQ4_NL reference method>','exec'),scope)
    return lambda blocks:scope[fn.name](SimpleNamespace(kvalues=lut,block_size=32),blocks)


def qualify(source):
    if sys.byteorder != 'little':raise ValueError('Pinned NumPy reference assumes little-endian host')
    reference=_reference(source);cases=[];digest=hashlib.sha256();nblocks=0
    finite=[bits for bits in range(65536) if bits&0x7c00!=0x7c00]
    for start in range(0,len(finite),2048):
        bits=finite[start:start+2048]
        raw=b''.join(struct.pack('<H',value)+bytes((value+i*17)&255 for i in range(16)) for value in bits)
        expected=reference(np.frombuffer(raw,dtype=np.uint8).reshape(-1,18))
        actual=decode_iq4_nl_rows(raw,32).numpy()
        exact=np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
        digest.update(actual.astype('<f4',copy=False).tobytes());nblocks+=len(bits)
        cases.append(dict(label='finite-F16-scale-exhaustive',first_bits=bits[0],last_bits=bits[-1],
                          blocks=len(bits),values=actual.size,byte_exact=bool(exact)))
    # Actual table row geometry: eight 32-value blocks, 144 packed bytes per row.
    raw=b''.join(struct.pack('<e',scale)+bytes([pair])*16
                 for scale in (.5,-2.,2**-24,65504.,0.,-0.,1.,-1.) for pair in range(256))
    expected=reference(np.frombuffer(raw,dtype=np.uint8).reshape(-1,18)).reshape(-1,256)
    actual=decode_iq4_nl_rows(raw,256).numpy()
    cases.append(dict(label='all-nibble-pairs-width256',rows=actual.shape[0],values=actual.size,
                      byte_exact=bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)))))
    return dict(finite_scale_patterns_tested=nblocks,exhaustive_finite_scale_output_sha256=digest.hexdigest(),
                all_byte_exact=all(c['byte_exact'] for c in cases),cases=cases)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);path=Path(a.reference_source)
    if path.stat().st_size != 64946:raise ValueError('Pinned reference source size differs')
    report=qualify(path.read_bytes())
    result=dict(scope='Synthetic packed CPU IQ4_NL decoding only; NOT real rows/weights, storage lookup, model quality, GPU or inference TPS',
        reference_repository='ggml-org/llama.cpp',reference_revision=REVISION,
        reference_source_sha256=SOURCE_SHA256,numpy_version=np.__version__,**report)
    with Path(a.output).open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_byte_exact']:raise SystemExit(1)


if __name__=='__main__':main()
