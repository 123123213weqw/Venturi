"""Official FP8 ONE-expert codec sample + synthetic store checks, not model TPS."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file, load_file
import torch
from torch.nn import functional as F

from collect_hf_headers import fetch_range
from sample_flash_next_fp8 import e4m3fn
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_moe import MoEGeometry


REVISION = '236dfdf285828023ca3bcd3f37366c58a3469b13'
PREFIX = 'model.language_model.layers.0.mlp.experts'
REFERENCE_SHA256 = '35bd871eb581e9b3f700b7672c20c867cbc26d3fcf2eaf431632e307501d5894'


def original_decoder(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Pinned SGLang FP8 codec reference source differs')
    tree = ast.parse(data)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'block_quant_dequant')
    scope = dict(torch=torch)
    code = ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+[function],
                                               type_ignores=[]))
    exec(compile(code, str(path), 'exec'), scope)
    return scope['block_quant_dequant'], hashlib.sha256(data).hexdigest()


def sample(source, output):
    output.mkdir(parents=True, exist_ok=False)
    api = json.loads((source/'hf-model-api.json').read_text())
    assert api['sha'] == REVISION
    repo = api['id']
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    names = [f'{PREFIX}.0.{p}.{suffix}' for p in ('gate_proj', 'up_proj', 'down_proj')
             for suffix in ('weight', 'weight_scale_inv')]
    headers, tensors, records = {}, {}, []
    for name in names:
        filename = index[name]
        if filename not in headers:
            old = source/'selected-headers'/(filename+'.json')
            if old.exists():
                record = json.loads(old.read_text())
            else:
                prefix, total = fetch_range(f'https://huggingface.co/{repo}/resolve/{REVISION}/{filename}', 0, 7)
                size = int.from_bytes(prefix, 'little')
                assert 2 <= size <= 16*2**20
                raw, again = fetch_range(f'https://huggingface.co/{repo}/resolve/{REVISION}/{filename}', 8, 7+size)
                assert again == total
                record = dict(repo=repo, revision=REVISION, filename=filename, header_bytes=size,
                              file_bytes=total, header=json.loads(raw), header_sha256=hashlib.sha256(raw).hexdigest())
            assert record['repo'] == repo and record['revision'] == REVISION
            headers[filename] = record
            (output/(filename+'.header.json')).write_text(json.dumps(record, indent=2)+'\n')
        header = headers[filename]
        spec = header['header'][name]
        expected = [640, 2560] if '.down_proj.' not in name else [2560, 640]
        if name.endswith('weight_scale_inv'):
            expected = [(v+127)//128 for v in expected]
            assert spec['dtype'] == 'BF16'
        else:
            assert spec['dtype'] == 'F8_E4M3'
        assert spec['shape'] == expected
        lo = 8+header['header_bytes']+spec['data_offsets'][0]
        hi = 8+header['header_bytes']+spec['data_offsets'][1]-1
        raw, total = fetch_range(f'https://huggingface.co/{repo}/resolve/{REVISION}/{filename}', lo, hi)
        assert total == header['file_bytes'] and len(raw) == hi-lo+1
        kind = torch.bfloat16 if name.endswith('weight_scale_inv') else torch.float8_e4m3fn
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=kind).reshape(expected).clone()
        outname = name.rsplit('.', 2)[-2]+('-scale' if name.endswith('weight_scale_inv') else '-codes')+'.bin'
        (output/outname).write_bytes(raw)
        records.append(dict(tensor=name, filename=filename, absolute_range=[lo, hi], bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(), raw_file=outname))
    save_file(tensors, output/'one-expert-codec.safetensors')
    (output/'config.json').write_text((source/'config.json').read_text())
    report = dict(repo=repo, revision=REVISION, samples=records, source_bytes=sum(r['bytes'] for r in records),
        artifact_sha256=hashlib.sha256((output/'one-expert-codec.safetensors').read_bytes()).hexdigest(),
        scope='Only original expert0 gate/up/down codes+scales; not a full MoE/checkpoint or pruned model')
    (output/'sample.json').write_text(json.dumps(report, indent=2)+'\n')
    return output


def qualify(checkpoint, reference, output, device, strategy='tiled'):
    output.mkdir(parents=True, exist_ok=False)
    oracle, sha = original_decoder(reference)
    cp = inspect_checkpoint(checkpoint)
    # One-expert codec geometry is intentional: no routing/model execution with
    # an E512 bank pruned to E1. Only read() and a standalone expert MLP tested.
    g = MoEGeometry(2560, 1, 1, 640, 640)
    store = BlockFP8ExpertStore(cp, g, PREFIX, decode_strategy=strategy)
    raw = load_file(str(checkpoint/'one-expert-codec.safetensors'))
    records = []
    dtypes = (torch.float32, torch.bfloat16) if device.type == 'cpu' else (torch.float32, torch.float16)
    independent_codes = {}
    for p in ('gate_proj', 'up_proj', 'down_proj'):
        name = f'{PREFIX}.0.{p}.weight'
        codes = raw[name]
        # Decode ALL sampled codes via256-entry independent scalar bit formula,
        # not just torch.float8 cast. Reject the two E4M3FN NaN codes.
        values = torch.tensor([e4m3fn(i) for i in range(256)], dtype=torch.float32)
        decoded = values[codes.view(torch.uint8).long()]
        torch.testing.assert_close(decoded, codes.float(), atol=0, rtol=0)
        independent_codes[p] = decoded
    for dtype in dtypes:
        with torch.inference_mode():
            gate_up, down = store.read(0, device=device, dtype=dtype)
            expected = []
            for p in ('gate_proj', 'up_proj', 'down_proj'):
                base = f'{PREFIX}.0.{p}'
                a = oracle(raw[base+'.weight'], raw[base+'.weight_scale_inv'], [128, 128], dtype)
                scales = raw[base+'.weight_scale_inv'].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
                b = (independent_codes[p]*scales).to(dtype)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                expected.append(a.to(device=device))
            torch.testing.assert_close(gate_up, torch.cat(expected[:2]), atol=0, rtol=0)
            torch.testing.assert_close(down, expected[2], atol=0, rtol=0)
            for batch in (1, 8):
                x = (torch.randn(batch, 2560, generator=torch.Generator().manual_seed(212))*.1).to(device=device, dtype=dtype)
                gate, up = F.linear(x, gate_up).chunk(2, -1)
                actual = F.linear(F.silu(gate)*up, down)
                # Both use same concatenated GEMM shape to isolate codec;
                # separate gate/up GEMM rounding is NOT claimed interchangeable.
                eg, eu = F.linear(x, torch.cat(expected[:2])).chunk(2, -1)
                wanted = F.linear(F.silu(eg)*eu, expected[2])
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
                assert torch.isfinite(actual).all()
                records.append(dict(batch=batch, dtype=str(dtype), device=str(device), decoded_weights_exact=True,
                    standalone_expert_output_exact=True, all_finite=True, weights='actual expert0 official FP8', inputs='synthetic'))
    report = dict(scope='One official FP8 expert CPU decode+standalone MLP, not MoE routing/full-model/FP8-GEMM',
        cases=records, sglang_commit='059ba6b66669b9acf4eca833e6f72abeccec9be4', reference_sha256=sha,
        fp8_revision=REVISION, all_sampled_codes_checked_by_independent_bit_formula=True,
        cpu_decode=True, decode_strategy=strategy, gpu_fp8_arithmetic=False, native_kernel_implemented=False, whole_model_tested=False,
        throughput_measured=False, quality_measured=False, torch=str(torch.__version__))
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source')
    p.add_argument('--checkpoint')
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--decode-strategy', choices=['tiled','vectorized'], default='tiled')
    a = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory, device)
    out = Path(a.output)
    cp = Path(a.checkpoint) if a.checkpoint else sample(Path(a.source), out.parent/'official-one-expert')
    qualify(cp, Path(a.reference), out, device, a.decode_strategy)


if __name__ == '__main__':
    main()
