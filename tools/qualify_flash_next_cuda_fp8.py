"""SM70 byte-decoder operator correctness, not a native full-model backend."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from qualify_flash_next_fp8_expert import PREFIX, original_decoder
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_cuda_fp8 import CUDABlockFP8ExpertStore
from venturi.models.flash_next_moe import MoEGeometry
from venturi.ops.flash_next_fp8 import NativeBlockFP8Decoder


def bit_equal(actual, wanted):
    actual = actual.cpu()
    kind = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(kind), wanted.view(kind))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    assert (props.major, props.minor) == (7, 0)
    torch.cuda.set_per_process_memory_fraction(1024*2**20/props.total_memory, device)
    native = NativeBlockFP8Decoder(a.library)
    original, source_sha = original_decoder(a.reference)
    cases, errors, mlps = [], [], []
    with torch.inference_mode():
        for shape, subnormal in [((2, 127), False), ((1, 1), False), ((127, 129), False),
                                 ((129, 257), False), ((256, 128), False), ((65536, 1), False), ((2, 127), True)]:
            m, n = shape
            if shape == (2, 127):
                codes = torch.tensor([v for v in range(256) if v not in (127, 255)], dtype=torch.uint8).reshape(shape)
            else:
                order = torch.arange(m*n).reshape(shape)
                codes = ((order % 127)+(order % 2)*128).to(torch.uint8)
            scales = torch.full(((m+127)//128, (n+127)//128), .0137, dtype=torch.bfloat16)
            if subnormal:
                scales = torch.ones(scales.shape, dtype=torch.int16).view(torch.bfloat16)
            raw = codes.to(device)
            grid = scales.view(torch.int16).to(device)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                expected = original(codes.view(torch.float8_e4m3fn), scales, [128, 128], dtype)
                result = native.decode(raw, grid, dtype=dtype)
                bit_equal(result, expected)
                cases.append(dict(shape=shape, dtype=str(dtype), bf16_subnormal_scale=subnormal,
                                  decoded_bits_exact=True, includes_signed_zero=shape == (2, 127)))
        raw = load_file(str(Path(a.checkpoint)/'one-expert-codec.safetensors'))
        store = CUDABlockFP8ExpertStore(inspect_checkpoint(a.checkpoint), MoEGeometry(2560, 1, 1, 640, 640), PREFIX, decoder=native)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            cpu, pairs = [], []
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                base = f'{PREFIX}.0.{name}'
                codes, scales = raw[base+'.weight'], raw[base+'.weight_scale_inv']
                cpu.append(original(codes, scales, [128, 128], dtype))
                pairs.append((codes.view(torch.uint8).to(device), scales.view(torch.int16).to(device)))
            for result, expected in zip(native.decode_many(pairs, dtype=dtype), cpu):
                bit_equal(result, expected)
                cases.append(dict(actual_official_expert0=True, dtype=str(dtype), decoded_bits_exact=True))
            weights = store.read(0, device=device, dtype=dtype)
            bit_equal(weights[0], torch.cat(cpu[:2]))
            bit_equal(weights[1], cpu[2])
            if dtype == torch.bfloat16:
                continue  # BF16 storage/decode only, no SM70 BF16 GEMM claim
            expected_weights = (torch.cat(cpu[:2]).to(device), cpu[2].to(device))
            for batch in (1, 8):
                x = (torch.randn(batch, 2560, generator=torch.Generator().manual_seed(912))*.1).to(device=device, dtype=dtype)
                def mlp(w):
                    gate, up = F.linear(x, w[0]).chunk(2, -1)
                    return F.linear(F.silu(gate)*up, w[1])
                torch.testing.assert_close(mlp(weights), mlp(expected_weights), atol=0, rtol=0)
                mlps.append(dict(batch=batch, dtype=str(dtype), standalone_expert_output_exact=True))
        for code, scale, flag in [(127, 1., 1), (56, 0., 2), (126, 3e38, 4)]:
            try:
                native.decode(torch.tensor([[code]], dtype=torch.uint8, device=device),
                    torch.tensor([[scale]], dtype=torch.bfloat16).view(torch.int16).to(device), dtype=torch.float16)
            except ValueError as error:
                assert f'flags={flag}' in str(error)
                errors.append(dict(code=code, scale=scale, rejected_flags=flag))
            else:
                raise AssertionError('Native invalid code/scale/overflow was accepted')
        small = NativeBlockFP8Decoder(a.library, max_output_bytes=1)
        try:
            small.decode(pairs[0][0], pairs[0][1], dtype=torch.float32)
        except ValueError as error:
            assert 'byte budget' in str(error)
        else:
            raise AssertionError('Native output budget not enforced')
    report = dict(scope='One owned SM70 CUDA E4M3FN/BF16-block byte-decoder, not FP8 GEMM/full model',
        decoder_cases=cases, standalone_mlp_cases=mlps, rejected_inputs=errors, output_admission_passed=True,
        sglang_reference_sha256=source_sha, gpu=props.name, torch=str(torch.__version__),
        native_operator=True, native_full_model=False, fp8_hardware_required=False, activation_quantization=False,
        graph_capture_supported=False, validation='One synchronized error flag per matrix group; no invalid output published',
        throughput_measured=False, model_quality_measured=False)
    (out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
