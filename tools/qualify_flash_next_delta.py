"""Pinned floating GDN core/gated RMS oracle, synthetic tensors only.

No projection/convolution/checkpoint binding, native kernel or model TPS claim.
Official16->48 head expansion is a fixture contract, not a new native frontend.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from qualify_flash_next_qsa import UPSTREAM_SHA256
from venturi.models.flash_next_delta import SigmoidGatedRMSNorm, gated_delta_rule


def oracle(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Pinned GDN reference source hash differs")
    names = {"l2norm", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "Qwen4ExpTextRMSNormGated"}
    selected = [n for n in ast.parse(data).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned GDN reference definitions missing")
    for n in selected:
        n.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    scope = dict(torch=torch, nn=nn, F=F, ACT2FN={"sigmoid": torch.sigmoid, "silu": F.silu},
                 is_torchdynamo_exporting=lambda: False)
    exec(compile(module, str(path), "exec"), scope)
    return scope


def fixture(batch, length, kh, vh, dk, dv, dtype, device, seed):
    rng = torch.Generator().manual_seed(seed)
    # Q/K start at K heads and expand consecutively, not tiled/reordered heads.
    q, k = (torch.randn(batch, length, kh, dk, generator=rng).repeat_interleave(vh // kh, 2) * 0.15 for _ in range(2))
    v = torch.randn(batch, length, vh, dv, generator=rng) * 0.3
    decay = -torch.rand(batch, length, vh, generator=rng) * 2
    beta = torch.rand(batch, length, vh, generator=rng)
    return tuple(t.to(device=device, dtype=torch.float32 if i == 3 else dtype)
                 for i, t in enumerate((q, k, v, decay, beta)))


def qualify(reference, device, dtypes):
    report, cross = [], []
    shapes = ((2, 7, 2, 6, 4, 3), (1, 67, 2, 6, 4, 3),
              (1, 65, 16, 48, 128, 128), (8, 1, 16, 48, 128, 128))
    for dtype in dtypes:
        for b, length, kh, vh, dk, dv in shapes:
            values = fixture(b, length, kh, vh, dk, dv, dtype, device, 541)
            generator = torch.Generator().manual_seed(327)
            initial = torch.randn(b, vh, dk, dv, generator=generator).to(device=device) * 0.1
            for normalized in (True, False):
                for nonzero in (False, True):
                    past = initial if nonzero else None
                    preserved = None if past is None else past.clone()
                    outputs = {}
                    for mode in ("recurrent", "chunk"):
                        with torch.inference_mode():
                            actual, state = gated_delta_rule(*values, initial_state=past, mode=mode,
                                normalize_qk=normalized, max_work_elements=64 * 2**20)
                            function = reference[f"torch_{mode}_gated_delta_rule"]
                            expected, expected_state = function(*values, initial_state=past, output_final_state=True,
                                                               use_qk_l2norm_in_kernel=normalized)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
                        if past is not None and not torch.equal(past, preserved):
                            raise AssertionError("GDN changed the caller's recurrent snapshot")
                        report.append(dict(shape=[b, length, kh, vh, dk, dv], dtype=str(dtype), device=str(device),
                            mode=mode, normalized_qk=normalized, nonzero_initial_state=nonzero,
                            output_exact=True, proposed_fp32_state_exact=True))
                        outputs[mode] = actual, state
                    ro, rs = outputs["recurrent"]
                    co, cs = outputs["chunk"]
                    cross.append(dict(shape=[b, length, kh, vh, dk, dv], dtype=str(dtype),
                        normalized_qk=normalized, nonzero_initial_state=nonzero,
                        output_bit_equal=torch.equal(ro, co), state_bit_equal=torch.equal(rs, cs),
                        output_max_abs_difference=float((ro.float() - co.float()).abs().max().item()),
                        state_max_abs_difference=float((rs - cs).abs().max().item())))
    gate_report = []
    for dtype in dtypes:
        candidate = SigmoidGatedRMSNorm(128).to(device=device, dtype=dtype)
        expected_module = reference["Qwen4ExpTextRMSNormGated"](128, activation="sigmoid").to(device=device, dtype=dtype)
        with torch.no_grad():
            candidate.weight.copy_(torch.linspace(-0.5, 1.5, 128, device=device, dtype=dtype))
            expected_module.load_state_dict(candidate.state_dict())
        values = torch.linspace(-3, 3, 8 * 48 * 128, device=device, dtype=dtype).reshape(-1, 128)
        gate = torch.linspace(-20, 20, values.numel(), device=device, dtype=dtype).reshape_as(values)
        with torch.inference_mode():
            actual, expected = candidate(values, gate), expected_module(values, gate)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gate_report.append(dict(dtype=str(dtype), device=str(device), shape=list(values.shape),
                               direct_weight_sigmoid_gate_exact=True))
    return report, cross, gate_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1536 * 2**20 / torch.cuda.get_device_properties(device).total_memory, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    cases, cross, gates = qualify(oracle(a.upstream_modeling), device, dtypes)
    result = dict(scope="Floating GDN recurrence/chunk and sigmoid gated RMS, synthetic tensors only",
        cases=cases, cross_mode_differences=cross, gated_rms_cases=gates, torch=str(torch.__version__),
        upstream_sha256=UPSTREAM_SHA256, reference_export_branch="ordinary solve_triangular, not export fallback",
        state_layout="FP32 [B,HV,DK,DV], not the existing native backend's [HV,DV,DK]",
        gate_semantics="SiLU conv is not tested here; sigmoid output gate with direct RMS weight is tested",
        whole_gdn_frontend_tested=False, checkpoint_weights_tested=False, native_kernel_implemented=False,
        whole_model_tested=False, throughput_measured=False)
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
