"""Original-derived Q8 layer qualification; synthetic inputs, NOT trained inference.

Read and hash the complete artifact, then upload one declared layer at a time.
Check each actual-input stage against independent CPU FP64 represented-weight
math, with the existing per-Op error criteria. No benchmark or engine TPS claim.
FP16 explicitly casts small BF16 a/b/conv weights; Q8 planes and FP32 gamma/state
are preserved. Report those small-weight casts, including underflow/overflow.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import torch

from venturi.models import (
    QwenQ8Weights,
    ResidentQwenGDNLayer,
    ResidentQwenAttentionLayer,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, DenseWeight


def scalar_check(name, actual, ideal, *, floor=2e-5, state=False):
    dtype = actual.dtype
    value, ideal = actual.detach().cpu().double(), ideal.double()
    rounded = ideal if state else ideal.to(dtype).double()
    error = (value - rounded).abs()
    bound = (
        (2e-6 + 2e-4 * ideal.abs())
        if state
        else (2 * torch.finfo(dtype).eps * ideal.abs() + floor)
    )
    result = dict(
        stage=name,
        passed=bool(torch.isfinite(value).all()) and bool((error <= bound).all()),
        elements=value.numel(),
        max_rounded_abs_error=error.max().item() if error.numel() else 0.0,
        floor=2e-6 if state else floor,
        relative_bound=2e-4 if state else 2 * torch.finfo(dtype).eps,
    )
    if not result["passed"]:
        raise AssertionError(
            result | dict(failures=(error > bound).nonzero()[:8].tolist())
        )
    return result


def linear_ideal(weights, role, x, dtype):
    """Read encoded CPU planes, tile rows, and never round decoded Q8 weights."""
    descriptor = weights.read_linear(role)
    source = x.detach().cpu().double()
    if isinstance(descriptor, DenseWeight):
        return source @ descriptor.data.to(dtype).double().T
    out = torch.empty(source.shape[0], descriptor.out_features, dtype=torch.float64)
    for start in range(0, descriptor.out_features, 256):
        end = min(start + 256, descriptor.out_features)
        # Codes * binary FP16 scales are represented exactly in FP64. The
        # independent reference has a different reduction order from the kernel.
        decoded = descriptor.codes[start:end].double() * descriptor.scales[
            start:end
        ].double().repeat_interleave(32, -1)
        out[:, start:end] = source @ decoded[:, : descriptor.in_features].T
    return out


class Checks:
    def __init__(self, weights, layer, dtype):
        self.weights, self.layer, self.dtype = weights, layer, dtype
        self.prefix = f"text/layers/{layer}/"
        self.results = []

    def add(self, name, actual, ideal, **kwargs):
        result = scalar_check(name, actual, ideal, **kwargs)
        self.results.append(result)
        print(json.dumps(result), flush=True)

    def norm(self, op, x, out, role):
        op.run(x, out=out)
        v = x.detach().cpu().double()
        gamma = (1 + self.weights.read_float(self.prefix + role).float()).double()
        ideal = (
            v
            * torch.rsqrt(
                v.square().mean(-1, keepdim=True) + self.weights.config["rms_norm_eps"]
            )
            * gamma
        )
        self.add(role, out, ideal)

    def linear(self, op, x, out, role, scratch=None):
        op.run(x, out=out, workspace=scratch)
        self.add(
            role,
            out,
            linear_ideal(self.weights, self.prefix + role, x, self.dtype),
            floor=2e-4
            if self.weights._object(self.prefix + role)["format"] == "q8_g32_fp16"
            else 2e-5,
        )

    def mlp(self, p, workspace, residual, out):
        self.norm(p.norm, residual, workspace.hidden, "post_attention_norm")
        for op, target, role in (
            (p.gate, workspace.gate, "gate"),
            (p.up, workspace.up, "up"),
        ):
            self.linear(
                op,
                workspace.hidden,
                target,
                "mlp/" + role,
                workspace.linear if op.workspace_bytes else None,
            )
        gate, up = workspace.gate.cpu().double(), workspace.up.cpu().double()
        p.swiglu.run(workspace.gate, workspace.up, out=workspace.gate)
        self.add("swiglu", workspace.gate, gate * gate.sigmoid() * up)
        self.linear(
            p.down,
            workspace.gate,
            workspace.hidden,
            "mlp/down",
            workspace.linear if p.down.workspace_bytes else None,
        )
        ideal = residual.cpu().double() + workspace.hidden.cpu().double()
        p.add.run(residual, workspace.hidden, out=out)
        self.add("mlp_residual", out, ideal)


def recurrence_ideal(packed, g, beta, state, *, hq, hv, dk, dv):
    values = packed.cpu().double()
    g, beta = g.cpu().double(), beta.cpu().double()
    state = state.cpu().double().clone()
    out = torch.empty(values.shape[0], hv, dv, dtype=torch.float64)
    for t in range(values.shape[0]):
        for h in range(hv):
            kh = h // (hv // hq)
            q = values[t, kh * dk : (kh + 1) * dk]
            k = values[t, hq * dk + kh * dk : hq * dk + (kh + 1) * dk]
            v = values[t, 2 * hq * dk + h * dv : 2 * hq * dk + (h + 1) * dv]
            q = q * torch.rsqrt(q @ q + 1e-6)
            k = k * torch.rsqrt(k @ k + 1e-6)
            decayed = state[h] * g[t, h].exp()
            state[h] = decayed + torch.outer(beta[t, h] * (v - decayed @ k), k)
            out[t, h] = (state[h] @ q) / (dk**0.5)
    return out, state


def qualify_gdn(weights, resident, plan, x):
    p, r = plan, resident
    w = p.empty_workspace()
    s = r.zero_state()
    c = Checks(weights, r.layer_index, r.dtype)
    c.norm(p.norm, x, w.normalized, "input_norm")
    for op, target, role, scratch in zip(
        p.projections[:4],
        (w.qkv, w.z, w.a, w.b),
        ("qkv", "z", "a", "b"),
        w.projection_scratch[:4],
    ):
        c.linear(op, w.normalized, target, "gdn/" + role, scratch)
    taps = weights.read_float(c.prefix + "gdn/convolution").to(r.dtype).double()
    joined = torch.cat((s.history.cpu().double(), w.qkv.cpu().double()))
    conv = torch.stack(
        [
            (joined[t : t + taps.shape[0]] * taps).sum(0).to(r.dtype).double()
            for t in range(p.tokens)
        ]
    )
    final_history = torch.empty_like(s.history)
    final_state = torch.empty_like(s.recurrent)
    p.mixer.conv.run(
        w.qkv, history=s.history, out=w.mixer.convolved, history_out=final_history
    )
    c.add("convolution_silu", w.mixer.convolved, conv * conv.sigmoid())
    assert (
        torch.equal(final_history.cpu(), joined[-s.history.shape[0] :].to(r.dtype))
        if s.history.shape[0]
        else final_history.numel() == 0
    )
    p.mixer.gates.run(w.a, w.b, log_decay=w.mixer.log_decay, beta=w.mixer.beta)
    dt = w.a.cpu().double() + weights.read_float(c.prefix + "gdn/dt_bias").double()
    expected_g = -weights.read_float(c.prefix + "gdn/a_log").double().exp() * (
        torch.log1p((-dt.abs()).exp()) + dt.clamp(min=0)
    )
    for name, actual, ideal in (
        ("log_decay", w.mixer.log_decay, expected_g),
        ("beta", w.mixer.beta, w.b.cpu().double().sigmoid().to(r.dtype).double()),
    ):
        error = (actual.cpu().double() - ideal).abs()
        result = dict(
            stage=name,
            passed=bool((error <= 2e-7 + 2e-6 * ideal.abs()).all()),
            max_abs_error=error.max().item(),
            floor=2e-7,
            relative_bound=2e-6,
        )
        c.results.append(result)
        assert result["passed"], result
    p.mixer.recurrence.run_packed(
        w.mixer.convolved,
        w.mixer.log_decay,
        w.mixer.beta,
        state=s.recurrent,
        out=w.mixer.recurrent,
        state_out=final_state,
    )
    ideal, ideal_state = recurrence_ideal(
        w.mixer.convolved,
        w.mixer.log_decay,
        w.mixer.beta,
        s.recurrent,
        hq=r.frontend.qk_heads,
        hv=r.frontend.value_heads,
        dk=r.frontend.key_dim,
        dv=r.frontend.value_dim,
    )
    c.add("recurrence", w.mixer.recurrent, ideal)
    c.add("recurrent_state", final_state, ideal_state, state=True)
    v = w.mixer.recurrent.cpu().double()
    z = w.z_heads.cpu().double()
    gamma = weights.read_float(c.prefix + "gdn/norm")
    norm = (
        (
            v
            * torch.rsqrt(
                v.square().mean(-1, keepdim=True) + weights.config["rms_norm_eps"]
            )
        )
        .to(r.dtype)
        .double()
    )
    weighted = norm * gamma.double()
    if gamma.dtype != torch.float32:
        weighted = weighted.to(r.dtype).double()
    p.mixer.gated_norm.run(w.mixer.recurrent, w.z_heads, out=w.mixed_heads)
    c.add("gated_norm", w.mixed_heads, weighted * z * z.sigmoid())
    c.linear(p.output, w.mixed, w.projected, "gdn/output", w.projection_scratch[4])
    p.add.run(x, w.projected, out=w.residual)
    c.add(
        "attention_residual", w.residual, x.cpu().double() + w.projected.cpu().double()
    )
    expected = torch.empty_like(x)
    c.mlp(p.mlp, w.mlp, w.residual, expected)
    actual = torch.empty_like(x)
    p.run(
        x,
        out=actual,
        history=s.history,
        state=s.recurrent,
        history_out=s.history,
        state_out=s.recurrent,
        workspace=w,
    )
    assert (
        torch.equal(actual, expected)
        and torch.equal(s.history, final_history)
        and torch.equal(s.recurrent, final_state)
    ), "native GDN whole-layer/manual-composition mismatch"
    return dict(
        stages=c.results,
        manual_composition_bit_exact=True,
        state_history_bit_exact=True,
    )


def attention_ideal(q, k, v, scale):
    q, k, v = (t.cpu().double() for t in (q, k, v))
    out = torch.empty_like(q)
    for t in range(q.shape[0]):
        for h in range(q.shape[1]):
            kh = h // (q.shape[1] // k.shape[1])
            scores = (k[: t + 1, kh] @ q[t, h]) * scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ v[: t + 1, kh]
    return out


def rotate(norm, cos, sin, r, dtype):
    v = norm.cpu().double()
    co, si = cos.cpu().double()[:, None], sin.cpu().double()[:, None]
    first, second = v[..., : r // 2], v[..., r // 2 : r]
    # Match the observable activation-dtype product rounding, not a fused FP64
    # norm+RoPE cascade that could amplify earlier norm-rounding cancellation.
    return torch.cat(
        (
            (first * co).to(dtype).double() - (second * si).to(dtype).double(),
            (second * co).to(dtype).double() + (first * si).to(dtype).double(),
            v[..., r:],
        ),
        dim=-1,
    ).to(dtype)


def qualify_attention(weights, resident, plan, x):
    p, r = plan, resident
    w = p.empty_workspace()
    c = Checks(weights, r.layer_index, r.dtype)
    cache = r.empty_cache(capacity=p.tokens)
    step = p.at(position=0, cache=cache)
    co, si = r.rope.tables(
        r._backend, position=0, tokens=p.tokens, dtype=r.dtype, device=r.device
    )
    b = w.buffers
    d = r.dimensions
    c.norm(p.norm, x, b["normalized"], "input_norm")
    for op, target, role, scratch in zip(
        p.projections[:3],
        (b["qg"], b["key"], step.value_tail),
        ("q", "k", "v"),
        w.projection_scratch[:3],
    ):
        c.linear(op, b["normalized"], target, "attention/" + role, scratch)
    qn, kn = torch.empty_like(w.query_heads), torch.empty_like(w.key_heads)
    p.qk_norm.run_normalized(b["qg"], w.key_heads, query_out=qn, key_out=kn)
    qg = b["qg"].view(p.tokens, d.query_heads, 2 * d.head_dim)
    for name, raw, actual in (("q", qg[..., : d.head_dim], qn), ("k", w.key_heads, kn)):
        v = raw.cpu().double()
        gamma = (
            1 + weights.read_float(c.prefix + "attention/" + name + "_norm").float()
        ).double()
        c.add(
            name + "_norm",
            actual,
            v
            * torch.rsqrt(
                v.square().mean(-1, keepdim=True) + weights.config["rms_norm_eps"]
            )
            * gamma,
        )
    p.qk_norm.run(
        b["qg"], w.key_heads, co, si, query_out=w.query_heads, key_out=step.key_tail
    )
    assert torch.equal(w.query_heads.cpu(), rotate(qn, co, si, d.rotary_dim, r.dtype))
    assert torch.equal(step.key_tail.cpu(), rotate(kn, co, si, d.rotary_dim, r.dtype))
    step.attention.run(
        w.query_heads, step.key_prefix, step.value_prefix, out=w.attended_heads
    )
    c.add(
        "causal_attention",
        w.attended_heads,
        attention_ideal(
            w.query_heads, step.key_prefix, step.value_prefix, d.head_dim**-0.5
        ),
    )
    attended = w.attended_heads.cpu().double()
    gates = qg[..., d.head_dim :].cpu().double().sigmoid().to(r.dtype).double()
    p.gate.run(w.attended_heads, b["qg"], out=w.attended_heads)
    c.add("output_gate", w.attended_heads, attended * gates)
    c.linear(
        p.output, b["attended"], b["projected"], "attention/o", w.projection_scratch[3]
    )
    p.add.run(x, b["projected"], out=b["residual"])
    c.add(
        "attention_residual",
        b["residual"],
        x.cpu().double() + b["projected"].cpu().double(),
    )
    expected = torch.empty_like(x)
    c.mlp(p.mlp, w.mlp, b["residual"], expected)
    keys, values = step.key_tail.clone(), step.value_tail.clone()
    actual = torch.empty_like(x)
    step.run(x, co, si, out=actual, workspace=w)
    assert (
        torch.equal(actual, expected)
        and torch.equal(cache._keys, keys)
        and torch.equal(cache._values.view(p.tokens, d.kv_width), values)
    ), "native attention whole-layer/manual-composition mismatch"
    return dict(
        stages=c.results,
        manual_composition_bit_exact=True,
        cache_bit_exact=True,
        fused_actual_norm_rotary_bit_exact=True,
    )


def cast_statistics(weights, layer, dtype):
    result = []
    for binding in weights.layer_bindings(layer):
        obj = weights._object(binding)
        if obj["format"] == "bf16" and (
            binding.endswith(("/gdn/a", "/gdn/b", "/gdn/convolution"))
        ):
            value = weights.read_float(binding).float()
            converted = value.to(dtype).float()
            result.append(
                dict(
                    binding=binding,
                    source_dtype="BF16",
                    execution_dtype=str(dtype),
                    changed_elements=int((value != converted).sum()),
                    underflow_to_zero=int(((value != 0) & (converted == 0)).sum()),
                    nonfinite_after_cast=int((~converted.isfinite()).sum()),
                    max_abs_cast_error=(value - converted).abs().max().item(),
                )
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--blocks-library", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8])
    args = parser.parse_args(argv)
    if len(args.expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.expected_sha256
    ):
        raise ValueError("Expected an explicit lowercase SHA256")
    if not args.tokens or any(type(t) is not int or t <= 0 for t in args.tokens):
        raise ValueError("Positive token counts required")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = perf_counter()
    artifact = NInferArtifact(args.artifact)
    weights = QwenQ8Weights.from_ninfer(artifact)
    for layer in args.layers:
        weights.layer_bindings(layer)
    before = artifact.path.stat()

    def stamp(s):
        return (s.st_ino, s.st_size, s.st_mtime_ns)

    print(
        "Hashing the complete artifact before selected-layer qualification", flush=True
    )
    digest = hashlib.sha256()
    with artifact.path.open("rb") as stream:
        while chunk := stream.read(16 * 2**20):
            digest.update(chunk)
    if digest.hexdigest() != args.expected_sha256:
        raise ValueError("Original-derived Q8 artifact SHA256 mismatch")
    backend = CudaBackend(args.blocks_library)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    target = backend._device(dtype, "cuda:0")
    report = dict(
        scope=__doc__,
        artifact_id=artifact.artifact_id,
        artifact_sha256=digest.hexdigest(),
        artifact=str(artifact.path),
        library=str(backend.library),
        architecture=backend.architecture,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(target),
        dtype=str(dtype),
        cases=[],
        timing="Correctness/CPU oracle work only; no latency, TPS or matched engine timing",
        gpu_state=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        counters="RmProfilingAdminOnly=1 on this host; achieved occupancy/SM/DRAM/cache/stall counters not collected. No profiler campaign.",
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for layer in args.layers:
        specs = [
            s
            for s in weights.plan.bindings
            if s.binding in weights.layer_bindings(layer)
        ]
        if (
            sum(s.nbytes for s in specs) + 512 * 2**20
            > torch.cuda.mem_get_info(target)[0]
        ):
            raise MemoryError(
                "Selected layer plus reserve does not fit free VRAM; no service stopped"
            )
        cls = (
            ResidentQwenGDNLayer
            if weights.config["layer_types"][layer] == "linear_attention"
            else ResidentQwenAttentionLayer
        )
        resident = cls.from_q8(
            weights, backend, layer=layer, dtype=dtype, device=target
        )
        casts = cast_statistics(weights, layer, dtype)
        for tokens in args.tokens:
            print(f"Qualifying layer {layer}, T={tokens}, {dtype}", flush=True)
            p = resident.prepare(tokens=tokens)
            x = (
                torch.randn(
                    tokens,
                    resident.hidden_size,
                    generator=torch.Generator().manual_seed(1871 + tokens),
                )
                * 0.2
            ).to(device=target, dtype=dtype)
            try:
                checks = (
                    qualify_gdn if cls is ResidentQwenGDNLayer else qualify_attention
                )(weights, resident, p, x)
                torch.cuda.synchronize(target)
                report["cases"].append(
                    dict(
                        layer=layer,
                        kind=weights.config["layer_types"][layer],
                        tokens=tokens,
                        resident_weight_bytes=resident.storage_nbytes,
                        small_float_casts=casts,
                        **checks,
                    )
                )
                save()
            finally:
                torch.cuda.synchronize(target)
                p.close()
            del p, x
        del resident
        gc.collect()
        torch.cuda.empty_cache()
    if stamp(artifact.path.stat()) != stamp(before):
        raise ValueError("Artifact changed during qualification")
    report["qualification_wall_seconds_including_hash_upload_cpu_oracles"] = (
        perf_counter() - started
    )
    report["passed"] = True
    report["full_trained_model_forward_qualified"] = False
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
