"""Independent staged CPU FP64 Qwen GDN decoder-layer oracle.

Input is original floating fixture tensors, not the engine's binding table,
quantizer or prepared operators. Q8 codes/scales are reconstructed with Python
round (nearest-even). Native fused SwiGLU uses one final activation cast; q/k
L2 normalization and recurrent state are not rounded to BF16 mid-recurrence.
"""

import math

import torch


def q8_source_oracle(source):
    values = source.cpu().double()
    result = torch.empty_like(values)
    for row in range(values.shape[0]):
        for col in range(0, values.shape[1], 32):
            group = values[row, col : col + 32].tolist()
            scale = torch.tensor(max(map(abs, group)) / 127, dtype=torch.float16).item()
            result[row, col : col + 32] = torch.tensor(
                [
                    min(127, max(-127, round(x / scale))) * scale if scale else 0
                    for x in group
                ],
                dtype=torch.float64,
            )
    return result


def gdn_layer_oracle(raw, cfg, x, history, state, *, layer=0):
    """Return layer output, chronological history, state and intermediate stages."""
    dtype, source = x.dtype, x.cpu().double()
    hq, hv, dk, dv = (
        cfg[k]
        for k in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
    )
    epsilon = cfg["rms_norm_eps"]
    prefix = f"model.language_model.layers.{layer}."
    trace = {}

    def norm(x, key):
        value = x.double()
        gamma = 1 + raw[prefix + key].double()
        return (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * gamma
        ).to(dtype)

    def linear(x, key, *, q8=True):
        weight = (
            q8_source_oracle(raw[prefix + key]) if q8 else raw[prefix + key].double()
        )
        return (x.double() @ weight.T).to(dtype)

    normalized = trace["normalized"] = norm(source, "input_layernorm.weight")
    qkv = trace["qkv"] = linear(normalized, "linear_attn.in_proj_qkv.weight")
    z = trace["z"] = linear(normalized, "linear_attn.in_proj_z.weight")
    a = trace["a"] = linear(normalized, "linear_attn.in_proj_a.weight", q8=False)
    b = trace["b"] = linear(normalized, "linear_attn.in_proj_b.weight", q8=False)
    taps = raw[prefix + "linear_attn.conv1d.weight"].squeeze(1).T.double()
    joined = torch.cat((history.cpu().double(), qkv.double()))
    convolved = torch.empty_like(qkv)
    for t in range(x.shape[0]):
        accum = (joined[t : t + taps.shape[0]] * taps).sum(0).to(dtype).double()
        convolved[t] = (accum * accum.sigmoid()).to(dtype)
    final_history = (
        joined[-history.shape[0] :].to(dtype)
        if history.shape[0]
        else history.cpu().clone()
    )
    trace["convolved"] = convolved
    rate = raw[prefix + "linear_attn.A_log"].double().exp()
    dt = a.double() + raw[prefix + "linear_attn.dt_bias"].double()
    gates = -rate * (torch.log1p((-dt.abs()).exp()) + dt.clamp(min=0))
    beta = b.double().sigmoid().to(dtype).double()
    trace["log_decay"], trace["beta"] = gates, beta
    state = state.cpu().double().clone()
    recurrent = torch.empty(x.shape[0], hv, dv, dtype=dtype)
    for t in range(x.shape[0]):
        for head in range(hv):
            khead = head // (hv // hq)
            q = convolved[t, khead * dk : (khead + 1) * dk].double()
            k = convolved[t, hq * dk + khead * dk : hq * dk + (khead + 1) * dk].double()
            v = convolved[
                t, 2 * hq * dk + head * dv : 2 * hq * dk + (head + 1) * dv
            ].double()
            q, k = q / torch.sqrt(q @ q + 1e-6), k / torch.sqrt(k @ k + 1e-6)
            decayed = state[head] * gates[t, head].exp()
            delta = beta[t, head] * (v - decayed @ k)
            state[head] = decayed + torch.outer(delta, k)
            recurrent[t, head] = ((state[head] @ q) / math.sqrt(dk)).to(dtype)
    trace["recurrent"] = recurrent
    value = recurrent.double()
    normalized_mix = (
        (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon))
        .to(dtype)
        .double()
    )
    weighted = normalized_mix * raw[prefix + "linear_attn.norm.weight"].double()
    if raw[prefix + "linear_attn.norm.weight"].dtype != torch.float32:
        weighted = weighted.to(dtype).double()
    z = z.reshape_as(value).double()
    mixed = trace["mixed"] = (weighted * z * z.sigmoid()).to(dtype).flatten(1)
    projected = trace["projected"] = linear(mixed, "linear_attn.out_proj.weight")
    residual = trace["residual"] = (source + projected.double()).to(dtype)
    mlp_norm = trace["mlp_norm"] = norm(residual, "post_attention_layernorm.weight")
    gate = trace["mlp_gate"] = linear(mlp_norm, "mlp.gate_proj.weight")
    up = trace["mlp_up"] = linear(mlp_norm, "mlp.up_proj.weight")
    activated = trace["mlp_activated"] = (
        gate.double() * gate.double().sigmoid() * up.double()
    ).to(dtype)
    down = trace["mlp_down"] = linear(activated, "mlp.down_proj.weight")
    output = (residual.double() + down.double()).to(dtype)
    return output, final_history, state, trace


def qualify_layer_operations(plan, raw, x, history, state, *, workspace):
    """Qualify each Op against independent FP64 math on its ACTUAL stage inputs.

    A complete FP64 cascade is a diagnostic, not a per-Op FP32 error bound:
    early output-rounding differences can amplify through later cancellation.
    No existing per-Op tolerance is widened here. Returns actual output/state
    for exact manual-composition parity with the complete prepared layer run.
    """
    from test_gdn import gdn_oracle
    from test_state_space import conv_oracle

    r, w = plan.resident, workspace
    dtype, prefix = r.dtype, f"model.language_model.layers.{r.layer_index}."
    checks = []

    def check(name, actual, ideal, *, floor=2e-5):
        value, ideal = actual.cpu().double(), ideal.cpu().double()
        error = (value - ideal.to(dtype).double()).abs()
        bound = 2 * torch.finfo(dtype).eps * ideal.abs() + floor
        if dtype == torch.float64:
            torch.testing.assert_close(value, ideal, atol=2e-12, rtol=2e-11)
        else:
            assert bool((error <= bound).all()), dict(
                stage=name,
                max_abs_error=error.max().item(),
                failures=(error > bound).nonzero()[:8].tolist(),
            )
        checks.append(
            dict(
                stage=name,
                passed=True,
                max_rounded_abs_error=error.max().item(),
                floor=floor,
            )
        )

    def norm_ideal(source, key):
        value = source.cpu().double()
        gamma = 1 + raw[prefix + key].double()
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * gamma

    def linear(name, p, source, target, key, scratch, q8=True):
        p.run(source, out=target, workspace=scratch)
        weight = (
            q8_source_oracle(raw[prefix + key]) if q8 else raw[prefix + key].double()
        )
        check(
            name, target, source.cpu().double() @ weight.T, floor=2e-4 if q8 else 2e-5
        )

    plan.norm.run(x, out=w.normalized)
    check("input_norm", w.normalized, norm_ideal(x, "input_layernorm.weight"))
    for name, p, target, key, scratch, q8 in zip(
        ("qkv", "z", "a", "b"),
        plan.projections[:4],
        (w.qkv, w.z, w.a, w.b),
        ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
        w.projection_scratch[:4],
        (True, True, False, False),
    ):
        linear(
            name, p, w.normalized, target, "linear_attn." + key + ".weight", scratch, q8
        )
    h, s = torch.empty_like(history), torch.empty_like(state)
    plan.mixer.conv.run(w.qkv, history=history, out=w.mixer.convolved, history_out=h)
    conv, expected_history = conv_oracle(
        w.qkv, raw[prefix + "linear_attn.conv1d.weight"].squeeze(1).T, history
    )
    check("convolution", w.mixer.convolved, conv)
    assert torch.equal(h.cpu(), expected_history)
    plan.mixer.gates.run(w.a, w.b, log_decay=w.mixer.log_decay, beta=w.mixer.beta)
    dt = w.a.cpu().double() + raw[prefix + "linear_attn.dt_bias"].double()
    ideal_g = -raw[prefix + "linear_attn.A_log"].double().exp() * (
        torch.log1p((-dt.abs()).exp()) + dt.clamp(min=0)
    )
    ideal_beta = w.b.cpu().double().sigmoid().to(dtype).double()
    for name, actual, ideal in (
        ("log_decay", w.mixer.log_decay, ideal_g),
        ("beta", w.mixer.beta, ideal_beta),
    ):
        value = actual.cpu().double()
        error = (value - ideal).abs()
        assert bool(
            (
                error
                <= (
                    2e-12 + 2e-11 * ideal.abs()
                    if dtype == torch.float64
                    else 2e-7 + 2e-6 * ideal.abs()
                )
            ).all()
        ), name
        checks.append(dict(stage=name, passed=True, max_abs_error=error.max().item()))
    plan.mixer.recurrence.run_packed(
        w.mixer.convolved,
        w.mixer.log_decay,
        w.mixer.beta,
        state=state,
        out=w.mixer.recurrent,
        state_out=s,
    )
    rec = plan.mixer.recurrence
    width = rec.qk_heads * rec.key_dim
    packed = w.mixer.convolved.cpu()
    q, k, v = (
        packed[:, :width].reshape(rec.tokens, rec.qk_heads, rec.key_dim),
        packed[:, width : 2 * width].reshape(rec.tokens, rec.qk_heads, rec.key_dim),
        packed[:, 2 * width :].reshape(rec.tokens, rec.value_heads, rec.value_dim),
    )
    expected, final = gdn_oracle(q, k, v, w.mixer.log_decay, w.mixer.beta, state)
    check("recurrence", w.mixer.recurrent, expected)
    torch.testing.assert_close(
        s.cpu().double(),
        final,
        atol=2e-12 if dtype == torch.float64 else 2e-6,
        rtol=2e-11 if dtype == torch.float64 else 2e-4,
    )
    plan.mixer.gated_norm.run(w.mixer.recurrent, w.z_heads, out=w.mixed_heads)
    value = w.mixer.recurrent.cpu().double()
    normalized = (
        (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))
        .to(dtype)
        .double()
    )
    weighted = normalized * raw[prefix + "linear_attn.norm.weight"].double()
    if raw[prefix + "linear_attn.norm.weight"].dtype != torch.float32:
        weighted = weighted.to(dtype).double()
    z = w.z_heads.cpu().double()
    check("gated_norm", w.mixed_heads, weighted * z * z.sigmoid())
    linear(
        "output_projection",
        plan.output,
        w.mixed,
        w.projected,
        "linear_attn.out_proj.weight",
        w.projection_scratch[4],
    )
    plan.add.run(x, w.projected, out=w.residual)
    check("mixer_residual", w.residual, x.cpu().double() + w.projected.cpu().double())
    plan.mlp.norm.run(w.residual, out=w.mlp.hidden)
    check(
        "post_attention_norm",
        w.mlp.hidden,
        norm_ideal(w.residual, "post_attention_layernorm.weight"),
    )
    for name, p, target in (
        ("gate", plan.mlp.gate, w.mlp.gate),
        ("up", plan.mlp.up, w.mlp.up),
    ):
        linear(
            "mlp_" + name,
            p,
            w.mlp.hidden,
            target,
            "mlp." + name + "_proj.weight",
            w.mlp.linear if p.workspace_bytes else None,
        )
    gate, up = w.mlp.gate.cpu().double().clone(), w.mlp.up.cpu().double().clone()
    plan.mlp.swiglu.run(w.mlp.gate, w.mlp.up, out=w.mlp.gate)
    check("swiglu", w.mlp.gate, gate * gate.sigmoid() * up)
    linear(
        "mlp_down",
        plan.mlp.down,
        w.mlp.gate,
        w.mlp.hidden,
        "mlp.down_proj.weight",
        w.mlp.linear if plan.mlp.down.workspace_bytes else None,
    )
    out = torch.empty_like(x)
    plan.mlp.add.run(w.residual, w.mlp.hidden, out=out)
    check("mlp_residual", out, w.residual.cpu().double() + w.mlp.hidden.cpu().double())
    return out, h, s, checks
