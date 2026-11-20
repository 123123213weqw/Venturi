"""Hash-bound CPU qualification of DS41 residual-stream semantics.

Extracts selected pure-Torch methods, never imports the official model, loads a
checkpoint, compiles a GPU kernel or starts inference. Sinkhorn CPU equations
have independent scalar tests; official TileLang GPU equivalence is NOT tested.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from venturi.models.hyperconnections import (
    HyperConnectionSpec, reference_hc_coefficients, reference_hc_mixes,
    reference_hc_pre, reference_hc_post, reference_identity_pre,
)


def extract(source, class_name, name, **extra):
    tree = ast.parse(source)
    scope = tree if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = copy.deepcopy(next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name))
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = dict(torch=torch, F=F, **extra)
    exec(compile(module, '<explicit hashed official residual method>', 'exec'), env)
    return env[name]


def qualify(source):
    pre = extract(source, 'Block', 'hc_pre')
    post = extract(source, 'Block', 'hc_post')
    identity = extract(source, None, 'make_identity_pre_mix')
    spec = HyperConnectionSpec()
    captured = []

    def cpu_split(projected, scales, bases, h, iters, eps):
        captured.append(projected.clone())
        return reference_hc_coefficients(projected, scales, bases, HyperConnectionSpec(h, iters, eps))

    mixes = extract(source, 'Block', 'hc_mixes', hc_split_sinkhorn=cpu_split)
    block = extract(source, 'Block', 'forward')
    oracle = SimpleNamespace(norm_eps=spec.norm_epsilon, hc_mult=spec.streams,
        hc_sinkhorn_iters=spec.sinkhorn_iterations, hc_eps=spec.epsilon)
    rng = torch.Generator().manual_seed(4171017)
    records = []

    def record(kind, observed, expected, **metadata):
        records.append(dict(kind=kind, byte_exact=torch.equal(observed, expected),
            max_abs=(observed.float()-expected.float()).abs().max().item(), **metadata))

    with torch.no_grad():
        for norm_epsilon in (1e-20, 1e-6):
            spec = HyperConnectionSpec(norm_epsilon=norm_epsilon)
            oracle.norm_eps = norm_epsilon
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for b, t, d in ((1, 1, 8), (2, 3, 32), (1, 2, 5120)):
                    x = torch.randn(b, t, 4, d, generator=rng).to(dtype)
                    previous = torch.rand(b, t, 4, generator=rng)
                    out = torch.randn(b, t, d, generator=rng).to(dtype)
                    expansion = torch.rand(b, t, 4, generator=rng)
                    combination = torch.rand(b, t, 4, 4, generator=rng)
                    meta = dict(dtype=str(dtype), shape=list(x.shape), norm_epsilon=norm_epsilon)
                    record('collapse', reference_hc_pre(x, previous), pre(oracle, x, previous), **meta)
                    record('expand', reference_hc_post(out, x, expansion, combination),
                        post(oracle, out, x, expansion, combination), **meta)
                    record('initial_stream_selection', reference_identity_pre(x), identity(x, 4), **meta)
                    weight = torch.randn(24, 4*d, generator=rng)*.01
                    scales = torch.tensor([.1, .3, .2])
                    bases = torch.randn(24, generator=rng)
                    expected = mixes(oracle, x, weight, scales, bases)
                    observed = reference_hc_mixes(x, weight, scales, bases, spec)
                    # Compares projection/RMS integration only: both use the same
                    # injected CPU split, NOT the official TileLang Sinkhorn kernel.
                    for field, actual, target in zip(('pre', 'post', 'comb'), observed, expected):
                        record('projection_with_injected_cpu_split_'+field, actual, target, **meta)
        # Execute official Block.forward on deterministic fake sublayers to
        # check the one-sublayer lag, not checkpoint attention/FFN arithmetic.
        x = torch.randn(1, 2, 4, 8, generator=rng)
        prior = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        a_pre = torch.tensor([[[0., 0., 1., 0.], [0., 0., 0., 1.]]])
        f_pre = torch.ones_like(prior)*.25
        a_post, f_post = torch.rand(1, 2, 4, generator=rng), torch.rand(1, 2, 4, generator=rng)
        a_comb, f_comb = torch.rand(1, 2, 4, 4, generator=rng), torch.rand(1, 2, 4, 4, generator=rng)
        seq = iter(((a_pre, a_post, a_comb), (f_pre, f_post, f_comb)))
        fake = SimpleNamespace(hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None,
            hc_ffn_fn=None, hc_ffn_scale=None, hc_ffn_base=None,
            hc_mixes=lambda *args: next(seq), hc_pre=reference_hc_pre, hc_post=reference_hc_post,
            attn_norm=lambda v: v+1, attn=lambda v, *args: v*2,
            ffn_norm=lambda v: v-1, ffn=lambda v, *args: v*3)
        official_stream, official_pre = block(fake, x, 0, prior, None)
        intermediate = reference_hc_post((reference_hc_pre(x, prior)+1)*2, x, a_post, a_comb)
        expected = reference_hc_post((reference_hc_pre(intermediate, a_pre)-1)*3,
            intermediate, f_post, f_comb)
        record('lagged_block_order_fake_sublayers', expected, official_stream)
        record('next_block_receives_ffn_pre', f_pre, official_pre)
        wrong = reference_hc_post((reference_hc_pre(x, a_pre)+1)*2, x, a_post, a_comb)
        records.append(dict(kind='lag_counterexample_retained',
            byte_exact=not torch.equal(wrong, intermediate),
            max_abs=(wrong-intermediate).abs().max().item()))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    raw = Path(args.reference_source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.reference_sha256:
        raise ValueError('Explicit official source SHA256 mismatch')
    records = qualify(raw.decode())
    result = dict(scope='Synthetic CPU residual/projection/order only; no real weights, GPU or TPS',
        official_repository='deepseek-ai/DeepSeek-V4.1-Flash-1017',
        official_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',
        official_model_source_sha256=digest, torch_version=torch.__version__, device='cpu',
        sinkhorn_gpu_verified=False, all_passed=all(r['byte_exact'] for r in records), cases=records)
    with Path(args.output).open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
