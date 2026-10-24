"""Finite aligned cross-engine probability diagnostic, never a TPS/corpus test.

The source-pinned engines have different Q8 encodings/small parameters/internal
arithmetic. Native full logits retain their independently qualified98-row gate.
This diagnostic samples11 aligned next-token positions across four cases, checks
finite probabilities and records top1/overlap/log-probability differences without
claiming exact arithmetic or a full-vocabulary cross-engine KL. Requests use
pre-sampling probabilities; fail rather than compare token strings or a greedy
post-sampling point mass. See llama.cpp tools/server/README.md (official GitHub).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

PROFILE = 'q8-residual-column-mlp-head-gdn'
QUAL_SHA = '6e4c1623bc10488bbcb92f92d26d9f0346c3f727db65673acb9808e3970dba50'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_probabilities(response, vocab=248320):
    rows = response.get('completion_probabilities', response.get('probs'))
    assert isinstance(rows, list) and len(rows) == 1, 'Missing one next-token probability row'
    row = rows[0]
    choices = row.get('top_logprobs', row.get('probs'))
    assert isinstance(choices, list) and len(choices) == 64, 'Missing requested64 pre-sampling alternatives'
    result = {}
    for value in choices:
        token = value.get('id')
        assert type(token) is int and 0 <= token < vocab and token not in result, 'Missing/duplicate/out-of-range token IDs'
        if 'logprob' in value:
            lp = value['logprob']
            assert isinstance(lp, (int, float)) and math.isfinite(lp) and lp <= 1e-5
        else:
            p = value.get('prob')
            assert isinstance(p, (int, float)) and math.isfinite(p) and 0 < p <= 1
            lp = math.log(p)
        result[token] = lp
    mass = sum(math.exp(x) for x in result.values())
    assert 0 < mass <= 1.0001
    assert max(result.values()) < -1e-8, 'Reject a post-greedy point mass'
    assert max(result, key=result.get) == response['tokens'][0], 'Top1 and generated ID disagree'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['native', 'external'])
    p.add_argument('--artifact', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--qualified-report', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--output-directory', required=True)
    p.add_argument('--url', default='http://127.0.0.1:18953')
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    assert sha(a.qualified_report) == QUAL_SHA
    qualified = json.loads(Path(a.qualified_report).read_text())
    assert sha(a.library) == qualified['library_sha256']
    assert sha(a.cases) == qualified['cases_sha256']
    refs = qualified['profiles'][PROFILE]['candidate']['cases']
    cases = json.loads(Path(a.cases).read_text())
    selected = [(i, step) for i, ref in enumerate(refs) for step in sorted({0, 1, min(7, len(ref['generated_ids']) - 1)})]
    assert len(selected) == 11
    report_path = out / 'native.json'
    if a.mode == 'native':
        from venturi.models import NativeQwenModel, QwenQ8Weights
        from venturi.ninfer import NInferArtifact
        from venturi.ops import CudaBackend
        from qualify_qwen_projection_variant import tensor_signature
        report = dict(scope=__doc__, qualified_report_sha256=QUAL_SHA, library_sha256=sha(a.library), cases_sha256=sha(a.cases), rows=[], completed=False)
        probabilities = {}
        model = NativeQwenModel.from_q8(QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)), CudaBackend(a.library), weight_mode='resident', dtype=torch.float16, device='cuda:0', max_tokens=32, tensor_parallel_device='cuda:1', tensor_parallel_transport='host', projection_profile=PROFILE)
        try:
            for i, case in enumerate(cases):
                session = model.create_session(256)
                try:
                    for step in range(max(s for j, s in selected if i == j) + 1):
                        logits = session.forward(case['prompt_token_ids'] if step == 0 else [refs[i]['generated_ids'][step - 1]])
                        assert tensor_signature(logits) == refs[i]['all_logit_bytes'][step], 'Qualified native logits changed'
                        assert bool(torch.isfinite(logits).all())
                        if (i, step) not in selected:
                            continue
                        lp = torch.log_softmax(logits.cpu().double().flatten(), dim=0)
                        assert bool(torch.isfinite(lp).all())
                        key = f'{i}:{step}'
                        probabilities[key] = lp
                        values, tokens = torch.topk(lp, 64)
                        report['rows'].append(dict(key=key, case=case['name'], step=step, prompt=case['prompt_token_ids'] + refs[i]['generated_ids'][:step], top1=int(logits.argmax()), top64=[dict(id=int(t), logprob=float(v)) for t, v in zip(tokens, values)], full_logit_signature=refs[i]['all_logit_bytes'][step], finite_full_vocabulary=True))
                finally:
                    session.close()
        finally:
            model.close()
        path = out / 'native-logprobs.pt'
        torch.save(probabilities, path)
        report.update(completed=True, full_logprob_file=dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    else:
        native = json.loads(report_path.read_text())
        assert native['completed'] and native['qualified_report_sha256'] == QUAL_SHA
        pin = native['full_logprob_file']
        assert sha(pin['path']) == pin['sha256']
        probs = torch.load(pin['path'], map_location='cpu', weights_only=True)
        report = dict(scope=__doc__, native_report_sha256=sha(report_path), rows=[], completed=False, full_vocabulary_cross_engine_kl=None, original_bf16_corpus_quality=False)
        for row in native['rows']:
            payload = dict(prompt=row['prompt'], n_predict=1, id_slot=0, cache_prompt=False, ignore_eos=True, stream=False, return_tokens=True, temperature=0, top_k=1, top_p=1, min_p=0, repeat_penalty=1, seed=42, n_probs=64, post_sampling_probs=False)
            (out / (row['key'].replace(':', '-') + '-request.json')).write_text(json.dumps(payload, indent=2) + '\n')
            request = Request(a.url + '/completion', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=120) as r:
                response = json.load(r)
            (out / (row['key'].replace(':', '-') + '-response.json')).write_text(json.dumps(response, indent=2) + '\n')
            assert response['tokens_evaluated'] == response['timings']['prompt_n'] == len(row['prompt'])
            assert response['tokens_predicted'] == len(response['tokens']) == 1
            choices = parse_probabilities(response)
            lp = probs[row['key']]
            native_top = {x['id'] for x in row['top64']}
            differences = [float(lp[t]) - value for t, value in choices.items()]
            report['rows'].append(dict(key=row['key'], case=row['case'], step=row['step'], input_ids_aligned=True, native_top1=row['top1'], external_top1=response['tokens'][0], top1_equal=row['top1'] == response['tokens'][0], top64_overlap=len(native_top & choices.keys()), external_top64_mass=sum(math.exp(v) for v in choices.values()), max_abs_logprob_difference_over_external_top64=max(map(abs, differences)), native_logprobs_for_external_top64={str(t):float(lp[t]) for t in choices}, external_top64={str(t):v for t,v in choices.items()}, finite=True))
            (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        report.update(completed=True, aligned_rows=len(report['rows']), top1_equal_count=sum(r['top1_equal'] for r in report['rows']))
        (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Completed aligned', a.mode, 'diagnostic')


if __name__ == '__main__':
    main()
