"""Freeze qualified native semantic-head GDN integration, not final goal success."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / 'v100'
LIB = '33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8'
QUAL = '6e4c1623bc10488bbcb92f92d26d9f0346c3f727db65673acb9808e3970dba50'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never replace frozen evidence'
    prior = read(OUT / 'prior-capsule-audit.json')
    assert len(prior) == 42
    for rel, h in read(OUT / 'prior-reference-pins.json').items():
        assert sha(ROOT / rel) == h, rel
    verified = {}
    for name in prior:
        cap = ROOT / 'benchmarks' / name
        missing, count = [], 0
        for filename, base in [('source-sha256.json', cap / 'source'), ('evidence-sha256.json', cap)]:
            if not (cap / filename).exists():
                continue
            m = read(cap / filename)
            m = m.get('files', m)
            found = 0
            for rel, h in m.items():
                p = base / rel
                if not p.is_file() and name == 'decoder-layer':
                    p = base / Path(rel).name
                if p.is_file():
                    assert sha(p) == h, str(p)
                    count += 1
                    found += 1
                else:
                    missing.append(rel)
            assert not missing or (name == 'decoder-layer' and found == 4)
        verified[name] = dict(verified_files=count, legacy_missing=missing)
    terminal = read(OUT / 'terminal-audit.json')
    assert {k: v['exit_code'] for k, v in terminal['workers'].items()} == {
        'fixtures-v1': 1, 'fixtures-v2': 1, 'model-v3': 1,
        'mtp-v4': 0, 'campaign-v5': 0, 'profile-v6': 0,
    }
    assert not terminal['owned_processes'] and not terminal['remaining_periodic_samplers'] and not terminal['port18953']
    assert '3651 MiB' in terminal['gpu_after'] and '3189 MiB' in terminal['gpu_after']
    assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy']
    for n in range(1, 4):
        m = read(OUT / f'source-manifest-v{n}.json')
        assert m == terminal['source_manifests'][str(n)]
        assert all(sha(OUT / f'executed-source-v{n}' / r) == h for r, h in m.items())
    latest = read(OUT / 'source-manifest-v3.json')
    current = {str(p.relative_to(ROOT)): sha(p) for d in ['src', 'tests', 'tools']
               for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert current == latest
    build = read(next((RAW / 'fixtures-v2').glob('build/*/build.json')))
    assert build['library_sha256'] == LIB and len(build['source_sha256']) == 11
    for r, h in build['source_sha256'].items():
        assert sha(ROOT / 'src/venturi/ops/csrc' / r) == h
        assert sha(OUT / 'executed-source-v2/src/venturi/ops/csrc' / r) == h
    assert '26 passed' in (RAW / 'fixtures-v2/focused-tests.log').read_text()
    assert '1 failed, 2242 passed, 719 skipped' in (RAW / 'fixtures-v2/full-tests.log').read_text()
    assert '38 passed' in (RAW / 'model-v3/regression-closure.log').read_text()
    assert '831 passed, 2131 skipped' in (OUT / 'local-tests-v3.log').read_text()
    assert 'FileNotFoundError' in (RAW / 'fixtures-v1/focused-tests.log').read_text()
    assert 'skip-block-reference' in (RAW / 'model-v3/mtp2-qualification.log').read_text()
    qp = RAW / 'model-v3/qualification/report.json'
    q = read(qp)
    assert sha(qp) == QUAL and q['completed'] and q['library_sha256'] == LIB and q['teacher_rows_reused'] == 98
    candidate = q['profiles']['q8-residual-column-mlp-head-gdn']['candidate']
    assert candidate['passed'] and len(candidate['cases']) == 4
    quality = read(OUT / 'qualified-target-metrics.json')
    assert quality['qualified'] and quality['all_top1_equal'] and quality['rows'] == 98
    assert quality['max_relative_l2'] <= .02 and quality['min_centered_cosine'] >= .999 and quality['max_reference_kl_to_native'] <= .002
    mtp = read(RAW / 'mtp-v4/mtp2-qualification/report.json')
    assert mtp['completed'] and mtp['library_sha256'] == LIB and mtp['prior_target_report_sha256'] == QUAL
    assert len(mtp['block_reference']) == 2 and len(mtp['recurrence_reference']) == 3
    assert all(c['passed'] for c in mtp['block_reference'] + mtp['recurrence_reference'])
    assert len(mtp['cases']) == 4
    for c in mtp['cases']:
        assert all(c[k] for k in ['output_ids_equal', 'committed_target_state_bytes_equal', 'committed_draft_state_equal_mtp1', 'native_without_python_math'])
    batch = read(RAW / 'mtp-v4/mtp-batch-qualification/report.json')
    assert batch['completed'] and batch['library_sha256'] == LIB and batch['prior_target_report_sha256'] == QUAL
    assert len(batch['full_references']) == 4 and len(batch['cohorts']) == 2 and batch['selective_reset_untouched_state_exact']
    assert all(c['exact_target_state'] and c['exact_corrected_draft_state'] and c['native_without_python_math'] for c in batch['cohorts'])
    metrics = read(OUT / 'metrics.json')
    assert metrics['library_sha256'] == LIB and metrics['numeric_goal_against_retained_baselines_met'] and not metrics['full_goal_complete']
    assert metrics['ordinary256_candidate_not_tested'] and not metrics['automatic_strategy_router']
    assert all(c['trials'] == 3 for m in metrics['matrices'].values() for a in ['control', 'candidate'] for c in m[a])
    assert len(metrics['separate_profiles']) == 4 and metrics['achieved_hardware_counters'] is None
    profile = read(RAW / 'profile-v6/completed.json')
    resources = read(RAW / 'profile-v6/resources.json')
    assert profile['completed'] and profile['separate_from_unprofiled'] and profile['library_sha256'] == resources['library_sha256'] == LIB
    assert profile['achieved_hardware_counters'] is resources['achieved_occupancy'] is None
    pins = read(OUT / 'reused-full-payload-pins.json')
    assert pins['identity'] == terminal['payload_identity']
    assert mtp['mtp_artifact_sha256'] in pins['full_payload_hashes'].values()
    for tag in ['mtp-v4', 'campaign-v5', 'profile-v6']:
        assert read(RAW / tag / 'payload-before.json') == read(RAW / tag / 'payload-after.json') == pins['identity']
    files = [ROOT / r for r in latest] + [ROOT / r for r in ['AGENTS.md', 'README.md', 'pyproject.toml', 'docs/native-q8-goal.md', 'docs/native-mtp-plan.md']]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.md', '.py')]
    sources = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT)
        dst = OUT / 'source' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        sources[str(rel)] = sha(p)
    (OUT / 'source-sha256.json').write_text(json.dumps(sources, indent=2, sort_keys=True) + '\n')
    check = dict(frozen=True, created_utc=datetime.now(timezone.utc).isoformat(), source_files=len(sources), prior_capsules_verified=42,
                 prior_verification=verified, native_profile_integrated=True, qualified_library_sha256=LIB,
                 focused_two_gpu_tests_passed=26, remote_full_suite=dict(passed=2242, skipped=719, failed_stale_dependency_count=1),
                 corrected_dependency_assertion_and_control_closure_tests=38, local_tests_passed=831, local_skipped=2131,
                 trained_target_teacher_rows=98, trained_mtp_and_private_cache_gates_passed=True,
                 eight_screen_experiments_and_four_three_repeat_matrices=True, numeric_goal_against_retained_baselines_met=True,
                 final_matched_controls_and_completion_audit_pending=True, full_goal_complete=False,
                 achieved_hardware_counters=None, counter_policy_unchanged=True, six_workers_terminal=True,
                 original_gpu_footprint_restored=True, no_local_rust_cuda_compilation=True, no_commit_or_push=True)
    (OUT / 'evidence-check.json').write_text(json.dumps(check, indent=2) + '\n')
    evidence = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source' not in p.relative_to(OUT).parts
                and p.name != 'evidence-sha256.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    (OUT / 'evidence-sha256.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(source_files=len(sources), evidence_files=len(evidence), prior_capsules_verified=42, goal_complete=False)))

if __name__ == '__main__':
    main()
