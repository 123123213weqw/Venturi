"""Validate immutable inputs, executed sources and scoped acceptance evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LIB = '33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tag_for(name):
    kind = 'profile' if '-controls-profile-' in name else 'controls' if '-controls-' in name else 'aligned'
    return kind + '-' + name.rsplit('-', 1)[1]

def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    terminal = read(OUT / 'terminal-audit.json')
    expected = {1:1, 2:1, 4:1, 5:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:0, 14:1, 15:0, 16:1, 17:0, 18:0}
    source = read(OUT / 'source-manifest.json')
    assert terminal['source_manifest'] == source
    actual = {str(p.relative_to(ROOT)):sha(p) for folder in ['src','tests','tools'] for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert actual == source and len(source) == 295
    assert terminal['library_sha256'] == LIB
    assert not terminal['port18953'] and not terminal['remaining_artifact_processes']
    assert not [row for row in terminal['periodic_samplers'] if row['owned']]
    assert '3651 MiB' in terminal['gpu_after'] and '3189 MiB' in terminal['gpu_after']
    assert 'RmProfilingAdminOnly: 1' in terminal['counter_policy']
    seeds = read(OUT / 'source-deduplication-seeds.json')
    seed_audit = []
    captures = {}
    for name, state in terminal['workers'].items():
        version = int(name.rsplit('v', 1)[1])
        assert state['exit_code'] == expected[version]
        tag = tag_for(name)
        capture = read(OUT / (tag + '-capture-manifest.json'))
        directory = OUT / 'v100' / tag
        assert capture['exit_code'] == state['exit_code']
        assert sha(directory / 'worker.py') == state['worker_sha256']
        assert all(sha(directory / rel) == row['sha256'] and (directory / rel).stat().st_size == row['bytes'] for rel, row in capture['manifest'].items())
        # A preflight-failed job may never have emitted a seeded common source.
        # Remove only our logged, unchanged provisional files absent remotely.
        for seed in [row for row in seeds if row['tag'] == tag]:
            p = directory / seed['file']
            if seed['file'] not in capture['manifest']:
                if p.exists():
                    assert sha(p) == seed['sha256'], 'Unknown local modification: never remove it'
                    p.unlink()
                seed_audit.append(dict(tag=tag, file=seed['file'], status='Absent remotely; provisional seed removed, not treated as executed evidence'))
            else:
                seed_audit.append(dict(tag=tag, file=seed['file'], status='Remote manifest confirmed or replaced the provisional seed', sha256=sha(p)))
        found = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        assert found == set(capture['manifest']), (tag, sorted(found - set(capture['manifest'])))
        captures[tag] = dict(files=len(found), exit_code=state['exit_code'], manifest_sha256=sha(OUT / (tag + '-capture-manifest.json')))
    (OUT / 'source-deduplication-validation.json').write_text(json.dumps(seed_audit, indent=2) + '\n')
    assert len(captures) == 16
    last = OUT / 'v100/controls-v17'
    campaign = read(last / 'campaign-report.json')
    assert campaign['completed'] and campaign['source_sha256'] == source and campaign['library_sha256'] == LIB
    assert read(last / 'payload-before.json') == read(last / 'payload-after.json') == terminal['payload_identity']
    pins = read(last / 'fresh-external-pins.json')
    assert pins['identity'] == terminal['payload_identity']
    assert 'eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513' in json.dumps(pins)
    assert pins['binary_sha256'] == 'e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2'
    prior = read(OUT.parent / 'v100-native-head-gdn/reused-full-payload-pins.json')
    assert all(terminal['payload_identity'][key] == row for key, row in prior['identity'].items())
    assert 'b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8' in json.dumps(prior)
    assert 'a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641' in json.dumps(prior)
    qpath = OUT / 'v100/controls-v1/qualification256/report.json'
    assert sha(qpath) == campaign['qualified256_sha256']
    quality = read(qpath)
    assert quality['completed'] and quality['teacher_rows_reused'] == 98
    candidate = quality['profiles']['q8-residual-column-mlp-head-gdn']['candidate']
    assert candidate['passed'] and all(c['passed'] for case in candidate['cases'] for c in case['teacher_logit_checks'])
    diagnostic = read(OUT / 'v100/aligned-v18/completed.json')
    assert diagnostic['completed'] and diagnostic['finite_aligned_rows'] == 11 and diagnostic['native_full_logit_hashes_equal_qualified']
    profile = read(OUT / 'v100/profile-v15/completed.json')
    assert profile['completed'] and profile['separate_from_unprofiled'] and profile['library_sha256'] == LIB
    assert profile['production_source_unchanged'] and profile['achieved_hardware_counters'] is None
    assert '28 passed' in (OUT / 'recovery-parser-tests-v2.log').read_text()
    metrics = read(OUT / 'selected-comparison.json')
    assert metrics['numeric_goal_met'] and metrics['single_post_first']['all_trials_clear_floor']
    assert all(row['all_fresh_trials_clear_floor'] and row['all_prior_trials_clear_new_floor'] and row['all_fresh_trials_also_clear_highest_screen_floor'] for row in metrics['selected_http'].values())
    controls = read(OUT / 'recomputed-controls.json')
    mtp1 = {str(count):[dict(tag=row['tag'], label=row['label'], http_tps=row['http_tps']) for row in controls['repeated_cells'] if row['engine']=='llama' and row['depth']==1 and row['requests']==count] for count in [1,2,4,8]}
    result = dict(
        audited_utc=datetime.now(timezone.utc).isoformat(),
        scoped_primary_throughput_goal_met=True,
        qualification_passed=True,
        captures=captures,
        source_files_unchanged=295,
        native_library_sha256=LIB,
        source_checkpoint_and_precision_pins_verified=True,
        strongest_historical_or_fresh_controls_retained=True,
        native_fresh_and_prior_trials_clear_5pct_floors=True,
        native_fresh_trials_also_clear_highest_valid_screens=True,
        external_repeated_mtp1_coverage=mtp1,
        complete_paired_mtp1_matrix_available=all(mtp1.values()),
        comparison_scope='Strongest repeatably completed pinned build per primary cell, including best supported tested speculative configurations; unsuccessful arms are not zero TPS. Repeated MTP1 at unsupported counts is unavailable, not a claimed numerical win.',
        cross_engine_top1_equal_count=diagnostic['top1_equal_count'],
        cross_engine_aligned_rows=11,
        exact_arithmetic_or_corpus_parity=False,
        achieved_hardware_counters=None,
        streaming_client_ttft_available=False,
        native_first_ready_and_external_prompt_timing_labeled_separately=True,
        model_loading_excluded=True,
        automatic_strategy_router=False,
        paged_attention_or_continuous_batching_claim=False,
        universal_speed_or_latest_upstream_claim=False,
        exact_competitor_source_commit_verified=False,
        no_production_change_this_phase=True,
        no_local_rust_or_cuda_compile_this_phase=True,
        no_commit_or_push=True,
        terminal_cleanup_verified=True,
        evidence_freeze_pending=True,
    )
    (OUT / 'completion-audit-final.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(scoped_primary_throughput_goal_met=True, captured_jobs=len(captures), captured_files=sum(v['files'] for v in captures.values()), source_files=295, paired_mtp1_matrix_available=result['complete_paired_mtp1_matrix_available'], aligned_top1=diagnostic['top1_equal_count'], evidence_freeze_pending=True), indent=2))

if __name__ == '__main__':
    main()
