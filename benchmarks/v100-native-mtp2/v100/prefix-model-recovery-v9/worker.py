import pathlib,json,hashlib,time,traceback,shutil
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-native-mtp2-prefix-model-recovery-v9';prior=root/'artifacts/venturi-native-mtp2-prefix-model-v6';study=root/'artifacts/venturi-native-mtp2-prefix-v5';src=root/'repos/codex-build/Venturi/native-mtp2-src-v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 assert (prior/'exit-code.txt').read_text().strip()=='1' and "KeyError: 'recovery'" in (prior/'failure.txt').read_text()
 q=json.loads((prior/'qualification/report.json').read_text());assert q['completed'] and q['target_recovery']=='prefix' and q['draft_tokens']==2
 assert len(q['cases'])==4 and len(q['recurrence_reference'])==3 and all(x['passed'] for x in q['recurrence_reference'])
 for c in q['cases']:
  s=c['mtp_stats'];assert c['output_ids_equal'] and c['committed_target_state_bytes_equal'] and c['committed_draft_state_equal_mtp1'] and c['pinned_target_trajectory_state_reused'];assert s['forced_proposals']==s['replayed_input_tokens']==0 and s['target_recovery']=='prefix-trace' and s['prefix_trace_extra_primary_bytes']==s['primary_target_backup_bytes']
 library=pathlib.Path((study/'library.txt').read_text().strip());assert sha(library)==q['library_sha256']=='9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894'
 assert '1768 passed, 448 skipped' in (study/'full-tests.log').read_text()
 build=json.loads(library.with_name('build.json').read_text())
 for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 (out/'qualification').mkdir();shutil.copy2(prior/'qualification/report.json',out/'qualification/report.json');(out/'library.txt').write_text(str(library)+'\n')
 (out/'recovery.json').write_text(json.dumps(dict(scope='Reporting-only recovery: trained qualification command completed and all numerical assertions passed. Wrapper used recovery instead of target_recovery, then failed after completion. Verify and reuse the exact completed report; do not rerun unchanged GPU/CPU qualification.',prior_report=str(prior/'qualification/report.json'),prior_report_sha256=sha(prior/'qualification/report.json'),prior_failure_sha256=sha(prior/'failure.txt'),fixture_report=str(study/'full-tests.log'),fixture_report_sha256=sha(study/'full-tests.log')),indent=2)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):sha(p) for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=str(library),sha256=sha(library),ended_unix=time.time()),indent=2)+'\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
