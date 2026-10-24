"""Freeze recovered controls and explicit outstanding acceptance requirements."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never replace frozen evidence'
    assert not (OUT / 'source').exists(), 'Inspect an unfinished source snapshot before retrying'
    audit = read(OUT / 'completion-audit-final.json')
    assert audit['scoped_primary_throughput_goal_met'] and len(audit['captures']) == 16
    assert audit['complete_paired_mtp1_matrix_available'] is False
    prior = read(OUT / 'prior-capsule-audit.json')
    assert len(prior) == 43
    verified = {}
    for name in prior:
        cap = ROOT / 'benchmarks' / name
        missing, count = [], 0
        for filename, base in [('source-sha256.json', cap / 'source'), ('evidence-sha256.json', cap)]:
            if not (cap / filename).is_file():
                continue
            manifest = read(cap / filename)
            manifest = manifest.get('files', manifest)
            for rel, expected in manifest.items():
                p = base / rel
                if not p.is_file() and name == 'decoder-layer':
                    p = base / Path(rel).name
                if p.is_file():
                    assert sha(p) == expected, str(p)
                    count += 1
                else:
                    missing.append(rel)
        assert not missing or name == 'decoder-layer'
        verified[name] = dict(verified_files=count, legacy_missing=missing)
    source = read(OUT / 'source-manifest.json')
    actual = {str(p.relative_to(ROOT)):sha(p) for folder in ['src','tests','tools'] for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert actual == source and len(source) == 295
    for tag, record in audit['captures'].items():
        manifest = read(OUT / (tag + '-capture-manifest.json'))
        directory = OUT / 'v100' / tag
        assert len(manifest['manifest']) == record['files']
        assert all(sha(directory / rel) == item['sha256'] for rel, item in manifest['manifest'].items())
        assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == set(manifest['manifest'])
    assert '28 passed' in (OUT / 'recovery-parser-tests-final.log').read_text()
    assert read(OUT / 'selected-comparison.json')['numeric_goal_met']
    audit.update(evidence_freeze_pending=False, full_goal_complete=False, outstanding_requirements=['Repeatable complete paired one-proposal MTP comparator controls at2/8 requests'], prior_frozen_capsules_verified=43)
    (OUT / 'completion-audit-final.json').write_text(json.dumps(audit, indent=2) + '\n')
    pending = read(OUT / 'pending-files.json')
    pending.update(frozen=True, evidence_capture_incomplete=False, required_remote_connectivity_unresolved=False, final_comparator_and_diagnostic_pending=False, goal_complete=False, model_jobs_terminal=True, capture_jobs_finished=True)
    (OUT / 'pending-files.json').write_text(json.dumps(pending, indent=2) + '\n')
    files = set(source)
    files.update(['AGENTS.md', 'README.md', 'pyproject.toml', 'docs/native-q8-goal.md', 'docs/native-mtp-plan.md'])
    files.update(str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.py', '.md'])
    (OUT / 'source').mkdir()
    for rel in sorted(files):
        p = OUT / 'source' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, p)
    source_hashes = {rel:sha(OUT / 'source' / rel) for rel in sorted(files)}
    (OUT / 'source-sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    evidence = {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source' not in p.relative_to(OUT).parts and '__pycache__' not in p.parts and p.suffix != '.pyc' and p.name not in ['evidence-sha256.json','evidence-check.json']}
    (OUT / 'evidence-sha256.json').write_text(json.dumps(evidence, indent=2) + '\n')
    assert all(sha(OUT / rel) == expected for rel, expected in evidence.items())
    assert all(sha(OUT / 'source' / rel) == expected for rel, expected in source_hashes.items())
    result = dict(frozen=True, completed_utc=datetime.now(timezone.utc).isoformat(), scoped_numeric_goal_met=True, full_goal_complete=False, complete_paired_mtp1_matrix_available=False, source_files=len(source_hashes), evidence_files=len(evidence), raw_jobs=16, raw_job_files=sum(row['files'] for row in audit['captures'].values()), qualified_source_files_unchanged=295, prior_capsules_verified=43, prior_verification=verified, outstanding_requirements=audit['outstanding_requirements'], no_commit_or_push=True)
    (OUT / 'evidence-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'prior_verification'}, indent=2))

if __name__ == '__main__':
    main()
