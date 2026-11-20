from pathlib import Path
import json
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);out=Path('benchmarks/v100-native-head-gdn')
s=Path('/tmp/venturi-native-head-gdn-model-v3-worker.py').read_text();s=s[:s.index('QUALIFIER=')].replace("out=root/'artifacts/venturi-native-head-gdn-model-v3'", "out=root/'artifacts/venturi-native-head-gdn-mtp-v4'")
expected=json.loads((out/'source-manifest-v3.json').read_text())
s+='try:\n expected='+repr(expected)+r'''
 prior=root/'artifacts/venturi-native-head-gdn-model-v3';assert (prior/'exit-code.txt').read_text().strip()=='1'
 assert 'New draft arithmetic requires non-PP and fresh block reference checks' in (prior/'mtp2-qualification.log').read_text()
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 report=prior/'qualification/report.json';quality=json.loads(report.read_text());candidate=quality['profiles']['q8-residual-column-mlp-head-gdn']['candidate'];assert quality['completed'] and candidate['passed'] and len(candidate['cases'])==4 and sum(len(c['all_logit_bytes']) for c in candidate['cases'])==98
 library=pathlib.Path((prior/'library.txt').read_text().strip());build=json.loads(library.with_name('build.json').read_text());assert quality['library_sha256']==sha(library)==build['library_sha256'] and len(build['source_sha256'])==11
 for rel,h in build['source_sha256'].items():assert sha(src/'src/venturi/ops/csrc'/rel)==h
 (out/'reused-target-qualification.json').write_text(json.dumps(dict(path=str(report),sha256=sha(report),passed=True,teacher_rows=98,no_rerun=True),indent=2)+chr(10));(out/'library.txt').write_text(str(library)+chr(10))
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bf16=root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16'
 before={str(p):identity(p) for p in [target,draft]};assert before==json.loads((prior/'payload-before.json').read_text());(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 env['PYTHONPATH']+=':'+str(src/'tools')
 cmd=[py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--source',str(bf16),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual-column-mlp-head-gdn','--mtp-draft-projection-profile','q8-residual','--prior-target-report',str(report),'--prior-target-report-sha256',sha(report),'--draft-tokens','2','--recovery','prefix','--output-directory',str(out/'mtp2-qualification')]
 run(cmd,'mtp2-qualification.log',timeout=2700)
 cmd=[py,'tools/qualify_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--cases',str(cases),'--projection-profile','q8-residual-column-mlp-head-gdn','--mtp-draft-projection-profile','q8-residual','--prior-target-report',str(report),'--prior-target-report-sha256',sha(report),'--output-directory',str(out/'mtp-batch-qualification')]
 run(cmd,'mtp-batch-qualification.log',timeout=2700)
 after={str(p):identity(p) for p in [target,draft]};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 for sub in ['mtp2-qualification','mtp-batch-qualification']:assert json.loads((out/sub/'report.json').read_text())['completed']
 (out/'completed.json').write_text(json.dumps(dict(completed=True,target_report_sha256=sha(report),library_sha256=sha(library),fresh_independent_mtp_block_references=True,ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
(out/'mtp-v4-launcher.py').write_text(Path(__file__).read_text())
n['dispatch']('venturi-native-head-gdn-mtp-v4',s)
