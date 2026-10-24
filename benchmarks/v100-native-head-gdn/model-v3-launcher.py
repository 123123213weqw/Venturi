from pathlib import Path
import json,hashlib,subprocess,shlex,time
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n);root=Path.cwd();out=root/'benchmarks/v100-native-head-gdn';remote='/home/data/wangyue/repos/codex-build/Venturi/native-head-gdn-src-v3'
code=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir(exist_ok=True);assert p.is_dir() and not (p/'.git').exists();print('Snapshot destination exists; bounded rsync will complete current source:',p,flush=True)\n"
for attempt in range(3):
 if attempt:time.sleep(3 if attempt==1 else 8)
 r=subprocess.run(n['SSH'],input=code,text=True,capture_output=True,timeout=60)
 with (out/'model-v3-initialization-recovery.log').open('a') as f:f.write(str(attempt)+'\n'+r.stdout+r.stderr)
 print(r.stdout,r.stderr)
 if not r.returncode:break
 if attempt==2:r.check_returncode()
cmd=['rsync','-az','-e',shlex.join(n['SSH'][:-2])]
for p in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',p]
cmd+=['src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/']
for i in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v3.log').open('a') as f:f.write(str(i)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if i==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v3.json').write_text(json.dumps(expected,indent=2)+'\n')
for rel,h in expected.items():
 p=root/rel;dst=out/'executed-source-v3'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
old=Path('/tmp/venturi-native-head-gdn-fixtures-v2-worker.py').read_text();source=old[:old.index('try:\n expected=')].replace('native-head-gdn-src-v2','native-head-gdn-src-v3').replace('venturi-native-head-gdn-fixtures-v2','venturi-native-head-gdn-model-v3')
qual=(out/'qualify_head_profile.py').read_text();source+='QUALIFIER='+repr(qual)+'\ntry:\n expected='+repr(expected)+r'''
 fixture=root/'artifacts/venturi-native-head-gdn-fixtures-v2';deadline=time.monotonic()+1500
 while not (fixture/'exit-code.txt').exists():
  pid=json.loads((fixture/'process.json').read_text())['pid'];q=pathlib.Path(f'/proc/{pid}/cmdline');assert q.exists() and str(fixture/'worker.py').encode() in q.read_bytes(),'Prerequisite handle missing without terminal marker';assert time.monotonic()<deadline,'Verified live prerequisite still pending; do not restart';time.sleep(5)
 focused=(fixture/'focused-tests.log').read_text();assert '26 passed' in focused,'Focused distributed state tests did not pass'
 regression=(fixture/'full-tests.log').read_text()
 if (fixture/'exit-code.txt').read_text().strip()!='0':
  assert regression.count(chr(10)+'FAILED ')==1 and 'FAILED tests/test_ada_build.py::test_column_tp_header_is_a_blocks_build_dependency' in regression and '1 failed,' in regression,'Unexpected native regression failure; preserve and stop'
 actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};assert actual==expected
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+chr(10))
 library=pathlib.Path((fixture/'library.txt').read_text().strip());build=json.loads(library.with_name('build.json').read_text());assert len(build['source_sha256'])==11 and hashlib.sha256(library.read_bytes()).hexdigest()==build['library_sha256']
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h,rel
 (out/'reused-build.json').write_text(json.dumps(build,indent=2)+chr(10));(out/'library.txt').write_text(str(library)+chr(10))
 env['PYTHONPATH']+=':'+str(src/'tools');env['VENTURI_MTP_DRAFT_PROFILE_LIBRARY']=str(library);env['VENTURI_MTP_DRAFT_CONTROL_LIBRARY']=str(root/'artifacts/venturi-native-column-mlp-fixtures-v1/build/77588f610f53223e/libventuri_blocks.so')
 run([py,'-m','pytest','-q','tests/test_ada_build.py','tests/test_mtp_draft_profile.py','--tb=short'],'regression-closure.log',timeout=900)
 target=root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer';draft=root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer';cases=root/'repos/codex-build/Venturi/venturi-mtp-cases.json';bf16=root/'repos/codex-build/Venturi/models/qwen3.5-27b-bf16';teacher=root/'artifacts/venturi-native-column-mlp-model-v2/qualification/report.json'
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def identity(p):
  s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 assert sha(teacher)=='904bace8b87a2047191e4fdeab66a420dfacb100d5d9686beb1ef52ac0d655f7'
 before={str(p):identity(p) for p in [target,draft]};(out/'payload-before.json').write_text(json.dumps(before,indent=2)+chr(10))
 (out/'qualify_head_profile.py').write_text(QUALIFIER)
 run([py,str(out/'qualify_head_profile.py'),str(target),'--library',str(library),'--cases',str(cases),'--teacher-report',str(teacher),'--teacher-report-sha256',sha(teacher),'--output-directory',str(out/'qualification')],'qualification.log',timeout=2700)
 report=out/'qualification/report.json';quality=json.loads(report.read_text());assert quality['completed'] and quality['profiles']['q8-residual-column-mlp-head-gdn']['candidate']['passed']
 cmd=[py,'tools/qualify_qwen_mtp.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--source',str(bf16),'--cases',str(cases),'--mode','tp','--projection-profile','q8-residual-column-mlp-head-gdn','--mtp-draft-projection-profile','q8-residual','--prior-target-report',str(report),'--prior-target-report-sha256',sha(report),'--draft-tokens','2','--recovery','prefix','--skip-block-reference','--output-directory',str(out/'mtp2-qualification')]
 run(cmd,'mtp2-qualification.log',timeout=2700)
 cmd=[py,'tools/qualify_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(draft),'--library',str(library),'--cases',str(cases),'--projection-profile','q8-residual-column-mlp-head-gdn','--mtp-draft-projection-profile','q8-residual','--prior-target-report',str(report),'--prior-target-report-sha256',sha(report),'--output-directory',str(out/'mtp-batch-qualification')]
 run(cmd,'mtp-batch-qualification.log',timeout=2700)
 after={str(p):identity(p) for p in [target,draft]};assert after==before;(out/'payload-after.json').write_text(json.dumps(after,indent=2)+chr(10))
 (out/'completed.json').write_text(json.dumps(dict(completed=True,new_native_head_gdn_profile=True,library_sha256=sha(library),quality_report_sha256=sha(report),ended_unix=time.time()),indent=2)+chr(10));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
finally:
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(pathlib.Path('/proc/driver/nvidia/params').read_text())
(out/'exit-code.txt').write_text(str(rc)+chr(10));raise SystemExit(rc)
'''
(out/'model-v3-launcher.py').write_text(Path(__file__).read_text())
n['dispatch']('venturi-native-head-gdn-model-v3',source)
