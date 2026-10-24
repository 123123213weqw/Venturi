from pathlib import Path
import json,hashlib,subprocess,shlex,time
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);root=Path.cwd();out=root/'benchmarks/v100-mtp-draft-residual';remote='/home/data/wangyue/repos/codex-build/Venturi/mtp-draft-residual-src-v1'
r=subprocess.run(ns['SSH'],input=f"import pathlib\np=pathlib.Path({remote!r});p.mkdir()\nprint(p)\n",text=True,capture_output=True,timeout=60);r.check_returncode();print(r.stdout)
cmd=['rsync','-az','-e',shlex.join(ns['SSH'][:-2])]
for p in ['.git/','target/','node_modules/','.env','.env.*','.venv*/','__pycache__/','.pytest_cache/','.ruff_cache/','benchmarks/','*.ninfer','*.gguf','*.pem','*.key']:cmd+=['--exclude',p]
cmd+=['src','tests','tools','scripts','docs','pyproject.toml','README.md','AGENTS.md','WZU_Server:'+remote+'/']
for i in range(2):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 with (out/'sync-v1.log').open('a') as f:f.write(str(i)+'\n'+r.stdout+r.stderr)
 if not r.returncode:break
 if i==1:r.check_returncode()
 time.sleep(3)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};(out/'source-manifest-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
old=Path('benchmarks/v100-native-column-mlp/v100/fixtures-v1/worker.py').read_text();source=old[:old.index('try:\n expected=')].replace('native-column-mlp-src-v1','mtp-draft-residual-src-v1').replace('venturi-native-column-mlp-fixtures-v1','venturi-mtp-draft-residual-fixtures-v1')
source+='try:\n expected='+repr(expected)+'''\n actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 assert actual==expected,'Full source mismatch before compilation'
 (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+'\\n')
 buildcmd="from venturi.ops.build import build_blocks_backend;from pathlib import Path;p=build_blocks_backend("+repr(str(out/'build'))+",nvcc='/usr/local/cuda/bin/nvcc',architecture='sm70');Path("+repr(str(out/'library.txt'))+").write_text(str(p)+chr(10))"
 run([py,'-c',buildcmd],'build.log',timeout=900)
 library=(out/'library.txt').read_text().strip()
 env.update(VENTURI_MTP_DRAFT_PROFILE_LIBRARY=library,VENTURI_MTP_DRAFT_CONTROL_LIBRARY=str(root/'artifacts/venturi-native-column-mlp-fixtures-v1/build/77588f610f53223e/libventuri_blocks.so'))
 run([py,'-m','pytest','-x','-q','tests/test_mtp_draft_profile.py','--tb=short'],'focused-tests.log',timeout=600)
 baseline=root/'artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so'
 env.update(VENTURI_NATIVE_COLUMN_MLP_LIBRARY=library,VENTURI_COLUMN_TP_LIBRARY=library,VENTURI_DP4A_LIBRARY=library,VENTURI_DP4A_CONTROL_LIBRARY=str(baseline),VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 env.update(VENTURI_Q8_SHARED_LIBRARY=library,VENTURI_Q8_FUSED_LIBRARY=library,VENTURI_Q8_TILED_SHARED_LIBRARY=library,VENTURI_SHARED_LIBRARY=library,VENTURI_SHARED_BASELINE_LIBRARY=str(root/'artifacts/venturi-model-profile-fixtures-v1/build/d2a1f70d83ab530b/libventuri_blocks.so'),VENTURI_WMMA_LIBRARY=library)
 env.update(VENTURI_WORDS_LIBRARY=library,VENTURI_GDN_COMPACT_LIBRARY=library,VENTURI_GDN_CONTROL_LIBRARY=str(baseline),VENTURI_MTP2_LIBRARY=library,VENTURI_MTP_PREFIX_LIBRARY=library,VENTURI_T3_LIBRARY=library,VENTURI_MTP_BATCH_LIBRARY=library,VENTURI_PREFILL_BUDGET_LIBRARY=library)
 env.update(VENTURI_DP2A_PILOT_LIBRARY=str(root/'artifacts/venturi-dp2a-pilot-fixtures-v1/libventuri_dp2a_pilot.so'),VENTURI_DP2A_CONTROL_LIBRARY=library)
 for name in ['VENTURI_TILE_ORDER_LIBRARY','VENTURI_T16_LIBRARY','VENTURI_LAYER_GRAPH_LIBRARY']:env.pop(name,None)
 run([py,'-m','pytest','-q','--tb=short'],'full-tests.log',timeout=900)
 (out/'executed-source.json').write_text(json.dumps(actual,indent=2)+'\\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,library=library,library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),ended_unix=time.time()),indent=2)+'\\n');rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\\n');raise SystemExit(rc)
'''
# Materialize line escapes in generated worker source, not source string literals.
source=source.replace("+'\\\\n'", "+'\\n'")
ns['dispatch']('venturi-mtp-draft-residual-fixtures-v1',source)
