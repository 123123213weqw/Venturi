import ast,pathlib,subprocess
p=pathlib.Path('/tmp/venturi-dp4a-fixture-dispatch.py');tree=ast.parse(p.read_text());worker=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in n.targets)))
worker=worker.replace('dp4a-src-v1','model-profile-src-v3').replace('venturi-dp4a-fixtures-v1','venturi-model-profile-qualification-v2');a=worker.index(" run([py,'-m','venturi.ops.build'");b=worker.index(" (out/'completed.json')",a)
body=r'''
 previous=root/'artifacts/venturi-model-profile-fixtures-v1';assert (previous/'exit-code.txt').read_text().strip()=='0';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 env.update(VENTURI_PROFILE_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_MTP_TEST='1',VENTURI_TP_TEST='1',VENTURI_BATCH_TEST='1',VENTURI_PP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-x','-q','tests/test_qwen_projection_profile.py','--tb=short'],'profile-tests.log',timeout=180)
 run([py,'-u','tools/qualify_qwen_projection_profile.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--library',library,'--cases',str(root/'repos/codex-build/Venturi/venturi-mtp-cases.json'),'--cpu-reference-directory',str(root/'repos/codex-build/Venturi/greedy-validation-v3/full-model-run1'),'--output-directory',str(out/'qualification')],'qualification.log',timeout=480)
'''
worker=worker[:a]+body+worker[b:];compile(worker,'worker.py','exec')
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-model-profile-qualification-v2');out.mkdir();p=out/'worker.py';p.write_text({worker!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -'];r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
