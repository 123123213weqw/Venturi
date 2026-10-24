from pathlib import Path
import subprocess
base=Path('/tmp/venturi-tiled-staging-model-worker.py').read_text()
base=base.replace('tiled-staging-src-v3','projection-fusion-src-v2').replace('venturi-tiled-staging-model-v3','venturi-projection-fusion-model-v2')
start=base.index(" baseline=root/")
end=base.index(" target=root/",start)
base=base[:start]+''' baseline=root/'artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f'
 library=str(root/'artifacts/venturi-projection-fusion-v1/build/b08a62aa2ac57cc2/libventuri_blocks.so');assert hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest()=='6698f47bae3405279102890818782c992e8179aec966af8b5a08941160bb28b1'
 build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 (out/'library.txt').write_text(library+'\\n')
 rootgate=root/'artifacts/venturi-projection-fusion-v1';assert (rootgate/'exit-code.txt').read_text().strip()=='0'
 (out/'reused-fixture-gates.json').write_text(json.dumps(dict(path=str(rootgate),full_test_log_sha256=hashlib.sha256((rootgate/'full-tests.log').read_bytes()).hexdigest(),library_sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),scope='Identical nine native source hashes; 221 selected and 1517 full-suite passes reused, no new fixture-test execution.'),indent=2)+'\\n')
 run([py,'-u','tools/benchmark_q8_fused.py',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer'),'--library',library,'--baseline-library',str(baseline),'--output-directory',str(out/'micro')],'micro.log',timeout=300)
''' +base[end:]
base=base.replace("prior=root/'artifacts/venturi-shared-activation-model-v2/qualification/report.json'", "prior=root/'artifacts/venturi-tiled-staging-model-v3/qualification/report.json'")
base=base.replace('073e59da90afa6dc5822c1dee0c2fa9377b33bec107eb0ab710c39edbacf8529','dc3478d7dd188f03b0e8bcabfafcdc5a7ff6e8e9c15783b3d919b9ff11d514b9')
base=base.replace("'--reuse-prior-control','--reference-cases-sha256'", "'--profiles','q8-residual','--reuse-prior-control','--reference-cases-sha256'")
base=base.replace('changed T2 target pitched stores','changed same-input projection-group transport fences; unchanged FP16 draft arithmetic')
Path('/tmp/venturi-projection-fusion-model-worker.py').write_text(base)
code=f'''import pathlib,subprocess,json,time
out=pathlib.Path('/home/data/wangyue/artifacts/venturi-projection-fusion-model-v2');out.mkdir();p=out/'worker.py';p.write_text({base!r})
with (out/'worker.log').open('xb') as log:proc=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=proc.pid,started_unix=time.time());(out/'process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','StrictHostKeyChecking=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ConnectTimeout=20','WZU_Server','python3 -']
r=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60);print(r.stdout,r.stderr);r.check_returncode()
