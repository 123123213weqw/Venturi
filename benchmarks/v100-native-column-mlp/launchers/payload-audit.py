from pathlib import Path
import subprocess,json,hashlib
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns);out=Path.cwd()/'benchmarks/v100-native-column-mlp';pin=Path('benchmarks/v100-llama-comparison/v100/screen-report.json');reference=json.loads(pin.read_text());expected=reference['llama_library_hashes']
code=r'''import pathlib,json,hashlib,time
root=pathlib.Path('/home/data/wangyue');bundle=pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
expected=EXPECTED
libraries={}
for name,h in expected.items():
 p=bundle/name;s=p.stat();actual=sha(p);assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
 libraries[name]=dict(expected_sha256=h,actual_sha256=actual,matches=actual==h,bytes=s.st_size)
assert all(v['matches'] for k,v in libraries.items() if k=='llama-server' or '.so' in k),'Pinned inference bundle changed'
models={}
for path,h in [
 (root/'repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer','b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8'),
 (root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer','a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641'),
 (root/'models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf','eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513')]:
 s=path.stat();actual=sha(path);assert actual==h,str(path);assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)
 models[str(path)]=dict(bytes=s.st_size,sha256=actual,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
print(json.dumps(dict(fresh_after_all_inference_jobs=True,ended_unix=time.time(),checkpoint_payloads=models,pinned_release_files=libraries,all_eighty_release_files_match=all(v['matches'] for v in libraries.values()),no_exact_source_commit_claim=True),indent=2))
'''.replace('expected=EXPECTED','expected='+repr(expected))
r=subprocess.run(ns['SSH'],input=code,text=True,capture_output=True,timeout=300);r.check_returncode();j=json.loads(r.stdout);j['prior_release_pin_report_sha256']=hashlib.sha256(pin.read_bytes()).hexdigest();(out/'fresh-payload-and-external-bundle-pins.json').write_text(json.dumps(j,indent=2)+'\n');print('Fresh payloads verified:',len(j['checkpoint_payloads']),'release files:',len(j['pinned_release_files']),'all match:',j['all_eighty_release_files_match'])
