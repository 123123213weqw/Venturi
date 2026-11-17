from pathlib import Path
import json,hashlib
r=Path('/home/xs/codex-build/Venturi-pr14-coalesced-20261117'); e=r/'evidence'; old=Path('/home/xs/codex-build/Venturi-pr14-final-20261117-v2')
b=dict(tag='change',architecture='sm70',ops=str(r/'native/ops/libventuri_ops.so'),qwen=str(r/'native/qwen/libventuri_qwen.so'))
for name in ('change-sm70-build.json','native-active-build.json'):(e/name).write_text(json.dumps(b,indent=2))
py='/home/xs/codex-work/ornith-opt-20261006/onecat-optimization/src/onecat/.venv/bin/python'
base=json.loads((old/'sm70-real-v1.json').read_text())
base['env']={'VENTURI_SM70_LIBRARY':b['ops'],'VENTURI_DP4A_LIBRARY':b['ops']}
base['seconds']=1500
cmd=base['command'];cmd[cmd.index('--library')+1]=b['ops'];cmd[cmd.index('--qwen-library')+1]=b['qwen']
cmd[cmd.index('--cases')+1]=str(r/'cases.json');cmd[cmd.index('--output')+1]=str(e/'sm70-real-v1-report.json')
(r/'sm70-real-v1.json').write_text(json.dumps(base,indent=2))
small=dict(base);small['name']='sm70-small-v1';small['seconds']=600
small['env']=dict(base['env'],VENTURI_REFERENCE_OPS_LIBRARY='/home/xs/codex-build/Venturi-pr14-fix-20261117/native/ops-final/libventuri_ops.so',VENTURI_REFERENCE_QWEN_LIBRARY='/home/xs/codex-build/Venturi-pr14-fix-20261117/native/candidate-final-qwen/libventuri_qwen.so')
small['command']=[py,'-m','pytest','-q','tests/test_quantlinear_cuda.py','tests/test_row_codec_execution.py','tests/test_native_split.py','tests/test_qwen_native.py','tests/test_quantlinear_contract.py','--maxfail=1','-rs']
(r/'sm70-small-v1.json').write_text(json.dumps(small,indent=2))
tps=dict(base);tps['name']='sm70-tps-v1';tps['seconds']=2700;tps['command']=[py,str(r/'run_matched_http.py'),str(r)]
(r/'sm70-tps-v1.json').write_text(json.dumps(tps,indent=2))
prof=dict(base);prof['name']='sm70-profile-v1';prof['seconds']=600;prof['gpus']=base['gpus'][:1];prof['command']=[py,str(r/'profile_quantlinear.py'),str(r)]
(r/'sm70-profile-v1.json').write_text(json.dumps(prof,indent=2))
print(json.dumps({'ready':True,'candidate_ops_sha256':hashlib.sha256(Path(b['ops']).read_bytes()).hexdigest(),'candidate_qwen_sha256':hashlib.sha256(Path(b['qwen']).read_bytes()).hexdigest()}))
