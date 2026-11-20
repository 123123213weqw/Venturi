import pathlib,subprocess,json,os,time,traceback
root=pathlib.Path('/home/wzu/codex-build/Venturi');out=root/'v100-mtp-export-v1';src=root/'v100-mtp-src-v1'
code="import torch,json;torch.set_num_threads(4);from venturi.qwen_export import export_qwen_mtp_q8;r=export_qwen_mtp_q8('/home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16','/home/wzu/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer');print(json.dumps(r,indent=2))"
env=os.environ.copy();env.update(PYTHONPATH=str(src/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
try:
 with (out/'export.json').open('w') as log, (out/'export.log').open('w') as err:subprocess.run(['/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python','-u','-c',code],cwd=src,env=env,stdout=log,stderr=err,check=True,timeout=900)
 (out/'completed.json').write_text(json.dumps(dict(completed=True,unix=time.time()),indent=2));rc=0
except BaseException:(out/'failure.txt').write_text(traceback.format_exc());rc=1
(out/'exit-code.txt').write_text(str(rc)+'\n');raise SystemExit(rc)
