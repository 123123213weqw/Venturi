import pathlib,subprocess
root=pathlib.Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi');cap=root/'benchmarks/v100-native-mtp'
ssh='ssh -o BatchMode=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o ProxyCommand=none -o ConnectTimeout=20'
files=['src/venturi/ops/csrc/'+f for f in ['blocks.cu','qwen_runtime.cuh','qwen_pipeline.cuh','qwen_mtp.cuh']]
files+=['src/venturi/models/qwen_mtp.py','src/venturi/models/qwen_mtp_native.py','src/venturi/qwen_export.py','tests/test_qwen_mtp.py']
for version in [1,2,3]:
 selected=files + (['tools/qualify_qwen_mtp.py'] if version>1 else [])
 listed=pathlib.Path('/tmp/venturi-mtp-history-files.txt');listed.write_text('\n'.join(selected)+'\n')
 dest=cap/f'v100/source-history-v{version}';dest.mkdir(exist_ok=True)
 subprocess.run(['rsync','-az','--files-from='+str(listed),'-e',ssh,f'WZU_Server:/home/wzu/codex-build/Venturi/v100-mtp-src-v{version}/',str(dest)+'/'],check=True,timeout=60)
 print('historical source',version,flush=True)
