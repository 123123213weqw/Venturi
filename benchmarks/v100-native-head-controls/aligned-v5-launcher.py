from pathlib import Path
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
ssh=n['SSH'][:];ssh[ssh.index('ControlPath=none')]='ControlPath=/Users/wangyue/.ssh/cm-14974355c3f315e78f58843cae0595e186514110';ssh[ssh.index('ServerAliveCountMax=2')]='ServerAliveCountMax=6';n['SSH']=ssh
b=Path('benchmarks/v100-native-head-controls');s=Path('/tmp/venturi-native-head-aligned-v2-worker.py').read_text();s=s[s.index('import pathlib,subprocess'):]
s=s.replace('venturi-native-head-aligned-v2','venturi-native-head-aligned-v5').replace('venturi-native-head-controls-v1','venturi-native-head-controls-v4')
s=s.replace("server=subprocess.Popen(cmd,cwd=src,env=env,stdout=log", "server_env=env.copy();server_env['LD_LIBRARY_PATH']=str(bindir)+':/usr/local/cuda/lib64:'+str(root/'repos/codex-build/.venvs/b2-v100/lib/python3.10/site-packages/nvidia/nccl/lib');(out/'server-environment.json').write_text(json.dumps({k:server_env.get(k) for k in ['LD_LIBRARY_PATH','NCCL_P2P_DISABLE','CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS']},indent=2)+'\\n');server=subprocess.Popen(cmd,cwd=src,env=server_env,stdout=log")
source='DIAGNOSTIC='+repr((b/'aligned_teacher.py').read_text())+'\n'+s;compile(source,'worker','exec');(b/'aligned-v5-launcher.py').write_text(Path(__file__).read_text());n['dispatch']('venturi-native-head-aligned-v5',source)
