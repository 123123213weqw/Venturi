from pathlib import Path
import subprocess,json,time
b=Path('benchmarks/v100-native-head-controls')
for label,cmd in [('netcheck',['tailscale','netcheck']),('ipv4-route',['/sbin/route','-n','get','100.70.188.115']),('ipv6-route',['/sbin/route','-n','get','-inet6','fd7a:115c:a1e0::5e35:bc73'])]:
 try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
 except subprocess.TimeoutExpired as e:(b/f'turn3-{label}.log').write_text(str(e));print(label,'bounded timeout',flush=True);continue
 (b/f'turn3-{label}.log').write_text(r.stdout+r.stderr);print(label,r.returncode,r.stdout,r.stderr,flush=True)
