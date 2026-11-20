from pathlib import Path
import subprocess,json,re
b=Path('benchmarks/v100-native-head-controls');v4=subprocess.run(['tailscale','ip','-4'],capture_output=True,text=True,timeout=15).stdout.strip();v6=subprocess.run(['tailscale','ip','-6'],capture_output=True,text=True,timeout=15).stdout.strip();raw=subprocess.run(['/sbin/ifconfig','-a'],capture_output=True,text=True,timeout=10).stdout;matches=[];current=None
for line in raw.splitlines():
 if line and not line[0].isspace():current=line.split(':',1)[0]
 if any(x and x in line.split() for x in [v4,v6]):matches.append(dict(interface=current,address_line=line.strip()))
r=dict(self_ipv4=v4,self_ipv6=v6,matching_interface_addresses=matches)
(b/'turn3-interface-check.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
