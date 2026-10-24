from pathlib import Path
import json,subprocess
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
r=subprocess.run(ns['SSH'],input="import datetime,json,time\nprint(json.dumps(dict(utc_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),unix=time.time(),local=datetime.datetime.now().astimezone().isoformat())))\n",text=True,capture_output=True,timeout=60);r.check_returncode();d=json.loads(r.stdout);print(d);(Path.cwd()/'benchmarks/v100-dp2a-pilot/telemetry-timezone.json').write_text(json.dumps(d,indent=2)+'\n')
