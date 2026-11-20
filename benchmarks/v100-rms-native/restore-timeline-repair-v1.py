import sys
from pathlib import Path
sys.path.insert(0,'/home/data/wangyue/repos/codex-build/Venturi/rms-native-window-control-v4')
import restore_service_v1 as m
m.ROOT=Path('/home/data/wangyue/artifacts/venturi-rms-timeline-repair-v1')
# Terminate only this repair unit before restoring its captured container.
if '--watchdog' in sys.argv:
 import subprocess
 subprocess.run(['systemctl','--user','stop','venturi-rms-timeline-repair-v1'],timeout=60,check=False)
print(m.restore())
