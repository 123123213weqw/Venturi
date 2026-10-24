from pathlib import Path
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
b=Path('benchmarks/v100-native-head-controls');source=Path('/tmp/venturi-native-head-aligned-v2-worker.py').read_text();source='DIAGNOSTIC='+repr((b/'aligned_teacher.py').read_text())+'\n'+source
(b/'aligned-v2-launcher.py').write_text(Path(__file__).read_text());n['dispatch']('venturi-native-head-aligned-v2',source)
