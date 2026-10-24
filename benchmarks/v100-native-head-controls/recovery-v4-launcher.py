from pathlib import Path
n={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),n)
b=Path('benchmarks/v100-native-head-controls');(b/'recovery-v4-launcher.py').write_text(Path(__file__).read_text());n['dispatch']('venturi-native-head-controls-v4',Path('/tmp/venturi-native-head-controls-v4-source.py').read_text())
