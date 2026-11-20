"""Bounded post-integration operator/native-fixture gate; production stays up."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

source = Path('/home/data/wangyue/repos/codex-build/Venturi/q8-prefetch-src-v5')
out = Path('/home/data/wangyue/artifacts/venturi-q8-prefetch-v3')
python = '/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
library = out / 'build/f634328d5b15400b/libventuri_blocks.so'
control = '/home/data/wangyue/artifacts/venturi-native-head-gdn-fixtures-v2/build/68769d5f68c715ef/libventuri_blocks.so'
build = json.loads(library.with_name('build.json').read_text())
assert build['library_sha256'] == hashlib.sha256(library.read_bytes()).hexdigest()
for name, expected in build['source_sha256'].items():
    assert hashlib.sha256((source/'src/venturi/ops/csrc'/name).read_bytes()).hexdigest() == expected
assert (out/'default-off-tests-repaired.exit').read_text().strip() == '0'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           VENTURI_Q8_PREFETCH_LIBRARY=str(library), VENTURI_Q8_PREFETCH_CONTROL=control,
           VENTURI_EXPECT_ROW_PAIR='1', VENTURI_NATIVE_HEAD_GDN_LIBRARY=str(library),
           PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{source}/src:{source}/tests:/home/wzu/.local/lib/python3.10/site-packages')

def run(name, args, timeout):
    (out/f'{name}.command.json').write_text(json.dumps(args, indent=2)+'\n')
    with (out/f'{name}.log').open('xb') as log:
        result = subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    (out/f'{name}.exit').write_text(str(result.returncode)+'\n')
    assert result.returncode == 0, name

run('integrated-tests', [python, '-m', 'pytest', '-q', 'tests/test_q8_prefetch.py',
    'tests/test_q8_projection_dispatch.py', 'tests/test_native_head_gdn.py'], 300)
env['CUDA_VISIBLE_DEVICES'] = '0'
run('native-screen', [python, 'tools/projection/benchmark_q8_prefetch.py',
    '--gguf', '/home/wzu/models/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf',
    '--library', str(library), '--qualified-library', control,
    '--output-directory', str(out/'native-screen'), '--native-dispatch',
    '--tokens', '8,23,32', '--repeats', '7'], 180)
(out/'integrated-completed.json').write_text(json.dumps(dict(
    completed=True, library_sha256=build['library_sha256'], driver_source=str(source),
    default_off_tests_passed=True, native_fixture_and_operator_tests_passed=True,
    scope='Native opt-in wrapper plus small hybrid fixtures and packed operators; NOT full-model/serving performance or production Qwen3.8 quality.',
    ended_unix=time.time()), indent=2)+'\n')
