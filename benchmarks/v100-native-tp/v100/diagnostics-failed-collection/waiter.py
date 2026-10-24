import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-tp-full-v1/exit-code.txt')
for _ in range(120):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Benchmark did not finish within bounded wait; no diagnostics started')
assert q.read_text().strip()=='0','Full qualification/benchmark failed; diagnostics not started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-tp-diagnostics-v1/worker.sh']))
