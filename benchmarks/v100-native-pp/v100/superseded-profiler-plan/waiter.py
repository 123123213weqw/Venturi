import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/exit-code.txt')
for _ in range(240):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Unprofiled benchmarks did not finish within 40 minutes; no profiling started')
assert q.read_text().strip()=='0','Unprofiled benchmark failed; no profiler started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-diagnostics-v1/worker.sh']))
