import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/exit-code.txt')
for _ in range(120):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Initial comparison did not finish within bounded wait; no target jobs started')
assert q.read_text().strip()=='0','Initial comparison failed'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-final-v1/worker.sh']))
