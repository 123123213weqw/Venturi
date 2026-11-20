import pathlib,time,subprocess,sys
q=pathlib.Path('/home/wzu/codex-build/Venturi/v100-pp-full-qualification-v1/exit-code.txt')
for _ in range(180):
 if q.exists():break
 time.sleep(10)
else:raise RuntimeError('Qualification did not finish within 30 minutes; timing not started')
assert q.read_text().strip()=='0','Correctness gate failed: timing not started'
sys.exit(subprocess.call(['bash','/home/wzu/codex-build/Venturi/v100-pp-comparison-v1/worker.sh']))
