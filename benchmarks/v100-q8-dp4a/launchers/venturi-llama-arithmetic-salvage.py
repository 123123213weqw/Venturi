import pathlib,os,json,subprocess,signal,time,traceback
root=pathlib.Path('/home/data/wangyue');out=root/'artifacts/venturi-llama-arithmetic-v1';parent=939349
src=root/'repos/codex-build/Venturi/native-batch-src-v3';py=str(root/'repos/codex-build/.venvs/b2-v100/bin/python');case=root/'repos/codex-build/Venturi/v100-pp-final-v1/case.json'
def descendants(pid):
 pending=[pid];found=set()
 while pending:
  current=pending.pop();task=pathlib.Path(f'/proc/{current}/task')
  if not task.exists():continue
  for path in task.glob('*/children'):
   for text in path.read_text().split():
    child=int(text)
    if child not in found:pending.append(child);found.add(child)
 return found
report=dict(completed=False,profiled=True,observation_recovery='Original worker inspected children of the main profiler thread only. Reuse the confirmed live owned profiler/server; do not restart for observation failure.',nsys_pid=parent,started_unix=time.time())
server=None
try:
 for pid in descendants(parent):
  p=pathlib.Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().split(b'\0')
  if (p/'exe').resolve()==pathlib.Path('/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin/llama-server') and b'18946' in cmd:
   assert server is None;server=pid
 assert server is not None,'Original handle no longer contains the owned server'
 report['server_pid']=server;c=json.loads(case.read_text());c['expected_generated_ids']=c['expected_generated_ids'][:8];(out/'salvage-case.json').write_text(json.dumps(c,indent=2)+'\n')
 cmd=[py,'tools/benchmark_completion_api.py','--url','http://127.0.0.1:18946','--server-pid',str(server),'--case',str(out/'salvage-case.json'),'--concurrency','1,8','--repeats','1','--output-directory',str(out/'salvaged-instrumented-http'),'--expect-pinned-ids'];report['command']=cmd
 with (out/'salvaged-client.log').open('w') as f:subprocess.run(cmd,cwd=src,stdout=f,stderr=subprocess.STDOUT,timeout=120,check=True)
 report['completed']=True
except BaseException:report['failure']=traceback.format_exc()
finally:
 if server is not None and server in descendants(parent):os.kill(server,signal.SIGINT)
 (out/'observation-recovery.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
