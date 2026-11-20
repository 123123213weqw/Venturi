"""Authorized standalone production pause; exact argv/env/cwd/log restoration.

Private process environment is never emitted in public evidence. No systemd,
cron, routing, driver, privilege or other-user process changes.
"""
from pathlib import Path
import fcntl, hashlib, json, os, signal, stat, subprocess, sys, time, urllib.request

R = Path(sys.argv[1]); P = R / 'private'; E = R / 'evidence'
P.mkdir(mode=0o700, exist_ok=True)
UID = 1001
ORIGINAL_PID = 1374268


def proc(pid):
    p = Path('/proc') / str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return p, fields


def identity(pid):
    p, fields = proc(pid)
    return p.stat().st_uid, fields[19]


def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def capture(pid):
    p, fields = proc(pid)
    assert p.stat().st_uid == UID and (p/'comm').read_text().strip() == 'llama-server'
    argv = [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    env = dict(x.decode().split('=', 1) for x in (p/'environ').read_bytes().split(b'\0') if x)
    d = dict(argv=argv, env=env, cwd=str((p/'cwd').resolve()), exe=str((p/'exe').resolve()),
             log=str((p/'fd/1').resolve()), start_ticks=fields[19], pid=pid)
    assert argv[0].endswith('/llama-server') and '--port' in argv and argv[argv.index('--port')+1] == '8096'
    assert str((p/'fd/2').resolve()) == d['log']
    assert Path(d['log']).is_file() and os.access(d['log'], os.W_OK)
    d['umask'] = int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Umask:')),8)
    d['cpu_affinity'] = sorted(os.sched_getaffinity(pid))
    assert int(env.get('LISTEN_FDS','0')) == 0, 'Inherited activation FDs require their supervisor'
    d['binary_sha256'] = hashlib.sha256(Path(d['exe']).read_bytes()).hexdigest()
    d['fingerprint'] = hashlib.sha256(json.dumps({k:d[k] for k in ('argv','env','cwd','exe','binary_sha256','umask','cpu_affinity')},sort_keys=True).encode()).hexdigest()
    return d


def matching(original):
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if p.stat().st_uid != UID or (p/'comm').read_text().strip() != 'llama-server': continue
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if argv == original['argv'] and str((p/'cwd').resolve()) == original['cwd']:
                found.append(int(p.name))
        except (OSError,ValueError,UnicodeError): pass
    assert len(found) <= 1, 'Multiple matching services; never add another'
    return found


def own_group_members(pgid, ticks):
    members=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            _, fields=proc(int(p.name))
            if p.stat().st_uid == UID and int(fields[2]) == pgid and int(fields[19]) >= int(ticks):
                cwd=str((p/'cwd').resolve())
                if cwd == str(R/'change') or cwd.startswith(str(R)+'/'):
                    members.append(int(p.name))
        except (OSError,ValueError): pass
    return members


def cleanup_owned():
    files=list(P.glob('*-pgid.json'))+list(E.glob('sm70-*-launch.json'))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for f in files:
            try:
                d=json.loads(f.read_text()); pid=d['pid']; tick=d['start_ticks']
                if own_group_members(pid,tick): os.killpg(pid,sig)
            except (OSError,KeyError,ValueError): pass
        time.sleep(4)


def restore(state, actor):
    with (P/'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (P/'restored.json').exists(): return json.loads((P/'restored.json').read_text())
        if not (P/'stop-authorized.json').exists(): return dict(no_stop_occurred=True)
        cleanup_owned()
        old=state['original']; found=matching(old)
        if not found:
            # An externally replaced listener must not be overwritten.
            assert not health(), 'Different listener at original endpoint; no overwrite'
            assert hashlib.sha256(Path(old['exe']).read_bytes()).hexdigest() == old['binary_sha256']
            with Path(old['log']).open('ab') as output:
                p=subprocess.Popen(old['argv'],cwd=old['cwd'],env=old['env'],stdin=subprocess.DEVNULL,
                                   stdout=output,stderr=subprocess.STDOUT,start_new_session=True,umask=old['umask'],preexec_fn=lambda: os.sched_setaffinity(0,old['cpu_affinity']))
            found=[p.pid]
        ready=False
        for _ in range(180):
            if health(): ready=True; break
            if not Path('/proc',str(found[0])).exists(): break
            time.sleep(2)
        current=capture(found[0])
        receipt=dict(actor=actor,restored_running=True,http_200=ready,
                     same_command_environment_cwd_binary=current['fingerprint']==old['fingerprint'],
                     original_pid=old['pid'],restored_pid=found[0],original_log_preserved=current['log']==old['log'],
                     no_route_or_model_parameter_changes=True,unix=time.time())
        (E/f'production-restore-{actor}.json').write_text(json.dumps(receipt,indent=2))
        assert ready and receipt['same_command_environment_cwd_binary'] and receipt['original_log_preserved'], 'Restoration failed or unverified'
        (P/'restored.json').write_text(json.dumps(receipt))
        return receipt


if len(sys.argv)>2 and sys.argv[2]=='--restore-watch':
    state=json.loads((P/'state.json').read_text())
    while not (P/'restored.json').exists():
        try: alive=identity(state['parent_pid'])==(UID,state['parent_ticks'])
        except OSError: alive=False
        if not alive or time.monotonic()>=state['deadline_monotonic'] or time.time()>=state['deadline_unix']:
            if alive: os.kill(state['parent_pid'],signal.SIGTERM)
            restore(state,'watchdog'); break
        time.sleep(2)
    raise SystemExit()


def interrupted(sig,frame): raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
assert os.getuid()==UID and os.uname().nodename=='root1'
assert not (P/'state.json').exists(), 'No repeated production window'
if len(sys.argv)>2 and sys.argv[2]=='--preflight-only':
    original=capture(ORIGINAL_PID);saved=P/'preflight-original.json';saved.write_text(json.dumps(original,indent=2));saved.chmod(0o600)
    print(json.dumps(dict(preflight=True,original_pid=ORIGINAL_PID,health_200=health(),command_environment_cwd_binary_identity=original['fingerprint'],service_actions=[])))
    raise SystemExit()
ci=json.loads((E/'ci-approved.json').read_text());assert ci['conclusion']=='success' and ci['headSha']==json.loads((E/'source-head.json').read_text())['commit']
artifact=json.loads((E/'artifact.json').read_text());assert Path(artifact['path']).stat().st_size==artifact['bytes']
transfer=json.loads((R/'transfer-private/parallel-receipt.json').read_text());assert transfer['sha256']==artifact['sha256']
planned=json.loads((R/'cases.json').read_text())['cases']
work=[x for x in planned if x['prompt_token_ids']==[(7*i+13)%248320 for i in range(16)] and x['max_new_tokens']==32]
assert len(work)==1, 'Exact 16-in/32-out case required before stopping production'
old=capture(ORIGINAL_PID);assert health()
state=dict(original=old,parent_pid=os.getpid(),parent_ticks=identity(os.getpid())[1],deadline_monotonic=time.monotonic()+5400,deadline_unix=time.time()+5400)
p=P/'state.json';p.write_text(json.dumps(state,indent=2));p.chmod(0o600)
watch=subprocess.Popen([sys.executable,__file__,str(R),'--restore-watch'],stdout=(P/'restore-watch.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(stage='restorer_started',pid=watch.pid,deadline=state['deadline_unix'])),flush=True)
try:
    stable=0
    for _ in range(120):
        with urllib.request.urlopen('http://127.0.0.1:8096/slots',timeout=5) as response: slots=json.loads(response.read())
        assert len(slots)==4 and all('is_processing' in x for x in slots)
        stable=stable+1 if not any(x['is_processing'] for x in slots) else 0
        if stable>=3: break
        time.sleep(1)
    assert stable>=3, 'Production remained active; no forced stop'
    assert capture(ORIGINAL_PID)['fingerprint']==old['fingerprint']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert all(int(x)==ORIGINAL_PID for x in apps.splitlines() if x.strip()), 'Unrelated GPU task present'
    (P/'stop-authorized.json').write_text(json.dumps(dict(pid=ORIGINAL_PID,unix=time.time())))
    os.kill(ORIGINAL_PID,signal.SIGTERM)
    for _ in range(60):
        try: same=identity(ORIGINAL_PID)==(UID,old['start_ticks'])
        except OSError: same=False
        if not same: break
        time.sleep(1)
    if same: os.kill(ORIGINAL_PID,signal.SIGKILL); time.sleep(3)
    print(json.dumps(dict(stage='production_temporarily_paused',pid=ORIGINAL_PID)),flush=True)
    (E/'production-window.json').write_text(json.dumps(dict(initial_running=True,original_pid=ORIGINAL_PID,identity_sha256=old['fingerprint'],independent_restorer_pid=watch.pid,no_route_or_model_parameter_changes=True),indent=2))
    for name in ('sm70-small-v1','sm70-real-v1','sm70-tps-v1'):
        print(json.dumps(dict(stage='start',name=name)),flush=True)
        subprocess.run([sys.executable,str(R/'guard_job.py'),str(R/(name+'.json'))],check=True,cwd=R)
        final=json.loads((E/(name+'-final.json')).read_text());assert final['exit_code']==0 and not final['remaining_own_apps']
        if name=='sm70-real-v1':
            d=json.loads((E/'sm70-real-v1-report.json').read_text());assert d['passed'] and d['completed']
        print(json.dumps(dict(stage='complete',name=name)),flush=True)
    if (R/'sm70-profile-v1.json').exists() and time.monotonic()+600<state['deadline_monotonic']:
        # Diagnostics cannot qualify or alter the preceding unprofiled TPS.
        result=subprocess.run([sys.executable,str(R/'guard_job.py'),str(R/'sm70-profile-v1.json')],cwd=R)
        print(json.dumps(dict(stage='separate_profile_finished',exit_code=result.returncode)),flush=True)
finally:
    receipt=restore(state,'primary');print(json.dumps(dict(stage='production_restored',**receipt)),flush=True)
    if not (P/'stop-authorized.json').exists():(P/'restored.json').write_text(json.dumps(dict(no_stop_occurred=True)))
    try: watch.wait(timeout=15)
    except subprocess.TimeoutExpired: pass
