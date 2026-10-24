"""Bounded V100 SwiGLU/two-pass-quantization fusion screen; never stops or changes production."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

source = Path('/home/data/wangyue/repos/codex-build/Venturi/swiglu-quant-src-v1')
out = Path('/home/data/wangyue/artifacts/venturi-swiglu-quant-v1')
python = '/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
control = '/home/data/wangyue/artifacts/venturi-q8-prefetch-v3/build/f634328d5b15400b/libventuri_blocks.so'
library = out/'libswiglu_quant_probe.so'
assert hashlib.sha256(Path(control).read_bytes()).hexdigest()=='d569f3fb9f11598b8fd4c70263bd7948e4d43e5620441c48fcd8b038bcc71284'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4', CUDA_CACHE_PATH=str(out/'cuda-cache'),
           PYTHONPATH=f'/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:{source}/src:{source}/tests:/home/wzu/.local/lib/python3.10/site-packages')


def save(name, value):
    (out/name).write_text(json.dumps(value,indent=2)+'\n')


def audit(name):
    item = json.loads(subprocess.check_output(['docker','inspect','qwen38-27b'],timeout=15))[0]
    fingerprint = hashlib.sha256(json.dumps(dict(args=item['Args'],env=item['Config']['Env'],
        mounts=sorted(item['Mounts'],key=lambda m:(m.get('Destination',''),m.get('Source',''),m.get('Type','')))),sort_keys=True).encode()).hexdigest()
    record = dict(container_id=item['Id'],status=item['State']['Status'],health=item['State']['Health']['Status'],
                  started_at=item['State']['StartedAt'],restart_count=item['RestartCount'],canonical_config_fingerprint=fingerprint,
                  health_api=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)),
                  counter_policy=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s],
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader'],text=True,timeout=10).splitlines(),unix=time.time())
    save(name,record)
    assert fingerprint=='d5de4a5f11bd9b37624887cf472415ad3939f94366429d97846a0cdf05e579e0'
    assert record['container_id']=='fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d'
    assert record['status']=='running' and record['health']=='healthy'
    assert record['started_at']=='2026-10-21T06:07:14.516526843Z'


def run(name, args, timeout, root=False):
    save(name+'.command.json',args)
    with (out/(name+'.log')).open('xb') as log:
        proc = subprocess.Popen(args,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (out/(name+'.exit')).write_text(str(rc)+'\n')
    assert rc==0,name


try:
    audit('production-before.json')
    for _ in range(90):
        state=subprocess.check_output(['systemctl','--user','show','venturi-swiglu-quant-tests-v1','-p','ActiveState','--value'],text=True).strip()
        if state=='inactive':break
        assert state!='failed','Correctness unit failed'
        time.sleep(2)
    else:raise TimeoutError('Correctness unit unfinished; speed tests not started')
    assert subprocess.check_output(['systemctl','--user','show','venturi-swiglu-quant-tests-v1','-p','ExecMainStatus','--value'],text=True).strip()=='0'
    assert '122 passed' in (out/'tests.log').read_text()
    pins={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','tests','tools') for p in (source/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    save('executed-source-sha256.json',pins)
    common=[python,'tools/projection/benchmark_swiglu_quant.py',
        '--gguf','/home/wzu/models/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf',
        '--library',str(library),'--control-library',control]
    run('screen',common+['--output-directory',str(out/'screen'),'--tokens','1,2,4,8,23,32','--layers','0,31,63','--repeats','7'],360)
    run('confirm',common+['--output-directory',str(out/'confirm'),'--tokens','1,8','--layers','0','--repeats','7'],150)
    metrics=','.join(['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active',
        'smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active',
        'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
        'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum',
        'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
        'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
        'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct'])
    for arm in ('pinned-separate','fused-128','fused-256'):
        cmd=['sudo','-n','/usr/bin/timeout','100','/usr/bin/env']
        cmd += [k+'='+env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_CACHE_PATH','PYTHONPATH')]
        cmd += ['/usr/local/cuda/bin/ncu','--clock-control','none','--cache-control','none',
            '--target-processes','all','--profile-from-start','off','--kernel-name-base','demangled',
            '--launch-count','2' if arm=='pinned-separate' else '1','--metrics',metrics,
            '--csv','--page','raw','--disable-extra-suffixes','--log-file',str(out/f'counter-{arm}.csv')]
        cmd += common+['--output-directory',str(out/f'profile-{arm}'),'--tokens','8','--layers','0','--profile-arm',arm]
        run('counter-'+arm,cmd,115,root=True)
    save('completed.json',dict(completed=True,operator_screen_only=True,production_deployed=False,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),ended_unix=time.time()))
finally:
    audit('production-after.json')
