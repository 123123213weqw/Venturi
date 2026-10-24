"""Finite, explicitly authorized counter campaign with exact-container restore."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path('/home/data/wangyue/artifacts/venturi-batch8-decode-counters-v2')
REPO = Path('/home/data/wangyue/repos/codex-build/Venturi')
SOURCE = REPO / 'adaptive-mtp-src-v1'
PYTHON = '/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python'
LIBRARY = Path('/home/data/wangyue/artifacts/venturi-native-head-gdn-fixtures-v2/build/68769d5f68c715ef/libventuri_blocks.so')
CID = 'fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d'
EXPECTED_CONFIG = 'd5de4a5f11bd9b37624887cf472415ad3939f94366429d97846a0cdf05e579e0'
METRICS = [
    'dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
]


def save(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def inspect():
    return json.loads(subprocess.check_output(['docker', 'inspect', 'qwen38-27b'], timeout=15))[0]


def fingerprint(item):
    return hashlib.sha256(json.dumps(dict(
        args=item['Args'], env=item['Config']['Env'],
        mounts=sorted(item['Mounts'], key=lambda m: (
            m.get('Destination', ''), m.get('Source', ''), m.get('Type', ''),
        )),
    ), sort_keys=True).encode()).hexdigest()


def api(path):
    return json.load(urllib.request.urlopen('http://127.0.0.1:8000/' + path, timeout=5))


def interrupted(*args):
    raise KeyboardInterrupt('Finite profiling campaign interrupted')


def counter_rows(path, selected):
    raw = path.read_text()
    assert '==ERROR==' not in raw, raw[:2000]
    start = raw.find('"ID",')
    assert start >= 0, 'No counter CSV header'
    rows = [r for r in csv.DictReader(io.StringIO(raw[start:])) if r.get('ID')]
    assert len(rows) == 3, f'Expected three decode ranges, got {len(rows)}'
    values = []
    for row in rows:
        assert row['Kernel Name'] == 'range' and row['Device'] == str(selected)
        item = dict(id=row['ID'], device=row['Device'], context=row['Context'])
        for key in METRICS:
            item[key] = float(row[key].replace(',', ''))
        assert item['gpu__time_duration.sum'] > 0
        assert item['dram__bytes_read.sum'] > 0
        item['read_plus_write_GBs'] = (
            item['dram__bytes_read.sum'] + item['dram__bytes_write.sum']
        ) / item['gpu__time_duration.sum']
        values.append(item)
    return values


def main():
    assert os.geteuid() == 0, 'Root transient unit is required for bounded descendant cleanup'
    signal.signal(signal.SIGTERM, interrupted)
    fd = os.open(ROOT / 'controlled.started', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    before = inspect()
    assert before['Id'] == CID and before['State']['Running']
    assert fingerprint(before) == EXPECTED_CONFIG, 'Unexpected production configuration'
    assert api('health') == {'status': 'ok'}
    assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest() == '33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8'
    save('production-before.json', dict(
        id=CID, config_fingerprint=fingerprint(before), state=before['State'],
        utc=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly approved a short production pause followed by restoration for batch-8 decode counter sampling.',
    ))
    env = dict(
        PATH='/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', HOME='/root',
        CUDA_VISIBLE_DEVICES='1,0', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        CUDA_CACHE_PATH=str(ROOT / 'cache'),
        PYTHONPATH=':'.join(map(str, [
            REPO / 'v100-tp-diagnostics-v2/python-deps', SOURCE / 'src',
            SOURCE / 'tests', SOURCE / 'tools', Path('/home/wzu/.local/lib/python3.10/site-packages'),
        ])),
    )
    base = [PYTHON, str(ROOT / 'profile_decode.py'), '--artifact',
            str(REPO / 'models/qwen3.5-27b-text-q8.ninfer'), '--case',
            str(REPO / 'v100-pp-final-v1/case.json'), '--library', str(LIBRARY)]
    stopped = threading.Event()

    def telemetry():
        with (ROOT / 'gpu-telemetry.jsonl').open('w') as log:
            while not stopped.is_set():
                stamp = time.monotonic()
                try:
                    r = subprocess.run([
                        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu',
                        '--format=csv,noheader,nounits',
                    ], capture_output=True, text=True, timeout=8)
                    log.write(json.dumps(dict(monotonic=stamp, returncode=r.returncode,
                                              csv=r.stdout.strip(), error=r.stderr.strip())) + '\n')
                except Exception as e:
                    log.write(json.dumps(dict(monotonic=stamp, error=repr(e))) + '\n')
                log.flush()
                stopped.wait(0.5)

    sampler, paused, active = None, False, None
    results = []
    try:
        idle = 0
        for _ in range(90):
            slots = api('slots')
            idle = idle + 1 if slots and not any(s.get('is_processing') for s in slots) else 0
            if idle >= 3:
                break
            time.sleep(2)
        assert idle >= 3, 'Production remained busy; maintenance cancelled'
        paused = True
        subprocess.run(['docker', 'stop', '-t', '45', CID], check=True, timeout=75)
        save('production-paused.json', dict(utc=datetime.now(timezone.utc).isoformat(), id=CID))
        sampler = threading.Thread(target=telemetry)
        sampler.start()
        deadline = time.monotonic() + 900
        for name, selected in [('unprofiled-before', None), ('counters-logical0', 0),
                               ('counters-logical1', 1), ('unprofiled-after', None)]:
            mode = 'unprofiled' if selected is None else 'counters'
            command = base + ['--mode', mode, '--output-directory', str(ROOT / name)]
            if selected is not None:
                command += ['--selected', str(selected), '--counter-fast']
                command = [
                    '/usr/local/cuda/bin/ncu', '--clock-control', 'none',
                    '--cache-control', 'none', '--target-processes', 'all',
                    '--replay-mode', 'app-range', '--disable-extra-suffixes',
                    '--metrics', ','.join(METRICS), '--csv', '--page', 'raw',
                    '--log-file', str(ROOT / (name + '.csv')),
                ] + command
            remaining = deadline - time.monotonic()
            assert remaining > 15, 'Finite campaign time budget exhausted'
            save(name + '-launch.json', dict(command=command, utc=datetime.now(timezone.utc).isoformat()))
            with (ROOT / (name + '.log')).open('w') as log:
                active = subprocess.Popen(command, cwd=SOURCE, env=env,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = active.wait(timeout=min(300, remaining))
                except BaseException:
                    os.killpg(active.pid, signal.SIGTERM)
                    try:
                        active.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(active.pid, signal.SIGKILL)
                        active.wait(timeout=8)
                    raise
                finally:
                    active = None
            save(name + '-exit.json', dict(returncode=code))
            assert code == 0, f'{name} failed; see retained log'
            reports = [json.loads(p.read_text()) for p in (ROOT / name).glob('run-*.json')]
            assert reports and all(r.get('passed') for r in reports), 'Incomplete correctness report'
            result = dict(name=name, report_count=len(reports))
            if selected is not None:
                result['ranges'] = counter_rows(ROOT / (name + '.csv'), selected)
            else:
                result['unprofiled_summary'] = reports[0]['unprofiled_summary']
            results.append(result)
            save('measurements.json', results)
            print(json.dumps(result), flush=True)
        save('benchmark-complete.json', dict(passed=True, utc=datetime.now(timezone.utc).isoformat()))
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=12)
        if paused:
            assert inspect()['Id'] == CID, 'Production container identity changed externally'
            start = subprocess.run(['docker', 'start', CID], capture_output=True, text=True, timeout=90)
            restored = False
            for _ in range(100):
                try:
                    if api('health') == {'status': 'ok'}:
                        restored = True
                        break
                except Exception:
                    pass
                time.sleep(2)
            after = inspect()
            state = dict(id=after['Id'], running=after['State']['Running'], healthy=restored,
                         config_fingerprint=fingerprint(after), configuration_unchanged=fingerprint(after) == EXPECTED_CONFIG,
                         start_returncode=start.returncode, start_stderr=start.stderr,
                         utc=datetime.now(timezone.utc).isoformat())
            save('production-restored.json', state)
            assert restored and state['configuration_unchanged'], 'Production restoration needs attention'


if __name__ == '__main__':
    main()
