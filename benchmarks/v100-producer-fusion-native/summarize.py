"""Offline audit; no GPU execution or production changes."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def read(p):return json.loads(p.read_text())


def timing(p,expected,baseline,native):
    r=read(p/'report.json');assert r['completed'] and not r['profiled'] and len(r['cases'])==expected
    result=[]
    for c in r['cases']:
        assert c['torch_reserved_bytes']<1024**3
        if native:
            assert c['complete_workspace_bytes_equal'] and c['complete_down_bytes_equal'] and c['materialized_f16_bytes_equal']
            assert c['timed_arms_share_intermediate_workspace_and_down_output_addresses']
            assert c['sampled_independent_fp64_down_relative_l2']<1e-5
        else:
            assert c['full_workspace_bytes_equal'] and c['full_projection_bytes_equal'] and c['full_f16_intermediate_bytes_equal'] and c['timed_arms_share_addresses']
        scopes={}
        for scope,arms in c['median_ms'].items():
            data={}
            for name,gpu in arms.items():
                waves=[w for w in c['waves'] if w['scope']==scope and w['arm']==name]
                assert len(waves)==7 and all(w['calls']==100 for w in waves)
                assert statistics.median(w['gpu_event_ms_per_call'] for w in waves)==gpu
                data[name]=dict(gpu_ms=gpu,latency_reduction=1-gpu/arms[baseline],
                    range_ms=[min(w['gpu_event_ms_per_call'] for w in waves),max(w['gpu_event_ms_per_call'] for w in waves)],
                    wall_ms=statistics.median(1000*w['wall_seconds']/w['calls'] for w in waves),
                    process_cpu_ms=statistics.median(1000*w['process_cpu_seconds']/w['calls'] for w in waves),
                    process_cpu_cores=sum(w['process_cpu_seconds'] for w in waves)/sum(w['wall_seconds'] for w in waves))
            scopes[scope]=data
        result.append(dict(layer=c['layer'],tokens=c['tokens'],scopes=scopes))
    samples=[]
    for row in read(p/'gpu-telemetry.json'):
        if row.get('returncode')==0:
            values=next(csv.reader([row['csv']]));assert values[1].strip().removeprefix('GPU-')==r['device_uuid'].removeprefix('GPU-')
            samples.append(list(map(float,values[2:])))
    names=['device_MiB_including_production','gpu_busy_percent','power_watts','sm_MHz','memory_MHz','temperature_C'];assert samples
    return dict(cases=result,library_sha256=r['library_sha256'],control_library_sha256=r['control_library_sha256'],checkpoint=r['gguf'],
        telemetry_scope='All arms, warmup and gaps, not per-arm bandwidth.',telemetry={name:dict(min=min(x[i] for x in samples),median=statistics.median(x[i] for x in samples),max=max(x[i] for x in samples)) for i,name in enumerate(names)})


def counter(p,expected):
    s=p.read_text();a=s.index('"ID",');rows=[r for r in csv.DictReader(io.StringIO(s[a:])) if r.get('ID','').isdigit()];assert len(rows)==expected
    result=[]
    for row in rows:
        out=dict(kernel=row['Kernel Name'])
        for k,v in row.items():
            if '__' in k:
                try:out[k]=float(v.replace(',',''))
                except ValueError:out[k]=v
        for metric in ['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','dram__throughput.avg.pct_of_peak_sustained_elapsed']:assert isinstance(out[metric],float)
        result.append(out)
    return result


def gate(screen,confirm,arm,consumer):
    checks={}
    for name,data in [('screen',screen),('confirm',confirm)]:
        checks[name]={scope:dict(median=statistics.median(c['scopes'][scope][arm]['latency_reduction'] for c in data['cases']),min=min(c['scopes'][scope][arm]['latency_reduction'] for c in data['cases'])) for scope in ['producer',consumer]}
    passed=all(d['producer']['median']>=0.05 and d[consumer]['median']>=0.01 for d in checks.values()) and checks['confirm'][consumer]['min']>=-0.02
    return dict(predeclared_operator_gate_passed=passed,results=checks)


def main():
    folder=ROOT/'attempt-v1';ops=folder/'operators'
    assert read(folder/'completed.json')['completed'] and read(ops/'completed.json')['completed']
    for sub in [folder,ops]:
        for p in sub.glob('*.exit'):assert p.read_text().strip()=='0',p
        a,b=[read(sub/('production-'+n+'.json')) for n in ['before','after']]
        for k in ['container_id','started_at','restart_count','canonical_config_fingerprint','counter_policy']:assert a[k]==b[k]
        assert a['status']==b['status']=='running' and a['health']==b['health']=='healthy'
    counts={}
    for name in ['build-api-tests','native-fusion-tests','producer-byte-tests','native-fallback-tests']:
        s=(folder/(name+'.log')).read_text();match=re.search(r'(\d+) passed',s);assert match,name;counts[name]=int(match.group(1))
    s=(ops/'rms-tests.log').read_text();counts['rms-tests']=int(re.search(r'(\d+) passed',s).group(1))
    native=[timing(ops/n,e,'native-off',True) for n,e in [('native-screen',18),('native-confirm',2)]]
    rms=[timing(ops/n,e,'separate',False) for n,e in [('rms-screen',18),('rms-confirm',2)]]
    for pair in [native,rms]:
        assert pair[0]['library_sha256']==pair[1]['library_sha256'] and pair[0]['control_library_sha256']==pair[1]['control_library_sha256'] and pair[0]['checkpoint']==pair[1]['checkpoint']
    libs=read(folder/'libraries.json');builds={}
    for name in ['off','on']:
        files=list((folder/'build').glob('*/build.json'));record=next(read(p) for p in files if str(p.parent.name) in libs[name]);builds[name]=record
        assert all(hashlib.sha256((REPO/'src/venturi/ops/csrc'/p).read_bytes()).hexdigest()==h for p,h in record['source_sha256'].items())
    assert builds['off']['source_sha256']==builds['on']['source_sha256']
    def flags(b):return [s for s in b['command'] if s.startswith('-D')]
    assert [s for s in flags(builds['on']) if s not in flags(builds['off'])]==['-DVENTURI_Q8_SWIGLU_QUANT=1']
    assert [s for s in flags(builds['off']) if s not in flags(builds['on'])]==['-DVENTURI_Q8_SWIGLU_QUANT=0']
    assert builds['on']['library_sha256']==native[0]['library_sha256'] and builds['off']['library_sha256']==native[0]['control_library_sha256']==rms[0]['control_library_sha256']
    for p,h in read(ops/'executed-source-sha256.json').items():assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h,p
    best=read(ops/'rms-screen-selection.json')['best_by_screen_producer_median']
    result=dict(scope='Native opt-in integration with eligible synthetic hybrid fixtures plus isolated V100 operator timings. No full-model Qwen3.8 TPS or API-concurrency result.',
        precision_changed=False,production_deployed=False,default_enabled=False,correctness=counts,builds=builds,
        native=dict(screen=native[0],confirm=native[1],gate=gate(*native,'native-on','producer+down')),
        rms=dict(screen=rms[0],confirm=rms[1],selection=best,gates={arm:gate(*rms,arm,'producer+gate-up') for arm in ['chunk8','chunk32','chunk128']}),
        counters={},production_after=read(ops/'production-after.json'))
    for label,arms in [('native',['native-off','native-on']),('rms',['separate',best])]:
        for arm in arms:result['counters'][label+'-'+arm]=counter(ops/f'counter-{label}-{arm}.csv',2 if arm in ['native-off','separate'] else 1)
    (ROOT/'recomputed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Correctness',counts)
    for label,pair,arm in [('native',native,'native-on'),('rms',rms,best)]:
        print(label,arm)
        for c in pair[1]['cases']:
            print('T',c['tokens'],{scope:dict(gpu_us=round(data[arm]['gpu_ms']*1000,3),reduction_pct=round(100*data[arm]['latency_reduction'],3)) for scope,data in c['scopes'].items()})
    print('RMS gates',result['rms']['gates'])


if __name__=='__main__':main()
