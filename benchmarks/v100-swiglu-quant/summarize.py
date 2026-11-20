"""Recompute fusion measurements without running CUDA or changing production."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def read(p):return json.loads(p.read_text())


def performance(folder,expected,shared):
    report=read(folder/'report.json')
    assert report['completed'] and not report['profiled'] and len(report['cases'])==expected
    cases=[]
    for c in report['cases']:
        assert c['complete_workspace_bytes_equal'] and c['complete_down_bytes_equal'] and c['clone_f16_bytes_equal']
        assert c['sampled_independent_fp64_down_relative_l2']<1e-5
        assert c['torch_reserved_bytes']<1024**3
        if shared:assert c['timed_arms_share_intermediate_workspace_and_down_output_addresses']
        scopes={}
        for scope,medians in c['median_ms'].items():
            arms={}
            for name,value in medians.items():
                waves=[w for w in c['waves'] if w['scope']==scope and w['arm']==name]
                assert len(waves)==7 and all(w['calls']==100 for w in waves)
                assert statistics.median(w['gpu_event_ms_per_call'] for w in waves)==value
                arms[name]=dict(gpu_event_ms=value,
                    gpu_event_range_ms=[min(w['gpu_event_ms_per_call'] for w in waves),max(w['gpu_event_ms_per_call'] for w in waves)],
                    wall_ms=statistics.median(1000*w['wall_seconds']/w['calls'] for w in waves),
                    process_cpu_ms=statistics.median(1000*w['process_cpu_seconds']/w['calls'] for w in waves),
                    process_cpu_cores=sum(w['process_cpu_seconds'] for w in waves)/sum(w['wall_seconds'] for w in waves),
                    latency_reduction_vs_pinned=1-value/medians['pinned-separate'],
                    latency_reduction_vs_clone=1-value/medians['local-separate'])
            scopes[scope]=dict(arms=arms,clone_control_drift=medians['local-separate']/medians['pinned-separate']-1)
        cases.append(dict(layer=c['layer'],tokens=c['tokens'],scopes=scopes,weight_pins=c['weight_pins'],
                          fp64_sample_relative_l2=c['sampled_independent_fp64_down_relative_l2'],
                          process_torch_reserved_bytes=c['torch_reserved_bytes']))
    samples=[]
    for row in read(folder/'gpu-telemetry.json'):
        if row.get('returncode')==0:
            fields=next(csv.reader([row['csv']]))
            assert fields[1].strip().removeprefix('GPU-')==report['device_uuid'].removeprefix('GPU-')
            samples.append([float(v) for v in fields[2:]])
    assert samples
    names=['device_used_MiB_including_production','gpu_busy_percent','power_watts','sm_MHz','memory_MHz','temperature_C']
    return dict(library_sha256=report['library_sha256'],control_library_sha256=report['control_library_sha256'],
                checkpoint=report['gguf'],cases=cases,
                telemetry_scope='All arms, warmups and idle gaps; not per-arm utilization or bandwidth.',
                telemetry={n:dict(min=min(v[i] for v in samples),median=statistics.median(v[i] for v in samples),max=max(v[i] for v in samples)) for i,n in enumerate(names)})


def counters(folder,arm):
    p=folder/f'counter-{arm}.csv';s=p.read_text();offset=s.find('"ID",');assert offset>=0
    rows=[r for r in csv.DictReader(io.StringIO(s[offset:])) if r.get('ID','').isdigit()]
    assert len(rows)==(2 if arm=='pinned-separate' else 1)
    result=[]
    for r in rows:
        row={'kernel':r['Kernel Name']}
        for k,v in r.items():
            if '__' in k:
                try:row[k]=float(v.replace(',',''))
                except ValueError:row[k]=v
        result.append(row)
    return dict(profiled=True,kernels=result,
                sum_kernel_duration_ns_not_wall=sum(r['gpu__time_duration.sum'] for r in result),
                global_load_request_bytes=sum(r['l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum'] for r in result),
                global_store_request_bytes=sum(r['l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum'] for r in result))


def gate(screen,confirm,arm):
    checks={}
    for name,data in [('screen',screen),('confirm',confirm)]:
        gains={scope:[c['scopes'][scope]['arms'][arm]['latency_reduction_vs_pinned'] for c in data['cases']] for scope in ['producer','producer+down','mlp-shard']}
        checks[name]={scope:dict(median=statistics.median(v),min=min(v),max=max(v)) for scope,v in gains.items()}
    passed=all(checks[n]['producer']['median']>=0.05 and checks[n]['producer+down']['median']>=0.01 for n in checks)
    passed=passed and checks['confirm']['producer+down']['min']>=-0.02
    return dict(predeclared_operator_gate_passed=passed,results=checks)


def main():
    assert '122 passed' in (ROOT/'attempt-v1/tests.log').read_text()
    result=dict(scope='SwiGLU/two-pass-quantization fusion operators and one-rank MLP projection chains only; not full-model TPS. Actual Qwen3.8 Q8_0 weights with synthetic hidden inputs.',
                correctness_tests=122,precision_changed=False,production_deployed=False,attempts={})
    for v in (1,2):
        folder=ROOT/f'attempt-v{v}'
        assert read(folder/'completed.json')['completed']
        for command in ['screen','confirm','counter-pinned-separate','counter-fused-128','counter-fused-256']:
            assert (folder/f'{command}.exit').read_text().strip()=='0'
        screen,confirm=performance(folder/'screen',18,v==2),performance(folder/'confirm',2,v==2)
        assert screen['library_sha256']==confirm['library_sha256']==read(folder/'completed.json')['library_sha256']
        assert screen['checkpoint']==confirm['checkpoint']
        before,after=[read(folder/f'production-{name}.json') for name in ['before','after']]
        for name in ['container_id','canonical_config_fingerprint','started_at','restart_count','counter_policy']:
            assert before[name]==after[name]
        assert before['health']==after['health']=='healthy'
        result['attempts'][f'v{v}']=dict(screen=screen,confirm=confirm,
            counters={arm:counters(folder,arm) for arm in ['pinned-separate','fused-128','fused-256']},
            production_before=before,production_after=after,
            gates={arm:gate(screen,confirm,arm) for arm in ['fused-128','fused-256']})
    old,current=result['attempts']['v1'],result['attempts']['v2']
    assert old['screen']['library_sha256']==current['screen']['library_sha256']
    for a,b in zip(old['screen']['cases'],current['screen']['cases']):
        assert (a['layer'],a['tokens'],a['weight_pins'])==(b['layer'],b['tokens'],b['weight_pins'])
    pins=read(ROOT/'attempt-v2/executed-source-sha256.json')
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in pins.items())
    initial=read(ROOT/'initial-source-pins.json')
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in initial.items())
    result.update(local_matches_executed_source_files=len(pins),existing_core_sources_unchanged=True,
        canonical_result='v2 shared-address control. v1 preserved, but its larger downstream gain must not be attributed entirely to fusion.',
        limitations=['Producer fusion also specializes fixed two-pass quantization; not all gain can be attributed solely to removing a launch.',
            'F16 rounding and both A8 passes preserved. Invalid-group scale bytes remain canonical; debug NaN payloads are not a public contract.',
            'All Down kernels are pinned and identical across arms. Same timed workspace/intermediate/output addresses in v2.',
            'FP16 intermediate removed from the fast producer path: theoretical 4 bytes/element of writes plus reads avoided, not necessarily DRAM traffic.',
            'NCU data is separate from normal timings. XU/ALU pipe metrics are partial instruction-pipe diagnostics, not total compute utilization.',
            'No default/native model integration or production deployment. Full-model/native integration correctness and serving speed remain untested.'])
    (ROOT/'recomputed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for c in current['confirm']['cases']:
        print('T',c['tokens'])
        for scope,data in c['scopes'].items():
            old,new=data['arms']['pinned-separate'],data['arms']['fused-256']
            print(scope,round(old['gpu_event_ms']*1000,3),'->',round(new['gpu_event_ms']*1000,3),'us',
                  f"saving {100*new['latency_reduction_vs_pinned']:.2f}%")
    print('Gates:',current['gates'])


if __name__=='__main__':main()
