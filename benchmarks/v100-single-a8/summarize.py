"""Offline audit: operator speed and arithmetic-error screens are not model gates."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parent
ATTEMPT = ROOT/'attempt-v1'


def read(path):return json.loads(path.read_text())


def summarize_performance(path, expected_cases):
    report = read(path/'report.json')
    assert report['completed'] and not report['profiled']
    assert len(report['cases'])==expected_cases
    results, errors = [], []
    for case in report['cases']:
        assert case['complete_two_pass_bytes_equal'] and case['one_pass_first_plane_bytes_equal']
        assert case['torch_reserved_bytes']<1024**3
        scopes = {}
        for scope, recorded in case['median_ms'].items():
            arms = {}
            for arm, value in recorded.items():
                waves = [w for w in case['waves'] if w['scope']==scope and w['arm']==arm]
                assert len(waves)==7 and all(w['calls']==100 for w in waves)
                assert statistics.median(w['gpu_event_ms_per_call'] for w in waves)==value
                arms[arm] = dict(gpu_event_ms=value,
                    wall_ms=statistics.median(1000*w['wall_seconds']/w['calls'] for w in waves),
                    process_cpu_ms=statistics.median(1000*w['process_cpu_seconds']/w['calls'] for w in waves),
                    process_cpu_cores=sum(w['process_cpu_seconds'] for w in waves)/sum(w['wall_seconds'] for w in waves),
                    gpu_event_range_ms=[min(w['gpu_event_ms_per_call'] for w in waves),max(w['gpu_event_ms_per_call'] for w in waves)])
            scopes[scope] = dict(arms=arms,
                latency_reduction_vs_pinned=1-recorded['single-a8']/recorded['pinned-two'],
                latency_reduction_vs_same_source=1-recorded['single-a8']/recorded['local-two'],
                same_source_control_latency_drift=recorded['local-two']/recorded['pinned-two']-1)
        for quality in case['quality']:
            assert max(quality['implementation_fp64_sample_relative_l2'].values())<0.001
            errors.append(dict(case=case['name'],tokens=case['tokens'],rows=quality['rows'],family=quality['family'],
                               single=quality['versus_unquantized_activation']['single-a8'],
                               two=quality['versus_unquantized_activation']['pinned-two'],
                               single_versus_two=quality['single_versus_two']))
        results.append(dict(name=case['name'],tokens=case['tokens'],rows=case['rows'],k=case['k'],scopes=scopes))
    by_token = {}
    for t in sorted(set(c['tokens'] for c in results)):
        reductions = [c['scopes']['quantize+projection']['latency_reduction_vs_pinned'] for c in results if c['tokens']==t]
        by_token[t] = dict(median_operator_latency_reduction=statistics.median(reductions),
                          worst_operator_latency_reduction=min(reductions),
                          speed_screen_passed=statistics.median(reductions)>=0.05 and min(reductions)>=-0.02)
    telemetry = []
    for row in read(path/'gpu-telemetry.json'):
        if row.get('returncode')==0:
            fields = next(csv.reader([row['csv']]))
            assert fields[1].strip().removeprefix('GPU-')==report['device_uuid'].removeprefix('GPU-')
            telemetry.append([float(v) for v in fields[2:]])
    names = ['device_used_MiB_including_production','gpu_busy_percent','power_watts','sm_MHz','memory_MHz','temperature_C']
    assert telemetry
    return dict(library_sha256=report['library_sha256'],control_library_sha256=report['control_library_sha256'],
                weights=report['gguf'],cases=results,synthetic_quality=errors,speed_screens=by_token,
                telemetry_scope='All trial arms and warmups, not per-arm utilization/bandwidth.',
                telemetry={n:dict(min=min(v[i] for v in telemetry),median=statistics.median(v[i] for v in telemetry),max=max(v[i] for v in telemetry)) for i,n in enumerate(names)})


def counter(path):
    s = path.read_text();offset = s.find('"ID",');assert offset>=0
    rows = [r for r in csv.DictReader(io.StringIO(s[offset:])) if r.get('ID','').isdigit()]
    assert len(rows)==1
    row = rows[0];result = {'kernel':row['Kernel Name'],'profiled':True}
    for k,v in row.items():
        if '__' in k:
            try:result[k] = float(v.replace(',',''))
            except ValueError:result[k] = v
    result['dram_read_plus_write_GB_per_s_profiled'] = (result['dram__bytes_read.sum']+result['dram__bytes_write.sum'])/result['gpu__time_duration.sum']
    return result


def main():
    assert '225 passed' in (ATTEMPT/'tests.log').read_text()
    assert read(ATTEMPT/'completed.json')['completed']
    for name in ['build','screen','confirm','counter-local-two','counter-single-a8']:
        assert (ATTEMPT/f'{name}.exit').read_text().strip()=='0'
    before,after = [read(ATTEMPT/f'production-{v}.json') for v in ['before','after']]
    for field in ['container_id','canonical_config_fingerprint','started_at','restart_count','counter_policy']:
        assert before[field]==after[field],field
    assert before['health']==after['health']=='healthy'
    source = read(ATTEMPT/'executed-source-sha256.json')
    repo = ROOT.parents[1]
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in source.items())
    screen,confirm = summarize_performance(ATTEMPT/'screen',30),summarize_performance(ATTEMPT/'confirm',10)
    assert screen['library_sha256']==confirm['library_sha256']==read(ATTEMPT/'completed.json')['library_sha256']
    assert screen['weights']==confirm['weights']
    families = {}
    for family in ('gaussian','group-outlier','mixed-range'):
        values = [v for v in screen['synthetic_quality'] if v['family']==family]
        families[family] = dict(relative_l2_range=[min(v['single']['relative_l2'] for v in values),max(v['single']['relative_l2'] for v in values)],
            minimum_centered_cosine=min(v['single']['centered_cosine'] for v in values),
            all_operator_screens_passed=all(v['single']['operator_screen_passed'] for v in values))
    small = struct.unpack('e',struct.pack('e',0.003))[0]
    actual_counterexample = read(ATTEMPT/'small-feature-counterexample.json')
    assert actual_counterexample['oracle_f64']==small*127
    assert actual_counterexample['passes']['1']['output']==0
    assert abs(actual_counterexample['passes']['2']['output']-small*127)/(small*127)<1e-5
    result = dict(scope='Packed projection and quantize+projection only; actual Qwen3.8 GGUF weight bytes and synthetic F16 activations. Not full-model/native Qwen3.8/serving TPS or corpus-quality validation.',
        correctness_tests=225,screen=screen,confirm=confirm,synthetic_quality_ranges=families,
        counters={name:counter(ATTEMPT/f'counter-{name}.csv') for name in ['local-two','single-a8']},
        counter_limit='ALU/issue percentages are not total DP4A compute utilization; counter runs are separate from normal performance.',
        small_feature_counterexample=dict(input='F16 x[0]=1, x[1]=0.003; other elements zero',
            weight='Q8 q[0]=0, q[1]=127, other codes zero, scale=1',reference_output=small*127,
            single_pass_output=actual_counterexample['passes']['1']['output'],
            two_pass_output=actual_counterexample['passes']['2']['output'],
            two_pass_relative_error_bound_verified_by_test=1e-5,
            interpretation='Single pass loses the small feature completely; high global cosine is not a general accuracy guarantee.'),
        production_before=before,production_after=after,local_matches_executed_source_files=len(source),
        decision=dict(default_or_production_enabled=False,
            single_request='Rejected by predeclared speed screen: little gain, MLP-down regresses about 4%.',
            parallel='Synthetic operator speed/quality screens only. Candidate for an explicitly lower-precision opt-in after full-model quality validation, not an approved model mode.',
            full_model_gate='Not run. Requires compatible native model support and a maintenance window; no production shutdown authorized in this task.'))
    (ROOT/'recomputed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for c in confirm['cases']:
        if c['tokens']==8:
            s = c['scopes']['quantize+projection'];arms = s['arms']
            print(c['name'],round(arms['pinned-two']['gpu_event_ms'],6),'->',round(arms['single-a8']['gpu_event_ms'],6),
                  f"latency {-100*s['latency_reduction_vs_pinned']:.2f}%")
    print('Quality:',families)


if __name__=='__main__':main()
