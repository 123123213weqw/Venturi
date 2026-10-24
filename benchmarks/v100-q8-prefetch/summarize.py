"""Recompute local operator evidence. No remote calls or GPU execution."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def counters(path):
    text = path.read_text()
    offset = text.find('"ID",')
    assert offset >= 0, path
    rows = [r for r in csv.DictReader(io.StringIO(text[offset:])) if r.get('ID', '').isdigit()]
    assert len(rows) == 1, path
    row = rows[0]
    result = {'kernel': row['Kernel Name'], 'profiled': True}
    for key, value in row.items():
        if '__' in key:
            try:
                result[key] = float(value.replace(',', ''))
            except ValueError:
                result[key] = value
    duration = result['gpu__time_duration.sum'] * 1e-9
    result['dram_read_plus_write_GB_per_s_profiled'] = (
        result['dram__bytes_read.sum'] + result['dram__bytes_write.sum']) / duration / 1e9
    return result


def telemetry(path):
    samples = read(path)
    values = []
    for sample in samples:
        if sample.get('returncode') == 0:
            fields = next(csv.reader([sample['csv']]))
            assert fields[1].strip() == 'GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28'
            values.append([float(v) for v in fields[2:]])
    assert values
    names = ['device_total_used_MiB_including_production', 'gpu_busy_percent',
             'power_watts', 'sm_MHz', 'memory_MHz', 'temperature_C']
    return dict(scope='All arms, warmups, and idle gaps; NOT per-arm utilization or bandwidth.',
                samples=len(values), errors=len(samples)-len(values),
                values={name: dict(min=min(v[i] for v in values),
                                   median=statistics.median(v[i] for v in values),
                                   max=max(v[i] for v in values)) for i,name in enumerate(names)})


def performance(path):
    report = read(path/'report.json')
    assert report['completed'] and not report['profiled'] and len(report['cases']) == 15
    results = []
    for case in report['cases']:
        assert case['complete_output_bytes_equal']
        assert max(case['sampled_fp64_relative_l2']) < 0.001
        assert case['additional_reserved_bytes'] < 1024**3
        waves = case['waves']
        median = {name: statistics.median(w['gpu_event_ms_per_call'] for w in waves if w['arm']==name)
                  for name in case['median_ms']}
        assert median == case['median_ms']
        arms = {}
        for name in median:
            own = [w for w in waves if w['arm']==name]
            arms[name] = dict(
                gpu_event_ms_per_call=median[name],
                gpu_event_range_ms=[min(w['gpu_event_ms_per_call'] for w in own), max(w['gpu_event_ms_per_call'] for w in own)],
                wall_ms_per_call=statistics.median(1000*w['wall_seconds']/w['calls'] for w in own),
                process_cpu_ms_per_call=statistics.median(1000*w['process_cpu_seconds']/w['calls'] for w in own),
                process_cpu_cores=sum(w['process_cpu_seconds'] for w in own)/sum(w['wall_seconds'] for w in own),
                relative_operator_rate_vs_qualified=median['qualified']/median[name],
                latency_reduction_fraction_vs_qualified=1-median[name]/median['qualified'],
                latency_reduction_fraction_vs_same_build_control=1-median[name]/median['probe-control'])
        results.append(dict(name=case['name'], tokens=case['tokens'], rows=case['rows'], k=case['k'],
                            partial_fp32=case['partial_fp32'], arms=arms,
                            sampled_fp64_relative_l2=case['sampled_fp64_relative_l2'],
                            process_torch_reserved_bytes=case['additional_reserved_bytes']))
    return dict(library_sha256=report['library_sha256'],
                qualified_library_sha256=report['qualified_library_sha256'],
                precision=report['precision'], cases=results,
                telemetry=telemetry(path/'gpu-telemetry.json'))


def main():
    expected_tests = {
        'attempt-v1/tests.log': '47 passed',
        'attempt-v2/tests.log': '49 passed',
        'attempt-v3/default-off-tests.log': '1 failed, 118 passed',
        'attempt-v3/default-off-tests-repaired.log': '119 passed',
        'attempt-v3/integrated-tests.log': '138 passed',
    }
    for path, expected in expected_tests.items():
        assert expected in (ROOT/path).read_text(), path
    result = dict(
        scope='Packed projection operators on physical V100 GPU0, production Qwen3.8 GGUF Q8_0 bytes; synthetic F16 activations, two-pass residual A8. Quantization, activation functions, transport, attention/GDN recurrence, and serving excluded from timings. Not full-model TPS.',
        tests=dict(v1_research=47, v2_research=49, default_off_repaired=119,
                   integrated_operator_and_small_native=138),
        preserved_failures=['v1 initial missing math_constants.h build',
                            'v3 first default-off suite: 118 passed, stale build-dependency count assertion failed; repaired to 13, then 119 passed'],
        attempts={name: performance(ROOT/path) for name,path in [
            ('prefetch_v1','attempt-v1/screen'), ('row_pair_v2_screen','attempt-v2/screen'),
            ('row_pair_v2_confirm','attempt-v2/confirm'), ('integrated_native_dispatch','attempt-v3/native-screen')]},
        counters={str(p.relative_to(ROOT)): counters(p) for p in sorted(ROOT.glob('attempt-v*/counter-*.csv'))},
        production=read(ROOT/'attempt-v3/production-audit.json'),
        source_checks={},
        current_source_verification=read(ROOT/'current-source-verified.json'),
        decision=dict(retained='Mode3 row-pair, native compile-time opt-in --q8-row-pair, OFF by default.',
                      rejected='Mode1 early and Mode2 one-tile lookahead: no repeatable gain / substantial regression.',
                      dispatch_guards='SM70, two-pass residual, measured 27B TP shapes, T8..32 only. Neutral GDN-out T8 falls back; all single-request, MTP T1..3, tiny or unqualified shapes fall back.',
                      full_model_validation='Pending a maintenance window; no full-model Qwen3.8/native quality, single-request speed, or serving TPS claim. Production llama.cpp was never replaced.',
                      occupancy='Row-pair can be faster with lower achieved occupancy; do not maximize occupancy as the objective.',
                      compute_counter_limit='ALU-pipe metric is a partial instruction-pipe diagnostic, NOT achieved DP4A TOPS or total compute utilization. Earlier IPA metric returned zero and is not useful as DP4A utilization.'),
        prior_capsule_checks=read(ROOT/'prior-capsule-unchanged.json'))
    build = read(ROOT/'attempt-v3/build/f634328d5b15400b/build.json')
    completed = read(ROOT/'attempt-v3/integrated-completed.json')
    assert completed['completed'] and completed['library_sha256'] == build['library_sha256']
    assert result['attempts']['integrated_native_dispatch']['library_sha256'] == build['library_sha256']
    assert result['production']['unchanged'] and result['production']['health'] == 'healthy'
    for path in ROOT.glob('attempt-v*/counter-*.exit'):
        assert path.read_text().strip() == '0', path
    for name, pin in build['source_sha256'].items():
        actual = hashlib.sha256((ROOT/'attempt-v3/source-tests-benchmark-v5/src/venturi/ops/csrc'/name).read_bytes()).hexdigest()
        assert actual == pin
    for folder in ['attempt-v1/source','attempt-v2/source','attempt-v3/source-build-v3','attempt-v3/source-tests-benchmark-v5']:
        result['source_checks'][folder] = {
            str(p.relative_to(ROOT/folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
    (ROOT/'recomputed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for case in result['attempts']['integrated_native_dispatch']['cases']:
        if case['tokens']==8:
            old, new = case['arms']['qualified'], case['arms']['native-dispatch']
            print(case['name'], round(old['gpu_event_ms_per_call'],6), '->',
                  round(new['gpu_event_ms_per_call'],6),
                  f"latency {-100*new['latency_reduction_fraction_vs_qualified']:.2f}%")


if __name__ == '__main__':
    main()
