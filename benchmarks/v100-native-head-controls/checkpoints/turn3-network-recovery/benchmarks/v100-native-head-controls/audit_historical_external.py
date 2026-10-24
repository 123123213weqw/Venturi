"""Retain every discoverable matched, repeated pinned external timing report.

Raw command paths plus original-work/precision/slot settings filter comparable
cells. The binary/payload hashes are established separately in each capsule and
reconfirmed in fresh-external-pins; command paths alone are not source provenance.
This audit never erases a stronger previously recorded reference.
"""
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SERVER = '/home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin/llama-server'
GGUF = '/home/data/wangyue/models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf'
REFERENCE = ROOT / 'benchmarks/v100-native-head-gdn/v100/campaign-v5/matrix-mtp2-candidate/timed/report.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def option(cmd, key):
    return cmd[cmd.index(key)+1] if key in cmd else None


def main():
    case = json.loads(REFERENCE.read_text())['case']
    records, excluded = [], []
    for cap in sorted((ROOT / 'benchmarks').glob('v100-*')):
        if cap.resolve() == OUT:
            continue
        for p in sorted((cap / 'v100').rglob('report.json')):
            if 'llama' not in str(p.relative_to(cap)) or 'source' in p.relative_to(cap).parts:
                continue
            cmdpath = p.parent.parent / 'command.json'
            if not cmdpath.is_file():
                continue
            cmd = json.loads(cmdpath.read_text())
            if not isinstance(cmd, list) or not cmd or cmd[0] != SERVER or option(cmd, '-m') != GGUF:
                continue
            data = json.loads(p.read_text())
            if not data.get('completed') or 'waves' not in data:
                continue
            assert option(cmd, '-ctk') == option(cmd, '-ctv') == 'f16'
            assert option(cmd, '-c') == '2048' and option(cmd, '--parallel') == '8'
            assert data['case']['prompt_token_ids'] == case['prompt_token_ids'] and data['case']['max_new_tokens'] == 32
            for count in (1, 2, 4, 8):
                waves = [w for w in data['waves'] if w['request_count'] == count]
                if not waves:
                    continue
                if len(waves) < 3:
                    excluded.append(dict(report=str(p.relative_to(ROOT)), requests=count, trials=len(waves), reason='Fewer than3 repeats, historical screen retained but not a promoted baseline'))
                    continue
                for w in waves:
                    assert w['emitted_tokens'] == count * 32 and len(w['requests']) == count
                    for r in w['requests']:
                        x = r['response']
                        assert x['tokens_evaluated'] == x['timings']['prompt_n'] == 23 and x['tokens_predicted'] == len(x['tokens']) == 32
                cell = dict(report=str(p.relative_to(ROOT)), report_sha256=sha(p), command_sha256=sha(cmdpath), requests=count, trials=len(waves), mode=option(cmd, '--split-mode'), depth=int(option(cmd, '--spec-draft-n-max') or 0), http_tps=count * 32 / statistics.mean(w['wall_seconds'] for w in waves))
                if count == 1:
                    cell['post_first_tps'] = 31 / statistics.mean(w['requests'][0]['response']['timings']['predicted_ms'] / 1000 for w in waves)
                records.append(cell)
    old = json.loads((ROOT / 'benchmarks/v100-native-head-gdn/metrics.json').read_text())
    best_http = {}
    for count in (1, 2, 4, 8):
        values = [r for r in records if r['requests'] == count]
        assert values
        best = max(values, key=lambda x:x['http_tps'])
        previous = old['selected_http'][str(count)]['retained_external']
        assert best['http_tps'] >= previous['http_tps'] - 1e-9, 'Historical stronger reference disappeared'
        best_http[str(count)] = best
    best_single = max((r for r in records if 'post_first_tps' in r), key=lambda x:x['post_first_tps'])
    assert best_single['post_first_tps'] >= old['single_decode_comparison']['best_retained_external_tps'] - 1e-9
    result = dict(scope=__doc__, command_path_filter_is_not_standalone_hash_provenance=True, comparable_repeated_cells=records, excluded_screens=excluded, strongest_http_by_count=best_http, strongest_single=best_single, completed=True)
    (OUT / 'historical-external-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cells=len(records), reports=len({r['report'] for r in records}), screens=len(excluded), best_http={k:v['http_tps'] for k,v in best_http.items()}, best_single=best_single), indent=2))


if __name__ == '__main__':
    main()
