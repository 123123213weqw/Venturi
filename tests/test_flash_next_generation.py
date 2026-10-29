"""Tiny complete reference generation only; not original-model quality/TPS."""

import json

import pytest
from safetensors.torch import load_file, save_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_loader import fixture as floating_fixture, OPTIONS
from test_flash_next_fp8_loader import fixture as fp8_fixture
from venturi.cli import main
from venturi.flash_next_generation import generate_flash_next_tokens, generate_flash_next_text
from venturi.models.flash_next_loader import (
    load_floating_text, load_weight_decoded_fp8_text, plan_floating_text,
)
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest


@pytest.mark.parametrize('encoding', ['floating', 'fp8'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('chunk', [1, 2, 16])
def test_generation_matches_same_schedule_request_and_is_private(tmp_path, encoding, dtype, chunk):
    if encoding == 'fp8':
        cp, independent, _ = fp8_fixture(tmp_path, execution_dtype=dtype)
        loader = load_weight_decoded_fp8_text
    else:
        cp, independent, _ = floating_fixture(tmp_path, dtype=torch.bfloat16)
        independent.to(dtype=dtype)
        loader = load_floating_text
    loaded = loader(cp, **dict(OPTIONS, execution_dtype=dtype))
    prompt = (2, 31, 5, 6, 7)
    with PreparedNGramTable(loaded.tables[1]) as table, sdpa_kernel(SDPBackend.MATH):
        manual = FlashNextRequest(independent, table)
        for offset in range(0, len(prompt), chunk):
            proposal = manual.preview(prompt[offset:offset+chunk])
            manual.commit(proposal)
        expected = [int(manual.last_logits.argmax())]
        for _ in range(5):
            proposal = manual.preview([expected[-1]])
            manual.commit(proposal)
            expected.append(int(manual.last_logits.argmax()))
        for _ in range(2):
            actual = generate_flash_next_tokens(loaded.core, table, prompt, max_new_tokens=6,
                eos_token_ids=(), prefill_chunk_size=chunk)
            assert actual.generated_ids == tuple(expected)
            assert actual.prompt_ids == prompt and actual.finish_reason == 'length'
            assert actual.committed_tokens == len(prompt)+5 and actual.decode_tokens == 5
            assert actual.total_seconds >= actual.prefill_seconds+actual.decode_seconds >= 0
            assert actual.total_seconds >= actual.time_to_first_token_seconds >= actual.prefill_seconds
            assert actual.total_cpu_seconds >= actual.prefill_cpu_seconds+actual.decode_cpu_seconds >= 0


def test_eos_and_zero_do_not_forward_final_or_empty_output(tmp_path, monkeypatch):
    cp, _, _ = floating_fixture(tmp_path)
    loaded = load_floating_text(cp, **OPTIONS)
    core, table = loaded.core, loaded.tables[1]
    expected = generate_flash_next_tokens(core, table, [2, 3], max_new_tokens=1).generated_ids[0]
    calls, original = [], core.forward
    def counted(ids, **kwargs):
        calls.append(ids.shape[1])
        return original(ids, **kwargs)
    monkeypatch.setattr(core, 'forward', counted)
    result = generate_flash_next_tokens(core, table, [2, 3], max_new_tokens=5, eos_token_ids=(expected,))
    assert result.generated_ids == (expected,) and result.finish_reason == 'eos'
    assert calls == [2] and result.decode_tokens == 0 and result.committed_tokens == 2
    monkeypatch.setattr(table, 'lookup', lambda *a: pytest.fail('zero-output/invalid request read rows'))
    result = generate_flash_next_tokens(core, table, [2, 3], max_new_tokens=0)
    assert result.generated_ids == () and result.committed_tokens == 0
    assert result.time_to_first_token_seconds is None
    assert calls == [2]


@pytest.mark.parametrize('ids,options', [([], {}), ([True], {}), ([32], {}), ([2], {'max_new_tokens': -1}),
    ([2], {'max_new_tokens': True}), ([2], {'prefill_chunk_size': 0}), ([2], {'prefill_chunk_size': True}),
    ([2], {'max_new_tokens': 128}), ([2], {'eos_token_ids': (32,)}), ([2], {'eos_token_ids': (True,)})])
def test_invalid_request_fails_before_row_io(tmp_path, monkeypatch, ids, options):
    cp, _, _ = floating_fixture(tmp_path)
    loaded = load_floating_text(cp, **OPTIONS)
    monkeypatch.setattr(loaded.tables[1], 'lookup', lambda *a: pytest.fail('invalid request performed I/O'))
    with pytest.raises(ValueError):
        generate_flash_next_tokens(loaded.core, loaded.tables[1], ids, **options)


def test_nonfinite_prefill_or_decode_never_selects_a_token(tmp_path, monkeypatch):
    cp, _, _ = floating_fixture(tmp_path)
    loaded = load_floating_text(cp, **OPTIONS)
    original = loaded.core.forward
    for phase in ('prefill', 'decode'):
        def bad(ids, **kwargs):
            logits, state = original(ids, **kwargs)
            if phase == 'prefill' or kwargs.get('past') is not None:
                logits.fill_(float('nan'))
            return logits, state
        monkeypatch.setattr(loaded.core, 'forward', bad)
        with pytest.raises(RuntimeError, match='nonfinite'):
            generate_flash_next_tokens(loaded.core, loaded.tables[1], [2, 3], max_new_tokens=3, eos_token_ids=())


def tokenizer_file(path):
    tokenizers = pytest.importorskip('tokenizers')
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({f't{i}': i for i in range(32)}, unk_token='t0'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path/'tokenizer.json'))
    return path/'tokenizer.json'


@pytest.mark.parametrize('fp8', [False, True])
def test_cli_real_local_tokenizer_plan_and_completion(tmp_path, capsys, fp8):
    (fp8_fixture if fp8 else floating_fixture)(tmp_path)
    tokenizer_file(tmp_path)
    args = ['flash-next', 'generate-reference', str(tmp_path), '--weight-encoding',
        'fp8-weight-only' if fp8 else 'floating', '--dtype', 'float32', '--core-budget-mib', '1',
        '--max-sequence', '32', '--prefill-chunk-size', '2', '--expert-cache-mib-per-layer', '1']
    assert main(args+['--plan-only']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['loaded'] is False and report['plan_only'] is True
    assert main(args+['--prompt', 't2 t31 t5', '--max-new-tokens', '4', '--ignore-eos']) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report['result']['generated_ids']) == 4
    assert report['result']['text'] == ' '.join(f't{i}' for i in report['result']['generated_ids'])
    assert report['plan']['native_generation_supported'] is False
    assert report['plan']['whole_official_model_qualified'] is False
    assert report['tokenizer']['chat_template_applied'] is False
    assert report['loading_seconds'] > 0 and report['reader_preparation_seconds'] > 0


def test_execution_preflight_is_metadata_only_and_counts_cast_bytes(tmp_path, monkeypatch):
    cp, _, _ = floating_fixture(tmp_path, dtype=torch.bfloat16)
    import venturi.models.flash_next_loader as loader
    monkeypatch.setattr(loader, 'safe_open', lambda *a, **k: pytest.fail('plan read payload'))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *a: pytest.fail('plan initialized CUDA'))
    report = plan_floating_text(cp, **dict(OPTIONS, execution_device='cuda:0', execution_dtype=torch.float32))
    assert report['execution_core_parameter_bytes'] == 2*report['resident_parameter_bytes']
    assert report['execution_device'] == 'cuda:0'
    for override in (dict(execution_device='cuda'), dict(execution_device='cpu:0'), dict(execution_device='mps'),
                     dict(execution_dtype=torch.int8), dict(max_execution_core_bytes=1), dict(max_execution_core_bytes=True)):
        with pytest.raises(ValueError):
            load_floating_text(cp, **dict(OPTIONS, **override))


def test_static_execution_cast_overflow_is_an_error(tmp_path):
    cp, _, _ = floating_fixture(tmp_path, dtype=torch.bfloat16)
    weights = load_file(str(cp.files[0]))
    weights['lm_head.weight'].fill_(70000.)
    save_file(weights, cp.files[0])
    with pytest.raises(ValueError, match='overflowing static'):
        load_floating_text(cp, **dict(OPTIONS, execution_dtype=torch.float16))


def test_prefill_admission_before_io_and_empty_prompt(tmp_path, monkeypatch):
    cp, _, _ = floating_fixture(tmp_path)
    loaded = load_floating_text(cp, **OPTIONS)
    monkeypatch.setattr(loaded.tables[1], 'lookup', lambda *a: pytest.fail('admission read rows'))
    loaded.core.max_logits_elements = 32
    with pytest.raises(ValueError, match='smaller chunk'):
        generate_flash_next_tokens(loaded.core, loaded.tables[1], [2, 3], prefill_chunk_size=2)
    from venturi.generation import LocalTokenizer
    tokenizer = LocalTokenizer(tokenizer_file(tmp_path), vocab_size=32)
    with pytest.raises(ValueError, match='Empty tokenized'):
        generate_flash_next_text(loaded.core, loaded.tables[1], tokenizer, '')


def test_generation_stops_are_separate_from_ngram_segment_eos(tmp_path):
    from venturi.flash_next_generation import resolve_flash_next_eos
    cp, _, _ = floating_fixture(tmp_path)
    assert resolve_flash_next_eos(cp) == ((31,), 'text_config.eos_token_id')
    path = tmp_path/'generation_config.json'
    path.write_text(json.dumps(dict(eos_token_id=[30, 31, 30], do_sample=True)))
    assert resolve_flash_next_eos(cp) == ((30, 31), 'generation_config.json')
    assert resolve_flash_next_eos(cp, override=(29,)) == ((29,), 'explicit')
    assert cp.config['text_config']['eos_token_id'] == 31
    path.write_text('{invalid')
    assert resolve_flash_next_eos(cp, ignore=True) == ((), 'ignore-eos')
    assert resolve_flash_next_eos(cp, override=(29,)) == ((29,), 'explicit')


@pytest.mark.parametrize('value', [True, '31', [], [32], [True], [31]*129])
def test_invalid_eos_metadata_is_not_silently_applied(tmp_path, value):
    from venturi.flash_next_generation import resolve_flash_next_eos
    cp, _, _ = floating_fixture(tmp_path)
    (tmp_path/'generation_config.json').write_text(json.dumps(dict(eos_token_id=value)))
    with pytest.raises(ValueError):
        resolve_flash_next_eos(cp)
