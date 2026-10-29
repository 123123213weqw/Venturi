"""Offline intake/PLE contracts; no full-model or CUDA speed claims."""

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
import random
import struct
import threading

import pytest
import torch
from safetensors.torch import save_file

from venturi.checkpoint import inspect_checkpoint
from venturi.cli import main
from venturi.flash_next import FlashNextConfig, FlashNextManifest
from venturi.models.flash_next_ngram import (
    FP8ShardedNGramTable, NGramRequest, PreparedNGramTable, ShardedNGramTable,
)


def raw_config():
    return dict(model_type="qwen4_exp", text_config=dict(
        model_type="qwen4_exp_text", hidden_size=8, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], vocab_size=32,
        eos_token_id=31, hc_count=2, hc_lowrank=3, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=101, ple_embed_dim=16,
        make_ngram_vocab_size_divisible_by=8, split_ngram_parts=8,
        ple_layer_ids=[2], num_key_value_heads=1, head_dim=4,
        max_position_embeddings=4096,
    ))


@pytest.fixture
def fixture(tmp_path):
    raw = raw_config()
    c = FlashNextConfig.from_dict(raw)
    weights = {}

    def add(name, shape):
        weights[name] = torch.zeros(shape, dtype=torch.bfloat16)

    add("lm_head.weight", (c.vocab_size, c.hidden_size))
    add("model.language_model.embed_tokens.weight", (c.vocab_size, c.hidden_size))
    for layer in range(c.layers):
        p = f"model.language_model.layers.{layer}"
        add(p + ".mlp.experts.gate_up_proj", (4, 8, 8))
        add(p + ".mlp.experts.down_proj", (4, 8, 4))
        for part in ("attn_hyper_connection", "mlp_hyper_connection"):
            add(f"{p}.{part}.hc_norm.weight", (16,))
            add(f"{p}.{part}.input_mix_weight_down.weight", (3, 16))
            add(f"{p}.{part}.input_mix_weight_up.weight", (16, 3))
            add(f"{p}.{part}.block_inject_weight.weight", (2, 16))
    p = "model.language_model.layers.1.ple.ple_embedding"
    weights[p + ".layer_multipliers"] = torch.tensor([7, 13, 19], dtype=torch.int64)
    _, _, padded = c.ngram_geometry()
    table = torch.arange(padded * 4).reshape(padded, 4).to(torch.bfloat16)
    rows_per_part = padded // 8
    for part in range(8):
        weights[f"{p}.ngram_embedding.shard_{part}.weight"] = table[
            part * rows_per_part:(part + 1) * rows_per_part
        ].clone()
    add("model.visual.fixture.weight", (4, 8))
    add("mtp.fixture.weight", (8, 8))
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    mapping = {name: path.name for name in weights}
    index = dict(weight_map=mapping, metadata=dict(total_size=sum(
        t.numel() * t.element_size() for t in weights.values())))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    record = dict(repo="Qwen/fixture", revision="a" * 40, filename=path.name,
                  file_bytes=len(data), header_bytes=length,
                  header=json.loads(data[8:8 + length]))
    headers = tmp_path / "headers"
    headers.mkdir()
    (headers / (path.name + ".json")).write_text(json.dumps(record))
    return tmp_path, headers, c, table, record


def naive_rows(c, tokens, multipliers=(7, 13, 19)):
    """Absolute-position oracle: never uses incremental request state."""
    sizes, offsets, _ = c.ngram_geometry()
    result = []
    for pos in range(len(tokens)):
        boundary = max([i for i in range(pos) if tokens[i] == c.eos_token_id], default=-1)
        out = []
        for order in range(2, c.ngram_size + 1):
            value = 0
            for shift in range(order):
                source = pos - shift
                token = tokens[source] if source > boundary else c.eos_token_id
                value ^= token * multipliers[shift]
            for head in range((order - 2) * c.heads_per_ngram, (order - 1) * c.heads_per_ngram):
                out.append(value % sizes[head] + offsets[head])
        result.append(tuple(out))
    return tuple(result)


def test_header_and_real_checkpoint_intake_agree(fixture):
    path, headers, c, _, _ = fixture
    local = FlashNextManifest.from_checkpoint(inspect_checkpoint(path))
    remote = FlashNextManifest.from_headers(path, headers)
    assert local.config == remote.config == c
    assert sorted(local.tensors, key=lambda t: t.name) == list(remote.tensors)
    assert "unverified" in remote.source_scope
    assert remote.revision == "a" * 40
    report = remote.plan()
    assert report["native_generation_supported"] is False
    assert "MoE routing/expert execution" in report["missing_native_components"]


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 0), ("hc_count", True), ("eos_token_id", None),
    ("ple_layer_ids", [0]), ("ple_layer_ids", [2, 2]),
    ("ngram_size", 1), ("ple_embed_dim", 17),
    ("num_experts_per_tok", 5), ("layer_types", ["full_attention"]),
])
def test_bad_configs(field, value):
    raw = raw_config()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        FlashNextConfig.from_dict(raw)


def test_dense_model_not_misidentified():
    raw = raw_config()
    raw["model_type"] = "qwen3_5"
    with pytest.raises(ValueError, match="not dense"):
        FlashNextConfig.from_dict(raw)


@pytest.mark.parametrize("mutation", ["offset", "shape", "overlap", "total", "filename", "missing"])
def test_bad_headers_rejected(fixture, mutation):
    path, headers, _, _, record = fixture
    record = copy.deepcopy(record)
    names = [n for n in record["header"] if n != "__metadata__"]
    if mutation == "offset":
        record["header"][names[0]]["data_offsets"][1] += 1
    elif mutation == "shape":
        record["header"][names[0]]["shape"] = [1]
    elif mutation == "overlap":
        record["header"][names[1]]["data_offsets"] = record["header"][names[0]]["data_offsets"]
    elif mutation == "total":
        index = json.loads((path / "model.safetensors.index.json").read_text())
        index["metadata"]["total_size"] += 1
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
    elif mutation == "filename":
        record["filename"] = "wrong.safetensors"
    else:
        record["header"].pop(names[0])
    (headers / "model.safetensors.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        FlashNextManifest.from_headers(path, headers)


def test_schema_does_not_accept_missing_experts_or_misplaced_ple(fixture):
    path, headers, _, _, _ = fixture
    m = FlashNextManifest.from_headers(path, headers)
    with pytest.raises(ValueError, match="experts"):
        replace(m, tensors=tuple(t for t in m.tensors if not t.is_expert)).validate_schema()
    bad = replace(m.config, ple_layers=(1,))
    with pytest.raises(ValueError, match="layer_multipliers"):
        replace(m, config=bad).validate_schema()


def test_plans_count_private_kv_and_cpu_tables(fixture):
    path, headers, c, table, _ = fixture
    m = FlashNextManifest.from_headers(path, headers)
    a = m.plan(gpu_budgets=(2**30,), ram_budget=2**30, context=1024)
    b = m.plan(gpu_budgets=(2**30,), ram_budget=2**30, context=1024, batch_size=8)
    assert a["cpu_ngram_table_bytes"] == table.numel() * table.element_size()
    assert a["dense_fp16_kv_lower_bound_bytes"] == 1024 * c.kv_heads * c.head_dim * 4
    assert b["dense_fp16_kv_lower_bound_bytes"] == 8 * a["dense_fp16_kv_lower_bound_bytes"]
    assert a["all_weights_gpu_lower_bound_fits"]
    assert not m.plan(gpu_budgets=(0,), ram_budget=0)["capacity_lower_bound_fits"]
    assert not m.plan()["capacity_lower_bound_fits"]
    for profile in ("expert-q8-g32", "expert-q4-g64"):
        q = m.plan(profile=profile)
        # This tiny K=8 fixture grows with K128 padding: quantized != always smaller.
        assert q["proposed_component_bytes"]["text_core"] > a["proposed_component_bytes"]["text_core"]
        assert q["cpu_ngram_table_bytes"] == a["cpu_ngram_table_bytes"]
    included = m.plan(include_mtp=True, include_vision=True)
    assert included["proposed_component_bytes"] == included["source_component_bytes"]


@pytest.mark.parametrize("kwargs", [
    {"context": 4097}, {"batch_size": True}, {"gpu_budgets": (-1,)},
    {"ram_budget": -1}, {"profile": "fp8"}, {"include_mtp": 1},
])
def test_bad_plan(fixture, kwargs):
    path, headers, _, _, _ = fixture
    with pytest.raises(ValueError):
        FlashNextManifest.from_headers(path, headers).plan(**kwargs)


@pytest.mark.parametrize("tokens", [[], [0], [31], [31, 31, 1], [1, 2, 31, 3, 4], [31, 1, 31, 31, 2, 3, 31]])
def test_ngram_eos_semantics(tokens):
    c = FlashNextConfig.from_dict(raw_config())
    request = NGramRequest(c, (7, 13, 19))
    p = request.preview(tokens)
    assert p.rows == naive_rows(c, tokens)
    assert request.history == ()
    request.commit(p)
    last_eos = max([i for i, x in enumerate(tokens) if x == 31], default=-1)
    assert request.history == tuple(tokens[last_eos + 1:][-2:])


def test_random_chunking_and_max_i64_multipliers():
    c = FlashNextConfig.from_dict(raw_config())
    rng = random.Random(743)
    ids = [rng.randrange(c.vocab_size) for _ in range(93)]
    bound = (2**63 - 1) // (c.vocab_size - 1)
    multipliers = (bound, bound - 1, bound - 2)
    expected = naive_rows(c, ids, multipliers)
    for chunk in (1, 2, 3, 8, 32, 93):
        request, actual = NGramRequest(c, multipliers), []
        for start in range(0, len(ids), chunk):
            p = request.preview(ids[start:start + chunk])
            actual.extend(p.rows)
            request.commit(p)
        assert tuple(actual) == expected


def test_rejected_suffix_does_not_leak_and_private_slots():
    c = FlashNextConfig.from_dict(raw_config())
    a, b = NGramRequest(c, (7, 13, 19)), NGramRequest(c, (7, 13, 19))
    p = a.preview([1, 2, 31, 5])
    assert a.preview([1, 2, 31, 5]).rows == p.rows
    a.commit(p, 2)
    assert a.history == (1, 2)
    assert a.preview([7]).rows == naive_rows(c, [1, 2, 7])[-1:]
    assert b.history == ()
    with pytest.raises(ValueError, match="Stale/foreign"):
        b.commit(p)
    with pytest.raises(ValueError, match="Stale/foreign"):
        a.commit(p)
    q = a.preview([8])
    a.reset()
    with pytest.raises(ValueError):
        a.commit(q)


@pytest.mark.parametrize("multipliers", [(), (0, 1, 2), (True, 1, 2), (2**63, 1, 2)])
def test_invalid_multipliers(multipliers):
    with pytest.raises(ValueError):
        NGramRequest(FlashNextConfig.from_dict(raw_config()), multipliers)


def test_lookup_mmap_and_async_prefetch_exact(fixture):
    path, _, c, expected_table, _ = fixture
    store = ShardedNGramTable(inspect_checkpoint(path), c, max_tokens=8)
    request = NGramRequest(c, store.multipliers)
    p = request.preview([1, 2, 31, 4, 4, 4])
    expected = expected_table[torch.tensor(p.rows)].flatten(-2)
    assert torch.equal(store.lookup(p), expected)
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = store.prefetch(p, executor)
        b = store.prefetch(request.preview([0, 31]), executor)
        assert torch.equal(a.result(), expected)
        assert b.result().shape == (2, 16)
    assert request.history == ()  # lookup/prefetch is not a token commit
    assert store.lookup(request.preview([])).shape == (0, 16)
    with pytest.raises(ValueError, match="bounded"):
        store.lookup(request.preview([0] * 9))
    with pytest.raises(ValueError, match="different table"):
        store.lookup(NGramRequest(c, (3, 5, 7)).preview([0]))
    with pytest.raises(ValueError, match="outside"):
        store.lookup(replace(p, rows=((-1,) * 4,) * 6))


def test_cli_metadata_only_never_initializes_cuda(fixture, monkeypatch, capsys):
    path, headers, _, _, _ = fixture
    def forbidden(*args, **kwargs):
        raise AssertionError("Intake must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    assert main(["flash-next", "inspect", str(path), "--headers-dir", str(headers),
                 "--gpu-budget-gib", "30", "30", "--ram-budget-gib", "200"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["native_generation_supported"] is False
    assert report["gpu_budgets_bytes"] == [30 * 2**30] * 2


def test_coalesced_requests_keep_private_state_and_own_results(fixture):
    path, _, c, _, _ = fixture
    store = ShardedNGramTable(inspect_checkpoint(path), c, max_tokens=8)
    requests = [NGramRequest(c, store.multipliers) for _ in range(3)]
    proposals = [r.preview(ids) for r, ids in zip(requests, ([0, 1], [], [0, 1, 31]))]
    expected = [store.lookup(p) for p in proposals]
    results = store.lookup_many(proposals)
    assert all(torch.equal(a, b) for a, b in zip(results, expected))
    retained = results[0].clone()
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert all(torch.equal(a, b) for a, b in zip(
            store.prefetch_many(proposals, executor).result(), expected))
    assert torch.equal(results[0], retained)
    assert all(r.history == () for r in requests)
    requests[0].commit(proposals[0], 1)
    requests[2].commit(proposals[2], 2)
    assert [r.history for r in requests] == [(0,), (), (0, 1)]
    assert store.lookup_many([]) == ()
    with pytest.raises(ValueError, match="bounded"):
        store.lookup_many([proposals[2]] * 3)


def make_fp8_fixture(fixture, scale=0.125):
    path, _, c, _, _ = fixture
    from safetensors.torch import load_file
    weights = load_file(path / "model.safetensors")
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    codes = []
    for name in sorted(weights):
        if name.startswith(prefix + ".shard_"):
            shape = weights[name].shape
            raw = torch.arange(weights[name].numel(), dtype=torch.int32).remainder(256)
            raw[raw == 127] = 0
            raw[raw == 255] = 128  # All finite E4M3FN codes, including signed zero.
            weights[name] = raw.to(torch.uint8).view(torch.float8_e4m3fn).reshape(shape)
    for part in range(c.split_ngram_parts):
        codes.append(weights[f"{prefix}.shard_{part}.weight"])
    weights[prefix + ".weight_scale"] = torch.tensor([scale], dtype=torch.bfloat16)
    save_file(weights, path / "model.safetensors")
    index = dict(weight_map={k: "model.safetensors" for k in weights},
                 metadata=dict(total_size=sum(t.numel() * t.element_size() for t in weights.values())))
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return inspect_checkpoint(path), torch.cat(codes).to(torch.bfloat16) * weights[prefix + ".weight_scale"]


def test_explicit_fp8_ple_reader_dequantizes_only_requested_rows(fixture):
    _, _, c, _, _ = fixture
    checkpoint, table = make_fp8_fixture(fixture)
    store = FP8ShardedNGramTable(checkpoint, c, max_tokens=8)
    p = NGramRequest(c, store.multipliers).preview([0, 1, 31, 2, 3])
    expected = table[torch.tensor(p.rows)].flatten(-2)
    # Compare raw BF16 encodings, including negative zero.
    assert torch.equal(store.lookup(p).view(torch.int16), expected.view(torch.int16))
    with pytest.raises(ValueError, match="quantized"):
        ShardedNGramTable(checkpoint, c)
    results = store.lookup_many([p])
    assert torch.equal(results[0].view(torch.int16), expected.view(torch.int16))


def test_fp8_scale_failure_is_not_silent_float_fallback(fixture):
    _, _, c, _, _ = fixture
    checkpoint, _ = make_fp8_fixture(fixture, scale=float("nan"))
    with pytest.raises(ValueError, match="positive and finite"):
        FP8ShardedNGramTable(checkpoint, c)


@pytest.mark.parametrize("fp8", [False, True])
def test_prepared_bits_batch_history_and_owned_lifetime(fixture, fp8):
    path, _, c, _, _ = fixture
    checkpoint = make_fp8_fixture(fixture)[0] if fp8 else inspect_checkpoint(path)
    cls = FP8ShardedNGramTable if fp8 else ShardedNGramTable
    original = cls(checkpoint, c, max_tokens=8)
    requests = [NGramRequest(c, original.multipliers) for _ in range(3)]
    proposals = [r.preview(ids) for r, ids in zip(requests, ([0, 1, 31], [], [4, 2]))]
    expected = original.lookup_many(proposals)
    prepared = PreparedNGramTable(original, max_open_files=1)
    with pytest.raises(RuntimeError, match="prepared"):
        prepared.lookup(proposals[0])
    with prepared as store:
        assert store.prepare() is store
        actual = store.lookup_many(proposals)
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                   for a, b in zip(actual, expected))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [store.prefetch(p, executor) for p in proposals]
            assert all(torch.equal(f.result().view(torch.int16), b.view(torch.int16))
                       for f, b in zip(futures, expected))
        assert store.lookup_many([]) == ()
        assert all(r.history == () for r in requests)
        with pytest.raises(ValueError, match="bounded"):
            store.lookup_many([proposals[0]] * 3)
        with pytest.raises(ValueError, match="different table"):
            store.lookup(NGramRequest(c, (3, 5, 7)).preview([0]))
        with pytest.raises(ValueError, match="outside"):
            store.lookup(replace(proposals[0], rows=((-1,) * 4,) * 3))
    prepared.close()  # idempotent, terminal
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
               for a, b in zip(actual, expected))
    with pytest.raises(RuntimeError, match="closed"):
        prepared.prepare()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.lookup(proposals[0])


def test_prepared_opens_each_file_once_without_full_table_reads(fixture, monkeypatch):
    import venturi.models.flash_next_ngram as module
    path, _, c, _, _ = fixture
    original = ShardedNGramTable(inspect_checkpoint(path), c)
    real_open, events = module.safe_open, []

    class Tracked:
        def __init__(self, *args, **kwargs):
            self.context = real_open(*args, **kwargs)
            events.append("open")

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def get_slice(self, name):
            events.append("slice")
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            raise AssertionError("Prepared path must not materialize the table")

        def __exit__(self, *args):
            events.append("close")
            return self.context.__exit__(*args)

    monkeypatch.setattr(module, "safe_open", Tracked)
    with PreparedNGramTable(original, max_open_files=1) as store:
        p = NGramRequest(c, original.multipliers).preview([1, 2, 31])
        store.lookup(p)
        store.lookup(p)
        assert events == ["open"] + ["slice"] * 8
    assert events[-1] == "close"


def test_prepared_budget_and_metadata_failure_cleanup(fixture, monkeypatch):
    path, _, c, _, _ = fixture
    original = ShardedNGramTable(inspect_checkpoint(path), c)
    with pytest.raises(ValueError, match="budget"):
        PreparedNGramTable(original, max_open_files=True)
    second = path / "second.safetensors"
    second.write_bytes((path / "model.safetensors").read_bytes())
    original.parts[1] = replace(original.parts[1], file=second)
    store = PreparedNGramTable(original, max_open_files=1)
    with pytest.raises(ValueError, match="exceed"):
        store.prepare()
    assert store._stack is None and store._views == ()
    original.parts[1] = replace(original.parts[1], shape=(1, 1))
    broken = PreparedNGramTable(original, max_open_files=2)
    with pytest.raises(ValueError, match="metadata changed"):
        broken.prepare()
    assert broken._stack is None and broken._views == ()
    broken.close()


def test_prepared_close_waits_for_active_read_and_rejects_later_work(fixture):
    path, _, c, _, _ = fixture
    original = ShardedNGramTable(inspect_checkpoint(path), c)
    store = PreparedNGramTable(original).prepare()
    entered, release = threading.Event(), threading.Event()

    class SlowSlice:
        def __init__(self, view):
            self.view = view

        def __getitem__(self, index):
            entered.set()
            assert release.wait(5)
            return self.view[index]

    store._views = tuple(SlowSlice(v) for v in store._views)
    p = NGramRequest(c, original.multipliers).preview([0])
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            read = store.prefetch(p, executor)
            assert entered.wait(5)
            close = executor.submit(store.close)
            assert not close.done()
            release.set()
            actual = read.result(timeout=5)
            close.result(timeout=5)
            assert torch.equal(actual, original.lookup(p))
            with pytest.raises(RuntimeError, match="closed"):
                store.prefetch(p, executor).result(timeout=5)
    finally:
        release.set()
        store.close()
