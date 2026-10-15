import importlib
import json
import os
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace
import sys

import pytest
import torch
from safetensors.torch import save_file

from venturi import CheckpointError, inspect_checkpoint, load_checkpoint
from venturi.cli import main


@pytest.mark.parametrize("sharded", [False, True])
def test_round_trip(make_checkpoint, weights, raw_config, sharded):
    directory = make_checkpoint(sharded=sharded)
    checkpoint = inspect_checkpoint(directory)
    loaded = load_checkpoint(directory, device="cpu", loader="safetensors")
    assert dict(loaded.config) == raw_config
    assert len(checkpoint.files) == (2 if sharded else 1)
    assert loaded.weights.keys() == weights.keys()
    for name, expected in weights.items():
        actual = loaded.weights[name]
        assert actual.device.type == "cpu"
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.stats.tensor_bytes == sum(
        x.numel() * x.element_size() for x in weights.values()
    )
    assert loaded.stats.tensor_count == len(weights)
    assert loaded.stats.total_seconds >= loaded.stats.load_seconds >= 0
    assert loaded.stats.cuda_allocated_bytes_delta is None
    assert all(
        m["custom"] == "opaque metadata" for m in checkpoint.file_metadata.values()
    )
    if sharded:
        assert checkpoint.index_metadata["custom"] == "preserved"
    with pytest.raises(TypeError):
        loaded.weights["extra"] = torch.zeros(1)
    assert json.loads(json.dumps(loaded.summary()))["config"] == raw_config


@pytest.mark.parametrize(
    "config", [False, {}, {"model_type": "unknown", "architectures": ["Anything"]}]
)
def test_no_architecture_or_config_required(make_checkpoint, config):
    loaded = load_checkpoint(
        make_checkpoint(config=config), device="cpu", loader="safetensors"
    )
    assert dict(loaded.config) == ({} if config is False else config)


def test_explicit_file(make_checkpoint, weights):
    directory = make_checkpoint(sharded=True)
    path = directory / "model-00001-of-00002.safetensors"
    loaded = load_checkpoint(path, device="cpu", loader="safetensors")
    assert loaded.checkpoint.files == (path,)
    assert set(loaded.weights) == set(list(weights)[::2])


def test_original_layout_is_preserved(make_checkpoint, weights):
    loaded = load_checkpoint(make_checkpoint(), device="cpu", loader="safetensors")
    assert loaded.weights["arbitrary.weight"].shape == (11, 7)
    assert loaded.weights["quant.packed_int4"].dtype == torch.int32
    assert loaded.weights["quant.packed_int4"].shape == (2, 2)
    assert loaded.config["quantization_config"]["bits"] == 4


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_scalar_and_empty_tensor_storage(make_checkpoint, dtype):
    tensors = {
        "scalar": torch.tensor(1, dtype=dtype),
        "empty": torch.empty((0, 3), dtype=dtype),
    }
    loaded = load_checkpoint(
        make_checkpoint(tensors=tensors), device="cpu", loader="safetensors"
    )
    assert loaded.weights["scalar"].shape == ()
    assert loaded.weights["empty"].shape == (0, 3)
    assert loaded.stats.tensor_bytes == tensors["scalar"].element_size()


def test_cpu_index_is_normalized(make_checkpoint):
    loaded = load_checkpoint(make_checkpoint(), device="cpu:0", loader="safetensors")
    assert loaded.stats.device == "cpu"


def test_inspection_does_not_materialize_tensors(make_checkpoint, monkeypatch):
    directory = make_checkpoint()
    module = importlib.import_module("venturi.checkpoint")
    original = module.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def metadata(self):
            return self.reader.metadata()

        def get_slice(self, key):
            return self.reader.get_slice(key)

        def get_tensor(self, key):
            pytest.fail("Inspection must not materialize any tensors")

    monkeypatch.setattr(module, "safe_open", HeaderOnly)
    assert inspect_checkpoint(directory).tensor_bytes > 0


@pytest.mark.parametrize(
    "case,match",
    [
        ("absent", "Missing checkpoint shard"),
        ("wrong_shard", "Shard index disagrees"),
        ("missing_entry", "Shard index disagrees"),
        ("phantom", "references missing tensors"),
        ("traversal", "Invalid shard index entry"),
        ("absolute", "Invalid shard index entry"),
        ("nonstring", "Invalid shard index entry"),
        ("empty", "nonempty weight_map"),
        ("bad_metadata", "metadata must be a JSON object"),
    ],
)
def test_invalid_shard_index(make_checkpoint, case, match):
    directory = make_checkpoint(sharded=True)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mapping = index["weight_map"]
    name = next(iter(mapping))
    if case == "absent":
        (directory / mapping[name]).unlink()
    elif case == "wrong_shard":
        mapping[name] = "model-00002-of-00002.safetensors"
    elif case == "missing_entry":
        del mapping[name]
    elif case == "phantom":
        mapping["phantom"] = mapping[name]
    elif case == "traversal":
        mapping[name] = "../outside.safetensors"
    elif case == "absolute":
        mapping[name] = "/outside.safetensors"
    elif case == "nonstring":
        mapping[name] = 42
    elif case == "bad_metadata":
        index["metadata"] = []
    else:
        index["weight_map"] = {}
    index_path.write_text(json.dumps(index))
    with pytest.raises(CheckpointError, match=match):
        inspect_checkpoint(directory)


def test_duplicate_weights_across_unindexed_shards(make_checkpoint, weights):
    directory = make_checkpoint(sharded=True)
    (directory / "model.safetensors.index.json").unlink()
    save_file(weights, str(directory / "extra.safetensors"))
    with pytest.raises(CheckpointError, match="Duplicate tensor"):
        inspect_checkpoint(directory)


def test_index_ignores_unrelated_shards(make_checkpoint, weights):
    directory = make_checkpoint(sharded=True)
    save_file(weights, str(directory / "unrelated.safetensors"))
    assert len(inspect_checkpoint(directory).files) == 2


def test_alternative_index_and_ambiguity(make_checkpoint):
    directory = make_checkpoint(sharded=True)
    canonical = directory / "model.safetensors.index.json"
    first = directory / "custom.safetensors.index.json"
    canonical.rename(first)
    assert len(inspect_checkpoint(directory).files) == 2
    (directory / "another.safetensors.index.json").write_bytes(first.read_bytes())
    with pytest.raises(CheckpointError, match="Multiple safetensors indexes"):
        inspect_checkpoint(directory)


def test_hf_cache_symlink(make_checkpoint, tmp_path):
    directory = make_checkpoint()
    original = directory / "model.safetensors"
    blob = tmp_path / "blob"
    original.rename(blob)
    original.symlink_to(blob)
    assert inspect_checkpoint(directory).tensor_bytes > 0
    explicit = inspect_checkpoint(original)
    assert explicit.directory == directory
    assert explicit.files == (original,)
    assert explicit.config["model_type"] == "custom-model"


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_path", "path does not exist"),
        ("no_weights", "No safetensors"),
        ("config_list", "JSON object"),
        ("duplicate_json", "Duplicate JSON key"),
        ("corrupt", "Cannot inspect"),
        ("empty_shard", "contains no tensors"),
        ("non_safetensors", "directory or a .safetensors file"),
    ],
)
def test_malformed_checkpoint(make_checkpoint, tmp_path, case, match):
    directory = make_checkpoint()
    if case == "missing_path":
        directory = tmp_path / "missing"
    elif case == "no_weights":
        (directory / "model.safetensors").unlink()
    elif case == "config_list":
        (directory / "config.json").write_text("[]")
    elif case == "duplicate_json":
        (directory / "config.json").write_text('{"a":1,"a":2}')
    elif case == "corrupt":
        (directory / "model.safetensors").write_bytes(b"broken")
    elif case == "empty_shard":
        save_file({}, str(directory / "model.safetensors"))
    else:
        directory = directory / "config.json"
    with pytest.raises(CheckpointError, match=match):
        inspect_checkpoint(directory)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"loader": "unknown"}, "Unknown loader"),
        ({"device": "cpu"}, "Siphon requires CUDA"),
        ({"loader": "safetensors", "device": "mps"}, "Only CUDA"),
        ({"buffer_size": 0}, "positive integer"),
        ({"loader": "safetensors", "buffer_size": 1024}, "Siphon options"),
    ],
)
def test_explicit_backend_errors(make_checkpoint, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_checkpoint(make_checkpoint(), **kwargs)


def test_cuda_unavailable_does_not_fallback(make_checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no automatic CPU fallback"):
        load_checkpoint(make_checkpoint())


def test_missing_siphon_has_install_message(make_checkpoint, monkeypatch):
    from venturi.loader import _siphon_tensors

    monkeypatch.setitem(sys.modules, "siphon", None)
    checkpoint = inspect_checkpoint(make_checkpoint())
    with pytest.raises(RuntimeError, match="Siphon native loader"):
        dict(_siphon_tensors(checkpoint, torch.device("cuda:0"), None, None))


def test_unavailable_dtype_fails_before_cuda(make_checkpoint, monkeypatch):
    module = importlib.import_module("venturi.loader")

    def unsupported(code):
        raise ValueError(f"Installed PyTorch does not support {code}")

    def must_not_initialize():
        pytest.fail("Unsupported storage dtypes must fail before CUDA setup")

    monkeypatch.setattr(module, "torch_dtype", unsupported)
    monkeypatch.setattr(torch.cuda, "is_available", must_not_initialize)
    with pytest.raises(ValueError, match="does not support"):
        load_checkpoint(make_checkpoint())


@pytest.mark.parametrize(
    "requested", [None, "MMAP", "mmap", "BUFFERED", "URING,MMAP", "invalid"]
)
def test_siphon_adapter_contract(make_checkpoint, weights, monkeypatch, requested):
    from venturi.loader import _siphon_tensors

    calls = []

    class FakeReader:
        def __init__(self, filenames, **kwargs):
            calls.append((filenames, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def tensors(self):
            yield from weights.items()

    backend = Enum("Backend", ["MMAP", "URING"])
    policy = Enum("BackendPolicy", ["BUFFERED"])
    monkeypatch.setitem(
        sys.modules,
        "siphon",
        SimpleNamespace(safe_open=FakeReader, Backend=backend, BackendPolicy=policy),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    checkpoint = inspect_checkpoint(make_checkpoint(sharded=True))
    if requested == "invalid":
        with pytest.raises(ValueError, match="Unknown Siphon backend"):
            dict(_siphon_tensors(checkpoint, torch.device("cuda:1"), requested, 8192))
        assert not calls
        return
    result = dict(_siphon_tensors(checkpoint, torch.device("cuda:1"), requested, 8192))
    assert result.keys() == weights.keys()
    files, kwargs = calls[0]
    assert len(files) == 2
    expected = {
        "framework": "pt",
        "device": "cuda:1",
        "copy": True,
        "buffer_size": 8192,
    }
    if requested is not None:
        expected["backend"] = {
            "MMAP": backend.MMAP,
            "mmap": backend.MMAP,
            "BUFFERED": policy.BUFFERED,
            "URING,MMAP": [backend.URING, backend.MMAP],
        }[requested]
    assert kwargs == expected
    assert calls[-1] == "closed"


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "dtype", "shape", "unexpected", "device"]
)
def test_runtime_validation_closes_reader(
    make_checkpoint, weights, monkeypatch, mutation
):
    module = importlib.import_module("venturi.loader")
    items = list(weights.items())
    if mutation == "missing":
        items.pop()
    elif mutation == "duplicate":
        items.append(items[0])
    elif mutation == "dtype":
        items[0] = (items[0][0], items[0][1].float())
    elif mutation == "shape":
        items[0] = (items[0][0], items[0][1].reshape(-1))
    elif mutation == "device":
        items[0] = (items[0][0], torch.empty_like(items[0][1], device="meta"))
    else:
        items.append(("extra", torch.zeros(1)))
    closed = []

    def reader(*args):
        try:
            yield from items
        finally:
            closed.append(True)

    monkeypatch.setattr(module, "_reference_tensors", reader)
    with pytest.raises(CheckpointError):
        load_checkpoint(make_checkpoint(), device="cpu", loader="safetensors")
    assert closed == [True]


def test_cli(make_checkpoint, capsys):
    directory = str(make_checkpoint())
    assert main(["inspect", directory, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tensor_count"] == 10
    assert (
        main(
            ["load", directory, "--device", "cpu", "--loader", "safetensors", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["load"]["loader"] == "safetensors"
    assert main(["inspect", directory]) == 0
    assert "10 tensors" in capsys.readouterr().out
    assert main(["inspect", directory + "/missing"]) == 1
    assert "venturi:" in capsys.readouterr().err


@pytest.mark.cuda
@pytest.mark.parametrize("io_backend", [None, "MMAP", "BUFFERED"])
def test_real_siphon_cuda_round_trip(make_checkpoint, weights, io_backend):
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 on a CUDA host with Siphon installed")
    assert torch.cuda.is_available(), (
        "CUDA integration was requested but CUDA is unavailable"
    )
    importlib.import_module("siphon")
    directory = make_checkpoint(sharded=True)
    loaded = load_checkpoint(
        directory, device="cuda:0", siphon_backend=io_backend, buffer_size=8 * 2**20
    )
    assert loaded.stats.cuda_allocated_bytes_delta > 0
    for name, expected in weights.items():
        torch.testing.assert_close(loaded.weights[name].cpu(), expected, rtol=0, atol=0)
    # Returned tensors must remain valid after Siphon closes and allocations recur.
    scratch = torch.full((2**20,), 42, dtype=torch.float16, device="cuda:0")
    for name, expected in weights.items():
        torch.testing.assert_close(loaded.weights[name].cpu(), expected, rtol=0, atol=0)
    assert scratch[0].item() == 42
