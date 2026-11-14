"""Shard identity and schema guards; CPU transport doubles never initialize CUDA."""

from dataclasses import replace
import importlib
import os
from types import MappingProxyType

import pytest
import torch
from safetensors.torch import save_file

from venturi.container.checkpoint import CheckpointError, inspect_checkpoint
from venturi.container.loader import _staged_tensors, load_checkpoint
from venturi.container.safetensors import SafetensorsContainer
from venturi.container import staged_transfer


@pytest.fixture
def host_transfer(monkeypatch):
    instances = []

    class HostTransfer:
        def __init__(self, *args, **kwargs):
            self.closed = False
            instances.append(self)

        def upload(self, tensor, read_chunk):
            raw = tensor.reshape(-1).view(torch.uint8)
            if raw.numel():
                data = bytearray(raw.numel())
                assert read_chunk(0, len(data), data) == len(data)
                raw.copy_(torch.frombuffer(data, dtype=torch.uint8))

        def close(self):
            self.closed = True

    monkeypatch.setattr(staged_transfer, "StagedTransfer", HostTransfer)
    return instances


def read_on_host(checkpoint):
    return dict(_staged_tensors(checkpoint, torch.device("cpu"), None, None, None))


def replace_file(path, tensors):
    replacement = path.with_suffix(".replacement")
    save_file(tensors, str(replacement))
    os.replace(replacement, path)


@pytest.mark.parametrize("sharded", [False, True])
def test_hf_symlinks_keep_snapshot_names_and_load(
    make_checkpoint, weights, tmp_path, host_transfer, sharded
):
    directory = make_checkpoint(sharded=sharded)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for path in directory.glob("*.safetensors"):
        blob = blobs / path.stem
        path.rename(blob)
        path.symlink_to(blob)
        # The model-specific strict input policy is unchanged.
        with pytest.raises(ValueError, match="symlink"):
            SafetensorsContainer(path)
    checkpoint = inspect_checkpoint(directory)
    loaded = read_on_host(checkpoint)
    reference = load_checkpoint(directory, device="cpu", loader="safetensors")
    assert loaded.keys() == weights.keys()
    for name, expected in weights.items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(reference.weights[name], expected, rtol=0, atol=0)
        assert checkpoint.tensors[name].file.parent == directory
    assert all(t.closed for t in host_transfer)


@pytest.mark.parametrize(
    "mutation",
    ["dtype", "shape", "names", "payload", "same_bytes", "missing", "retarget"],
)
def test_changed_inspected_source_is_rejected_before_transfer(
    tmp_path, host_transfer, mutation
):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.tensor([1.0, 2.0])}, str(path))
    if mutation == "retarget":
        blob = tmp_path / "blob"
        path.rename(blob)
        path.symlink_to(blob)
    checkpoint = inspect_checkpoint(path)
    if mutation == "dtype":
        replace_file(
            path, {"x": torch.tensor([0x3F800000, 0x40000000], dtype=torch.int32)}
        )
    elif mutation == "shape":
        replace_file(path, {"x": torch.ones(1, 2)})
    elif mutation == "names":
        replace_file(path, {"y": torch.tensor([1.0, 2.0])})
    elif mutation == "payload":
        replace_file(path, {"x": torch.tensor([3.0, 4.0])})
    elif mutation == "same_bytes":
        data = path.read_bytes()
        other = tmp_path / "identical"
        other.write_bytes(data)
        os.replace(other, path)
    elif mutation == "missing":
        path.unlink()
    else:
        other = tmp_path / "other_blob"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        read_on_host(checkpoint)
    assert host_transfer == []


@pytest.mark.parametrize("mutation", ["dtype", "shape", "names"])
def test_actual_layout_is_compared_before_copy(tmp_path, host_transfer, mutation):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.tensor([1.0, 2.0])}, str(path))
    checkpoint = inspect_checkpoint(path)
    spec = checkpoint.tensors["x"]
    if mutation == "dtype":
        tensors = {"x": replace(spec, dtype="I32")}
    elif mutation == "shape":
        tensors = {"x": replace(spec, shape=(1, 2))}
    else:
        tensors = {"y": replace(spec, name="y")}
    checkpoint = replace(checkpoint, tensors=MappingProxyType(tensors))
    with pytest.raises(
        CheckpointError, match="disagrees with inspection|tensor set changed"
    ):
        read_on_host(checkpoint)
    assert len(host_transfer) == 1 and host_transfer[0].closed


def test_inspection_rejects_replacement_during_header_read(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.ones(2)}, str(path))
    module = importlib.import_module("venturi.container.checkpoint")
    original = module.safe_open

    class ReplacingReader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            return self.reader.__enter__()

        def __exit__(self, *args):
            result = self.reader.__exit__(*args)
            replace_file(path, {"x": torch.ones(2)})
            return result

    monkeypatch.setattr(module, "safe_open", ReplacingReader)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        inspect_checkpoint(path)


def test_replacement_between_header_and_open_is_rejected(
    tmp_path, monkeypatch, host_transfer
):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.ones(2)}, str(path))
    checkpoint = inspect_checkpoint(path)
    original = os.open
    descriptors = []

    def replacing_open(filename, flags, *args, **kwargs):
        if filename == path:
            replace_file(path, {"x": torch.zeros(2)})
        fd = original(filename, flags, *args, **kwargs)
        descriptors.append(fd)
        return fd

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        read_on_host(checkpoint)
    assert host_transfer[0].closed
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_in_place_change_during_read_is_rejected(tmp_path, monkeypatch, host_transfer):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.ones(2)}, str(path))
    checkpoint = inspect_checkpoint(path)
    original = staged_transfer.pread_fill

    def changing_read(*args):
        result = original(*args)
        status = path.stat()
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(staged_transfer, "pread_fill", changing_read)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        read_on_host(checkpoint)
    assert host_transfer[0].closed


def test_generator_cancellation_closes_transfer(make_checkpoint, host_transfer):
    checkpoint = inspect_checkpoint(make_checkpoint())
    iterator = _staged_tensors(checkpoint, torch.device("cpu"), None, None, None)
    next(iterator)
    iterator.close()
    assert host_transfer[0].closed


def test_earlier_shard_change_is_detected_at_load_end(
    make_checkpoint, monkeypatch, host_transfer
):
    checkpoint = inspect_checkpoint(make_checkpoint(sharded=True))
    first, second = checkpoint.files
    original = staged_transfer.pread_fill
    changed = False

    def read_then_change_earlier_shard(fd, *args):
        nonlocal changed
        result = original(fd, *args)
        if not changed and os.fstat(fd).st_ino == second.stat().st_ino:
            status = first.stat()
            os.utime(first, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))
            changed = True
        return result

    monkeypatch.setattr(staged_transfer, "pread_fill", read_then_change_earlier_shard)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        read_on_host(checkpoint)
    assert changed and host_transfer[0].closed


@pytest.mark.cuda
def test_real_cuda_rejects_replaced_inspected_file(tmp_path, monkeypatch):
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 on a CUDA host")
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.tensor([1.0, 2.0])}, str(path))
    module = importlib.import_module("venturi.container.loader")

    def inspect_then_replace(filename):
        checkpoint = inspect_checkpoint(filename)
        replace_file(
            path, {"x": torch.tensor([0x3F800000, 0x40000000], dtype=torch.int32)}
        )
        return checkpoint

    monkeypatch.setattr(module, "inspect_checkpoint", inspect_then_replace)
    with pytest.raises(CheckpointError, match="changed since inspection"):
        load_checkpoint(path, device="cuda:0")


@pytest.mark.cuda
@pytest.mark.parametrize("sharded", [False, True])
def test_real_cuda_hf_cache_round_trip(make_checkpoint, weights, tmp_path, sharded):
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 on a CUDA host")
    directory = make_checkpoint(sharded=sharded)
    for index, path in enumerate(directory.glob("*.safetensors")):
        blob = tmp_path / f"blob-{index}"
        path.rename(blob)
        path.symlink_to(blob)
    loaded = load_checkpoint(directory, device="cuda:0")
    for name, expected in weights.items():
        torch.testing.assert_close(loaded.weights[name].cpu(), expected, rtol=0, atol=0)
