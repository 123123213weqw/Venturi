"""Pinned transport correctness, including allocation reuse and failed reads."""

import ctypes
import os
import threading
import time

import pytest
import torch

from venturi.container import staged_transfer as transport


@pytest.mark.parametrize("short", [False, True])
def test_pread_fallback_accepts_tensor_memoryview(tmp_path, monkeypatch, short):
    path = tmp_path / "bytes"
    path.write_bytes(b"0123456789")
    tensor = torch.full((8,), 255, dtype=torch.uint8)
    monkeypatch.setattr(transport, "_PREADV", False)
    fd = os.open(path, os.O_RDONLY)
    try:
        size = transport.pread_fill(
            fd, 7 if short else 2, 8, transport._writable_region(tensor)
        )
    finally:
        os.close(fd)
    assert size == (3 if short else 8)
    assert bytes(tensor[:size].tolist()) == (b"789" if short else b"23456789")
    if short:
        assert tensor[3:].tolist() == [255] * 5


@pytest.fixture
def cuda_device():
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 on a CUDA host")
    assert torch.cuda.is_available()
    return torch.device("cuda:0")


def fill_bytes(offset, count, dest):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(dest)), 3, count)
    return count


@pytest.mark.cuda
@pytest.mark.parametrize("reuse", [False, True])
def test_upload_waits_for_allocation_stream(cuda_device, reuse):
    size = 4 << 20
    transfer = transport.StagedTransfer(
        cuda_device, slot_bytes=size, slots=1, readers=1
    )
    try:
        target = torch.empty(size, dtype=torch.uint8, device=cuda_device)
        target.zero_()
        torch.cuda.synchronize(cuda_device)
        pointer = target.data_ptr()
        # Queue a delayed old writer. No profiler or inference timing involved.
        torch.cuda._sleep(500_000_000)
        target.fill_(9)
        if reuse:
            del target
            target = torch.empty(size, dtype=torch.uint8, device=cuda_device)
            if target.data_ptr() != pointer:
                pytest.skip("Allocator did not reuse the retired allocation")
        transfer.upload(target, fill_bytes)
        # Verify readiness on the caller's stream, without a device-wide sync.
        assert target.cpu().unique().tolist() == [3]
    finally:
        transfer.close()


@pytest.mark.cuda
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("failure", ["short", "exception"])
def test_failed_upload_drains_partial_dma_and_closes(
    cuda_device, monkeypatch, readers, failure
):
    chunk = 1 << 20
    transfer = transport.StagedTransfer(
        cuda_device, slot_bytes=chunk, slots=2, readers=readers
    )
    target = torch.empty(3 * chunk, dtype=torch.uint8, device=cuda_device)
    original_copy = transfer._copy

    def delayed_copy(*args):
        torch.cuda._sleep(500_000_000)
        original_copy(*args)

    def read(offset, count, dest):
        if offset:
            if failure == "short":
                return 0
            raise OSError("injected read failure")
        return fill_bytes(offset, count, dest)

    monkeypatch.setattr(transfer, "_copy", delayed_copy)
    try:
        with pytest.raises(OSError, match="Short read|injected read failure"):
            transfer.upload(target, read)
        assert transfer._stream.query(), (
            "Partial DMA is still using the failed destination"
        )
        assert transfer._pool is None
        assert transfer._closed
        assert target[:chunk].cpu().unique().tolist() == [3]
        with pytest.raises(RuntimeError, match="closed"):
            transfer.upload(target, fill_bytes)
    finally:
        transfer.close()
        transfer.close()


@pytest.mark.cuda
def test_failed_upload_joins_other_reader(cuda_device):
    transfer = transport.StagedTransfer(
        cuda_device, slot_bytes=4096, slots=2, readers=2
    )
    target = torch.empty(8192, dtype=torch.uint8, device=cuda_device)
    rendezvous = threading.Barrier(2)
    other_finished = threading.Event()

    def read(offset, count, dest):
        rendezvous.wait(timeout=10)
        if offset == 0:
            raise OSError("first reader failed")
        time.sleep(0.05)
        size = fill_bytes(offset, count, dest)
        other_finished.set()
        return size

    try:
        with pytest.raises(OSError, match="first reader failed"):
            transfer.upload(target, read)
        assert other_finished.is_set()
        assert transfer._stream.query()
        assert transfer._closed
    finally:
        transfer.close()


@pytest.mark.cuda
def test_noncontiguous_target_is_rejected_before_read(cuda_device):
    transfer = transport.StagedTransfer(cuda_device, slots=1)
    try:
        target = torch.empty((2, 3), device=cuda_device).T
        with pytest.raises(ValueError, match="contiguous"):
            transfer.upload(target, lambda *args: pytest.fail("Reader must not run"))
    finally:
        transfer.close()
