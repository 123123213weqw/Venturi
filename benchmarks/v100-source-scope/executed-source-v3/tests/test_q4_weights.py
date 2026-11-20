"""Exact codec and metadata rejection tests, requiring no CUDA toolchain."""

import pytest
import torch

from venturi.ops import AdaBackend, Q4RowSplitWeight
from venturi.ops.build import build_cuda_backend


@pytest.mark.parametrize("k", [1, 63, 64, 65, 127, 128, 129, 4097])
def test_pack_signed_codes_and_tail(k):
    values = (torch.arange(3 * k).reshape(3, k) % 16 - 8).to(torch.int8)
    scales = (torch.arange(3 * ((k + 63) // 64)).reshape(3, -1) % 7 + 1).to(
        torch.float16
    ) / 8
    weight = Q4RowSplitWeight.from_codes(values, scales)
    expected = torch.empty((3, k), dtype=torch.float64)
    # Independent scalar storage oracle, not the vectorized codec's operations.
    for row in range(3):
        for col in range(k):
            byte = int(weight.codes[row, col // 2])
            nibble = (byte // (16 if col % 2 else 1)) % 16
            integer = nibble if nibble <= 7 else nibble - 16
            assert integer == int(values[row, col])
            expected[row, col] = integer * float(scales[row, col // 64])
    torch.testing.assert_close(weight.dequantize_reference(), expected, rtol=0, atol=0)
    assert weight.codes.shape == (3, ((k + 127) // 128) * 64)


def test_known_bytes_and_group_scales():
    # Low nibble first: 0,1,-8,-1,7,-7. No offset-zero convention.
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    codes[0, :3] = torch.tensor([0x10, 0xF8, 0x97], dtype=torch.uint8)
    scales = torch.tensor([[0.5, -2]], dtype=torch.float16)
    weight = Q4RowSplitWeight(codes, scales, 128)
    assert weight.dequantize_reference()[0, :6].tolist() == [
        0,
        0.5,
        -4,
        -0.5,
        3.5,
        -3.5,
    ]
    assert weight.nbytes == 68


def test_payload_alignment_and_shared_views():
    payload = torch.zeros(260, dtype=torch.uint8)
    payload[0] = 0xF8
    payload[256:].view(torch.float16).copy_(torch.tensor([0.5, 2], dtype=torch.float16))
    weight = Q4RowSplitWeight.from_payload(payload, out_features=1, in_features=65)
    assert weight.codes.data_ptr() == payload.data_ptr()
    assert weight.scales.data_ptr() == payload.data_ptr() + 256
    assert weight.dequantize_reference()[0, :2].tolist() == [-4, -0.5]
    for wrong in [
        payload[:-1],
        torch.cat((payload, torch.zeros(1, dtype=torch.uint8))),
    ]:
        with pytest.raises(ValueError, match="byte count"):
            Q4RowSplitWeight.from_payload(wrong, out_features=1, in_features=65)


@pytest.mark.parametrize("value", [-9, 8])
def test_reject_out_of_range(value):
    with pytest.raises(ValueError, match=r"in \[-8, 7\]"):
        Q4RowSplitWeight.from_codes(
            torch.tensor([[value]], dtype=torch.int8),
            torch.ones(1, 1, dtype=torch.float16),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_reject_nonfinite_pack_scales(value):
    with pytest.raises(ValueError, match="finite"):
        Q4RowSplitWeight.from_codes(
            torch.zeros(1, 64, dtype=torch.int8),
            torch.tensor([[value]], dtype=torch.float16),
        )


@pytest.mark.parametrize("case", ["dtype", "width", "scales", "stride", "k", "empty"])
def test_reject_bad_storage(case):
    codes = torch.zeros((2, 64), dtype=torch.uint8)
    scales = torch.ones((2, 2), dtype=torch.float16)
    k = 128
    if case == "dtype":
        codes = codes.to(torch.int32)
    if case == "width":
        codes = codes[:, :32].contiguous()
    if case == "scales":
        scales = scales[:, :1].contiguous()
    if case == "stride":
        codes = torch.zeros(2, 128, dtype=torch.uint8)[:, ::2]
    if case == "k":
        k = 0
    if case == "empty":
        codes = codes[:0]
    with pytest.raises(ValueError):
        Q4RowSplitWeight(codes, scales, k)


def test_preparation_contract_before_cuda():
    backend = object.__new__(AdaBackend)
    weight = Q4RowSplitWeight.from_codes(
        torch.zeros(1, 1, dtype=torch.int8), torch.ones(1, 1, dtype=torch.float16)
    )
    with pytest.raises(TypeError, match="Q4RowSplitWeight"):
        backend.prepare_linear(object(), tokens=1)
    for kwargs in [
        dict(tokens=0),
        dict(tokens=True),
        dict(tokens=262141),
        dict(tokens=1, dtype=torch.float32),
        dict(tokens=1, route="awq"),
        dict(tokens=1, device="cpu"),
    ]:
        with pytest.raises(ValueError):
            backend.prepare_linear(weight, **kwargs)


def test_build_is_explicit_and_linux_only(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux CUDA"):
        build_cuda_backend(tmp_path)


def test_missing_library(tmp_path):
    with pytest.raises(FileNotFoundError):
        AdaBackend(tmp_path / "missing.so")
