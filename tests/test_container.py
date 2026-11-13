"""Container layer tests: SafetensorsContainer."""
import json
import struct
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from venturi.container import (
    ByteSpan, SafetensorsContainer, WeightContainer, TensorInfo,
)


def _make_quantized_safetensors(tmp_path, n=64, k=128, bits=4):
    """Create a synthetic safetensors file with quantized tensors."""
    group = 64
    groups = (k + group - 1) // group
    gen = torch.Generator().manual_seed(42)

    codes = torch.randint(0, 256, (n, groups * 32), generator=gen, dtype=torch.uint8)
    scales = (torch.rand(n, groups, generator=gen) * 0.25 + 0.03125).to(torch.float16)
    norm = torch.randn(n, generator=gen).to(torch.bfloat16)

    tensors = {
        "layer.0.attn.q.codes": codes,
        "layer.0.attn.q.scales": scales,
        "layer.0.norm.weight": norm,
    }
    metadata = {
        "quantization_config": json.dumps({
            "quant_method": "venturi",
            "bits": bits,
            "group_size": group,
            "format": f"q{bits}_g{group}_fp16",
        }),
    }
    path = tmp_path / "quantized.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


class TestSafetensorsContainer:
    def test_list_and_info(self, tmp_path):
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)

        names = c.list_tensors()
        assert "layer.0.attn.q.codes" in names
        assert "layer.0.norm.weight" in names

        info = c.tensor_info("layer.0.attn.q.codes")
        assert info.shape == (64, 128 // 64 * 32)  # (n, groups*32)
        assert info.dtype == "uint8"
        assert info.byte_size > 0
        assert info.byte_offset >= 8  # after header length field

    def test_metadata(self, tmp_path):
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)
        meta = c.metadata()
        assert "quantization_config" in meta
        config = json.loads(meta["quantization_config"])
        assert config["bits"] == 4
        assert config["group_size"] == 64

    def test_read_bytes(self, tmp_path):
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)
        info = c.tensor_info("layer.0.attn.q.codes")

        # Read first 16 bytes
        data = c.read_bytes("layer.0.attn.q.codes", 0, 16)
        assert len(data) == 16

        # Read last 4 bytes
        tail = c.read_bytes("layer.0.attn.q.codes", info.byte_size - 4, 4)
        assert len(tail) == 4

        # Out of bounds
        with pytest.raises(ValueError):
            c.read_bytes("layer.0.attn.q.codes", 0, info.byte_size + 1)

    def test_byte_spans(self, tmp_path):
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)
        spans = c.byte_spans()

        assert len(spans) >= 2  # at least codes + norm
        for span in spans:
            assert span.path == str(path)
            assert span.bytes > 0
            assert span.file_offset >= 8

        # Total payload should equal sum of span bytes
        total = sum(s.bytes for s in spans)
        assert total == c.total_payload_bytes()

    def test_payload_round_trip(self, tmp_path):
        """Verify read_bytes returns the same data as safetensors.torch.load."""
        from safetensors.torch import load_file
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)

        loaded = load_file(str(path))
        codes = loaded["layer.0.attn.q.codes"]
        expected = codes.flatten().numpy().tobytes()

        actual = c.read_bytes("layer.0.attn.q.codes", 0, codes.numel())
        assert actual == expected


class TestWeightContainerProtocol:
    def test_satisfies_protocol(self, tmp_path):
        path = _make_quantized_safetensors(tmp_path)
        c = SafetensorsContainer(path)
        # Protocol check (runtime_checkable)
        assert isinstance(c, WeightContainer)
