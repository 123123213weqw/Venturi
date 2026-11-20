"""Tiny mixed-dtype storage fixtures, deliberately unrelated to any architecture."""

import json

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def raw_config():
    return {
        "model_type": "custom-model",
        "quantization_config": {"bits": 4, "quant_method": "custom-packed"},
        "custom_setting": {"arbitrary": [1, 2, 3]},
    }


@pytest.fixture
def weights():
    generator = torch.Generator().manual_seed(731)
    return {
        "arbitrary.weight": torch.randn(
            (11, 7), generator=generator, dtype=torch.float16
        ),
        "another.component": torch.randn(
            (5, 3), generator=generator, dtype=torch.bfloat16
        ),
        "quant.qweight": torch.arange(-64, 64, dtype=torch.int8).reshape(8, 16),
        "quant.packed_int4": torch.tensor(
            [[0x12345678, 0x76543210], [-1, 0]], dtype=torch.int32
        ),
        "quant.packed_uint4": torch.tensor([0x01, 0xAF, 0xFF, 0x70], dtype=torch.uint8),
        "quant.scales": torch.tensor([0.125, 0.5], dtype=torch.float32),
        "quant.zeros": torch.tensor([1, 3], dtype=torch.int16),
        "positions": torch.tensor([1, 7, 19], dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.75, dtype=torch.float64),
    }


@pytest.fixture
def make_checkpoint(tmp_path, raw_config, weights):
    def make(*, config=True, tensors=None, sharded=False, metadata=None):
        directory = tmp_path / "checkpoint"
        directory.mkdir(exist_ok=True)
        if config is not False:
            (directory / "config.json").write_text(
                json.dumps(raw_config if config is True else config)
            )
        tensors = weights if tensors is None else tensors
        metadata = (
            {"format": "pt", "custom": "opaque metadata"}
            if metadata is None
            else metadata
        )
        if sharded:
            entries = list(tensors.items())
            mapping = {}
            for index, part in enumerate((entries[::2], entries[1::2]), 1):
                filename = f"model-{index:05d}-of-00002.safetensors"
                save_file(dict(part), str(directory / filename), metadata=metadata)
                mapping.update({name: filename for name, _ in part})
            index_metadata = {
                "total_size": sum(
                    x.numel() * x.element_size() for x in tensors.values()
                ),
                "custom": "preserved",
            }
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": index_metadata, "weight_map": mapping})
            )
        else:
            save_file(tensors, str(directory / "model.safetensors"), metadata=metadata)
        return directory

    return make
