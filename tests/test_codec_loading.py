"""Complete per-codec admission using physical safetensors tensors."""

import pytest, torch
from venturi.models.qwen.qwen_weights import QwenWeights
from venturi.quantization.codecs import codec_for
from venturi.quantization.layout import row_payload_bytes
from venturi.quantization.storage_verify import verify_checkpoint
from qwen_safetensors import write_quantized_checkpoint
from qwen_fixtures import source_config


def _full_artifact(tmp_path, matrix_format):
    cfg = source_config()["text_config"]
    cfg.update(
        hidden_size=64,
        vocab_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        intermediate_size=128,
        tie_word_embeddings=True,
    )
    path, _, _ = write_quantized_checkpoint(
        tmp_path / "model.safetensors", config=cfg, matrix_format=matrix_format
    )
    return path


@pytest.mark.parametrize(
    "fmt", ["q8_g32_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "fp16"]
)
def test_full_checkpoint_admits_and_verifies(tmp_path, fmt):
    path = _full_artifact(tmp_path, fmt)
    weights = QwenWeights.open(path)
    assert weights.plan.tied_embeddings
    assert verify_checkpoint(path)["status"] == "storage-readable"
    assert weights.descriptor("text/token_embedding").format == fmt


@pytest.mark.parametrize(
    "fmt,bits,group",
    [
        ("q8_g32_fp16", 8, 32),
        ("q4_g64_fp16", 4, 64),
        ("q5_g64_fp16", 5, 64),
        ("q6_g64_fp16", 6, 64),
    ],
)
def test_codec_geometry_is_independent_of_container(fmt, bits, group):
    codec = codec_for(fmt)
    assert (codec.bits, codec.group) == (bits, group)
    n, k = 6, 130
    padded = (k + 127) // 128 * 128
    assert codec.encoded_bytes(n, k) == row_payload_bytes((n, k), fmt)
    assert codec.upload_bytes(n, k) == n * (padded * bits // 8 + padded // group * 2)


@pytest.mark.parametrize("fmt", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"])
def test_reference_plane_decode(tmp_path, fmt):
    weights = QwenWeights.open(_full_artifact(tmp_path, fmt))
    linear = weights.read_linear("text/token_embedding")
    assert linear.data.shape == (32, 64) and linear.data.dtype == torch.float16
    # 0x01 independently represents low nibble +1, high nibble 0.
    expected = torch.zeros(32, 64, dtype=torch.float16)
    expected[:, ::2] = 0.125
    assert torch.equal(linear.data, expected)


def test_dense_fp16_round_trip(tmp_path):
    weights = QwenWeights.open(_full_artifact(tmp_path, "fp16"))
    assert weights.read_float("text/token_embedding").shape == (32, 64)


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        codec_for("q3_g17_invalid")


def test_execution_gate_does_not_treat_storage_support_as_execution(tmp_path):
    from venturi.models.qwen_load.spec import ensure_execution_qualified

    with pytest.raises(ValueError):
        ensure_execution_qualified(QwenWeights.open(_full_artifact(tmp_path, "fp16")))
    ensure_execution_qualified(
        QwenWeights.open(_full_artifact(tmp_path, "q8_g32_fp16"))
    )


@pytest.mark.parametrize(
    "fmt,bits", [("q4_g64_fp16", 4), ("q5_g64_fp16", 5), ("q6_g64_fp16", 6)]
)
def test_all_signed_codes_and_high_bit_positions(fmt, bits):
    """Independent scalar pack covers sign, nibble and each high-bit lane."""
    from venturi.quantization.codecs import dequantize_planes

    values = [(i % (1 << bits)) - (1 << (bits - 1)) for i in range(128)]
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    high = torch.zeros((1, 16 * (bits - 4)), dtype=torch.uint8) if bits > 4 else None
    for group in range(2):
        for lane in range(64):
            unsigned = values[group * 64 + lane] % (1 << bits)
            codes[0, group * 32 + lane // 2] |= (unsigned & 15) << (4 * (lane % 2))
            if high is not None:
                hb = bits - 4
                high[0, group * 8 * hb + lane * hb // 8] |= (unsigned >> 4) << (
                    lane * hb % 8
                )
    scales = torch.tensor([[0.125, 0.25]], dtype=torch.float16)
    actual = dequantize_planes(codes, scales, high, (1, 117), fmt)
    expected = torch.tensor(
        [[v * (0.125 if i < 64 else 0.25) for i, v in enumerate(values[:117])]],
        dtype=torch.float16,
    )
    assert torch.equal(actual, expected)
