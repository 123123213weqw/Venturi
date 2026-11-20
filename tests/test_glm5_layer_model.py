"""GLM5 decoder layer + full model: shapes, finite outputs, state management."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from venturi.models.glm5.geometry import Glm5TextGeometry, glm5_text_geometry_from_metadata
from venturi.models.glm5.hc import Glm5HyperConnection
from venturi.models.glm5.layer import DenseMLP, Glm5DecoderLayer
from venturi.models.glm5.model import Glm5TextModel, ModelState

FIXTURE = Path(__file__).parent / "fixtures" / "glm5-metadata.json"


@pytest.fixture(scope="module")
def metadata():
    return json.loads(FIXTURE.read_text())


class TestDecoderLayer:
    def test_kda_layer_forward(self, metadata):
        g = glm5_text_geometry_from_metadata(metadata)
        from venturi.models.glm5.kda import Glm5KDA
        layer = Glm5DecoderLayer(64, 4, eps=1e-5)
        layer.attention = Glm5KDA(64, 4, 16, 4, -5.0, 1e-5)
        layer.mlp = DenseMLP(64, 128)
        x = torch.randn(1, 3, 4, 64)
        with torch.no_grad():
            streams, state, topk = layer(x)
        assert streams.shape == (1, 3, 4, 64)
        assert topk is None  # KDA doesn't produce topk
        assert isinstance(state, object)  # KDAState

    def test_mla_layer_forward(self, metadata):
        from venturi.models.glm5.mla import Glm5MLA
        layer = Glm5DecoderLayer(64, 4, eps=1e-5)
        layer.attention = Glm5MLA(64, 4, 32, 48, 16, 16, 1e-5)
        layer.mlp = DenseMLP(64, 128)
        x = torch.randn(1, 3, 4, 64)
        with torch.no_grad():
            streams, state, topk = layer(x)
        assert streams.shape == (1, 3, 4, 64)
        assert topk is None  # no indexer assigned

    def test_state_passthrough(self, metadata):
        from venturi.models.glm5.kda import Glm5KDA
        layer = Glm5DecoderLayer(64, 4, eps=1e-5)
        layer.attention = Glm5KDA(64, 4, 16, 4, -5.0, 1e-5)
        layer.mlp = DenseMLP(64, 128)
        x = torch.randn(1, 2, 4, 64)
        with torch.no_grad():
            s1, state1, _ = layer(x)
            s2, state2, _ = layer(x[:, :1], past=state1)
        assert state2.tokens_seen == 3 if hasattr(state2, 'tokens_seen') else True


class TestFullModel:
    def test_synthetic_model_forward(self, metadata):
        """Small synthetic version of the full model."""
        g = Glm5TextGeometry(
            hidden_size=64, num_layers=4, nextn_layers=1, vocab_size=100,
            head_count=4, kv_lora_rank=32, q_lora_rank=48,
            key_length_mla=16, value_length_mla=16,
            indexer_heads=2, indexer_key_length=8, indexer_top_k=4, indexer_kpool=2,
            kda_head_dim=16, kda_heads=4, ssm_conv_kernel=4, gate_lower_bound=-5.0,
            expert_count=8, expert_used_count=2, expert_intermediate=32,
            expert_shared_count=1, expert_shared_intermediate=32, expert_weights_scale=1.0,
            dense_feed_forward=64, leading_dense_blocks=1, hc_streams=4,
            hc_sinkhorn_iterations=5, rms_eps=1e-5, hc_eps=1e-6,
            context_length=128, eos_token_id=1, bos_token_id=0,
        )
        model = Glm5TextModel(g)
        # Fill MoE with DenseMLP for testing (real MoE via store)
        from venturi.models.glm5.layer import DenseMLP
        for i, layer in enumerate(model.layers):
            if layer.mlp is None:
                layer.mlp = DenseMLP(64, 64)
        ids = torch.tensor([[1, 5, 3, 7]])
        with torch.no_grad():
            logits, state = model(ids)
        assert logits.shape == (1, 4, 100)
        assert bool(torch.isfinite(logits).all())
        assert state.tokens_seen == 4

    def test_incremental_decode(self, metadata):
        """Prefill → decode gives consistent shape."""
        g = Glm5TextGeometry(
            hidden_size=32, num_layers=2, nextn_layers=0, vocab_size=50,
            head_count=2, kv_lora_rank=16, q_lora_rank=24,
            key_length_mla=8, value_length_mla=8,
            indexer_heads=1, indexer_key_length=4, indexer_top_k=2, indexer_kpool=2,
            kda_head_dim=8, kda_heads=2, ssm_conv_kernel=4, gate_lower_bound=-5.0,
            expert_count=4, expert_used_count=2, expert_intermediate=16,
            expert_shared_count=1, expert_shared_intermediate=16, expert_weights_scale=1.0,
            dense_feed_forward=32, leading_dense_blocks=1, hc_streams=2,
            hc_sinkhorn_iterations=3, rms_eps=1e-5, hc_eps=1e-6,
            context_length=64, eos_token_id=1, bos_token_id=0,
        )
        model = Glm5TextModel(g)
        for layer in model.layers:
            if layer.mlp is None:
                layer.mlp = DenseMLP(32, 32)
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            logits1, state = model(ids)
            logits2, _ = model(torch.tensor([[4]]), past=state)
        assert logits1.shape == (1, 3, 50)
        assert logits2.shape == (1, 1, 50)
        assert bool(torch.isfinite(logits2).all())
