"""GLM5 geometry parsing against the real measured GGUF metadata fixture."""

import json
from pathlib import Path

import pytest

from venturi.models.glm5.geometry import (
    Glm5TextGeometry,
    glm5_text_geometry_from_metadata,
    layer_tensor_names,
)

FIXTURE = Path(__file__).parent / "fixtures" / "glm5-metadata.json"


@pytest.fixture(scope="module")
def metadata():
    return json.loads(FIXTURE.read_text())


def test_real_fixture_parses(metadata):
    g = glm5_text_geometry_from_metadata(metadata)
    assert g.hidden_size == 4096 and g.num_layers == 45 and g.nextn_layers == 1
    assert g.vocab_size == 154880 and g.context_length == 1048576
    assert g.expert_count == 288 and g.expert_used_count == 8
    assert g.expert_shared_count == 1
    assert g.hc_streams == 4 and g.hc_sinkhorn_iterations == 20
    assert g.indexer_top_k == 2048 and g.indexer_kpool == 4
    assert g.kda_heads == 64 and g.kda_head_dim == 128
    assert g.ssm_conv_kernel == 4 and g.gate_lower_bound == -5.0
    assert g.leading_dense_blocks == 3
    assert g.sparse_mla_layers == tuple(range(3, 45, 4))
    assert g.attention_kind(3) == "sparse_mla"
    assert g.attention_kind(4) == "kda"
    assert g.attention_kind(0) == "kda"
    assert g.mlp_kind(0) == "dense" and g.mlp_kind(3) == "sparse"


def test_layer_tensor_names_match_measured_sets(metadata):
    g = glm5_text_geometry_from_metadata(metadata)
    mla = set(layer_tensor_names(g, 3))
    assert "blk.3.attn_kv_a_mqa.weight" in mla and "blk.3.ssm_a" not in mla
    assert "blk.3.indexer.proj.weight" in mla
    assert "blk.3.ffn_gate_exps.weight" in mla
    kda = set(layer_tensor_names(g, 4))
    assert "blk.4.ssm_a" in kda and "blk.4.attn_kv_a_mqa.weight" not in kda
    assert "blk.4.attn_q.weight" in kda  # KDA blocks also carry plain attn tensors
    dense = set(layer_tensor_names(g, 0))
    assert "blk.0.ffn_gate.weight" in dense
    assert "blk.0.ffn_gate_exps.weight" not in dense
    assert "blk.0.ffn_gate_inp.weight" not in dense  # router only on sparse layers


def test_fail_closed(metadata):
    broken = dict(metadata)
    broken["glm5-next.attention.head_count"] = 0
    with pytest.raises(ValueError):
        glm5_text_geometry_from_metadata(broken)
    broken = dict(metadata)
    broken["glm5-next.expert_gating_func"] = 1
    with pytest.raises(ValueError):
        glm5_text_geometry_from_metadata(broken)
    broken = dict(metadata)
    broken["general.architecture"] = "llama"
    with pytest.raises(ValueError):
        glm5_text_geometry_from_metadata(broken)
