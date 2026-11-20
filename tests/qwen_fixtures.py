"""Independent small floating-HF source fixtures; no runtime or packed container."""

import hashlib, json, struct
from pathlib import Path
import torch


def source_config():
    return dict(
        model_type="qwen3_5",
        tie_word_embeddings=False,
        text_config=dict(
            model_type="qwen3_5_text",
            dtype="bfloat16",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=64,
            linear_value_head_dim=13,
            linear_conv_kernel_dim=4,
            vocab_size=17,
            max_position_embeddings=128,
            hidden_act="silu",
            attention_bias=False,
            attn_output_gate=True,
            rms_norm_eps=1e-6,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters=dict(
                rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25
            ),
        ),
    )


def fixture_tensors():
    # Explicit HF reference shapes, independent of the production plan builder.
    shapes = {
        "lm_head.weight": (17, 32),
        "model.language_model.embed_tokens.weight": (17, 32),
        "model.language_model.norm.weight": (32,),
    }
    for i in range(2):
        p = f"model.language_model.layers.{i}."
        shapes.update(
            {
                p + "input_layernorm.weight": (32,),
                p + "post_attention_layernorm.weight": (32,),
                p + "mlp.gate_proj.weight": (64, 32),
                p + "mlp.up_proj.weight": (64, 32),
                p + "mlp.down_proj.weight": (32, 64),
            }
        )
    p = "model.language_model.layers.0.linear_attn."
    shapes.update(
        {
            p + "in_proj_qkv.weight": (308, 32),
            p + "in_proj_z.weight": (52, 32),
            p + "in_proj_a.weight": (4, 32),
            p + "in_proj_b.weight": (4, 32),
            p + "out_proj.weight": (32, 52),
            p + "conv1d.weight": (308, 1, 4),
            p + "norm.weight": (13,),
            p + "A_log": (4,),
            p + "dt_bias": (4,),
        }
    )
    p = "model.language_model.layers.1.self_attn."
    shapes.update(
        {
            p + "q_proj.weight": (64, 32),
            p + "k_proj.weight": (16, 32),
            p + "v_proj.weight": (16, 32),
            p + "o_proj.weight": (32, 32),
            p + "q_norm.weight": (16,),
            p + "k_norm.weight": (16,),
        }
    )
    result = {}
    for i, (name, shape) in enumerate(shapes.items()):
        count = torch.tensor(shape).prod().item()
        data = (((torch.arange(count) + i) % 251 - 125) / 16384).reshape(shape)
        result[name] = data.to(
            torch.float32
            if name.endswith(("A_log", "linear_attn.norm.weight"))
            else torch.bfloat16
        )
    # Deliberately asymmetric taps and per-head Q/gate rows detect reordering.
    result["model.visual.unused.weight"] = torch.ones(3, 5, dtype=torch.bfloat16)
    result["mtp.layers.0.unused.weight"] = torch.ones(2, 5, dtype=torch.bfloat16)
    return result


def write_source(root, *, raw=None, tensors=None, verified=True, shards=2):
    root.mkdir()
    raw = source_config() if raw is None else raw
    tensors = fixture_tensors() if tensors is None else tensors
    (root / "config.json").write_text(json.dumps(raw))
    weight_map = {}
    for i in range(shards):
        selected = {k: v for j, (k, v) in enumerate(tensors.items()) if j % shards == i}
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        # Independent safetensors framing, with no optional NumPy save helper.
        # All fixture payloads are small CPU tensors; production export uses
        # bounded zero-copy writes, not this diagnostic byte-list construction.
        header, chunks, offset = {}, [], 0
        for key, tensor in selected.items():
            data = bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
            header[key] = dict(
                dtype={torch.bfloat16: "BF16", torch.float32: "F32", torch.int8: "I8"}[
                    tensor.dtype
                ],
                shape=list(tensor.shape),
                data_offsets=[offset, offset + len(data)],
            )
            chunks.append(data)
            offset += len(data)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        (root / name).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
        )
        weight_map.update({k: name for k in selected})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=weight_map))
    )
    if verified:
        record = dict(
            status="complete",
            repo="synthetic/test-fixture",
            revision="fixture-not-trained-checkpoint",
            files=[],
        )
        for p in sorted(root.iterdir()):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            record["files"].append(
                dict(
                    name=p.name,
                    bytes=p.stat().st_size,
                    actual_sha256=digest,
                    lfs_sha256=None,
                )
            )
        (root / "venturi-source.json").write_text(json.dumps(record))
    return root
