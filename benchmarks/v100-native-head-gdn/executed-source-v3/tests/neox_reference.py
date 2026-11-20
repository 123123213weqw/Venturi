"""Small, independent FP64 GPT-NeoX equations for CPU correctness tests."""

import math

import torch


def tiny_checkpoint(parallel=True, rotary_pct=0.5):
    config = {
        "model_type": "gpt_neox",
        "vocab_size": 31,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "max_position_embeddings": 32,
        "layer_norm_eps": 1e-5,
        "rotary_emb_base": 10000,
        "rotary_pct": rotary_pct,
        "hidden_act": "gelu",
        "tie_word_embeddings": False,
        "use_parallel_residual": parallel,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    generator = torch.Generator().manual_seed(3901)

    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1

    weights = {
        "gpt_neox.embed_in.weight": random((31, 16)),
        "embed_out.weight": random((31, 16)),
        "gpt_neox.final_layer_norm.weight": 1 + random((16,)),
        "gpt_neox.final_layer_norm.bias": random((16,)),
    }
    for index in range(2):
        root = f"gpt_neox.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[root + name + ".weight"] = 1 + random((16,))
            weights[root + name + ".bias"] = random((16,))
        for name, n, k in [
            ("attention.query_key_value", 48, 16),
            ("attention.dense", 16, 16),
            ("mlp.dense_h_to_4h", 24, 16),
            ("mlp.dense_4h_to_h", 16, 24),
        ]:
            weights[root + name + ".weight"] = random((n, k))
            weights[root + name + ".bias"] = random((n,))
    return config, weights


def full_logits(ids, config, weights):
    def linear(x, root):
        return x @ weights[root + ".weight"].T + weights[root + ".bias"]

    def norm(x, root):
        centered = x - x.mean(-1, keepdim=True)
        normalized = (
            centered
            / (
                centered.square().mean(-1, keepdim=True) + config["layer_norm_eps"]
            ).sqrt()
        )
        return normalized * weights[root + ".weight"] + weights[root + ".bias"]

    hidden = weights["gpt_neox.embed_in.weight"][list(ids)]
    heads = config["num_attention_heads"]
    dim = config["hidden_size"] // heads
    rotation = int(dim * config["rotary_pct"])
    for index in range(config["num_hidden_layers"]):
        prefix = f"gpt_neox.layers.{index}."
        original = hidden
        fused = linear(
            norm(hidden, prefix + "input_layernorm"),
            prefix + "attention.query_key_value",
        )
        projected = []
        for head in range(heads):
            base = head * 3 * dim
            q, k, v = (
                fused[:, base + j * dim : base + (j + 1) * dim].clone()
                for j in range(3)
            )
            for position in range(len(ids)):
                for pair in range(rotation // 2):
                    angle = position / config["rotary_emb_base"] ** (
                        2 * pair / rotation
                    )
                    a, b = pair, pair + rotation // 2
                    for vector in (q, k):
                        x, y = vector[position, a].item(), vector[position, b].item()
                        vector[position, a] = x * math.cos(angle) - y * math.sin(angle)
                        vector[position, b] = x * math.sin(angle) + y * math.cos(angle)
            scores = q @ k.T / math.sqrt(dim)
            for position in range(len(ids)):
                scores[position, position + 1 :] = -math.inf
            exp = (scores - scores.max(-1, keepdim=True).values).exp()
            projected.append((exp / exp.sum(-1, keepdim=True)) @ v)
        attention = linear(torch.cat(projected, dim=-1), prefix + "attention.dense")
        mlp_source = (
            original if config["use_parallel_residual"] else original + attention
        )
        mlp = linear(
            norm(mlp_source, prefix + "post_attention_layernorm"),
            prefix + "mlp.dense_h_to_4h",
        )
        mlp = 0.5 * mlp * (1 + torch.erf(mlp / math.sqrt(2)))
        mlp = linear(mlp, prefix + "mlp.dense_4h_to_h")
        hidden = (
            (mlp + attention) + original
            if config["use_parallel_residual"]
            else mlp + mlp_source
        )
    return norm(hidden, "gpt_neox.final_layer_norm") @ weights["embed_out.weight"].T
