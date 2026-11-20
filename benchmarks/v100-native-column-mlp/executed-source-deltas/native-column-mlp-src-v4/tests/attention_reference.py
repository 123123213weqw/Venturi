"""Independent scalar-layout FP64 oracles; no Venturi execution helpers."""

import torch


def rope_reference(x, cos, sin, rotary_dim, layout):
    result = x.cpu().double().clone()
    data = x.cpu().double()
    c, s = cos.cpu().double(), sin.cpu().double()
    for token in range(x.shape[0]):
        for pair in range(rotary_dim // 2):
            first = pair if layout == "split_half" else 2 * pair
            second = pair + rotary_dim // 2 if layout == "split_half" else first + 1
            result[token, :, first] = (
                data[token, :, first] * c[token, pair]
                - data[token, :, second] * s[token, pair]
            )
            result[token, :, second] = (
                data[token, :, first] * s[token, pair]
                + data[token, :, second] * c[token, pair]
            )
    return result.to(x.dtype)


def attention_reference(q, k, v, scale):
    query, key, value = (t.cpu().double() for t in (q, k, v))
    result = torch.empty_like(query)
    repeats = q.shape[1] // k.shape[1]
    for token in range(q.shape[0]):
        length = k.shape[0] - q.shape[0] + token + 1
        for head in range(q.shape[1]):
            kv_head = head // repeats
            logits = (key[:length, kv_head] * query[token, head]).sum(-1) * scale
            probabilities = (logits - logits.max()).exp()
            probabilities /= probabilities.sum()
            result[token, head] = (
                probabilities[:, None] * value[:length, kv_head]
            ).sum(0)
    return result.to(q.dtype)
