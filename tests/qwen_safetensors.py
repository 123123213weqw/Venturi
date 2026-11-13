"""Synthetic canonical checkpoints with explicit code/scale tensors."""

import json
import torch
from safetensors.torch import save_file
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.quantization.codecs import codec_for
from qwen_fixtures import source_config


def write_quantized_checkpoint(
    path,
    *,
    config=None,
    matrix_format="q8_g32_fp16",
    updates=None,
    drop=(),
    component="text",
    reverse=False,
):
    cfg = source_config() if config is None else config
    if component == "mtp":
        from venturi.models.qwen.qwen_mtp import QwenMTPPlan

        plan = QwenMTPPlan.from_config(cfg)
    else:
        plan = QwenSourcePlan.from_config(cfg, matrix_format=matrix_format)
    tensors = {}
    for spec in plan.bindings:
        codec = codec_for(spec.format)
        if codec.quantized:
            n, k = spec.shape
            groups = ((k + 127) // 128 * 128) // codec.group
            tensors[spec.source + ".codes"] = torch.ones(
                (n, groups * 32), dtype=torch.int8 if codec.bits == 8 else torch.uint8
            )
            if codec.high_plane:
                tensors[spec.source + ".high"] = torch.zeros(
                    (n, groups * 8 * (codec.bits - 4)), dtype=torch.uint8
                )
            tensors[spec.source + ".scales"] = torch.full(
                (n, groups), 0.125, dtype=torch.float16
            )
        else:
            tensors[spec.source] = torch.full(
                spec.shape,
                0.125,
                dtype={
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                }[spec.format],
            )
    tensors.update(updates or {})
    for name in drop:
        tensors.pop(name)
    if reverse:
        tensors = dict(reversed(list(tensors.items())))
    metadata = {
        "venturi.config": json.dumps(cfg),
        "venturi.component": component,
        "venturi.matrix_format": matrix_format,
        "venturi.source": json.dumps({"fixture": "synthetic same source"}),
    }
    save_file(tensors, str(path), metadata=metadata)
    return path, plan, tensors
