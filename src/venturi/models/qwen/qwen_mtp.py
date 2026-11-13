"""Explicit Qwen3.5 single-block MTP checkpoint schema, separate from the target."""

from dataclasses import dataclass
import json
from venturi.container.safetensors import SafetensorsContainer

from venturi.container.checkpoint import CheckpointError
from venturi.models.qwen.qwen_source import QwenSourceBinding, QwenSourcePlan
from venturi.models.qwen.qwen_weights import QwenWeights


@dataclass(frozen=True)
class QwenMTPPlan:
    config: dict
    bindings: tuple
    tied_embeddings: bool = False

    @classmethod
    def from_config(cls, raw):
        cfg = QwenSourcePlan.from_config(raw).config
        if (
            cfg.get("mtp_num_hidden_layers") != 1
            or cfg.get("mtp_use_dedicated_embeddings") is not False
        ):
            raise ValueError(
                "MTP requires one explicit block and shared target embeddings"
            )
        h, i, hq, hk, d = (
            cfg[k]
            for k in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            )
        )
        specs = []

        def add(source, role, shape, format="q8_g32_fp16"):
            specs.append(
                QwenSourceBinding("mtp." + source, "mtp/" + role, shape, shape, format)
            )

        add("fc.weight", "input_projection", (h, 2 * h))
        for source, role in (
            ("pre_fc_norm_embedding", "embedding_norm"),
            ("pre_fc_norm_hidden", "hidden_norm"),
            ("norm", "output_norm"),
        ):
            add(source + ".weight", role, (h,), "bf16")
        p = "layers.0."
        for source, role in (
            ("input_layernorm", "input_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
        ):
            add(p + source + ".weight", role, (h,), "bf16")
        for name, shape in (
            ("q", (2 * hq * d, h)),
            ("k", (hk * d, h)),
            ("v", (hk * d, h)),
            ("o", (h, hq * d)),
        ):
            add(p + f"self_attn.{name}_proj.weight", "attention/" + name, shape)
        for name in ("q", "k"):
            add(
                p + f"self_attn.{name}_norm.weight",
                "attention/" + name + "_norm",
                (d,),
                "bf16",
            )
        for name, shape in (("gate", (i, h)), ("up", (i, h)), ("down", (h, i))):
            add(p + f"mlp.{name}_proj.weight", "mlp/" + name, shape)
        return cls(cfg, tuple(specs))

    def validate_checkpoint(self, checkpoint):
        QwenSourcePlan.from_config(dict(checkpoint.config)).validate_checkpoint(
            checkpoint
        )
        if type(self).from_config(dict(checkpoint.config)) != self:
            raise CheckpointError("MTP source plan/config mismatch")
        expected = {s.source for s in self.bindings}
        actual = {s for s in checkpoint.tensors if s.startswith("mtp.")}
        if actual != expected:
            raise CheckpointError("MTP requires exactly all 15 original source tensors")
        for s in self.bindings:
            stored = checkpoint.tensors[s.source]
            if stored.shape != s.source_shape or stored.dtype not in s.source_dtypes:
                raise CheckpointError(f"MTP source shape/dtype mismatch: {s.source}")
        return tuple(sorted(set(checkpoint.tensors) - expected))


@dataclass(frozen=True)
class QwenMTPWeights(QwenWeights):
    @classmethod
    def open(cls, path):
        source = SafetensorsContainer(path)
        meta = source.metadata()
        if (
            meta.get("venturi.component") != "mtp"
            or meta.get("venturi.matrix_format") != "q8_g32_fp16"
        ):
            raise ValueError("Expected a separate Q8 MTP safetensors checkpoint")
        plan = QwenMTPPlan.from_config(json.loads(meta["venturi.config"]))
        return cls._admit(source, plan, "mtp")

    def validate_target(self, target):
        if (
            not isinstance(target, QwenWeights)
            or isinstance(target, QwenMTPWeights)
            or target.config != self.config
        ):
            raise ValueError("MTP/target configuration mismatch")
        a = self.provenance
        b = target.provenance
        if not a or a != b:
            raise ValueError(
                "MTP and target must derive from identical verified source files/revision"
            )
