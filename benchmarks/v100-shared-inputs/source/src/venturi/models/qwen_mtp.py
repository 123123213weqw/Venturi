"""Explicit Qwen3.5 single-block MTP checkpoint schema, separate from the target."""

from dataclasses import dataclass

from ..checkpoint import CheckpointError
from ..ninfer import NInferArtifact
from .qwen_source import QwenSourceBinding, QwenSourcePlan
from .qwen_weights import QwenQ8Weights


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
class QwenMTPWeights(QwenQ8Weights):
    @classmethod
    def from_ninfer(cls, artifact):
        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        parts = artifact.manifest["components"]
        if "mtp" not in parts or parts["mtp"].get("target") != "text":
            raise ValueError(
                "Expected separate MTP artifact with external target vocabulary"
            )
        plan = QwenMTPPlan.from_config(parts["text"]["config"])
        if parts["mtp"]["config"] != plan.config:
            raise ValueError("MTP component config mismatch")
        expected = {s.binding: s for s in plan.bindings}
        if artifact.manifest["bindings"].keys() != expected.keys():
            raise ValueError("MTP requires exactly all 15 direct role bindings")
        seen = set()
        for name, spec in expected.items():
            binding = artifact.manifest["bindings"][name]
            if binding.keys() != {"object"} or binding["object"] in seen:
                raise ValueError("MTP bindings must be direct and untied")
            seen.add(binding["object"])
            obj = artifact.objects[binding["object"]]
            if (
                obj["shape"] != list(spec.shape)
                or obj["format"] != spec.format
                or obj["layout"]
                != (
                    "row_split_k128_v1"
                    if spec.format == "q8_g32_fp16"
                    else "contiguous_le_v1"
                )
            ):
                raise ValueError(f"MTP role shape/format/layout mismatch: {name}")
        return cls(artifact, plan)

    def validate_target(self, target):
        if (
            not isinstance(target, QwenQ8Weights)
            or isinstance(target, QwenMTPWeights)
            or target.config != self.config
        ):
            raise ValueError("MTP/target configuration mismatch")
        a = self.artifact.manifest.get("provenance", {}).get("source")
        b = target.artifact.manifest.get("provenance", {}).get("source")
        if not a or a != b:
            raise ValueError(
                "MTP and target must derive from identical verified source files/revision"
            )
