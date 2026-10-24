"""Complete text-role preflight and bounded, encoded Q8 layer-weight access.

This is checkpoint binding, not hybrid model forward/session implementation.
Readers return owned CPU descriptors for explicit native uploads; they never
materialize Q8 matrices as dense floats or execute an external model backend.
"""

from dataclasses import dataclass

import torch

from ..ninfer import NInferArtifact
from ..ops import DenseWeight, Q8RowSplitWeight, Q8VocabularyWeights
from .qwen_source import QwenSourcePlan


@dataclass(frozen=True)
class QwenQ8Weights:
    artifact: NInferArtifact
    plan: QwenSourcePlan

    @classmethod
    def from_ninfer(cls, artifact):
        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        plan = QwenSourcePlan.from_config(
            artifact.manifest["components"]["text"]["config"]
        )
        expected = {s.binding: s for s in plan.bindings}
        if plan.tied_embeddings:
            expected["text/output_head"] = expected["text/token_embedding"]
        bindings = artifact.manifest["bindings"]
        if bindings.keys() != expected.keys():
            raise ValueError(
                "Qwen Q8 requires exactly the complete unfused text binding set"
            )
        seen = {}
        # Preflight ALL roles before any payload read or device allocation.
        for name, spec in expected.items():
            binding = bindings[name]
            if binding.keys() != {"object"}:
                raise ValueError(f"Qwen Q8 role requires a direct object: {name}")
            obj = artifact.objects[binding["object"]]
            layout = (
                "row_split_k128_v1"
                if spec.format == "q8_g32_fp16"
                else "contiguous_le_v1"
            )
            if (
                obj["shape"] != list(spec.shape)
                or obj.get("format") != spec.format
                or obj.get("layout") != layout
            ):
                raise ValueError(f"Qwen Q8 role shape/format/layout mismatch: {name}")
            if spec.format != "q8_g32_fp16" and obj["bytes"] > 8 * 2**20:
                raise ValueError(f"Floating Qwen role exceeds the read bound: {name}")
            previous = seen.get(obj["id"])
            if previous is not None and not (
                plan.tied_embeddings
                and previous == "text/token_embedding"
                and name == "text/output_head"
            ):
                raise ValueError(f"Unexpected weight tying: {previous} and {name}")
            seen[obj["id"]] = name
        return cls(artifact, plan)

    @property
    def config(self):
        return self.plan.config

    def layer_bindings(self, layer):
        if type(layer) is not int or not 0 <= layer < self.config["num_hidden_layers"]:
            raise ValueError("Layer index out of range")
        prefix = f"text/layers/{layer}/"
        return tuple(
            s.binding for s in self.plan.bindings if s.binding.startswith(prefix)
        )

    def _object(self, binding):
        spec = self.artifact.manifest["bindings"].get(binding)
        if not isinstance(spec, dict) or spec.keys() != {"object"}:
            raise ValueError(f"Unknown direct Qwen role: {binding}")
        return self.artifact.objects[spec["object"]]

    def read_float(self, binding):
        """Own one bounded parameter; raw norm gamma is NOT offset here."""
        obj = self._object(binding)
        if obj["format"] not in ("bf16", "fp32") or obj["layout"] != "contiguous_le_v1":
            raise ValueError("Floating role access never silently dequantizes Q8")
        raw = bytearray(self.artifact.read_object(obj["id"], max_bytes=8 * 2**20))
        dtype = torch.bfloat16 if obj["format"] == "bf16" else torch.float32
        value = torch.frombuffer(raw, dtype=dtype).reshape(obj["shape"])
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite Qwen parameter: {binding}")
        return value

    def read_linear(self, binding):
        """One encoded matrix at a time, including preserved BF16 a/b projections."""
        obj = self._object(binding)
        if len(obj["shape"]) != 2:
            raise ValueError("Linear role requires a matrix")
        if obj["format"] == "q8_g32_fp16":
            weight = Q8VocabularyWeights.from_ninfer(self.artifact, binding)
            return Q8RowSplitWeight(weight.codes, weight.scales, weight.hidden_size)
        return DenseWeight(self.read_float(binding))
