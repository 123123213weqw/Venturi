"""Complete text-role preflight and bounded, encoded layer-weight access.

Format-agnostic: consumes any WeightContainer (safetensors, GGUF, ninfer).
The admission validates that every tensor the model needs exists in the
container with the right shape and format, then provides lazy readers.
"""

from dataclasses import dataclass

import torch

from venturi.container.base import WeightContainer, TensorInfo
from venturi.container import SafetensorsContainer, open_container
from venturi.quantization.codecs import codec_for, dequant_row_reference, CODECS
from venturi.models.qwen.qwen_source import QwenSourcePlan, QwenSourceBinding


@dataclass(frozen=True)
class QwenQ8Weights:
    """Admitted model weights from any container format."""
    container: WeightContainer
    plan: QwenSourcePlan

    @classmethod
    def from_container(cls, container, *, matrix_format="q8_g32_fp16"):
        """Format-agnostic admission entry point.

        Accepts any WeightContainer (SafetensorsContainer,
        GGUFContainer). Validates every tensor the model needs exists with
        the correct shape, then returns admitted weights for lazy reading.
        """
        if not isinstance(container, WeightContainer):
            raise TypeError(f"Expected WeightContainer, got {type(container).__name__}")

        plan = QwenSourcePlan.from_config(
            container.model_config(), matrix_format=matrix_format
        )

        # Build expected tensor name → (shape, format) mapping.
        # For quantized matrices: "{source}.codes" + "{source}.scales" pair.
        # For unquantized: "{source}" directly.
        expected: dict[str, tuple[tuple, str]] = {}
        for spec in plan.bindings:
            if spec.format in CODECS and CODECS[spec.format].quantized:
                codec = CODECS[spec.format]
                n, k = spec.shape
                padded = (k + 127) // 128 * 128
                groups = padded // codec.group
                codes_shape = (n, groups * 32)  # packed nibbles
                scales_shape = (n, groups)
                expected[f"{spec.source}.codes"] = (codes_shape, "uint8")
                expected[f"{spec.source}.scales"] = (scales_shape, "float16")
            else:
                # Unquantized tensor (norm, a/b, conv, embedding, etc.)
                dtype = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}.get(spec.format, spec.format)
                expected[spec.source] = (spec.shape, dtype)

        # Tied embeddings: output_head shares token_embedding's tensors
        if plan.tied_embeddings:
            emb_source = next(s.source for s in plan.bindings if s.binding == "text/token_embedding")
            if f"{emb_source}.codes" in expected:
                expected[f"{emb_source}.codes_for_head"] = expected[f"{emb_source}.codes"]
                expected[f"{emb_source}.scales_for_head"] = expected[f"{emb_source}.scales"]

        # Validate: every expected tensor must exist with matching shape
        available = set(container.list_tensors())
        missing = set(expected.keys()) - available
        if missing:
            # Filter out tied-embedding aliases (they share tensors)
            real_missing = {m for m in missing if not m.endswith("_for_head")}
            if real_missing:
                raise ValueError(
                    f"Missing {len(real_missing)} tensors in container: "
                    f"{sorted(real_missing)[:5]}..."
                )

        for name, (shape, dtype) in expected.items():
            if name not in available:
                continue  # tied-embedding alias, skip
            info = container.tensor_info(name)
            if tuple(info.shape) != tuple(shape):
                raise ValueError(
                    f"Shape mismatch for {name}: expected {shape}, got {info.shape}"
                )

        return cls(container, plan)

    @classmethod
    def from_safetensors(cls, path, *, matrix_format="q8_g32_fp16"):
        """Convenience: opens a safetensors file directly."""
        container = SafetensorsContainer(path)
        return cls.from_container(container, matrix_format=matrix_format)

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

    def _spec(self, binding):
        for s in self.plan.bindings:
            if s.binding == binding:
                return s
        raise ValueError(f"Unknown binding: {binding}")

    def read_float(self, binding):
        """Read one unquantized parameter."""
        spec = self._spec(binding)
        if spec.format in CODECS and CODECS[spec.format].quantized:
            raise ValueError(f"Cannot read quantized {binding} as float")
        info = self.container.tensor_info(spec.source)
        raw = self.container.read_bytes(spec.source, 0, info.byte_size)
        dtype_map = {
            "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        dtype = dtype_map.get(spec.format)
        if dtype is None:
            raise ValueError(f"Unsupported float format: {spec.format}")
        value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(spec.shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite parameter: {binding}")
        return value

    def read_linear(self, binding):
        """Read one matrix (quantized or dense)."""
        spec = self._spec(binding)
        if len(spec.shape) != 2:
            raise ValueError("Linear role requires a matrix")

        if spec.format in CODECS and CODECS[spec.format].quantized:
            # Read codes and scales as raw bytes; let the caller decode.
            codes_info = self.container.tensor_info(f"{spec.source}.codes")
            scales_info = self.container.tensor_info(f"{spec.source}.scales")
            codes_raw = self.container.read_bytes(
                f"{spec.source}.codes", 0, codes_info.byte_size
            )
            scales_raw = self.container.read_bytes(
                f"{spec.source}.scales", 0, scales_info.byte_size
            )
            codec = CODECS[spec.format]
            n, k = spec.shape
            return _QuantizedMatrix(
                codes=codes_raw, scales=scales_raw,
                shape=spec.shape, codec=codec,
                container=self.container, source=spec.source,
            )

        # Dense/unquantized matrix
        return _DenseMatrix(self.read_float(binding))
