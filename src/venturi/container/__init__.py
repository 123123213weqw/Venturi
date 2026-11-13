"""Safetensors storage metadata and explicit bounded tensor reads."""
from .base import ByteSpan, TensorInfo
from .safetensors import SafetensorsContainer
__all__ = ["TensorInfo", "ByteSpan", "SafetensorsContainer"]
