"""Lazy matrix representations returned by QwenQ8Weights.read_linear()."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class _DenseMatrix:
    """Unquantized matrix, fully materialized."""
    value: torch.Tensor


@dataclass(frozen=True)
class _QuantizedMatrix:
    """Quantized matrix: raw codes + scales, decoded on demand."""
    codes: bytes
    scales: bytes
    shape: tuple
    codec: object  # Codec from codecs.py
    container: object  # WeightContainer (for lazy re-read if needed)
    source: str

    @property
    def codes_tensor(self) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(self.codes), dtype=torch.uint8
        ).reshape(self._codes_shape)

    @property
    def scales_tensor(self) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(self.scales), dtype=torch.float16
        ).reshape(self._scales_shape)

    @property
    def _codes_shape(self) -> tuple:
        n, k = self.shape
        padded = (k + 127) // 128 * 128
        groups = padded // self.codec.group
        return (n, groups * 32)

    @property
    def _scales_shape(self) -> tuple:
        n, k = self.shape
        padded = (k + 127) // 128 * 128
        groups = padded // self.codec.group
        return (n, groups)

    def dequantize(self) -> torch.Tensor:
        """Reference dequantization: codes × scales → FP16 matrix."""
        n, k = self.shape
        padded = (k + 127) // 128 * 128
        groups = padded // self.codec.group
        bits = self.codec.bits
        sign = 1 << (bits - 1)
        span = 1 << bits

        codes = np.frombuffer(self.codes, dtype=np.uint8).reshape(n, groups, 32)
        scales = np.frombuffer(self.scales, dtype="<f2").reshape(n, groups).astype(np.float32)

        lanes = np.empty((n, groups, 64), dtype=np.int32)
        lanes[:, :, 0::2] = codes & 0x0F
        lanes[:, :, 1::2] = codes >> 4
        if bits > 4:
            # High-bit plane unpacking for q5/q6
            hb = bits - 4
            hb_stride = groups * hb * 8
            high = np.frombuffer(
                self.codes, dtype=np.uint8, count=n * hb_stride,
                offset=self._high_offset
            ).reshape(n, groups, hb * 8)
            extra = np.zeros((n, groups, 64), dtype=np.int32)
            for lane in range(64):
                v = np.zeros((n, groups), dtype=np.int32)
                for b in range(hb):
                    byte_i = (lane * hb + b) >> 3
                    bit_i = (lane * hb + b) & 7
                    v |= ((high[:, :, byte_i].astype(np.int32) >> bit_i) & 1) << b
                extra[:, :, lane] = v
            lanes |= extra << 4

        signed = np.where(lanes >= sign, lanes - span, lanes).astype(np.float32)
        values = (signed * scales[:, :, None]).reshape(n, groups * 64)[:, :k]
        return torch.from_numpy(np.ascontiguousarray(values)).to(torch.float16)

    @property
    def _high_offset(self) -> int:
        """Byte offset of the high plane within the codes tensor."""
        n, k = self.shape
        padded = (k + 127) // 128 * 128
        groups = padded // self.codec.group
        base = n * groups * 32
        return (base + 255) // 256 * 256  # 256-byte aligned
