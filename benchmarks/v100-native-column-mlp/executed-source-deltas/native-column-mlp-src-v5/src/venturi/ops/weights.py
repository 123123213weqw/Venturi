"""Explicit dense and exact Q4 storage descriptors; no implicit quantization."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseWeight:
    """A stored contiguous [out_features, in_features] floating-point matrix.

    No bias, transpose, conversion, or quantization is inferred. Preparation
    takes an owned copy; callers must not mutate prepared weight storage.
    """

    data: torch.Tensor

    format = "dense"

    def __post_init__(self):
        self.validate()

    @property
    def in_features(self):
        return self.data.shape[1]

    @property
    def out_features(self):
        return self.data.shape[0]

    @property
    def nbytes(self):
        return self.data.numel() * self.data.element_size()

    def validate(self):
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("DenseWeight requires a tensor")
        if self.data.ndim != 2 or not all(0 < d < 2**31 for d in self.data.shape):
            raise ValueError("DenseWeight requires a nonempty int32 [N, K] matrix")
        if self.data.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("DenseWeight requires floating-point storage")
        if (
            self.data.device.type not in ("cpu", "cuda")
            or not self.data.is_contiguous()
        ):
            raise ValueError("DenseWeight requires contiguous CPU/CUDA storage")


@dataclass(frozen=True)
class Q4RowSplitWeight:
    """Signed two's-complement nibbles, group-64 FP16 scales, K padded to 128.

    codes is uint8 [N, padded_K/2], with weight 2j in the low nibble.
    scales is float16 [N, padded_K/64]. Values represent q * scale,
    with q in [-8, 7], no zero point, and no extra scale rounding.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    in_features: int

    format = "q4_g64_fp16"
    layout = "row_split_k128_v1"
    group_size = 64

    def __post_init__(self):
        self.validate()

    @property
    def out_features(self) -> int:
        return self.codes.shape[0]

    @property
    def padded_k(self) -> int:
        return (self.in_features + 127) // 128 * 128

    @property
    def nbytes(self) -> int:
        return self.codes.numel() + 2 * self.scales.numel()

    def validate(self) -> None:
        if type(self.in_features) is not int or not 0 < self.in_features < 2**31:
            raise ValueError("in_features must be a positive int32")
        if self.padded_k >= 2**31:
            raise ValueError("padded K must fit int32")
        if self.codes.ndim != 2 or not 0 < self.codes.shape[0] < 2**31:
            raise ValueError("codes must be a nonempty [N, padded_K/2] matrix")
        if self.codes.dtype != torch.uint8 or self.scales.dtype != torch.float16:
            raise ValueError("Q4 requires uint8 codes and float16 scales")
        if self.codes.shape[1] != self.padded_k // 2:
            raise ValueError("codes width must match K padded to 128")
        if self.scales.shape != (self.out_features, self.padded_k // 64):
            raise ValueError("scales shape must be [N, padded_K/64]")
        if not self.codes.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError("Q4 planes must be contiguous")
        if self.codes.device != self.scales.device:
            raise ValueError("codes and scales must be on the same device")
        if self.codes.device.type not in ("cpu", "cuda"):
            raise ValueError("Q4 planes require CPU or CUDA storage")
        if self.scales.data_ptr() % 2:
            raise ValueError("FP16 scales must be aligned")

    @classmethod
    def from_codes(cls, codes: torch.Tensor, scales: torch.Tensor):
        """Pack CPU integer codes exactly. This is not a floating-weight quantizer."""
        if codes.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("from_codes is an explicit CPU preparation operation")
        if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("codes must contain signed integers")
        if codes.ndim != 2 or min(codes.shape) <= 0:
            raise ValueError("codes must be a nonempty [N, K] matrix")
        n, k = codes.shape
        if scales.dtype != torch.float16 or scales.shape != (n, (k + 63) // 64):
            raise ValueError("scales must be float16 [N, ceil(K/64)]")
        if bool(((codes < -8) | (codes > 7)).any()):
            raise ValueError("Q4 codes must be in [-8, 7]")
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("scales must be finite")
        padded = (k + 127) // 128 * 128
        values = torch.zeros((n, padded), dtype=torch.uint8)
        values[:, :k] = (codes.to(torch.int16) & 15).to(torch.uint8)
        packed = values[:, 0::2] | (values[:, 1::2] << 4)
        scale_plane = torch.zeros((n, padded // 64), dtype=torch.float16)
        scale_plane[:, : scales.shape[1]] = scales
        return cls(packed, scale_plane, k)

    @classmethod
    def from_payload(
        cls, payload: torch.Tensor, *, out_features: int, in_features: int
    ):
        """View an encoded Q4 tensor payload, not an entire .ninfer container.

        Codes precede a 256-byte-aligned FP16 scale plane. Views share payload
        storage; prepare_linear takes owned copies for execution.
        """
        if type(out_features) is not int or not 0 < out_features < 2**31:
            raise ValueError("out_features must be a positive int32")
        if type(in_features) is not int or not 0 < in_features < 2**31:
            raise ValueError("in_features must be a positive int32")
        if payload.dtype != torch.uint8 or payload.ndim != 1:
            raise ValueError("payload must be a contiguous uint8 vector")
        if not payload.is_contiguous() or payload.data_ptr() % 2:
            raise ValueError("payload must be contiguous and FP16 aligned")
        padded = (in_features + 127) // 128 * 128
        code_bytes = out_features * padded // 2
        scale_offset = (code_bytes + 255) // 256 * 256
        scale_bytes = out_features * padded // 32
        if payload.numel() != scale_offset + scale_bytes:
            raise ValueError("payload byte count does not match the Q4 layout")
        codes = payload[:code_bytes].view(out_features, padded // 2)
        scales = (
            payload[scale_offset:].view(torch.float16).view(out_features, padded // 64)
        )
        return cls(codes, scales, in_features)

    def dequantize_reference(self) -> torch.Tensor:
        """CPU FP64 diagnostic reference, never called by native execution."""
        self.validate()
        if self.codes.device.type != "cpu":
            raise ValueError("reference decoding requires explicit CPU weights")
        unsigned = torch.stack((self.codes & 15, self.codes >> 4), dim=-1)
        unsigned = unsigned.reshape(self.out_features, self.padded_k).to(torch.int16)
        signed = torch.where(unsigned < 8, unsigned, unsigned - 16)
        dense = signed.to(torch.float64) * self.scales.to(
            torch.float64
        ).repeat_interleave(64, 1)
        return dense[:, : self.in_features]


@dataclass(frozen=True)
class Q8RowSplitWeight:
    """Signed Q8 [-127,127], group-32 FP16 scales, K padded to 128.

    Generic projection storage, not restricted to a vocabulary matrix. There is
    no zero point. Source quantization is explicit, CPU-only, bounded by row
    chunks; converting a Q4 descriptor is intentionally not an accepted input.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    in_features: int
    format = "q8_g32_fp16"
    layout = "row_split_k128_v1"
    group_size = 32

    def __post_init__(self):
        self.validate()

    @property
    def out_features(self):
        return self.codes.shape[0]

    @property
    def padded_k(self):
        return (self.in_features + 127) // 128 * 128

    @property
    def nbytes(self):
        return self.codes.numel() + 2 * self.scales.numel()

    def validate(self):
        if (
            type(self.in_features) is not int
            or not 0 < self.in_features < 2**31
            or self.padded_k >= 2**31
        ):
            raise ValueError("Q8 in_features/padded K must be positive int32")
        if not isinstance(self.codes, torch.Tensor) or not isinstance(
            self.scales, torch.Tensor
        ):
            raise TypeError("Q8 requires code and scale tensors")
        if self.codes.ndim != 2 or not 0 < self.codes.shape[0] < 2**31:
            raise ValueError("Q8 codes require nonempty [N,padded_K]")
        if self.codes.shape[1] != self.padded_k or self.codes.dtype != torch.int8:
            raise ValueError("Q8 codes require int8 [N,padded_K]")
        if (
            self.scales.shape != (self.out_features, self.padded_k // 32)
            or self.scales.dtype != torch.float16
        ):
            raise ValueError("Q8 scales require float16 [N,padded_K/32]")
        if self.codes.device != self.scales.device or self.codes.device.type not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("Q8 planes require the same CPU/CUDA device")
        if not self.codes.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError("Q8 planes must be contiguous")

    @classmethod
    def from_dense(cls, source, *, rows_per_chunk=256):
        """Explicit nearest-even quantization using the stored FP16 scale.

        scale = FP16(max(abs(group))/127); codes = clip(round(x/scale),-127,127).
        Groups whose scale underflows to zero become all zero. Scale overflow
        and nonfinite sources are rejected. Source tensors remain unchanged.
        This creates NEW quantized weights, not a lossless storage conversion.
        """
        dense = DenseWeight(source)
        if source.device.type != "cpu":
            raise ValueError("Q8 quantization requires an explicit CPU source")
        if type(rows_per_chunk) is not int or not 0 < rows_per_chunk < 2**31:
            raise ValueError("rows_per_chunk must be a positive int32")
        n, k = dense.out_features, dense.in_features
        padded = (k + 127) // 128 * 128
        if padded >= 2**31:
            raise ValueError("Padded Q8 K must fit int32")
        codes = torch.zeros(n, padded, dtype=torch.int8)
        scales = torch.zeros(n, padded // 32, dtype=torch.float16)
        work = torch.float64 if source.dtype == torch.float64 else torch.float32
        with torch.no_grad():
            for begin in range(0, n, rows_per_chunk):
                end = min(n, begin + rows_per_chunk)
                values = torch.zeros(end - begin, padded, dtype=work)
                values[:, :k].copy_(source[begin:end].detach())
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Q8 source must be finite")
                groups = values.view(end - begin, -1, 32)
                stored = (groups.abs().amax(-1) / 127).to(torch.float16)
                if not bool(torch.isfinite(stored).all()):
                    raise ValueError("Q8 group scale overflows FP16")
                divisor = stored.to(work).unsqueeze(-1)
                normalized = groups / torch.where(
                    divisor > 0, divisor, torch.ones_like(divisor)
                )
                quantized = torch.where(
                    divisor > 0, normalized.round().clamp(-127, 127), 0
                ).to(torch.int8)
                codes[begin:end].copy_(quantized.reshape(end - begin, padded))
                scales[begin:end].copy_(stored)
        return cls(codes, scales, k)

    def dequantize_reference(self):
        """Explicit CPU FP64 diagnostic, never used by native execution."""
        self.validate()
        if self.codes.device.type != "cpu":
            raise ValueError("Reference decoding requires explicit CPU planes")
        if (
            bool((self.codes == -128).any())
            or not bool(torch.isfinite(self.scales).all())
            or bool((self.scales < 0).any())
        ):
            raise ValueError("Invalid Q8 codes/scales")
        return (self.codes.double() * self.scales.double().repeat_interleave(32, -1))[
            :, : self.in_features
        ]
