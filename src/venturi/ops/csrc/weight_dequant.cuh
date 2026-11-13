// Venturi row-codec weight dequantization kernels (q4/q5/q6_g64 family).
// Included inside the private operator namespace vrow by blocks.cu. Contract:
// Signed row-split group decode: per row, groups = padded/64;
// group codes are 32 packed bytes (even lane = low nibble, odd = high nibble);
// q5/q6 add a high-bit plane ((bits-4) bits per lane, LSB-first little-endian
// stream, (bits-4)*64/8 bytes per group); values are bits-wide two's
// complement scaled by one little-endian FP16 scale per group; padding lanes
// (col >= k) are never written. Output is a contiguous FP16 [rows, k] matrix
// with ld = k, ready for venturi_dense_run.
#pragma once

__global__ inline void row_dequant_rows_kernel(
    const unsigned char* codes, const unsigned char* high, const __half* scales,
    __half* out, int row_begin, int rows, int k, int padded, int bits) {
    const int groups = padded >> 6;
    const int hb = bits - 4;  // high bits per lane (0 for q4)
    const int hstride = hb ? groups * hb * 8 : 0;  // high-plane bytes per row
    const int local = blockIdx.y;
    const int row = row_begin + local;
    if (local >= rows) return;
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const unsigned char* crow = codes + (std::uint64_t)row * groups * 32 + (std::uint64_t)group * 32;
    const unsigned char* hrow = high ? high + (std::uint64_t)row * hstride + (std::uint64_t)group * hb * 8 : nullptr;
    const float scale = __half2float(scales[(std::uint64_t)row * groups + group]);
    __half* dst = out + (std::uint64_t)local * k + (std::uint64_t)group * 64;
    const int first = group * 64;
    const int span = 1 << bits, sign = 1 << (bits - 1);
    for (int lane = 0; lane < 64; lane += 2) {
        const unsigned char packed = crow[lane >> 1];
        int lo = packed & 0x0F, hi = packed >> 4;
        if (hb) {
            const int blo = lane * hb, bhi = (lane + 1) * hb;
            lo |= ((hrow[blo >> 3] >> (blo & 7)) & ((1 << hb) - 1)) << 4;
            hi |= ((hrow[bhi >> 3] >> (bhi & 7)) & ((1 << hb) - 1)) << 4;
        }
        if (lo >= sign) lo -= span;
        if (hi >= sign) hi -= span;
        if (first + lane < k) dst[lane] = __float2half(lo * scale);
        if (first + lane + 1 < k) dst[lane + 1] = __float2half(hi * scale);
    }
}

__global__ inline void row_dequant_indexed_kernel(
    const unsigned char* codes, const unsigned char* high, const __half* scales,
    const std::int64_t* ids, __half* out, int count, int k, int padded, int bits) {
    const int groups = padded >> 6;
    const int hb = bits - 4;
    const int hstride = hb ? groups * hb * 8 : 0;
    const int slot = blockIdx.y;
    if (slot >= count) return;
    const int row = static_cast<int>(ids[slot]);
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const unsigned char* crow = codes + (std::uint64_t)row * groups * 32 + (std::uint64_t)group * 32;
    const unsigned char* hrow = high ? high + (std::uint64_t)row * hstride + (std::uint64_t)group * hb * 8 : nullptr;
    const float scale = __half2float(scales[(std::uint64_t)row * groups + group]);
    __half* dst = out + (std::uint64_t)slot * k + (std::uint64_t)group * 64;
    const int first = group * 64;
    const int span = 1 << bits, sign = 1 << (bits - 1);
    for (int lane = 0; lane < 64; lane += 2) {
        const unsigned char packed = crow[lane >> 1];
        int lo = packed & 0x0F, hi = packed >> 4;
        if (hb) {
            const int blo = lane * hb, bhi = (lane + 1) * hb;
            lo |= ((hrow[blo >> 3] >> (blo & 7)) & ((1 << hb) - 1)) << 4;
            hi |= ((hrow[bhi >> 3] >> (bhi & 7)) & ((1 << hb) - 1)) << 4;
        }
        if (lo >= sign) lo -= span;
        if (hi >= sign) hi -= span;
        if (first + lane < k) dst[lane] = __float2half(lo * scale);
        if (first + lane + 1 < k) dst[lane + 1] = __float2half(hi * scale);
    }
}
