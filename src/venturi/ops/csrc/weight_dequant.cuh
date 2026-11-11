// Venturi q4_g64_fp16 weight dequantization for the dense qualification path.
// Included inside namespace vr by qwen_runtime.cuh. Contract mirrors
// NInfer row_split_k128_v1 decode_row exactly: per row, groups = padded/64;
// group codes are 32 packed bytes (even lane = low nibble, odd = high nibble),
// 4-bit two's complement (-8..7), one little-endian FP16 scale per group;
// padding lanes (col >= k) are never written. Output is a contiguous FP16
// [rows, k] matrix with ld = k, ready for venturi_dense_run.
#pragma once

__global__ inline void q4_dequant_rows_kernel(
    const unsigned char* codes, const __half* scales, __half* out,
    int row_begin, int rows, int k, int padded) {
    const int groups = padded >> 6;
    const int local = blockIdx.y;
    const int row = row_begin + local;
    if (local >= rows) return;
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const unsigned char* crow = codes + (std::uint64_t)row * groups * 32 + (std::uint64_t)group * 32;
    const float scale = __half2float(scales[(std::uint64_t)row * groups + group]);
    __half* dst = out + (std::uint64_t)local * k + (std::uint64_t)group * 64;
    const int first = group * 64;
    for (int lane = 0; lane < 64; lane += 2) {
        const unsigned char packed = crow[lane >> 1];
        const int lo = packed & 0x0F, hi = packed >> 4;
        if (first + lane < k) dst[lane] = __float2half((lo >= 8 ? lo - 16 : lo) * scale);
        if (first + lane + 1 < k) dst[lane + 1] = __float2half((hi >= 8 ? hi - 16 : hi) * scale);
    }
}

__global__ inline void q4_dequant_indexed_kernel(
    const unsigned char* codes, const __half* scales, const std::int64_t* ids,
    __half* out, int count, int k, int padded) {
    const int groups = padded >> 6;
    const int slot = blockIdx.y;
    if (slot >= count) return;
    const int row = static_cast<int>(ids[slot]);
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const unsigned char* crow = codes + (std::uint64_t)row * groups * 32 + (std::uint64_t)group * 32;
    const float scale = __half2float(scales[(std::uint64_t)row * groups + group]);
    __half* dst = out + (std::uint64_t)slot * k + (std::uint64_t)group * 64;
    const int first = group * 64;
    for (int lane = 0; lane < 64; lane += 2) {
        const unsigned char packed = crow[lane >> 1];
        const int lo = packed & 0x0F, hi = packed >> 4;
        if (first + lane < k) dst[lane] = __float2half((lo >= 8 ? lo - 16 : lo) * scale);
        if (first + lane + 1 < k) dst[lane + 1] = __float2half((hi >= 8 ? hi - 16 : hi) * scale);
    }
}
