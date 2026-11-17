// Q5 W5A16: decode stored q*scale to FP16, then accumulate in FP32.
// The decode rounding boundary matches HMMA and the cuBLAS reference lane.
#pragma once

__device__ __forceinline__ __half2 q5_input_pair(const __half* x, int col, int k) {
    // Odd K makes alternate token rows half2-misaligned. Preserve a lone tail.
    if (col + 1 < k && (reinterpret_cast<std::uintptr_t>(x + col) & 3u) == 0)
        return *reinterpret_cast<const __half2*>(x + col);
    return __halves2half2(col < k ? x[col] : __float2half(0.f),
                         col + 1 < k ? x[col + 1] : __float2half(0.f));
}
__device__ __forceinline__ float q5_pair_dot(unsigned char packed, int two,
                                            __half scale, __half2 x) {
    const int a = (packed & 15) | ((two & 1) << 4);
    const int b = (packed >> 4) | ((two >> 1) << 4);
    const __half2 w = __hmul2(__halves2half2(__int2half_rn((a ^ 16) - 16),
                                           __int2half_rn((b ^ 16) - 16)),
                             __half2half2(scale));
    return __half2float(__low2half(w)) * __half2float(__low2half(x)) +
           __half2float(__high2half(w)) * __half2float(__high2half(x));
}
__global__ inline void q5_fused_run_kernel(
    const __half* x, const unsigned char* codes, const unsigned char* high,
    const __half* scales, __half* out, int tokens, int n_rows, int k, int padded) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (t >= tokens || n >= n_rows) return;
    const int groups = padded >> 6;
    const __half* xr = x + (std::uint64_t)t * k;
    const auto* cr = codes + (std::uint64_t)n * (padded >> 1);
    const auto* hr = high + (std::uint64_t)n * (padded >> 3);
    const auto* sr = scales + (std::uint64_t)n * groups;
    float acc = 0.f;
    for (int g = 0; g < groups; g += 4) {
        unsigned char p[4]; int hb[4]; __half s[4]; __half2 v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (g + j < groups) {
                p[j] = cr[(g + j) * 32 + lane];
                hb[j] = (hr[(g + j) * 8 + (lane >> 2)] >> ((lane * 2) & 7)) & 3;
                s[j] = sr[g + j];
                v[j] = q5_input_pair(xr, (g + j) * 64 + lane * 2, k);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            if (g + j < groups) acc += q5_pair_dot(p[j], hb[j], s[j], v[j]);
    }
    for (int d = 16; d; d >>= 1) acc += __shfl_down_sync(0xffffffff, acc, d);
    if (lane == 0) out[(std::uint64_t)t * n_rows + n] = __float2half_rn(acc);
}
