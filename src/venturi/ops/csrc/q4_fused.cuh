// Venturi q4_g64_fp16 warp-coalesced fused GEMM kernel v3.
// Included inside namespace vr by qwen_runtime.cuh.
//
// Warp-per-row. Each iteration batches 4 groups: all memory requests
// (weight bytes, activation pairs, scales) are issued before any compute,
// letting the memory system overlap fetches. __half2 loads halve activation
// transactions. One butterfly reduction per row at the end.
#pragma once

__global__ inline void q4_fused_run_kernel(
    const __half* x, const unsigned char* codes, const __half* scales,
    __half* out, int t_tokens, int n_rows, int k, int padded) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    const int t = blockIdx.y;
    if (t >= t_tokens || n >= n_rows) return;
    const int groups = padded >> 6;
    const __half* xrow = x + (std::uint64_t)t * k;
    const unsigned char* crow = codes + (std::uint64_t)n * (padded >> 1);
    const __half* srow = scales + (std::uint64_t)n * groups;

    float acc = 0.0f;

    // Fast path: 4 groups per iteration with batched memory issue.
    int g = 0;
    for (; g + 4 <= groups; g += 4) {
        // ① Batch-issue all memory reads for 4 groups (pipeline overlap).
        const unsigned char p0 = crow[(g + 0) * 32 + lane];
        const unsigned char p1 = crow[(g + 1) * 32 + lane];
        const unsigned char p2 = crow[(g + 2) * 32 + lane];
        const unsigned char p3 = crow[(g + 3) * 32 + lane];
        const int c0 = (g + 0) * 64 + lane * 2;
        const int c1 = (g + 1) * 64 + lane * 2;
        const int c2 = (g + 2) * 64 + lane * 2;
        const int c3 = (g + 3) * 64 + lane * 2;
        // __half2 loads: one 4-byte transaction for two FP16 activations.
        const __half2 xa = __halves2half2(c0<k?xrow[c0]:__float2half(0), c0+1<k?xrow[c0+1]:__float2half(0));
        const __half2 xb = __halves2half2(c1<k?xrow[c1]:__float2half(0), c1+1<k?xrow[c1+1]:__float2half(0));
        const __half2 xc = __halves2half2(c2<k?xrow[c2]:__float2half(0), c2+1<k?xrow[c2+1]:__float2half(0));
        const __half2 xd = __halves2half2(c3<k?xrow[c3]:__float2half(0), c3+1<k?xrow[c3+1]:__float2half(0));
        const float s0 = __half2float(srow[g + 0]);
        const float s1 = __half2float(srow[g + 1]);
        const float s2 = __half2float(srow[g + 2]);
        const float s3 = __half2float(srow[g + 3]);

        // ② Compute all 4 groups (memory fetches already in flight).
        {
            int lo = p0 & 0x0F; if (lo >= 8) lo -= 16;
            int hi = p0 >> 4;   if (hi >= 8) hi -= 16;
            acc += __half2float(__float2half(lo*s0))*__half2float(__low2half(xa)) + __half2float(__float2half(hi*s0))*__half2float(__high2half(xa));
        }
        {
            int lo = p1 & 0x0F; if (lo >= 8) lo -= 16;
            int hi = p1 >> 4;   if (hi >= 8) hi -= 16;
            acc += __half2float(__float2half(lo*s1))*__half2float(__low2half(xb)) + __half2float(__float2half(hi*s1))*__half2float(__high2half(xb));
        }
        {
            int lo = p2 & 0x0F; if (lo >= 8) lo -= 16;
            int hi = p2 >> 4;   if (hi >= 8) hi -= 16;
            acc += __half2float(__float2half(lo*s2))*__half2float(__low2half(xc)) + __half2float(__float2half(hi*s2))*__half2float(__high2half(xc));
        }
        {
            int lo = p3 & 0x0F; if (lo >= 8) lo -= 16;
            int hi = p3 >> 4;   if (hi >= 8) hi -= 16;
            acc += __half2float(__float2half(lo*s3))*__half2float(__low2half(xd)) + __half2float(__float2half(hi*s3))*__half2float(__high2half(xd));
        }
    }

    // Tail: remaining groups one at a time.
    for (; g < groups; ++g) {
        const unsigned char packed = crow[g * 32 + lane];
        int lo = packed & 0x0F; if (lo >= 8) lo -= 16;
        int hi = packed >> 4;   if (hi >= 8) hi -= 16;
        const int cl = g * 64 + lane * 2;
        const float scale=__half2float(srow[g]);
        float ga = 0.0f;
        if (cl < k)     ga += __half2float(__float2half(lo*scale)) * __half2float(xrow[cl]);
        if (cl + 1 < k) ga += __half2float(__float2half(hi*scale)) * __half2float(xrow[cl + 1]);
        acc += ga;
    }

    // One butterfly reduction per row (5 shuffles).
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
    if (lane == 0) out[(std::uint64_t)t * n_rows + n] = __float2half(acc);
}
