// QuantLinear row-codec dispatch: prepared linear plans (PR-A).
//
// Design contract (night-tps-20261117 architecture review):
//   * weight FORMAT (bits/group) is per-matrix; precision POLICY (WxA16 vs
//     WxA8) is explicit and never switched by batch size or shape;
//   * plans are resolved at load time from (bits, policy); decode time is a
//     token-threshold comparison plus a direct call — no search, no compile;
//   * a plan exposes only kernels that share its policy's activation type;
//     a shape with no valid kernel under the policy fails explicitly.
// This header is the C++ mirror of codecs.py EXECUTION_QUALIFIED: python
// gates loading per architecture, this table routes execution per matrix.
#pragma once
#include "ops_api.h"

namespace rowcodec {

enum class Precision : int { WxA16 = 0, WxA8 = 1 };

struct LinearPlan {
    int bits;
    int group;       // codec scale-group width (64 for q4/q5/q6 row codecs)
    Precision precision;
    // WxA16 kernels: FP16 activations in, no activation workspace.
    int (*scalar_run)(const void* x, const void* codes, const void* high,
                      const void* scales, void* out, int t, int rows, int k,
                      int padded, int dtype, void* stream);
    int (*hmma_run)(const void* x, const void* codes, const void* high,
                    const void* scales, void* out, int tokens, int rows,
                    int k, int padded, void* stream);
    int min_hmma_tokens;      // 0 disables the tensor-core lane entirely
    int hmma_k_align;         // k must be a multiple of this for hmma_run
    // WxA8 kernels: quantized activations, workspace = int8 codes per 32
    // columns then fp32 scales (vdq layout).
    int (*dp4a_run)(const void* workspace, std::uint64_t available,
                    const void* codes, const void* high, const void* scales,
                    void* out, int rows, int stride, int tokens, int k,
                    int padded, void* stream);
    int activation_bytes_per_column;  // workspace need per token per column; 0 = none
};

// q4 predates the high plane; the adapter keeps one uniform signature without
// touching the qualified q4 ABI.
inline int q4_scalar_adapter(const void* x, const void* codes, const void* high,
                             const void* scales, void* out, int t, int rows, int k,
                             int padded, int dtype, void* stream) {
    (void)high;
    return venturi_q4_fused_run(x, codes, scales, out, t, rows, k, padded, dtype, stream);
}

// Load-time resolution. Returns nullptr when the (bits, policy) pair has no
// fully registered family — callers must fail explicitly, never substitute a
// different precision.
inline const LinearPlan* prepare(int bits, Precision precision) {
    static const LinearPlan q4_a16 = {
        4, 64, Precision::WxA16,
        q4_scalar_adapter, nullptr, 0, 0,
        nullptr, 0};
    static const LinearPlan q5_a16 = {
        5, 64, Precision::WxA16,
        venturi_q5_fused_run, venturi_q5_hmma_run, 4, 64,
        nullptr, 0};
    static const LinearPlan q5_a8 = {
        5, 64, Precision::WxA8,
        nullptr, nullptr, 0, 0,
        venturi_q5_dp4a_fused_run, 36};
    if (precision == Precision::WxA16) {
        if (bits == 4) return &q4_a16;
        if (bits == 5) return &q5_a16;
    } else if (precision == Precision::WxA8) {
        if (bits == 5) return &q5_a8;
    }
    return nullptr;
}

}  // namespace rowcodec
