// Adapted for Venturi: remove engine Weight dependencies; K=0 selects runtime x_ld.
// From NInfer/Cinference and qwen38-4090-decode; see NOTICE and provenance.json.
#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Single-pass small-T groupwise GEMM for Ada: out^T[T x N] = x^T[T x K] * W^T[K x N] on
// mma.m16n8k16 with tokens as M (zero-padded to 16) and eight weight rows as N, for the row-split
// Q4_G64_FP16 and Q5_G64_FP16 layouts (32 code bytes per 64-weight group, weight k in byte k / 2,
// low nibble first; Q5 adds 8 high-bit bytes per group, bit b of byte j holding weight 8j + b).
//
// A CTA owns 8 * RowTiles weight rows over the whole K and streams them through a cp.async pipeline
// of GroupsPerStage groups per stage. Global reads are contiguous 16-byte chunks per row; the row
// paddings below keep the fragment reads out of shared memory bank-conflict free. x is L2-resident
// at these widths and is prefetched into registers one stage ahead instead of taking pipeline
// space. Warps split each stage's groups and reduce through shared memory at the end, so every
// weight byte is read once for any T <= 8 * TokenTiles.
//
// Per group a lane owns 16 consecutive weights (8 code bytes) of one row. MMA step s maps K slots
// {2q, 2q+1} to weights 16q + 4s + {0,1} and slots {2q+8, 2q+9} to 16q + 4s + {2,3}; x is read with
// the same permutation, so the dot product is unchanged. Codes decode to exact bf16 integers and
// the fp16 group scale is applied in fp32 after the group's four MMAs.
enum class AdaWeightFormat { Q4, Q5, Q3, Q3Odd };

template <AdaWeightFormat Format, int RowTiles, int Warps, int TokenTiles, int GroupsPerStage,
          int Stages, bool EvictFirst = true, bool Packed = false>
struct AdaSmallTSchedule {
    static constexpr AdaWeightFormat kFormat = Format;
    // Weight fills carry an L2 evict-first hint: each weight byte is read once per call.
    static constexpr bool kEvictFirst        = EvictFirst;
    static constexpr bool kHasHigh           = Format != AdaWeightFormat::Q4;
    // Q3 planes: 2-bit low codes plus a high-bit plane; Q3Odd holds odd codes 2c + 1 (the half grid).
    static constexpr bool kQ3                = Format == AdaWeightFormat::Q3 || Format == AdaWeightFormat::Q3Odd;
    // Low-plane bytes per 64-weight group: 4-bit codes (Q4, Q5) or 2-bit codes (Q3, Q3Odd).
    static constexpr int kLowBytes           = kQ3 ? 16 : 32;
    static constexpr int kRowTiles           = RowTiles;
    static constexpr int kWarps              = Warps;
    static constexpr int kTokenTiles         = TokenTiles;
    static constexpr int kG                  = GroupsPerStage;
    static constexpr int kStages             = Stages;
    static constexpr int kRowsPerCta         = 8 * RowTiles;
    static constexpr int kThreads            = 32 * Warps;
    static constexpr int kMaxTokens          = 8 * TokenTiles;
    static constexpr int kGroupsPerWarp      = kG / kWarps;
    // Research (NINFER_Q3_TILED): `codes` holds stage-packed tiles [low | high | LS8 bytes] (q3_pack_tiles); the
    // high and scale pointers are unused and super must be set.
    static constexpr bool kPacked            = Packed;
    // Packed four-warp Q3 CTAs need three per SM (<= 168 registers): 76.6 us vs 80.6+ at two (bench_diag2.cu).
    static constexpr int kMinBlocks          = (Packed && Warps == 4) ? 3 : 1;
    static constexpr int kLowTileBytes       = 8 * RowTiles * kLowBytes * GroupsPerStage;
    static constexpr int kHighTileBytes      = (Format != AdaWeightFormat::Q4) ? 8 * RowTiles * 8 * GroupsPerStage : 0;
    static constexpr int kTileBytes          = kLowTileBytes + kHighTileBytes + 8 * RowTiles * GroupsPerStage;

    static constexpr int kCodeStride  = kLowBytes * kG + kLowBytes;
    static constexpr int kHighStride  = kHasHigh ? 8 * kG + 16 : 0;
    static constexpr int kScaleStride = kG <= 8 ? 16 : 48;

    static constexpr int kCodeBytes  = kRowsPerCta * kCodeStride;
    static constexpr int kHighBytes  = kRowsPerCta * kHighStride;
    static constexpr int kScaleBytes = kRowsPerCta * kScaleStride;
    static constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes;
    static constexpr int kReductionBytes =
        kWarps * kRowsPerCta * kMaxTokens * static_cast<int>(sizeof(float));
    static constexpr int kSmemBytes = std::max(kStages * kStageBytes, kReductionBytes);

    static_assert(RowTiles >= 1 && Warps >= 1 && (TokenTiles == 1 || TokenTiles == 2));
    static_assert(kG == 4 || kG == 8 || kG == 16);
    static_assert(kG % kWarps == 0);
    static_assert(kStages >= 2 && kStages <= 8);
};

template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q5AdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q5, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;
template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q3AdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q3, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;
template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q3OddAdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q3Odd, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;
template <int RowTiles, int Warps, int TokenTiles, int GroupsPerStage, int Stages>
using Q4AdaSmallTSchedule =
    AdaSmallTSchedule<AdaWeightFormat::Q4, RowTiles, Warps, TokenTiles, GroupsPerStage, Stages>;

// CTA b owns weight rows [b * rows, (b + 1) * rows).
struct AdaContiguousRows {
    __device__ __forceinline__ static int weight_row(int block, int local, int rows) {
        return block * rows + local;
    }
};

// CTA b owns output rows [b * rows / 2, (b + 1) * rows / 2) of a gate/up parent whose up half starts
// at weight row Offset: local rows [0, rows / 2) are gate rows and the rest the matching up rows.
template <int Offset>
struct AdaPairedHalfRows {
    __device__ __forceinline__ static int weight_row(int block, int local, int rows) {
        const int half = rows / 2;
        return local < half ? block * half + local : Offset + block * half + local - half;
    }
};

// Two Q5 codes of one byte plus their inverted high bits (bits 0 and 1) -> exact bf16x2 integers.
// bf16 0x4300 is 128 with a unit mantissa step, so 128 + u is exact for u < 128; u = code +
// 16 * (1 - high) and subtracting 144 yields the signed 5-bit value code - 16 * high.
__device__ __forceinline__ unsigned q5_ada_decode_pair(unsigned byte, unsigned inv_high2) {
    const unsigned bits = 0x43004300u | (byte & 0x0fu) | ((byte & 0xf0u) << 12) |
                          ((inv_high2 & 1u) << 4) | ((inv_high2 & 2u) << 19);
    const unsigned bias = 0x43104310u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// One Q3 pair -> exact bf16x2 integers: `low` carries the pair's two low bits at bits 0-1 and 16-17,
// `high` its high bits at bits 2 and 18 (u = low + 4 * high), and (128 + u) - 132 is the signed value.
__device__ __forceinline__ unsigned q3_ada_decode_pair(unsigned low, unsigned high) {
    const unsigned bits = 0x43004300u | (low & 0x00030003u) | (high & 0x00040004u);
    const unsigned bias = 0x43044304u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// One Q3Odd pair -> exact bf16x2 odd integers: `low` carries the pair's two low bits at bits 1-2 and
// 17-18, `high` its high bits at bits 3 and 19, and (128 + 2u) - 135 = 2u - 7 is the odd code.
__device__ __forceinline__ unsigned q3_odd_ada_decode_pair(unsigned low, unsigned high) {
    const unsigned bits = 0x43004300u | (low & 0x00060006u) | (high & 0x00080008u);
    const unsigned bias = 0x43074307u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// Two signed Q4 codes of one byte -> exact bf16x2 integers: flipping the sign bit gives the offset
// value u = code + 8, and (128 + u) - 136 is the two's-complement code.
__device__ __forceinline__ unsigned q4_ada_decode_pair(unsigned byte) {
    const unsigned bits =
        0x43004300u | (((byte & 0x0fu) | ((byte & 0xf0u) << 12)) ^ 0x00080008u);
    const unsigned bias = 0x43084308u;
    const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                                         *reinterpret_cast<const __nv_bfloat162*>(&bias));
    return *reinterpret_cast<const unsigned*>(&value);
}

// research33: an epilogue with kConvRows also gets preload(first_weight_row, tid) right after the pipeline prologue
// (cp.async copies that join the first in-loop group), stage(local_row, token, dot) beside each element call, and
// rows_done(first_weight_row, tid, threads) once every element of the CTA is staged.
template <class E>
concept AdaConvRowsEpilogue = requires { E::kConvRows; };

// research35: any epilogue with preload(first_weight_row, tid) gets it right after the pipeline prologue.
template <class E>
concept AdaPreloadEpilogue = requires(const E& e) { e.preload(0, 0); };

// research35: kRowsHook epilogues get element(weight_row, local_row, token, dot) for every live element and then
// rows_done(block, rows, tokens, tid, threads) from every thread (single launches only).
template <class E>
concept AdaRowsHookEpilogue = requires { E::kRowsHook; };

// Epilogue::kPairedHalves == false: epilogue(weight_row, token, dot) for every CTA row and live
// token. true: epilogue(output_row, token, gate_dot, up_dot) for the CTA's AdaPairedHalfRows pairs.
// One CTA's work; `block` is the CTA index within this problem's grid.
// super != nullptr (LS8): `scales` holds one byte j per group, the group scale is 2^(-j/32) and the
// row's FP32 super-scale multiplies the reduced dot product.
template <class S, int K, class RowMap, class Epilogue>
__device__ __forceinline__ void ada_small_t_mma_cta(
    const __nv_bfloat16* __restrict__ x, std::int32_t x_ld, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    std::int32_t tokens, const Epilogue& epilogue, int block,
    const float* __restrict__ super = nullptr) {
    constexpr int kGroupK = 64;
    static_assert(K % (kGroupK * S::kG) == 0);
    // Venturi's dimension-independent route uses K=0 and a validated runtime K.
    const int kGroups     = (K == 0 ? x_ld : K) / kGroupK;
    const int kPipeStages = kGroups / S::kG;
    constexpr int R           = S::kRowTiles;
    constexpr int kRows       = S::kRowsPerCta;

    extern __shared__ __align__(16) std::uint8_t smem[];

    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int q        = lane & 3;
    const int group_id = lane >> 2;

    unsigned long long weight_policy = 0;
    if constexpr (S::kEvictFirst) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(weight_policy));
    }
    const auto fill16 = [&](void* dst, const void* src) {
        if constexpr (S::kEvictFirst) {
            asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                         :
                         : "r"(smem_addr(dst)), "l"(src), "l"(weight_policy));
        } else {
            cp_async<16, Cache::cg>(dst, src);
        }
    };
    const auto issue = [&](int stage) {
        std::uint8_t* code_dst  = smem + (stage % S::kStages) * S::kStageBytes;
        std::uint8_t* high_dst  = code_dst + S::kCodeBytes;
        std::uint8_t* scale_dst = high_dst + S::kHighBytes;
        const int g0            = stage * S::kG;

        constexpr int kCodeChunks = S::kLowBytes * S::kG / 16;
        if constexpr (S::kPacked) {
            static_assert(S::kQ3, "packed tiles hold Q3 planes");
            const std::uint8_t* tile =
                codes + (static_cast<std::int64_t>(block) * kPipeStages + stage) * S::kTileBytes;
            for (int i = tid; i < kRows * kCodeChunks; i += S::kThreads) {
                fill16(code_dst + (i / kCodeChunks) * S::kCodeStride + 16 * (i % kCodeChunks), tile + 16 * i);
            }
            constexpr int kHighChunks = S::kG / 2;
            for (int i = tid; i < kRows * kHighChunks; i += S::kThreads) {
                fill16(high_dst + (i / kHighChunks) * S::kHighStride + 16 * (i % kHighChunks),
                       tile + S::kLowTileBytes + 16 * i);
            }
            for (int row = tid; row < kRows; row += S::kThreads) {
                const std::uint8_t* src = tile + S::kLowTileBytes + S::kHighTileBytes + row * S::kG;
                if constexpr (S::kG == 4) {
                    cp_async<4>(scale_dst + row * S::kScaleStride, src);
                } else if constexpr (S::kG == 8) {
                    cp_async<8>(scale_dst + row * S::kScaleStride, src);
                } else {
                    fill16(scale_dst + row * S::kScaleStride, src);
                }
            }
            return;
        }
        for (int i = tid; i < kRows * kCodeChunks; i += S::kThreads) {
            const int row = i / kCodeChunks;
            const int c   = i % kCodeChunks;
            const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
            fill16(code_dst + row * S::kCodeStride + 16 * c,
                   codes + (src_row * kGroups + g0) * S::kLowBytes + 16 * c);
        }
        if constexpr (S::kHasHigh) {
            constexpr int kHighChunks = S::kG / 2;
            for (int i = tid; i < kRows * kHighChunks; i += S::kThreads) {
                const int row = i / kHighChunks;
                const int c   = i % kHighChunks;
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                fill16(high_dst + row * S::kHighStride + 16 * c,
                       high + (src_row * kGroups + g0) * 8 + 16 * c);
            }
        }
        if (super != nullptr) {
            for (int row = tid; row < kRows; row += S::kThreads) {
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                const std::uint8_t* src    = scales + src_row * kGroups + g0;
                if constexpr (S::kG == 4) {
                    cp_async<4>(scale_dst + row * S::kScaleStride, src);
                } else if constexpr (S::kG == 8) {
                    cp_async<8>(scale_dst + row * S::kScaleStride, src);
                } else {
                    fill16(scale_dst + row * S::kScaleStride, src);
                }
            }
        } else if constexpr (S::kG == 4) {
            for (int row = tid; row < kRows; row += S::kThreads) {
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                cp_async<8>(scale_dst + row * S::kScaleStride,
                            scales + (src_row * kGroups + g0) * 2);
            }
        } else {
            constexpr int kScaleChunks = S::kG / 8;
            for (int i = tid; i < kRows * kScaleChunks; i += S::kThreads) {
                const int row = i / kScaleChunks;
                const int c   = i % kScaleChunks;
                const std::int64_t src_row = RowMap::weight_row(block, row, kRows);
                fill16(scale_dst + row * S::kScaleStride + 16 * c,
                       scales + (src_row * kGroups + g0) * 2 + 16 * c);
            }
        }
    };

    const bool lo_valid = group_id < tokens;
    const bool hi_valid = S::kTokenTiles == 2 && group_id + 8 < tokens;
    const __nv_bfloat16* x_lo_src = x + static_cast<std::int64_t>(group_id) * x_ld + 16 * q;
    const __nv_bfloat16* x_hi_src = x_lo_src + static_cast<std::int64_t>(8) * x_ld;

    struct XFragment {
        uint4 lo[2];
        uint4 hi[2];
    };
    constexpr int J = S::kGroupsPerWarp;
    const auto load_x = [&](int stage, XFragment (&dst)[J]) {
        const uint4 zero{0u, 0u, 0u, 0u};
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int k0 = (stage * S::kG + warp + j * S::kWarps) * kGroupK;
            const auto* lo = reinterpret_cast<const uint4*>(x_lo_src + k0);
            const auto* hi = reinterpret_cast<const uint4*>(x_hi_src + k0);
            dst[j].lo[0] = lo_valid ? __ldg(lo) : zero;
            dst[j].lo[1] = lo_valid ? __ldg(lo + 1) : zero;
            dst[j].hi[0] = hi_valid ? __ldg(hi) : zero;
            dst[j].hi[1] = hi_valid ? __ldg(hi + 1) : zero;
        }
    };

    float acc[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[r][i] = 0.0f; }
    }

    const auto compute = [&](int stage, const XFragment (&xf)[J]) {
        const std::uint8_t* code_s  = smem + (stage % S::kStages) * S::kStageBytes;
        const std::uint8_t* high_s  = code_s + S::kCodeBytes;
        const std::uint8_t* scale_s = high_s + S::kHighBytes;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int gi           = warp + j * S::kWarps;
            const unsigned a_lo[8] = {xf[j].lo[0].x, xf[j].lo[0].y, xf[j].lo[0].z, xf[j].lo[0].w,
                                      xf[j].lo[1].x, xf[j].lo[1].y, xf[j].lo[1].z, xf[j].lo[1].w};
            const unsigned a_hi[8] = {xf[j].hi[0].x, xf[j].hi[0].y, xf[j].hi[0].z, xf[j].hi[0].w,
                                      xf[j].hi[1].x, xf[j].hi[1].y, xf[j].hi[1].z, xf[j].hi[1].w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = 8 * r + group_id;
                uint2 word{0u, 0u};
                if constexpr (S::kQ3) {
                    word.x = *reinterpret_cast<const unsigned*>(code_s + row * S::kCodeStride +
                                                                gi * 16 + 4 * q);
                } else {
                    word = *reinterpret_cast<const uint2*>(code_s + row * S::kCodeStride + gi * 32 +
                                                           8 * q);
                }
                unsigned inv = 0;
                if constexpr (S::kQ3) {
                    // The chunk's high bits: pair firsts at bits 2-9, pair seconds at bits 18-25 (one
                    // higher for Q3Odd).
                    const unsigned hw = *reinterpret_cast<const unsigned short*>(
                        high_s + row * S::kHighStride + gi * 8 + 2 * q);
                    inv = ((hw & 0xffu) | ((hw & 0xff00u) << 8))
                          << (S::kFormat == AdaWeightFormat::Q3Odd ? 3 : 2);
                } else if constexpr (S::kHasHigh) {
                    const unsigned hw = *reinterpret_cast<const unsigned*>(
                        high_s + row * S::kHighStride + gi * 8 + 4 * (q >> 1));
                    inv = ~(hw >> (16 * (q & 1)));
                }
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const unsigned w     = s < 2 ? word.x : word.y;
                    const unsigned shift = 16u * static_cast<unsigned>(s & 1);
                    unsigned b0;
                    unsigned b1;
                    if constexpr (S::kFormat == AdaWeightFormat::Q3Odd) {
                        b0 = q3_odd_ada_decode_pair(s == 0 ? word.x << 1 : word.x >> (4 * s - 1),
                                                    inv >> (2 * s));
                        b1 = q3_odd_ada_decode_pair(word.x >> (4 * s + 1), inv >> (2 * s + 1));
                    } else if constexpr (S::kFormat == AdaWeightFormat::Q3) {
                        b0 = q3_ada_decode_pair(word.x >> (4 * s), inv >> (2 * s));
                        b1 = q3_ada_decode_pair(word.x >> (4 * s + 2), inv >> (2 * s + 1));
                    } else if constexpr (S::kHasHigh) {
                        b0 = q5_ada_decode_pair((w >> shift) & 0xffu, inv >> (4 * s));
                        b1 = q5_ada_decode_pair((w >> (shift + 8u)) & 0xffu, inv >> (4 * s + 2));
                    } else {
                        b0 = q4_ada_decode_pair((w >> shift) & 0xffu);
                        b1 = q4_ada_decode_pair((w >> (shift + 8u)) & 0xffu);
                    }
                    mma_bf16(d0, d1, d2, d3, a_lo[2 * s], a_hi[2 * s], a_lo[2 * s + 1],
                             a_hi[2 * s + 1], b0, b1);
                }
                const int srow = 8 * r + 2 * q;
                float s0, s1;
                if (super != nullptr) {
                    s0 = exp2f(-0.03125f * static_cast<float>(scale_s[srow * S::kScaleStride + gi]));
                    s1 = exp2f(-0.03125f *
                               static_cast<float>(scale_s[(srow + 1) * S::kScaleStride + gi]));
                } else {
                    s0 = __half2float(
                        *reinterpret_cast<const __half*>(scale_s + srow * S::kScaleStride + 2 * gi));
                    s1 = __half2float(*reinterpret_cast<const __half*>(
                        scale_s + (srow + 1) * S::kScaleStride + 2 * gi));
                }
                acc[r][0] = fmaf(s0, d0, acc[r][0]);
                acc[r][1] = fmaf(s1, d1, acc[r][1]);
                acc[r][2] = fmaf(s0, d2, acc[r][2]);
                acc[r][3] = fmaf(s1, d3, acc[r][3]);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < S::kStages - 1; ++s) {
        if (s < kPipeStages) { issue(s); }
        cp_commit();
    }
    if constexpr (AdaPreloadEpilogue<Epilogue>) { epilogue.preload(RowMap::weight_row(block, 0, kRows), tid); }
    XFragment current[J];
    load_x(0, current);
    for (int s = 0; s < kPipeStages; ++s) {
        cp_wait<S::kStages - 2>();
        __syncthreads();
        if (s + S::kStages - 1 < kPipeStages) { issue(s + S::kStages - 1); }
        cp_commit();
        XFragment next[J];
        if (s + 1 < kPipeStages) { load_x(s + 1, next); }
        compute(s, current);
#pragma unroll
        for (int j = 0; j < J; ++j) { current[j] = next[j]; }
    }
    cp_wait<0>();
    __syncthreads();

    auto* reduction = reinterpret_cast<float*>(smem);
    const auto slot = [](int w, int row, int token) {
        return (w * kRows + row) * S::kMaxTokens + token;
    };
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row                            = 8 * r + 2 * q;
        reduction[slot(warp, row, group_id)]     = acc[r][0];
        reduction[slot(warp, row + 1, group_id)] = acc[r][1];
        if constexpr (S::kTokenTiles == 2) {
            reduction[slot(warp, row, group_id + 8)]     = acc[r][2];
            reduction[slot(warp, row + 1, group_id + 8)] = acc[r][3];
        }
    }
    __syncthreads();

    const auto reduce = [&](int row, int token) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < S::kWarps; ++w) { sum += reduction[slot(w, row, token)]; }
        return sum;
    };
    if constexpr (Epilogue::kPairedHalves) {
        constexpr int kHalf = kRows / 2;
        static_assert(kRows % 2 == 0);
        for (int index = tid; index < kHalf * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kHalf;
            const int token = index / kHalf;
            if (token >= tokens) { continue; }
            float gate = reduce(row, token), up = reduce(row + kHalf, token);
            if (super != nullptr) {
                gate *= super[RowMap::weight_row(block, row, kRows)];
                up *= super[RowMap::weight_row(block, row + kHalf, kRows)];
            }
            epilogue(block * kHalf + row, token, gate, up);
        }
    } else if constexpr (AdaRowsHookEpilogue<Epilogue>) {
        for (int index = tid; index < kRows * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kRows;
            const int token = index / kRows;
            if (token >= tokens) { continue; }
            const int weight_row = RowMap::weight_row(block, row, kRows);
            float value          = reduce(row, token);
            if (super != nullptr) { value *= super[weight_row]; }
            epilogue.element(weight_row, row, token, value);
        }
        epilogue.rows_done(block, kRows, tokens, tid, S::kThreads);
    } else if constexpr (AdaConvRowsEpilogue<Epilogue>) {
        for (int index = tid; index < kRows * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kRows;
            const int token = index / kRows;
            if (token >= tokens) { continue; }
            const int weight_row = RowMap::weight_row(block, row, kRows);
            float value          = reduce(row, token);
            if (super != nullptr) { value *= super[weight_row]; }
            epilogue(weight_row, token, value);
            epilogue.stage(row, token, value);
        }
        epilogue.rows_done(RowMap::weight_row(block, 0, kRows), tid, S::kThreads);
    } else {
        for (int index = tid; index < kRows * S::kMaxTokens; index += S::kThreads) {
            const int row   = index % kRows;
            const int token = index / kRows;
            if (token >= tokens) { continue; }
            const int weight_row = RowMap::weight_row(block, row, kRows);
            float value          = reduce(row, token);
            if (super != nullptr) { value *= super[weight_row]; }
            epilogue(weight_row, token, value);
        }
    }
}

template <class S, int K, class RowMap, class Epilogue>
__launch_bounds__(S::kThreads) __global__ void ada_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t x_ld, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    std::int32_t tokens, Epilogue epilogue, const float* __restrict__ super) {
    ada_small_t_mma_cta<S, K, RowMap>(x, x_ld, codes, high, scales, tokens, epilogue,
                                      static_cast<int>(blockIdx.x), super);
}

// The same CTA under __launch_bounds__(threads, S::kMinBlocks) (packed Q3 schedules; see kMinBlocks).
template <class S, int K, class RowMap, class Epilogue>
__launch_bounds__(S::kThreads, S::kMinBlocks) __global__ void ada_small_t_mma_min_blocks_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t x_ld, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    std::int32_t tokens, Epilogue epilogue, const float* __restrict__ super) {
    ada_small_t_mma_cta<S, K, RowMap>(x, x_ld, codes, high, scales, tokens, epilogue,
                                      static_cast<int>(blockIdx.x), super);
}

// One weight matrix of a dual launch.
template <class Epilogue>
struct AdaSmallTOperand {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    Epilogue epilogue;
    const float* super = nullptr;  // LS8 (see ada_small_t_mma_cta)
};

// Two independent problems over the same x in one grid: CTAs [0, blocks_a) run A, the rest run B,
// each exactly as its own launch would (same per-CTA code, so the same bits). One launch instead of
// two removes a grid drain/fill between them, and the shorter problem's CTAs fill the longer one's
// tail.
template <class SA, int KA, class MapA, class EpiA, class SB, int KB, class MapB, class EpiB>
__launch_bounds__(SA::kThreads) __global__ void ada_small_t_mma_dual_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t x_ld, std::int32_t tokens,
    AdaSmallTOperand<EpiA> a, AdaSmallTOperand<EpiB> b, int blocks_a) {
    static_assert(SA::kThreads == SB::kThreads, "dual launch needs one block size");
    const int block = static_cast<int>(blockIdx.x);
    if (block < blocks_a) {
        ada_small_t_mma_cta<SA, KA, MapA>(x, x_ld, a.codes, a.high, a.scales, tokens, a.epilogue,
                                          block, a.super);
    } else {
        ada_small_t_mma_cta<SB, KB, MapB>(x, x_ld, b.codes, b.high, b.scales, tokens, b.epilogue,
                                          block - blocks_a, b.super);
    }
}

// Launches one CTA per 8 * RowTiles weight rows; `rows` must be a multiple of that.
template <class S, int K, class RowMap = AdaContiguousRows, class Epilogue>
void ada_small_t_mma_launch(const __nv_bfloat16* x, std::int32_t x_ld, const std::uint8_t* codes,
                            const std::uint8_t* high, const std::uint8_t* scales, std::int32_t rows,
                            std::int32_t tokens, Epilogue epilogue, cudaStream_t stream,
                            const float* super = nullptr) {
    if constexpr (S::kMinBlocks > 1) {
        static const bool configured_min_blocks = [] {
            CUDA_CHECK(cudaFuncSetAttribute(ada_small_t_mma_min_blocks_kernel<S, K, RowMap, Epilogue>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmemBytes));
            return true;
        }();
        (void)configured_min_blocks;
        ada_small_t_mma_min_blocks_kernel<S, K, RowMap, Epilogue>
            <<<rows / S::kRowsPerCta, S::kThreads, S::kSmemBytes, stream>>>(x, x_ld, codes, high, scales, tokens,
                                                                            epilogue, super);
        return;
    }
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(ada_small_t_mma_kernel<S, K, RowMap, Epilogue>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        S::kSmemBytes));
        return true;
    }();
    (void)configured;
    ada_small_t_mma_kernel<S, K, RowMap, Epilogue>
        <<<rows / S::kRowsPerCta, S::kThreads, S::kSmemBytes, stream>>>(x, x_ld, codes, high,
                                                                        scales, tokens, epilogue,
                                                                        super);
}

// Dual launch of A (rows_a rows) and B (rows_b rows) over one x; see ada_small_t_mma_dual_kernel.
template <class SA, int KA, class SB, int KB, class MapA = AdaContiguousRows,
          class MapB = AdaContiguousRows, class EpiA, class EpiB>
void ada_small_t_mma_dual_launch(const __nv_bfloat16* x, std::int32_t x_ld, std::int32_t tokens,
                                 const AdaSmallTOperand<EpiA>& a, std::int32_t rows_a,
                                 const AdaSmallTOperand<EpiB>& b, std::int32_t rows_b,
                                 cudaStream_t stream) {
    constexpr int kSmem = std::max(SA::kSmemBytes, SB::kSmemBytes);
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(
            ada_small_t_mma_dual_kernel<SA, KA, MapA, EpiA, SB, KB, MapB, EpiB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        return true;
    }();
    (void)configured;
    const int blocks_a = rows_a / SA::kRowsPerCta;
    const int blocks_b = rows_b / SB::kRowsPerCta;
    ada_small_t_mma_dual_kernel<SA, KA, MapA, EpiA, SB, KB, MapB, EpiB>
        <<<blocks_a + blocks_b, SA::kThreads, kSmem, stream>>>(x, x_ld, tokens, a, b, blocks_a);
}

} // namespace ninfer::ops::detail
