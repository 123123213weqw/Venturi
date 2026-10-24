#include "ops/linear_add/q4/q4_linear_add_dispatch.h"

#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/ada_small_t_mma.cuh"
#include "ops/common/norm_handoff.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ResidualEpilogue {
    __device__ __forceinline__ void operator()(__nv_bfloat16* destination, float value) const {
        *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
    }
};

struct GemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16*, int row,
                                               float value) const {
        static_assert(!SplitOutput);
        ResidualEpilogue{}(out + row, value);
    }
};

struct AdaResidualEpilogue {
    static constexpr bool kPairedHalves = false;

    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        __nv_bfloat16* slot = out + static_cast<std::int64_t>(token) * out_ld + row;
        *slot               = __float2bfloat16(value + __bfloat162float(*slot));
    }
};

struct KSplitResidualEpilogue {
    __nv_bfloat16* residual;
    std::int32_t tokens;

    template <int Capacity>
    __device__ __forceinline__ void store(int row, int col, float4 value) const {
        if (col < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row, value.x);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row + 8, value.z);
        }
        if (col + 1 < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row, value.y);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row + 8,
                               value.w);
        }
    }
};

using GemvR1W8 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 6144 / 64, 1>;
using GemvR1W8K17408 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 0, 1>;
using MmaR32C32  = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64  = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C128 = Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

template <int Capacity, int K = 6144>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<5120, K>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    q4_ksplit_mma_kernel<Geometry, (Capacity + 7) / 8 * 8, Capacity, KSplitResidualEpilogue,
                         Q4KSplitIdentityRows, true>
        <<<5120 / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output,
            KSplitResidualEpilogue{output, x.ne[1]}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class S, int K, class Epilogue>
void launch_ada_q3(const Tensor& x, const Weight& w, const Epilogue& epilogue, cudaStream_t stream) {
    static_assert(5120 % S::kRowsPerCta == 0);
    ada_small_t_mma_launch<S, K>(static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16)),
                                 static_cast<const std::uint8_t*>(w.q3_low),
                                 static_cast<const std::uint8_t*>(w.q3_high), ada_scales(w), 5120,
                                 x.ne[1], epilogue, stream, ada_super(w));
    CUDA_CHECK(cudaGetLastError());
}

template <int K, int TokenTiles, class Epilogue>
void launch_ada_with(const Tensor& x, const Weight& w, const Epilogue& epilogue, cudaStream_t stream) {
    // K = 6144 (GDN / attention output): 16 rows per CTA (320 CTAs) beat 40 (128 CTAs) by 2.9% in
    // kernels/mma/bench_sweep2.cu; the warp split is unchanged, so the result is bit-identical.
    using S = Q4AdaSmallTSchedule<K == 6144 ? 2 : 5, 8, TokenTiles, 8, 2>;
    static_assert(5120 % S::kRowsPerCta == 0);
    if (w.padded_shape[1] != K) {
        throw std::invalid_argument("q4 linear_add ada small-T: padded K differs");
    }
    if (w.q3_low != nullptr) {
        // Q3 shadow planes (NINFER_Q3_SHADOW): 16 groups per stage, 42.1 us at K = 17408, T = 16
        // against 50.1 us for the Q4 codes (kernels/q3/bench_q3_sweep.cu); the same bits.
        constexpr int kG = K == 17408 ? 16 : 8;
        // NINFER_Q3_TILED: the planes are packed per (40-row CTA, 8-group stage) tile, three stages (40.9 vs
        // 42.0 us for G16 S2 packed); warp w still sums the groups w mod 8 in order, so the bits are unchanged.
        if (w.q3_tile != 0) {
            if (K != 17408 || w.q3_tile != 2 || ada_super(w) == nullptr) {
                throw std::invalid_argument("q4 linear_add ada small-T: Q3 tiles packed for another route");
            }
            if (w.q3_odd) {
                launch_ada_q3<AdaSmallTSchedule<AdaWeightFormat::Q3Odd, 5, 8, TokenTiles, 8, 3, true, true>, K>(
                    x, w, epilogue, stream);
            } else {
                launch_ada_q3<AdaSmallTSchedule<AdaWeightFormat::Q3, 5, 8, TokenTiles, 8, 3, true, true>, K>(
                    x, w, epilogue, stream);
            }
            return;
        }
        if (w.q3_odd) {
            launch_ada_q3<Q3OddAdaSmallTSchedule<5, 8, TokenTiles, kG, 2>, K>(x, w, epilogue, stream);
        } else {
            launch_ada_q3<Q3AdaSmallTSchedule<5, 8, TokenTiles, kG, 2>, K>(x, w, epilogue, stream);
        }
        return;
    }
    ada_small_t_mma_launch<S, K>(static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16)),
                                 static_cast<const std::uint8_t*>(w.qdata), nullptr,
                                 ada_scales(w), 5120, x.ne[1], epilogue, stream, ada_super(w));
    CUDA_CHECK(cudaGetLastError());
}

template <int K, int TokenTiles>
void launch_ada(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    launch_ada_with<K, TokenTiles>(
        x, w,
        AdaResidualEpilogue{static_cast<__nv_bfloat16*>(residual.data),
                            static_cast<std::int32_t>(residual.nb[1] / sizeof(__nv_bfloat16))},
        stream);
}

} // namespace

void q4_linear_add_ada_handoff_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                      const NormHandoff& handoff, cudaStream_t stream) {
    const AdaNormHandoffResidualEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual.data),
        static_cast<std::int32_t>(residual.nb[1] / sizeof(__nv_bfloat16)),
        static_cast<const __nv_bfloat16*>(handoff.norm),
        static_cast<__nv_bfloat16*>(handoff.xs),
        handoff.d,
        handoff.slot,
        handoff.zero,
        handoff.zero_count};
    const std::int32_t t = x.ne[1];
    if (w.n != 5120 || t < 2 || t > 16 || handoff.d != 5120) {
        throw std::invalid_argument("q4 linear_add norm handoff: unsupported shape");
    }
    if (w.k == 17408) {
        if (t <= 8) {
            launch_ada_with<17408, 1>(x, w, epilogue, stream);
        } else {
            launch_ada_with<17408, 2>(x, w, epilogue, stream);
        }
    } else if (w.k == 6144) {
        if (t <= 8) {
            launch_ada_with<6144, 1>(x, w, epilogue, stream);
        } else {
            launch_ada_with<6144, 2>(x, w, epilogue, stream);
        }
    } else {
        throw std::invalid_argument("q4 linear_add norm handoff: unsupported K");
    }
}

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows != 5120 || (k != 6144 && k != 17408) || tokens <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
    }
    if (k == 17408) {
        if (tokens == 1) return launch_q4_gemv<GemvR1W8K17408, GemvResidualEpilogue>;
        if (tokens <= 8) return launch_ada<17408, 1>;
        if (tokens <= 16) return launch_ada<17408, 2>;
        if (tokens <= 24) return launch_ksplit<24, 17408>;
        if (tokens <= 32) return launch_ksplit<32, 17408>;
        if (tokens <= 96) return launch_q4_mma<MmaR32C32, ResidualEpilogue>;
        if (tokens <= 192) return launch_q4_mma<MmaR32C64, ResidualEpilogue>;
        return launch_q4_mma<MmaR64C128, ResidualEpilogue>;
    }
    if (tokens == 1) return launch_q4_gemv<GemvR1W8, GemvResidualEpilogue>;
    if (tokens <= 8) return launch_ada<6144, 1>;
    if (tokens <= 16) return launch_ada<6144, 2>;
    if (tokens <= 4) return launch_ksplit<4>;
    if (tokens <= 8) return launch_ksplit<8>;
    if (tokens <= 16) return launch_ksplit<16>;
    if (tokens <= 24) return launch_ksplit<24>;
    if (tokens <= 32) return launch_ksplit<32>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32, ResidualEpilogue>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64, ResidualEpilogue>;
    return launch_q4_mma<MmaR64C128, ResidualEpilogue>;
}

} // namespace ninfer::ops::detail
