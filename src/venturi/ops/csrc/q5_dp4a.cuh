// Venturi q5_g64_fp16 fused W5A8 DP4A kernels (A8 activations like the q8
// family, weights unpacked from packed nibbles + 1-bit high plane in
// registers). Included inside namespace vq5 by blocks.cu.
//
// Mirrors the vqf q8 kernels' proven structure: warp/row-block geometry,
// vector int4 transport, one integer dot per scale group. Weight traffic per
// 32 columns is 16B codes + 4B high (vs 32B for q8), with the same dp4a
// integer path after in-register unpacking. Scale groups are 64 wide (fp16).
#pragma once

namespace vq5 {

struct Group {
    struct Item {
        const unsigned char* codes;
        const unsigned char* high;
        const __half* scales;
        __half* out;
        int n, stride;
    } item[3]{};
    int count = 0;
};

template <int Rows>
__device__ __forceinline__ typename Group::Item select(Group g, int& row) {
    int block = blockIdx.x;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (i >= g.count) break;
        const int blocks = (g.item[i].n + Rows - 1) / Rows;
        if (block < blocks) {
            row = block * Rows + threadIdx.x / 32;
            return g.item[i];
        }
        block -= blocks;
    }
    row = 0;
    return {};
}

// Unpack four q5 codes (two bytes of nibbles + four high bits) into one int32
// holding four sign-extended int8 lanes. (q ^ 16) - 16 maps 0..31 to the same
// signed value as q>=16 ? q-32 : q in two instructions.
__device__ __forceinline__ int q5_quad(unsigned int cb, unsigned int hb4) {
    int b0 = (cb & 15u) | ((hb4 & 1u) << 4);
    int b1 = ((cb >> 4) & 15u) | (((hb4 >> 1) & 1u) << 4);
    int b2 = ((cb >> 8) & 15u) | (((hb4 >> 2) & 1u) << 4);
    int b3 = ((cb >> 12) & 15u) | (((hb4 >> 3) & 1u) << 4);
    b0 = (b0 ^ 16) - 16;
    b1 = (b1 ^ 16) - 16;
    b2 = (b2 ^ 16) - 16;
    b3 = (b3 ^ 16) - 16;
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | (b3 << 24);
}

__device__ __forceinline__ int q5_dot32(
    const unsigned char* codes_row, const unsigned char* high_row, int col,
    const std::int8_t* activations) {
    // 32 columns: 16B nibbles, 4B high bits, aligned reads.
    const uint4 c = *reinterpret_cast<const uint4*>(codes_row + (col >> 1));
    const unsigned int hb =
        *reinterpret_cast<const unsigned int*>(high_row + ((col >> 6) << 3) + ((col & 63) >> 5) * 4);
    // Each 32-bit register of c holds eight codes (two bytes per quad); the
    // high-plane bits pair per weight index across the full 32-bit window.
    const int w0 = q5_quad(c.x & 0xFFFFu, hb);
    const int w1 = q5_quad(c.x >> 16, hb >> 4);
    const int w2 = q5_quad(c.y & 0xFFFFu, hb >> 8);
    const int w3 = q5_quad(c.y >> 16, hb >> 12);
    const int w4 = q5_quad(c.z & 0xFFFFu, hb >> 16);
    const int w5 = q5_quad(c.z >> 16, hb >> 20);
    const int w6 = q5_quad(c.w & 0xFFFFu, hb >> 24);
    const int w7 = q5_quad(c.w >> 16, hb >> 28);
    const int4* ap = reinterpret_cast<const int4*>(activations);
    const int4 a0 = ap[0];
    const int4 a1 = ap[1];
    int d0 = __dp4a(w0, a0.x, 0), d1 = __dp4a(w1, a0.y, 0);
    d0 = __dp4a(w2, a0.z, d0);
    d1 = __dp4a(w3, a0.w, d1);
    int d2 = __dp4a(w4, a1.x, 0), d3 = __dp4a(w5, a1.y, 0);
    d2 = __dp4a(w6, a1.z, d2);
    d3 = __dp4a(w7, a1.w, d3);
    return d0 + d1 + d2 + d3;
}

template <int Tokens, int Rows>
__global__ __launch_bounds__(Rows * 32, 1) void global_dot(
    const std::int8_t* activations, const float* activation_scales, Group group,
    int tokens, int k, int padded, int groups) {
    int row;
    const auto plan = select<Rows>(group, row);
    const int n = plan.n;
    const auto* weights = plan.codes;
    const auto* high = plan.high;
    const auto* weight_scales = plan.scales;
    auto* out = plan.out;
    const int lane = threadIdx.x & 31;
    const int token = blockIdx.y * Tokens;
    if (row >= n) return;
    const auto* crow = weights + static_cast<std::int64_t>(row) * (padded >> 1);
    const auto* hrow = high + static_cast<std::int64_t>(row) * (padded >> 3);
    float sum[Tokens]{};
    for (int col = lane * 32; col < k; col += 1024) {
        const float ws = __half2float(
            weight_scales[static_cast<std::int64_t>(row) * (padded >> 6) + (col >> 6)]);
        #pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (token + t < tokens) {
                const auto group_index =
                    static_cast<std::int64_t>(token + t) * groups + (col >> 5);
                sum[t] = __fmaf_rn(
                    static_cast<float>(q5_dot32(crow, hrow, col, activations + group_index * 32)),
                    __fmul_rn(ws, activation_scales[group_index]), sum[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        for (int delta = 16; delta; delta >>= 1)
            sum[t] += __shfl_down_sync(0xffffffff, sum[t], delta);
        if (lane == 0 && token + t < tokens)
            out[static_cast<std::int64_t>(token + t) * plan.stride + row] =
                __float2half_rn(sum[t]);
    }
}

template <int Tokens, int Rows>
__global__ __launch_bounds__(Rows * 32, 1) void tiled_dot(
    const std::int8_t* activations, const float* activation_scales, Group group,
    int tokens, int k, int padded, int groups) {
    int row;
    const auto plan = select<Rows>(group, row);
    const int n = plan.n;
    const auto* weights = plan.codes;
    const auto* high = plan.high;
    const auto* weight_scales = plan.scales;
    auto* out = plan.out;
    const int lane = threadIdx.x & 31;
    const int token = blockIdx.y * Tokens;
    extern __shared__ __align__(16) char shared[];
    constexpr int all = Tokens * 32;
    // Layout: activations (32B per group, two int4) then fp32 scales.
    auto* acts = reinterpret_cast<std::int8_t*>(shared);
    auto* shared_scales = reinterpret_cast<float*>(acts + all * 32);
    const auto* crow = weights + static_cast<std::int64_t>(row) * (padded >> 1);
    const auto* hrow = high + static_cast<std::int64_t>(row) * (padded >> 3);
    float sum[Tokens]{};
    for (int base = 0; base < k; base += 1024) {
        for (int i = threadIdx.x; i < all; i += blockDim.x) {
            const int t = i / 32, g = base / 32 + i % 32;
            if (token + t < tokens && g < groups) {
                const auto group_index = static_cast<std::int64_t>(token + t) * groups + g;
                reinterpret_cast<int4*>(acts + i * 32)[0] =
                    reinterpret_cast<const int4*>(activations + group_index * 32)[0];
                reinterpret_cast<int4*>(acts + i * 32)[1] =
                    reinterpret_cast<const int4*>(activations + group_index * 32)[1];
                shared_scales[i] = activation_scales[group_index];
            } else {
                *reinterpret_cast<int4*>(acts + i * 32) = make_int4(0, 0, 0, 0);
                *(reinterpret_cast<int4*>(acts + i * 32) + 1) = make_int4(0, 0, 0, 0);
                shared_scales[i] = 0.0f;
            }
        }
        __syncthreads();
        const int col = base + lane * 32;
        if (row < n && col < k) {
            const float ws = __half2float(
                weight_scales[static_cast<std::int64_t>(row) * (padded >> 6) + (col >> 6)]);
            #pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (token + t < tokens) {
                    sum[t] = __fmaf_rn(
                        static_cast<float>(q5_dot32(
                            crow, hrow, col, acts + (t * 32 + lane) * 32)),
                        __fmul_rn(ws, shared_scales[t * 32 + lane]), sum[t]);
                }
            }
        }
        __syncthreads();
    }
    if (row >= n) return;
    #pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        for (int delta = 16; delta; delta >>= 1)
            sum[t] += __shfl_down_sync(0xffffffff, sum[t], delta);
        if (lane == 0 && token + t < tokens)
            out[static_cast<std::int64_t>(token + t) * plan.stride + row] =
                __float2half_rn(sum[t]);
    }
}

inline void run(const void* w, Group g, int tokens, int k, int padded, cudaStream_t stream) {
    int maxn = 0;
    for (int i = 0; i < g.count; ++i) maxn = std::max(maxn, g.item[i].n);
    const int groups = (k + 31) / 32;
    auto* codes = static_cast<const std::int8_t*>(w);
    auto* scales = reinterpret_cast<const float*>(codes + static_cast<std::uint64_t>(tokens) * groups * 32);
    if (tokens == 1) {
        global_dot<1, 4><<<(static_cast<std::uint64_t>(g.item[0].n) + 3) / 4, 128, 0, stream>>>(
            codes, scales, g, tokens, k, padded, groups);
    } else if (tokens == 2) {
        global_dot<2, 4><<<((static_cast<std::uint64_t>(g.item[0].n) + 3) / 4), 128, 0, stream>>>(
            codes, scales, g, tokens, k, padded, groups);
    } else {
        std::uint64_t blocks = 0;
        for (int i = 0; i < g.count; ++i)
            blocks += (g.item[i].n + 15) / 16;
        const int tile = tokens > 8 ? 16 : 8;
        const dim3 grid(static_cast<unsigned>(blocks), (tokens + tile - 1) / tile);
        const auto dynamic = static_cast<std::size_t>(tile) * 32 * 36;
        if (tile == 16)
            tiled_dot<16, 16><<<grid, 16 * 32, dynamic, stream>>>(
                codes, scales, g, tokens, k, padded, groups);
        else
            tiled_dot<8, 16><<<grid, 16 * 32, dynamic, stream>>>(
                codes, scales, g, tokens, k, padded, groups);
    }
}

}  // namespace vq5

extern "C" int venturi_q5_dp4a_fused_run(
    const void* workspace, std::uint64_t available,
    const void* codes, const void* high, const void* scales, void* out,
    int rows, int stride, int tokens, int k, int padded, void* stream_ptr) {
    try {
        const auto needed = vdq::bytes(tokens, k, 1);
        if (VENTURI_BLOCKS_ARCH != 70 || !workspace || available < needed || !codes ||
            !high || !scales || !out || rows <= 0 || rows > INT32_MAX - 16 || stride < rows ||
            tokens <= 0 || static_cast<std::uint64_t>(tokens - 1) * stride + rows >= INT32_MAX ||
            padded != ((static_cast<std::uint64_t>(k) + 127) / 128) * 128 ||
            reinterpret_cast<std::uintptr_t>(workspace) % 16 ||
            reinterpret_cast<std::uintptr_t>(codes) % 16 ||
            reinterpret_cast<std::uintptr_t>(high) % 4)
            throw std::invalid_argument("invalid fused W5A8 input/geometry");
        vq5::Group g{};
        g.count = 1;
        g.item[0] = {static_cast<const unsigned char*>(codes),
                     static_cast<const unsigned char*>(high),
                     static_cast<const __half*>(scales), static_cast<__half*>(out),
                     rows, stride};
        vq5::run(workspace, g, tokens, k, padded, static_cast<cudaStream_t>(stream_ptr));
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& e) {
        error_text = e.what();
        return -1;
    }
}
