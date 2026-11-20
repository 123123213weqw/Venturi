// W5A16 wmma experiment (Volta tensor cores): q5_g64_fp16 weights are
// unpacked to FP16 tiles in shared memory (scale folded in before the half
// cast) and multiplied against FP16 activations via nvcuda::wmma 16x16x16.
// HBM traffic stays at the q5 plane size; the math rides the tensor cores.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

namespace q5mma {

// One block: M16 (token tile) x N128 (8 warps x 16 rows), K loop in k64
// scale groups. Shared: one A tile (16x64 fp16) + per-warp B tiles (16x64).
__global__ __launch_bounds__(256) void q5_wmma_kernel(
    const __half* x, const unsigned char* codes, const unsigned char* high,
    const __half* scales, __half* out,
    int tokens, int n_rows, int k, int padded) {
    __shared__ __align__(32) __half a_tile[16][72];
    __shared__ __align__(32) __half b_tile[8][16][72];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int m_base = blockIdx.y * 16;
    const int n_base = blockIdx.x * 128 + warp * 16;
    const bool warp_active = n_base < n_rows;   // never early-return: all warps
                                                // cooperate on the A tile staging

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int k_groups = k >> 6;
    for (int kg = 0; kg < k_groups; ++kg) {
        const int kbase = kg << 6;
        // Cooperatively stage the A tile (16 tokens x 64 k), zero padding.
        for (int idx = threadIdx.x; idx < 16 * 8; idx += 256) {
            const int row = idx >> 3, col8 = (idx & 7) << 3;
            const int m = m_base + row;
            if (m < tokens && (reinterpret_cast<uintptr_t>(x) & 15u)==0)
                *reinterpret_cast<uint4*>(&a_tile[row][col8]) =
                    *reinterpret_cast<const uint4*>(&x[(int64_t)m * k + kbase + col8]);
            else if (m < tokens) {
                // A contiguous FP16 view may start at a 2-byte storage offset.
                // Keep its declared arithmetic; only the load width changes.
                #pragma unroll
                for (int j=0;j<8;++j) a_tile[row][col8+j]=x[(int64_t)m*k+kbase+col8+j];
            } else
                *reinterpret_cast<uint4*>(&a_tile[row][col8]) = make_uint4(0, 0, 0, 0);
        }
        // Each warp unpacks its 16 weight rows x 64 k into fp16 with the
        // group scale folded in (scales are per row per k64 group).
        if (warp_active) {
            for (int idx = lane; idx < 16 * 8; idx += 32) {
                const int row = idx >> 3, col8 = (idx & 7) << 3;
                const int n = n_base + row;
                if (n < n_rows) {
                    const unsigned char* crow = codes + (int64_t)n * (padded >> 1);
                    const unsigned char* hrow = high + (int64_t)n * (padded >> 3);
                    const int c0 = kbase + col8;
                    const auto* packed=crow+(c0>>1);
                    const uint32_t cbytes=(reinterpret_cast<uintptr_t>(packed)&3u)==0
                        ?*reinterpret_cast<const uint32_t*>(packed)
                        :uint32_t(packed[0])|(uint32_t(packed[1])<<8)|(uint32_t(packed[2])<<16)|(uint32_t(packed[3])<<24);
                    const unsigned int hbits =
                        hrow[(c0 >> 6) << 3 | ((c0 >> 3) & 7)] >> (c0 & 7);
                    const __half2 ws2 = __half2half2(scales[(int64_t)n * (padded >> 6) + kg]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const unsigned int cb = (cbytes >> (8 * j)) & 0xFFu;
                        const unsigned int hb = (hbits >> (2 * j)) & 3u;
                        const int raw0 = static_cast<int>((cb & 15u) | ((hb & 1u) << 4));
                        const int raw1 = static_cast<int>(((cb >> 4) & 15u) | ((hb >> 1) << 4));
                        *reinterpret_cast<__half2*>(&b_tile[warp][row][col8 + 2 * j]) = __hmul2(
                            __halves2half2(
                                __float2half_rn(static_cast<float>((raw0 ^ 16) - 16)),
                                __float2half_rn(static_cast<float>((raw1 ^ 16) - 16))),
                            ws2);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        *reinterpret_cast<__half2*>(&b_tile[warp][row][col8 + 2 * j]) =
                            __half2half2(__float2half(0.f));
                }
            }
        }
        __syncthreads();
        if (warp_active) {
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_tile[0][step << 4], 72);
                wmma::load_matrix_sync(b_frag, &b_tile[warp][0][step << 4], 72);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        __syncthreads();
    }
    if (!warp_active) return;
    // Store through shared to clip token rows below `tokens`.
    __shared__ __align__(32) float c_stage[8][16][16];
    wmma::store_matrix_sync(&c_stage[warp][0][0], c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int idx = lane; idx < 16 * 16; idx += 32) {
        const int row = idx >> 4, col = idx & 15;
        const int m = m_base + row, n = n_base + col;
        if (m < tokens && n < n_rows)
            out[(int64_t)m * n_rows + n] = __float2half(c_stage[warp][row][col]);
    }
}
}  // namespace q5mma

extern "C" int venturi_q5_hmma_run(
    const void* x, const void* codes, const void* high, const void* scales, void* out,
    int tokens, int rows, int k, int padded, void* stream) {
    try {
        require(x && codes && high && scales && out && tokens>0 && tokens<=65535 && rows>0 && k>0 && k%64==0 && padded>=k && padded%128==0,
            "invalid q5 HMMA input/geometry");
        const dim3 grid((rows+127)/128,(tokens+15)/16);
        q5mma::q5_wmma_kernel<<<grid,256,0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __half*>(x),static_cast<const unsigned char*>(codes),static_cast<const unsigned char*>(high),
            static_cast<const __half*>(scales),static_cast<__half*>(out),tokens,rows,k,padded);
        check(cudaGetLastError());return 0;
    } catch(const std::exception& e) { error_text=e.what();return -1; }
}
