// Venturi C ABI for caller-owned tensors and streams. No PyTorch C++ ABI dependency.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
thread_local std::string last_error;
}

#define CUDA_CHECK(expression) check(expression)
#include "third_party/ninfer/ada_small_t_mma.cuh"
#undef CUDA_CHECK

namespace {
using namespace ninfer::ops::detail;

struct Bf16Store {
    static constexpr bool kPairedHalves = false;
    __nv_bfloat16* out;
    int n;
    __device__ void operator()(int row, int token, float value) const {
        out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16_rn(value);
    }
};

template<int K, int TokenTiles>
using Schedule = Q4AdaSmallTSchedule<5, 8, TokenTiles, 8, 2>;

// Runtime K, no model-specific N/K table. A CTA processes 32 rows and 16 tokens.
using GeneralSchedule = Q4AdaSmallTSchedule<4, 4, 2, 8, 2>;

__global__ __launch_bounds__(GeneralSchedule::kThreads) void q4_general_mma(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
    __nv_bfloat16* out, int n, int k, int tokens) {
    const int first_token = static_cast<int>(blockIdx.y) * 16;
    const int count = min(16, tokens - first_token);
    ada_small_t_mma_cta<GeneralSchedule, 0, AdaContiguousRows>(
        x + static_cast<std::int64_t>(first_token) * k, k, codes, nullptr, scales,
        count, Bf16Store{out + static_cast<std::int64_t>(first_token) * n, n},
        static_cast<int>(blockIdx.x), nullptr);
}

template<int K, int TokenTiles>
void configure() {
    using S = Schedule<K, TokenTiles>;
    check(cudaFuncSetAttribute(ada_small_t_mma_kernel<S, K, AdaContiguousRows, Bf16Store>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmemBytes));
}

template<int K, int TokenTiles>
void launch_tuned(const void* x, const void* codes, const void* scales, void* out,
                  int n, int tokens, cudaStream_t stream) {
    using S = Schedule<K, TokenTiles>;
    // Configuration is performed by prepare, not lazily in the captured hot path.
    ada_small_t_mma_kernel<S, K, AdaContiguousRows, Bf16Store>
        <<<n / S::kRowsPerCta, S::kThreads, S::kSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x), K,
            static_cast<const std::uint8_t*>(codes), nullptr,
            static_cast<const std::uint8_t*>(scales), tokens,
            Bf16Store{static_cast<__nv_bfloat16*>(out), n}, nullptr);
}

template<class T> __device__ float as_float(T value);
template<> __device__ float as_float(__half value) { return __half2float(value); }
template<> __device__ float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template<class T> __device__ T from_float(float value);
template<> __device__ __half from_float(float value) { return __float2half_rn(value); }
template<> __device__ __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }

// General correctness route: arbitrary positive N/K/T, including padded K and row tails.
// Four warps own four rows; each warp reuses its codes across up to four tokens.
template<class T>
__global__ void q4_generic(const T* __restrict__ x, const std::uint8_t* __restrict__ codes,
                          const __half* __restrict__ scales, T* __restrict__ out,
                          int n, int k, int padded_k, int tokens) {
    const int row = static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int first_token = static_cast<int>(blockIdx.y) * 4;
    if (row >= n) return;
    float sums[4] = {0, 0, 0, 0};
    const std::int64_t row_base = static_cast<std::int64_t>(row) * (padded_k / 2);
    const std::int64_t scale_base = static_cast<std::int64_t>(row) * (padded_k / 64);
    for (int group = 0; group < (k + 63) / 64; ++group) {
        const unsigned byte = codes[row_base + static_cast<std::int64_t>(group) * 32 + lane];
        const float scale = __half2float(scales[scale_base + group]);
        const float w0 = static_cast<float>(static_cast<int>((byte & 15) ^ 8) - 8) * scale;
        const float w1 = static_cast<float>(static_cast<int>((byte >> 4) ^ 8) - 8) * scale;
        const int col = group * 64 + 2 * lane;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int token = first_token + t;
            if (token < tokens) {
                const std::int64_t base = static_cast<std::int64_t>(token) * k;
                if (col < k) sums[t] = fmaf(as_float(x[base + col]), w0, sums[t]);
                if (col + 1 < k) sums[t] = fmaf(as_float(x[base + col + 1]), w1, sums[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sums[t] += __shfl_down_sync(0xffffffffu, sums[t], offset);
        if (lane == 0 && first_token + t < tokens)
            out[static_cast<std::int64_t>(first_token + t) * n + row] = from_float<T>(sums[t]);
    }
}

bool can_tune(int n, int k, int tokens, int dtype) {
    return dtype == 1 && n % 40 == 0 && tokens >= 2 && tokens <= 16 &&
           (k == 5120 || k == 6144 || k == 25600);
}

bool can_mma(int n, int k, int dtype) {
    return dtype == 1 && n % 32 == 0 && k % 512 == 0;
}

template<int K>
void prepare_tuned(int tokens) {
    if (tokens <= 8) configure<K, 1>();
    else configure<K, 2>();
}

template<int K>
void tuned(const void* x, const void* codes, const void* scales, void* out,
           int n, int tokens, cudaStream_t stream) {
    if (tokens <= 8) launch_tuned<K, 1>(x, codes, scales, out, n, tokens, stream);
    else launch_tuned<K, 2>(x, codes, scales, out, n, tokens, stream);
}
}

extern "C" int venturi_ada_abi() { return 1; }
extern "C" const char* venturi_ada_error() { return last_error.c_str(); }

// dtype: 0=FP16, 1=BF16. requested: -1=auto, 0=SIMT, 1=retained, 2=general MMA.
extern "C" int venturi_q4_prepare(int n, int k, int tokens, int dtype, int requested) {
    try {
        last_error.clear();
        if (n <= 0 || k <= 0 || tokens <= 0 || dtype < 0 || dtype > 1 ||
            requested < -1 || requested > 2)
            throw std::invalid_argument("invalid Linear dimensions, dtype, or route");
        int device;
        check(cudaGetDevice(&device));
        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props, device));
        if (props.major != 8 || props.minor != 9)
            throw std::invalid_argument("Venturi Ada currently requires compute capability 8.9");
        int route = 0;
        if (requested == -1) {
            if (can_tune(n, k, tokens, dtype)) route = 1;
            else if (tokens >= 2 && can_mma(n, k, dtype)) route = 2;
        } else route = requested;
        if (requested == 1 && route != 1)
            throw std::invalid_argument("tuned route does not support these operands");
        if (route == 1 && !can_tune(n, k, tokens, dtype))
            throw std::invalid_argument("tuned route does not support these operands");
        if (route == 2 && !can_mma(n, k, dtype))
            throw std::invalid_argument("general MMA requires BF16, N%32=0, K%512=0");
        if (route == 1) {
            if (k == 5120) prepare_tuned<5120>(tokens);
            else if (k == 6144) prepare_tuned<6144>(tokens);
            else prepare_tuned<25600>(tokens);
        } else if (route == 2) {
            check(cudaFuncSetAttribute(q4_general_mma,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GeneralSchedule::kSmemBytes));
        } else {
            cudaFuncAttributes attributes;
            if (dtype == 0) check(cudaFuncGetAttributes(&attributes, q4_generic<__half>));
            else check(cudaFuncGetAttributes(&attributes, q4_generic<__nv_bfloat16>));
        }
        // Generic grid.y is ceil(T/4). Keep its launch geometry explicit.
        if (tokens > 4 * 65535)
            throw std::invalid_argument("tokens exceeds generic grid limit");
        return route;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" int venturi_q4_run(const void* x, const void* codes, const void* scales, void* out,
                              int n, int k, int padded_k, int tokens, int dtype, int route,
                              void* stream_ptr) {
    try {
        last_error.clear();
        if (!x || !codes || !scales || !out || n <= 0 || k <= 0 ||
            tokens <= 0 || tokens > 4 * 65535 || dtype < 0 || dtype > 1 ||
            route < 0 || route > 2 ||
            padded_k != (static_cast<std::int64_t>(k) + 127) / 128 * 128 ||
            (route == 1 && !can_tune(n, k, tokens, dtype)) ||
            (route == 2 && !can_mma(n, k, dtype)))
            throw std::invalid_argument("invalid prepared Q4 launch");
        const auto stream = static_cast<cudaStream_t>(stream_ptr);
        if (route == 1) {
            if (k == 5120) tuned<5120>(x, codes, scales, out, n, tokens, stream);
            else if (k == 6144) tuned<6144>(x, codes, scales, out, n, tokens, stream);
            else tuned<25600>(x, codes, scales, out, n, tokens, stream);
        } else if (route == 2) {
            const dim3 grid(n / 32, (tokens + 15) / 16);
            q4_general_mma<<<grid, GeneralSchedule::kThreads, GeneralSchedule::kSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(x), static_cast<const std::uint8_t*>(codes),
                static_cast<const std::uint8_t*>(scales), static_cast<__nv_bfloat16*>(out),
                n, k, tokens);
        } else {
            const dim3 grid((static_cast<unsigned>(n) + 3) / 4, (tokens + 3) / 4);
            if (dtype == 0)
                q4_generic<<<grid, 128, 0, stream>>>(static_cast<const __half*>(x),
                    static_cast<const std::uint8_t*>(codes), static_cast<const __half*>(scales),
                    static_cast<__half*>(out), n, k, padded_k, tokens);
            else
                q4_generic<<<grid, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(x),
                    static_cast<const std::uint8_t*>(codes), static_cast<const __half*>(scales),
                    static_cast<__nv_bfloat16*>(out), n, k, padded_k, tokens);
        }
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
