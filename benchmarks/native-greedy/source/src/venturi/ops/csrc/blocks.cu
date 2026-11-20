// Venturi-owned operator implementations and a cuBLAS Linear bridge.
// Caller owns all tensor/workspace storage. No model runtime or PyTorch C++ ABI.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef VENTURI_BLOCKS_ARCH
#error "Choose an explicit blocks architecture using venturi.ops.build"
#endif
static_assert(VENTURI_BLOCKS_ARCH == 70 || VENTURI_BLOCKS_ARCH == 89);

namespace {
thread_local std::string error_text;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(static_cast<int>(status)));
}
cudaDataType_t data_type(int dtype) {
    if (dtype == 0) return CUDA_R_16F;
    if (dtype == 1) {
        if (VENTURI_BLOCKS_ARCH == 70) throw std::invalid_argument("SM70 does not support native BF16 execution");
        return CUDA_R_16BF;
    }
    if (dtype == 2) return CUDA_R_32F;
    throw std::invalid_argument("unsupported scalar type");
}
void validate(int tokens, int features, int dtype) {
    if (tokens <= 0 || features <= 0) throw std::invalid_argument("invalid shape");
    (void)data_type(dtype);
}
struct Linear {
    cublasHandle_t handle{};
    int tokens, n, k, dtype;
};
template<class T> __device__ float read(T x) { return static_cast<float>(x); }
template<> __device__ float read(__half x) { return __half2float(x); }
template<> __device__ float read(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> __device__ T cast(float x) { return static_cast<T>(x); }
template<> __device__ __half cast(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }

// FP32 statistics and multiplication; one observable output-dtype cast.
template<class T>
__global__ void rms_norm(const T* x, const float* weight, T* out, int features, float epsilon) {
    __shared__ float partial[8];
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * features;
    float sum = 0.0f;
    for (std::int64_t col = threadIdx.x; col < features; col += blockDim.x) {
        const float v = read(x[offset + col]);
        sum = fmaf(v, v, sum);
    }
    for (int delta = 16; delta; delta >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
        for (int delta = 16; delta; delta >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (threadIdx.x == 0) partial[0] = rsqrtf(sum / features + epsilon);
    }
    __syncthreads();
    for (std::int64_t col = threadIdx.x; col < features; col += blockDim.x)
        out[offset + col] = cast<T>((read(x[offset + col]) * partial[0]) * weight[col]);
}

template<class T, bool Swiglu>
__global__ void elementwise(const T* a, const T* b, T* out, std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const float x = read(a[i]), y = read(b[i]);
        if constexpr (Swiglu) {
            // Stable SiLU for large negative x; avoid exp(-x) overflow.
            const float sigmoid = x >= 0 ? 1.0f / (1.0f + expf(-x))
                                         : expf(x) / (1.0f + expf(x));
            out[i] = cast<T>((x * sigmoid) * y);
        } else {
            out[i] = cast<T>(x + y);
        }
    }
}

template<bool Swiglu>
void launch_elementwise(const void* a, const void* b, void* out, std::int64_t count,
                        int dtype, cudaStream_t stream) {
    if (count <= 0) throw std::invalid_argument("invalid element count");
    (void)data_type(dtype);  // Reject BF16 on SM70 even for direct C ABI callers.
    const int grid = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 65535));
    if (dtype == 0) elementwise<__half, Swiglu><<<grid,256,0,stream>>>(
        static_cast<const __half*>(a), static_cast<const __half*>(b), static_cast<__half*>(out), count);
    else if (dtype == 1) elementwise<__nv_bfloat16, Swiglu><<<grid,256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(out), count);
    else if (dtype == 2) elementwise<float, Swiglu><<<grid,256,0,stream>>>(
        static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
    else throw std::invalid_argument("unsupported scalar type");
    check(cudaGetLastError());
}

// One thread owns both members of a rotation pair, so exact in-place is safe.
template<class T>
__global__ void rope(const T* x, const float* cos, const float* sin, T* out,
                     std::int64_t pairs, int heads, int d, int r, bool interleaved) {
    const int half = r / 2;
    const int tail = d - r;
    const int items = half + tail;
    const std::int64_t count = pairs * items;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = idx / items;
        const int col = idx % items;
        const std::int64_t base = row * d;
        if (col >= half) { out[base + r + col - half] = x[base + r + col - half]; continue; }
        const int a = interleaved ? 2 * col : col;
        const int b = interleaved ? 2 * col + 1 : col + half;
        const std::int64_t table = (row / heads) * half + col;
        const float va = read(x[base + a]), vb = read(x[base + b]);
        const float c = cos[table], t = sin[table];
        out[base + a] = cast<T>(va * c - vb * t);
        out[base + b] = cast<T>(va * t + vb * c);
    }
}

__device__ float block_sum(float value, float* partial) {
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffff, value, delta);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
        for (int delta = 16; delta; delta >>= 1)
            value += __shfl_down_sync(0xffffffff, value, delta);
        if (threadIdx.x == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

// Correctness-first online softmax. No scores matrix, global scratch, allocation,
// cache mutation, or Tensor Core claim. Serial key traversal is deliberately a
// baseline for subsequent tiled implementations, not an optimized prefill path.
template<class T>
__global__ void causal_attention(const T* q, const T* k, const T* v, T* out,
                                 int tokens, int kv_tokens, int hq, int hkv,
                                 int d, float scale) {
    __shared__ float partial[8];
    __shared__ float factors[3];
    const int row = blockIdx.x;
    const int token = row / hq;
    const int head = row % hq;
    const int kv_head = head / (hq / hkv);
    const int col = threadIdx.x;
    const std::int64_t qbase = static_cast<std::int64_t>(row) * d;
    const float query = col < d ? read(q[qbase + col]) : 0.0f;
    float acc = 0.0f, maximum = -INFINITY, denom = 0.0f;
    const int last = kv_tokens - tokens + token;
    for (int key = 0; key <= last; ++key) {
        const std::int64_t base = (static_cast<std::int64_t>(key) * hkv + kv_head) * d;
        const float product = col < d ? query * read(k[base + col]) : 0.0f;
        const float score = block_sum(product, partial) * scale;
        if (col == 0) {
            const float next = fmaxf(maximum, score);
            const float alpha = expf(maximum - next), beta = expf(score - next);
            denom = denom * alpha + beta;
            maximum = next;
            factors[0] = alpha; factors[1] = beta; factors[2] = denom;
        }
        __syncthreads();
        if (col < d) acc = acc * factors[0] + read(v[base + col]) * factors[1];
        // All consumers must finish before thread 0 updates shared factors.
        __syncthreads();
    }
    if (col < d) out[qbase + col] = cast<T>(acc / factors[2]);
}

// A dense gather; IDs are validated on the host during plan preparation. The
// bounds check is still mandatory to prevent OOB reads if a caller violates the
// immutable-ID contract. Invalid device IDs produce zero, not unchecked memory.
template<class T>
__global__ void embedding(const T* weight, const std::int64_t* ids, T* out,
                          std::int64_t count, int vocab, int hidden) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t id = ids[i / hidden];
        out[i] = id >= 0 && id < vocab ? weight[id * hidden + i % hidden] : cast<T>(0.0f);
    }
}

__device__ float sum_sync(float value, float* partial) {
    const float result = block_sum(value, partial);
    // All threads must have read partial[0] before the next reduction reuses it.
    __syncthreads();
    return result;
}

// Correctness-first recurrent GDN: one 256-thread block per value row. Registers
// retain a state's key dimension across tokens. No dense attention substitution,
// allocations, Tensor Cores, causal convolution or gate transformations here.
template<class T, bool Packed>
__global__ void gdn_recurrent(const T* q, const T* k,
    const T* v, const float* g, const float* beta, const float* state,
    T* out, float* state_out, int tokens, int hqk, int hv, int dk,
    int dv, bool normalize, float scale) {
    __shared__ float partial[8];
    const int head = blockIdx.x / dv, row = blockIdx.x % dv;
    const int qhead = head / (hv / hqk), col = threadIdx.x;
    const std::int64_t state_idx = static_cast<std::int64_t>(blockIdx.x) * dk + col;
    float s = col < dk ? state[state_idx] : 0.0f;
    const std::int64_t qstride = Packed ? 2LL * hqk * dk + static_cast<std::int64_t>(hv) * dv : static_cast<std::int64_t>(hqk) * dk;
    const std::int64_t vstride = Packed ? qstride : static_cast<std::int64_t>(hv) * dv;
    for (int t = 0; t < tokens; ++t) {
        const std::int64_t base = static_cast<std::int64_t>(t) * qstride + static_cast<std::int64_t>(qhead) * dk;
        float qt = col < dk ? read(q[base + col]) : 0.0f;
        float kt = col < dk ? read(k[base + col]) : 0.0f;
        if (normalize) {
            const float qsum = sum_sync(qt * qt, partial);
            const float ksum = sum_sync(kt * kt, partial);
            qt *= rsqrtf(qsum + 1e-6f); kt *= rsqrtf(ksum + 1e-6f);
        }
        const std::int64_t gate_idx = static_cast<std::int64_t>(t) * hv + head;
        s *= expf(g[gate_idx]);
        const float memory = sum_sync(s * kt, partial);
        const float delta = beta[gate_idx] * (read(v[static_cast<std::int64_t>(t) * vstride + head * dv + row]) - memory);
        s = fmaf(delta, kt, s);
        const float result = sum_sync(s * qt, partial) * scale;
        if (col == 0) out[gate_idx * dv + row] = cast<T>(result);
    }
    if (col < dk) state_out[state_idx] = s;
}

template<class T>
__global__ void q8_embedding(const std::int8_t* codes, const __half* scales,
    const std::int64_t* ids, T* out, std::int64_t count, int vocab, int k, int padded) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row = ids[i / k];
        const int col = i % k;
        const float value = row >= 0 && row < vocab
            ? static_cast<float>(codes[row * padded + col]) * read(scales[row * (padded / 32) + col / 32]) : 0.0f;
        out[i] = cast<T>(value);
    }
}

// Direct signed-Q8 x activation dot products; FP32 dequantization/accumulation. A
// correctness baseline for last-token output heads, not a tiled GEMM claim.
template<class T>
__global__ void q8_head(const T* x, const std::int8_t* codes,
    const __half* scales, T* out, int vocab, int k, int padded) {
    __shared__ float partial[8];
    const int token = blockIdx.x / vocab, row = blockIdx.x % vocab;
    float accum = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const float weight = static_cast<float>(codes[static_cast<std::int64_t>(row) * padded + col])
            * read(scales[static_cast<std::int64_t>(row) * (padded / 32) + col / 32]);
        accum = fmaf(read(x[static_cast<std::int64_t>(token) * k + col]), weight, accum);
    }
    const float result = block_sum(accum, partial);
    if (threadIdx.x == 0) out[blockIdx.x] = cast<T>(result);
}

// Preserve FP32 small parameters with FP16/BF16 inputs; no intermediate
// weight or product cast to the activation type. One output cast, FP32 FMA.
template<class T>
__global__ void fp32_weight_linear(const T* x, const float* weight, T* out, int n, int k) {
    __shared__ float partial[8];
    const int token=blockIdx.x/n,row=blockIdx.x%n;
    float sum=0.0f;
    for (int col=threadIdx.x;col<k;col+=blockDim.x)
        sum=fmaf(read(x[static_cast<std::int64_t>(token)*k+col]),weight[static_cast<std::int64_t>(row)*k+col],sum);
    const float value=block_sum(sum,partial);
    if (threadIdx.x==0) out[blockIdx.x]=cast<T>(value);
}

__device__ float sigmoid(float x) {
    return x >= 0 ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

// Each thread owns one channel across time. In-place history update is safe:
// consume its old history before updating the suffix, in chronological order.
template<class T, class W>
__global__ void causal_conv(const T* x, const W* w, const T* history, T* out,
    T* history_out, int tokens, int channels, int kernel) {
    for (std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; col < channels;
         col += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        for (int t = 0; t < tokens; ++t) {
            float accum = 0.0f;
            for (int tap = 0; tap < kernel; ++tap) {
                const int pos = t + tap - (kernel - 1);
                const float v = pos < 0
                    ? read(history[static_cast<std::int64_t>(pos + kernel - 1) * channels + col])
                    : read(x[static_cast<std::int64_t>(pos) * channels + col]);
                accum = fmaf(v, read(w[static_cast<std::int64_t>(tap) * channels + col]), accum);
            }
            const float conv = read(cast<T>(accum));
            out[static_cast<std::int64_t>(t) * channels + col] = cast<T>(conv * sigmoid(conv));
        }
        for (int row = 0; row < kernel - 1; ++row) {
            const int pos = tokens - (kernel - 1) + row;
            history_out[static_cast<std::int64_t>(row) * channels + col] = pos < 0
                ? history[static_cast<std::int64_t>(pos + kernel - 1) * channels + col]
                : x[static_cast<std::int64_t>(pos) * channels + col];
        }
    }
}

template<class T>
__global__ void gdn_gates(const T* a, const T* b, const float* rate, const float* bias,
    float* g, float* beta, std::int64_t count, int heads, bool round_beta) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const float z = read(a[i]) + bias[i % heads];
        const float softplus = z > 20.0f ? z : log1pf(expf(z));
        g[i] = -rate[i % heads] * softplus;
        const float value = sigmoid(read(b[i]));
        beta[i] = round_beta ? read(cast<T>(value)) : value;
    }
}

template<class T, bool FP32Weight>
__global__ void gated_rms(const T* x, const T* z, const void* weight, T* out,
    int features, float epsilon) {
    __shared__ float partial[8];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * features;
    float sum = 0.0f;
    for (std::int64_t col = threadIdx.x; col < features; col += blockDim.x) {
        const float value = read(x[base + col]);
        sum = fmaf(value, value, sum);
    }
    const float inv = rsqrtf(block_sum(sum, partial) / features + epsilon);
    for (std::int64_t col = threadIdx.x; col < features; col += blockDim.x) {
        const float normal = read(cast<T>(read(x[base + col]) * inv));
        const float gamma = FP32Weight ? static_cast<const float*>(weight)[col] : read(static_cast<const T*>(weight)[col]);
        const float weighted = FP32Weight ? normal * gamma : read(cast<T>(normal * gamma));
        const float gate = read(z[base + col]);
        out[base + col] = cast<T>(weighted * (gate * sigmoid(gate)));
    }
}

// One block per Q/K head. Packed Q/G is never unpacked into a temporary.
// Norm -> activation cast -> two rounded rotary products -> rounded sum.
template<class T>
__global__ void qwen_qk(const T* qg, const T* key, const float* qgamma,
    const float* kgamma, const float* cos, const float* sin, T* qout, T* kout,
    int hq, int hkv, int d, int r, float epsilon) {
    __shared__ float partial[8];
    __shared__ float normalized[256];
    const int token = blockIdx.x / (hq+hkv), h = blockIdx.x % (hq+hkv);
    const bool query = h < hq;
    const int head = query ? h : h-hq;
    const std::int64_t row = static_cast<std::int64_t>(token) * (query ? hq : hkv) + head;
    const T* input = query ? qg + row * 2*d : key + row*d;
    T* output = query ? qout + row*d : kout + row*d;
    const float* gamma = query ? qgamma : kgamma;
    const int col = threadIdx.x;
    const float x = col < d ? read(input[col]) : 0.0f;
    const float inv = rsqrtf(block_sum(x*x,partial)/d + epsilon);
    if (col < d) normalized[col] = read(cast<T>((x*inv)*gamma[col]));
    __syncthreads();
    if (col >= d) return;
    if (col >= r) { output[col]=cast<T>(normalized[col]); return; }
    const int half = r/2, other = col < half ? col+half : col-half;
    const std::int64_t offset = static_cast<std::int64_t>(token)*half + col%half;
    const float c=read(cast<T>(cos[offset])), t=read(cast<T>(sin[offset]));
    const float first=read(cast<T>(__fmul_rn(normalized[col],c)));
    const float second=read(cast<T>(__fmul_rn(normalized[other],t)));
    output[col]=cast<T>(col < half ? first-second : first+second);
}

template<class T>
__global__ void qwen_gate(const T* x, const T* qg, T* out, std::int64_t count, int d) {
    for (std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
        const float gate=read(qg[(i/d)*2*d+d+i%d]);
        out[i]=cast<T>(read(x[i])*read(cast<T>(sigmoid(gate))));
    }
}

// Exact finite-logit greedy selection. Every comparison carries the original
// vocabulary index, so the lowest-index winner is independent of reduction order.
// Any NaN or infinity invalidates only its row; no arbitrary token is emitted.
struct alignas(16) GreedyCandidate {
    float score;
    int index;
    int invalid;
    int unused;
};
__device__ GreedyCandidate greedy_empty() {
    return {-CUDART_INF_F, INT32_MAX, 0, 0};
}
__device__ GreedyCandidate greedy_merge(GreedyCandidate a, GreedyCandidate b) {
    const int invalid = a.invalid | b.invalid;
    if (b.score > a.score || (b.score == a.score && b.index < a.index)) a = b;
    a.invalid = invalid;
    return a;
}
__device__ GreedyCandidate greedy_reduce(GreedyCandidate a, GreedyCandidate* partial) {
    for (int delta=16; delta; delta>>=1) {
        GreedyCandidate b{
            __shfl_down_sync(0xffffffff,a.score,delta),
            __shfl_down_sync(0xffffffff,a.index,delta),
            __shfl_down_sync(0xffffffff,a.invalid,delta),0};
        a = greedy_merge(a,b);
    }
    if ((threadIdx.x & 31)==0) partial[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x < 32) {
        a = threadIdx.x < 8 ? partial[threadIdx.x] : greedy_empty();
        for (int delta=16; delta; delta>>=1) {
            GreedyCandidate b{
                __shfl_down_sync(0xffffffff,a.score,delta),
                __shfl_down_sync(0xffffffff,a.index,delta),
                __shfl_down_sync(0xffffffff,a.invalid,delta),0};
            a = greedy_merge(a,b);
        }
    }
    return a; // Only thread zero consumes the final block result.
}
template<class T>
__global__ void greedy_partials(const T* logits, GreedyCandidate* scratch, int vocab) {
    __shared__ GreedyCandidate partial[8];
    auto best = greedy_empty();
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.y)*vocab;
    for (std::int64_t col=static_cast<std::int64_t>(blockIdx.x)*256+threadIdx.x;
         col<vocab; col+=static_cast<std::int64_t>(gridDim.x)*256) {
        const float score=read(logits[base+col]);
        if (!isfinite(score)) best.invalid=1;
        else best=greedy_merge(best,{score,static_cast<int>(col),0,0});
    }
    best=greedy_reduce(best,partial);
    if (threadIdx.x==0) scratch[static_cast<std::int64_t>(blockIdx.y)*gridDim.x+blockIdx.x]=best;
}
__global__ void greedy_finish(const GreedyCandidate* scratch, std::int64_t* out, int tiles) {
    __shared__ GreedyCandidate partial[8];
    auto best=greedy_empty();
    if (threadIdx.x < tiles) best=scratch[static_cast<std::int64_t>(blockIdx.x)*tiles+threadIdx.x];
    best=greedy_reduce(best,partial);
    if (threadIdx.x==0) {
        out[static_cast<std::int64_t>(blockIdx.x)*2]=best.invalid ? -1 : best.index;
        out[static_cast<std::int64_t>(blockIdx.x)*2+1]=best.invalid ? 1 : 0;
    }
}

}

extern "C" int venturi_blocks_abi() { return 11; }
extern "C" int venturi_blocks_arch() { return VENTURI_BLOCKS_ARCH; }
extern "C" const char* venturi_blocks_error() { return error_text.c_str(); }
extern "C" int venturi_greedy_run(const void* logits, std::int64_t* out, void* workspace,
    std::size_t workspace_bytes, int rows, int vocab, int dtype, void* raw_stream) {
    try {
        validate(rows,vocab,dtype);
        const int tiles=std::min((vocab-1)/4096+1,128);
        const std::size_t required=static_cast<std::size_t>(rows)*tiles*sizeof(GreedyCandidate);
        if (!logits || !out || !workspace || rows>65535 || workspace_bytes<required ||
            reinterpret_cast<std::uintptr_t>(workspace)%alignof(GreedyCandidate) ||
            reinterpret_cast<std::uintptr_t>(out)%alignof(std::int64_t))
            throw std::invalid_argument("invalid greedy selection operand/workspace");
        auto stream=static_cast<cudaStream_t>(raw_stream);
        auto* scratch=static_cast<GreedyCandidate*>(workspace);
        const dim3 grid(tiles,rows);
        if (dtype==0) greedy_partials<<<grid,256,0,stream>>>(static_cast<const __half*>(logits),scratch,vocab);
        else if (dtype==1) greedy_partials<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(logits),scratch,vocab);
        else greedy_partials<<<grid,256,0,stream>>>(static_cast<const float*>(logits),scratch,vocab);
        check(cudaGetLastError());
        greedy_finish<<<rows,256,0,stream>>>(scratch,out,tiles);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text=e.what(); return -1; }
}
extern "C" void* venturi_dense_create(int tokens, int n, int k, int dtype) {
    Linear* plan = nullptr;
    try {
        validate(tokens, n, dtype);
        if (k <= 0) throw std::invalid_argument("invalid K");
        plan = new Linear{nullptr, tokens, n, k, dtype};
        check(cublasCreate(&plan->handle));
        check(cublasSetPointerMode(plan->handle, CUBLAS_POINTER_MODE_HOST));
        check(cublasSetAtomicsMode(plan->handle, CUBLAS_ATOMICS_NOT_ALLOWED));
        check(cublasSetMathMode(plan->handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        return plan;
    } catch (const std::exception& e) {
        error_text = e.what();
        if (plan) { if (plan->handle) cublasDestroy(plan->handle); delete plan; }
        return nullptr;
    }
}
extern "C" void venturi_dense_destroy(void* opaque) {
    auto* plan = static_cast<Linear*>(opaque);
    if (plan) { cublasDestroy(plan->handle); delete plan; }
}
extern "C" int venturi_dense_run(void* opaque, const void* x, const void* w, void* out,
                                 void* workspace, std::size_t workspace_bytes, void* raw_stream) {
    try {
        auto* p = static_cast<Linear*>(opaque);
        if (!p || !x || !w || !out || !workspace) throw std::invalid_argument("null Linear operand");
        check(cublasSetStream(p->handle, static_cast<cudaStream_t>(raw_stream)));
        // SetStream resets the handle workspace, so restore the explicit caller buffer.
        check(cublasSetWorkspace(p->handle, workspace, workspace_bytes));
        const float alpha = 1.0f, beta = 0.0f;
        // Use the explicit FP32 arithmetic contract for every stored input type.
        const auto compute = CUBLAS_COMPUTE_32F_PEDANTIC;
        // Row-major Y[T,N] = X[T,K] W[N,K]^T via column-major Y^T.
        check(cublasGemmEx(p->handle, CUBLAS_OP_T, CUBLAS_OP_N, p->n, p->tokens, p->k,
            &alpha, w, data_type(p->dtype), p->k, x, data_type(p->dtype), p->k,
            &beta, out, data_type(p->dtype), p->n, compute, CUBLAS_GEMM_DEFAULT));
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_fp32_linear_run(const void* x,const float* w,void* out,
    int tokens,int n,int k,int dtype,void* raw_stream) {
    try {
        validate(tokens,n,dtype);
        if (!x || !w || !out || k<=0 || static_cast<std::int64_t>(tokens)*n>=INT32_MAX ||
            (dtype!=0 && dtype!=1)) throw std::invalid_argument("invalid FP32-parameter Linear operands");
        auto stream=static_cast<cudaStream_t>(raw_stream);
        if (dtype==0) fp32_weight_linear<<<tokens*n,256,0,stream>>>(static_cast<const __half*>(x),w,static_cast<__half*>(out),n,k);
        else fp32_weight_linear<<<tokens*n,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),w,static_cast<__nv_bfloat16*>(out),n,k);
        check(cudaGetLastError());return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}

extern "C" int venturi_rms_run(const void* x, const float* w, void* out, int tokens,
                               int features, int dtype, float epsilon, void* raw_stream) {
    try {
        validate(tokens, features, dtype);
        if (!x || !w || !out || !(epsilon > 0) || !std::isfinite(epsilon))
            throw std::invalid_argument("invalid RMSNorm operand/epsilon");
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (dtype == 0) rms_norm<<<tokens,256,0,stream>>>(static_cast<const __half*>(x),w,static_cast<__half*>(out),features,epsilon);
        else if (dtype == 1) rms_norm<<<tokens,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),w,static_cast<__nv_bfloat16*>(out),features,epsilon);
        else rms_norm<<<tokens,256,0,stream>>>(static_cast<const float*>(x),w,static_cast<float*>(out),features,epsilon);
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_swiglu_run(const void* a, const void* b, void* out,
                                  std::int64_t count, int dtype, void* stream) {
    try { launch_elementwise<true>(a,b,out,count,dtype,static_cast<cudaStream_t>(stream)); return 0; }
    catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_add_run(const void* a, const void* b, void* out,
                              std::int64_t count, int dtype, void* stream) {
    try { launch_elementwise<false>(a,b,out,count,dtype,static_cast<cudaStream_t>(stream)); return 0; }
    catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" int venturi_rope_run(const void* x, const float* cos, const float* sin,
    void* out, int tokens, int heads, int d, int r, int interleaved, int dtype, void* raw_stream) {
    try {
        validate(tokens, heads, dtype);
        if (!x || !cos || !sin || !out || d <= 0 || r <= 0 || r > d || r % 2 ||
            (interleaved != 0 && interleaved != 1)) throw std::invalid_argument("invalid RoPE operand");
        const std::int64_t rows = static_cast<std::int64_t>(tokens) * heads;
        const std::int64_t count = rows * (d - r / 2);
        const int grid = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 65535));
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (dtype == 0) rope<<<grid,256,0,stream>>>(static_cast<const __half*>(x),cos,sin,static_cast<__half*>(out),rows,heads,d,r,interleaved);
        else if (dtype == 1) rope<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),cos,sin,static_cast<__nv_bfloat16*>(out),rows,heads,d,r,interleaved);
        else rope<<<grid,256,0,stream>>>(static_cast<const float*>(x),cos,sin,static_cast<float*>(out),rows,heads,d,r,interleaved);
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_attention_run(const void* q, const void* k, const void* v,
    void* out, int tokens, int kv_tokens, int hq, int hkv, int d, int dtype,
    float scale, void* raw_stream) {
    try {
        validate(tokens, kv_tokens, dtype);
        if (!q || !k || !v || !out || hq <= 0 || hkv <= 0 || hq % hkv ||
            kv_tokens < tokens || (d != 64 && d != 128 && d != 256) ||
            static_cast<std::int64_t>(tokens) * hq >= INT32_MAX ||
            !(scale > 0) || !std::isfinite(scale))
            throw std::invalid_argument("invalid causal attention operand");
        auto stream = static_cast<cudaStream_t>(raw_stream);
        const int rows = tokens * hq;
        if (dtype == 0) causal_attention<<<rows,256,0,stream>>>(static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),static_cast<__half*>(out),tokens,kv_tokens,hq,hkv,d,scale);
        else if (dtype == 1) causal_attention<<<rows,256,0,stream>>>(static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),static_cast<__nv_bfloat16*>(out),tokens,kv_tokens,hq,hkv,d,scale);
        else causal_attention<<<rows,256,0,stream>>>(static_cast<const float*>(q),static_cast<const float*>(k),static_cast<const float*>(v),static_cast<float*>(out),tokens,kv_tokens,hq,hkv,d,scale);
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" int venturi_embedding_run(const void* weight, const std::int64_t* ids,
    void* out, int tokens, int vocab, int hidden, int dtype, void* raw_stream) {
    try {
        validate(tokens, hidden, dtype);
        if (!weight || !ids || !out || vocab <= 0)
            throw std::invalid_argument("invalid embedding operand");
        const std::int64_t count = static_cast<std::int64_t>(tokens) * hidden;
        const int grid = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 65535));
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (dtype == 0) embedding<<<grid,256,0,stream>>>(static_cast<const __half*>(weight),ids,static_cast<__half*>(out),count,vocab,hidden);
        else if (dtype == 1) embedding<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(weight),ids,static_cast<__nv_bfloat16*>(out),count,vocab,hidden);
        else embedding<<<grid,256,0,stream>>>(static_cast<const float*>(weight),ids,static_cast<float*>(out),count,vocab,hidden);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" int venturi_gdn_run(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* state, void* out, float* state_out,
    int tokens, int hqk, int hv, int dk, int dv, int normalize, int packed, int dtype, float scale, void* raw_stream) {
    try {
        if (!q || !k || !v || !g || !beta || !state || !out || !state_out ||
            tokens <= 0 || hqk <= 0 || hv <= 0 || hv % hqk || dv <= 0 ||
            (dk != 64 && dk != 128 && dk != 256) ||
            static_cast<std::int64_t>(hv) * dv >= INT32_MAX ||
            (normalize != 0 && normalize != 1) || (packed != 0 && packed != 1) || !(scale > 0) || !std::isfinite(scale))
            throw std::invalid_argument("invalid GDN operand/configuration");
        validate(tokens, dk, dtype);
        if (dtype != 0 && dtype != 1) throw std::invalid_argument("GDN supports FP16/BF16 activations");
        if (dtype == 0) {
            if (packed) gdn_recurrent<__half,true><<<hv*dv,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
                static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),
                g,beta,state,static_cast<__half*>(out),state_out,tokens,hqk,hv,dk,dv,normalize,scale);
            else gdn_recurrent<__half,false><<<hv*dv,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
                static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),
                g,beta,state,static_cast<__half*>(out),state_out,tokens,hqk,hv,dk,dv,normalize,scale);
        } else if (packed) gdn_recurrent<__nv_bfloat16,true><<<hv*dv,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
            static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),g,beta,state,static_cast<__nv_bfloat16*>(out),
            state_out,tokens,hqk,hv,dk,dv,normalize,scale);
        else gdn_recurrent<__nv_bfloat16,false><<<hv*dv,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
            static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),g,beta,state,static_cast<__nv_bfloat16*>(out),
            state_out,tokens,hqk,hv,dk,dv,normalize,scale);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" int venturi_q8_embedding_run(const std::int8_t* codes, const __half* scales,
    const std::int64_t* ids, void* out, int tokens, int vocab, int k, int padded, int dtype, void* raw_stream) {
    try {
        if (!codes || !scales || !ids || !out || tokens <= 0 || vocab <= 0 || k <= 0 ||
            padded < k || padded % 128 || static_cast<std::int64_t>(padded) != ((static_cast<std::int64_t>(k) + 127) / 128) * 128)
            throw std::invalid_argument("invalid Q8 embedding operand");
        const std::int64_t count = static_cast<std::int64_t>(tokens) * k;
        const int grid = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 65535));
        validate(tokens,k,dtype);
        if (dtype == 0) q8_embedding<<<grid,256,0,static_cast<cudaStream_t>(raw_stream)>>>(codes,scales,ids,
            static_cast<__half*>(out),count,vocab,k,padded);
        else if (dtype == 1) q8_embedding<<<grid,256,0,static_cast<cudaStream_t>(raw_stream)>>>(codes,scales,ids,
            static_cast<__nv_bfloat16*>(out),count,vocab,k,padded);
        else throw std::invalid_argument("Q8 supports FP16/BF16 activations");
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_q8_head_run(const void* x, const std::int8_t* codes, const __half* scales,
    void* out, int tokens, int vocab, int k, int padded, int dtype, void* raw_stream) {
    try {
        if (!x || !codes || !scales || !out || tokens <= 0 || vocab <= 0 || k <= 0 ||
            padded < k || padded % 128 || static_cast<std::int64_t>(padded) != ((static_cast<std::int64_t>(k) + 127) / 128) * 128 ||
            static_cast<std::int64_t>(tokens) * vocab >= INT32_MAX)
            throw std::invalid_argument("invalid Q8 head operand");
        validate(tokens,k,dtype);
        if (dtype == 0) q8_head<<<tokens*vocab,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
            static_cast<const __half*>(x),codes,scales,static_cast<__half*>(out),vocab,k,padded);
        else if (dtype == 1) q8_head<<<tokens*vocab,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
            static_cast<const __nv_bfloat16*>(x),codes,scales,static_cast<__nv_bfloat16*>(out),vocab,k,padded);
        else throw std::invalid_argument("Q8 supports FP16/BF16 activations");
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}

extern "C" int venturi_conv_run(const void* x, const void* w, const void* history,
    void* out, void* history_out, int tokens, int channels, int kernel, int dtype, int fp32_weight, void* raw_stream) {
    try {
        validate(tokens, channels, dtype);
        if (!x || !w || !out || kernel < 1 || kernel > 32 || (fp32_weight!=0 && fp32_weight!=1) ||
            (kernel > 1 && (!history || !history_out)))
            throw std::invalid_argument("invalid causal convolution operand");
        const int grid = std::min((channels - 1) / 256 + 1, 65535);
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (dtype == 0) {
            if (fp32_weight) causal_conv<<<grid,256,0,stream>>>(static_cast<const __half*>(x),static_cast<const float*>(w),static_cast<const __half*>(history),static_cast<__half*>(out),static_cast<__half*>(history_out),tokens,channels,kernel);
            else causal_conv<<<grid,256,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(w),static_cast<const __half*>(history),static_cast<__half*>(out),static_cast<__half*>(history_out),tokens,channels,kernel);
        }
        else if (dtype == 1) {
            if (fp32_weight) causal_conv<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const float*>(w),static_cast<const __nv_bfloat16*>(history),static_cast<__nv_bfloat16*>(out),static_cast<__nv_bfloat16*>(history_out),tokens,channels,kernel);
            else causal_conv<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(w),static_cast<const __nv_bfloat16*>(history),static_cast<__nv_bfloat16*>(out),static_cast<__nv_bfloat16*>(history_out),tokens,channels,kernel);
        }
        else causal_conv<<<grid,256,0,stream>>>(static_cast<const float*>(x),static_cast<const float*>(w),static_cast<const float*>(history),static_cast<float*>(out),static_cast<float*>(history_out),tokens,channels,kernel);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_gates_run(const void* a, const void* b, const float* rate, const float* bias,
    float* g, float* beta, int tokens, int heads, int dtype, int round_beta, void* raw_stream) {
    try {
        validate(tokens, heads, dtype);
        if (!a || !b || !rate || !bias || !g || !beta || (round_beta != 0 && round_beta != 1))
            throw std::invalid_argument("invalid GDN gate operand");
        const std::int64_t count = static_cast<std::int64_t>(tokens) * heads;
        const int grid = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 65535));
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (dtype == 0) gdn_gates<<<grid,256,0,stream>>>(static_cast<const __half*>(a),static_cast<const __half*>(b),rate,bias,g,beta,count,heads,round_beta);
        else if (dtype == 1) gdn_gates<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(a),static_cast<const __nv_bfloat16*>(b),rate,bias,g,beta,count,heads,round_beta);
        else gdn_gates<<<grid,256,0,stream>>>(static_cast<const float*>(a),static_cast<const float*>(b),rate,bias,g,beta,count,heads,round_beta);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_gated_rms_run(const void* x, const void* z, const void* w, void* out,
    int tokens, int heads, int features, int dtype, int fp32_weight, float epsilon, void* raw_stream) {
    try {
        validate(tokens,features,dtype);
        if (!x || !z || !w || !out || heads<=0 || static_cast<std::int64_t>(tokens)*heads>=INT32_MAX ||
            (fp32_weight!=0 && fp32_weight!=1) || !(epsilon>0) || !std::isfinite(epsilon)) throw std::invalid_argument("invalid gated RMSNorm operand");
        const int rows=tokens*heads;auto stream=static_cast<cudaStream_t>(raw_stream);
        if (dtype==0) {
            if (fp32_weight) gated_rms<__half,true><<<rows,256,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(z),w,static_cast<__half*>(out),features,epsilon);
            else gated_rms<__half,false><<<rows,256,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(z),w,static_cast<__half*>(out),features,epsilon);
        } else if (dtype==1) {
            if (fp32_weight) gated_rms<__nv_bfloat16,true><<<rows,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(z),w,static_cast<__nv_bfloat16*>(out),features,epsilon);
            else gated_rms<__nv_bfloat16,false><<<rows,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(z),w,static_cast<__nv_bfloat16*>(out),features,epsilon);
        } else gated_rms<float,true><<<rows,256,0,stream>>>(static_cast<const float*>(x),static_cast<const float*>(z),w,static_cast<float*>(out),features,epsilon);
        check(cudaGetLastError());return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}

extern "C" int venturi_qwen_qk_run(const void* qg, const void* key,
    const float* qgamma, const float* kgamma, const float* cos, const float* sin,
    void* qout, void* kout, int tokens, int hq, int hkv, int d, int r, int dtype,
    float epsilon, void* raw_stream) {
    try {
        validate(tokens,hq,dtype);
        if (!qg || !key || !qgamma || !kgamma || !qout || !kout || (r && (!cos || !sin)) ||
            hkv<=0 || hq%hkv || (d!=64 && d!=128 && d!=256) || r<0 || r>d || r%2 ||
            static_cast<std::int64_t>(tokens)*(static_cast<std::int64_t>(hq)+hkv)>=INT32_MAX ||
            !(epsilon>0) || !std::isfinite(epsilon)) throw std::invalid_argument("invalid Qwen QK operand");
        auto stream=static_cast<cudaStream_t>(raw_stream);
        const int grid=static_cast<int>(static_cast<std::int64_t>(tokens)*(hq+hkv));
        if (dtype==0) qwen_qk<<<grid,256,0,stream>>>(static_cast<const __half*>(qg),static_cast<const __half*>(key),qgamma,kgamma,cos,sin,static_cast<__half*>(qout),static_cast<__half*>(kout),hq,hkv,d,r,epsilon);
        else if (dtype==1) qwen_qk<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(qg),static_cast<const __nv_bfloat16*>(key),qgamma,kgamma,cos,sin,static_cast<__nv_bfloat16*>(qout),static_cast<__nv_bfloat16*>(kout),hq,hkv,d,r,epsilon);
        else qwen_qk<<<grid,256,0,stream>>>(static_cast<const float*>(qg),static_cast<const float*>(key),qgamma,kgamma,cos,sin,static_cast<float*>(qout),static_cast<float*>(kout),hq,hkv,d,r,epsilon);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text=e.what(); return -1; }
}

extern "C" int venturi_qwen_gate_run(const void* x, const void* qg, void* out,
    int tokens, int heads, int d, int dtype, void* raw_stream) {
    try {
        validate(tokens,heads,dtype);
        if (!x || !qg || !out || (d!=64 && d!=128 && d!=256)) throw std::invalid_argument("invalid Qwen output gate operand");
        const std::int64_t count=static_cast<std::int64_t>(tokens)*heads*d;
        const int grid=static_cast<int>(std::min<std::int64_t>((count+255)/256,65535));
        auto stream=static_cast<cudaStream_t>(raw_stream);
        if (dtype==0) qwen_gate<<<grid,256,0,stream>>>(static_cast<const __half*>(x),static_cast<const __half*>(qg),static_cast<__half*>(out),count,d);
        else if (dtype==1) qwen_gate<<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(qg),static_cast<__nv_bfloat16*>(out),count,d);
        else qwen_gate<<<grid,256,0,stream>>>(static_cast<const float*>(x),static_cast<const float*>(qg),static_cast<float*>(out),count,d);
        check(cudaGetLastError()); return 0;
    } catch (const std::exception& e) { error_text=e.what(); return -1; }
}
