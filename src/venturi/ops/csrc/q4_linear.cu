// Independent Q4 linear: FP32 accumulation, explicit FP16/BF16 output cast.
// One owned SIMT implementation, no external headers or retained MMA routes.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string last_error;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
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


}
extern "C" int venturi_ada_abi() { return 2; }
extern "C" const char* venturi_ada_error() { return last_error.c_str(); }
extern "C" int venturi_q4_prepare(int n, int k, int tokens, int dtype, int requested) {
    try {
        last_error.clear();
        if(n<=0 || k<=0 || tokens<=0 || tokens>4*65535 || dtype<0 || dtype>1 || (requested!=-1 && requested!=0))
            throw std::invalid_argument("invalid Q4 dimensions, dtype, or removed route");
        int device; check(cudaGetDevice(&device));
        cudaDeviceProp props; check(cudaGetDeviceProperties(&props,device));
        if(props.major!=8 || props.minor!=9) throw std::invalid_argument("Q4 requires compute capability 8.9");
        cudaFuncAttributes attributes;
        if(dtype==0)check(cudaFuncGetAttributes(&attributes,q4_generic<__half>));
        else check(cudaFuncGetAttributes(&attributes,q4_generic<__nv_bfloat16>));
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return -1;}
}
extern "C" int venturi_q4_run(const void* x,const void* codes,const void* scales,void* out,
 int n,int k,int padded_k,int tokens,int dtype,int route,void* stream_ptr) {
    try {
        last_error.clear();
        if(!x || !codes || !scales || !out || n<=0 || k<=0 || tokens<=0 || tokens>4*65535 || dtype<0 || dtype>1 || route!=0 || padded_k!=(static_cast<std::int64_t>(k)+127)/128*128)
            throw std::invalid_argument("invalid prepared Q4 launch");
        const dim3 grid((static_cast<unsigned>(n)+3)/4,(tokens+3)/4);
        const auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(dtype==0)q4_generic<<<grid,128,0,stream>>>(static_cast<const __half*>(x),static_cast<const std::uint8_t*>(codes),static_cast<const __half*>(scales),static_cast<__half*>(out),n,k,padded_k,tokens);
        else q4_generic<<<grid,128,0,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const std::uint8_t*>(codes),static_cast<const __half*>(scales),static_cast<__nv_bfloat16*>(out),n,k,padded_k,tokens);
        check(cudaGetLastError());return 0;
    } catch(const std::exception& e) {last_error=e.what();return -1;}
}
