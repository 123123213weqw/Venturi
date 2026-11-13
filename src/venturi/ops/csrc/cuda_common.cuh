// Stateless CUDA scalar/error helpers. Each library owns its own error channel.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace {
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
template<class T> __device__ float read(T x) { return static_cast<float>(x); }
template<> __device__ float read(__half x) { return __half2float(x); }
template<> __device__ float read(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> __device__ T cast(float x) { return static_cast<T>(x); }
template<> __device__ __half cast(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }

inline void require(bool value,const char* message) {
    if (!value) throw std::invalid_argument(message);
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
__device__ float sigmoid(float x) {
    return x >= 0 ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

__device__ __forceinline__ float gdn_compact_sum(float value,float inactive,float* partial) {
    for(int delta=16;delta;delta>>=1)value+=__shfl_down_sync(0xffffffff,value,delta);
    if((threadIdx.x&31)==0)partial[threadIdx.x>>5]=value;
    if(threadIdx.x>=blockDim.x/32 && threadIdx.x<8) {
        for(int delta=16;delta;delta>>=1)inactive+=inactive;
        partial[threadIdx.x]=inactive;
    }
    __syncthreads();
    if(threadIdx.x<32) {
        value=threadIdx.x<8?partial[threadIdx.x]:0.0f;
        for(int delta=16;delta;delta>>=1)value+=__shfl_down_sync(0xffffffff,value,delta);
        if(threadIdx.x==0)partial[0]=value;
    }
    __syncthreads();
    const float result=partial[0];
    __syncthreads();return result;
}
}
