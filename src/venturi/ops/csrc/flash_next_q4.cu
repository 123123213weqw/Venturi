// W4A16 reference SM70 kernel: packed signed K64, FP16 scale/weight boundary.
// One warp/output row, FP32 dot reduction. No FP8/INT4 Tensor Core assumptions.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float weight_at(const uint8_t* codes, const __half* scales,
                                          int row, int col, int padded) {
    uint8_t byte = codes[(long long)row*(padded/2)+col/2];
    int q = (col & 1) ? (byte >> 4) : (byte & 15);
    q = q < 8 ? q : q-16;
    // Match explicit FP16 decoded weights, not unrounded FP32 q*scale.
    return __half2float(__float2half_rn((float)q*__half2float(scales[(long long)row*(padded/64)+col/64])));
}

__global__ void q4_decode(const uint8_t* codes, const __half* scales, __half* out,
                          int rows, int cols, int padded) {
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    for (; i < (long long)rows*cols; i += (long long)gridDim.x*blockDim.x)
        out[i] = __float2half_rn(weight_at(codes,scales,i/cols,i%cols,padded));
}

__global__ void q4_linear(const __half* x, const uint8_t* codes, const __half* scales,
                          __half* out, int rows, int cols, int padded) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x*8+threadIdx.x/32;
    int token = blockIdx.y;
    if (row >= rows) return;
    float sum = 0;
    for (int col = lane; col < cols; col += 32)
        sum += weight_at(codes,scales,row,col,padded)*__half2float(x[(long long)token*cols+col]);
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff,sum,step);
    if (lane == 0) out[(long long)token*rows+row] = __float2half_rn(sum);
}

extern "C" int venturi_q4_abi() { return 1; }
extern "C" int venturi_q4_decode(const void* codes, const void* scales, void* out,
                                int rows, int cols, int padded, void* stream) {
    if (!codes || !scales || !out || rows <= 0 || cols <= 0 || padded < cols || padded%128)
        return (int)cudaErrorInvalidValue;
    long long blocks = ((long long)rows*cols+255)/256;
    if (blocks > 65535) blocks = 65535;
    q4_decode<<<(unsigned)blocks,256,0,(cudaStream_t)stream>>>((const uint8_t*)codes,
        (const __half*)scales,(__half*)out,rows,cols,padded);
    return (int)cudaGetLastError();
}
extern "C" int venturi_q4_linear(const void* x, const void* codes, const void* scales, void* out,
                                int tokens, int rows, int cols, int padded, void* stream) {
    if (!x || !codes || !scales || !out || tokens <= 0 || tokens > 64 || rows <= 0 || cols <= 0 ||
        padded < cols || padded%128) return (int)cudaErrorInvalidValue;
    q4_linear<<<dim3((rows+7)/8,tokens),256,0,(cudaStream_t)stream>>>((const __half*)x,
        (const uint8_t*)codes,(const __half*)scales,(__half*)out,rows,cols,padded);
    return (int)cudaGetLastError();
}
