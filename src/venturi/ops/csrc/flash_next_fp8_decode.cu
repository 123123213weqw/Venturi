// E4M3FN byte + BF16 block multiplier -> floating weight, no FP8 hardware.
// Reference operator only: no activation quantization or fused GEMM.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float decode_e4m3fn(uint8_t code) {
    unsigned sign = (unsigned)(code & 128) << 24;
    unsigned exponent = (code >> 3) & 15;
    unsigned mantissa = code & 7;
    if (exponent == 0) {
        float magnitude = (float)mantissa * 0.001953125f;
        return __uint_as_float(__float_as_uint(magnitude) | sign);
    }
    return __uint_as_float(sign | ((exponent + 120) << 23) | (mantissa << 20));
}

__global__ void decode_blocks(const uint8_t* codes, const uint16_t* scales, void* output,
                              unsigned* errors, long long m, long long n, int dtype) {
    long long total = m*n;
    long long scale_cols = (n+127)/128;
    for (long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
         index < total; index += (long long)blockDim.x*gridDim.x) {
        uint8_t code = codes[index];
        unsigned problem = 0;
        if ((code & 127) == 127) problem |= 1;
        long long row = index/n, col = index-row*n;
        float scale = __uint_as_float((unsigned)scales[(row/128)*scale_cols+(col/128)] << 16);
        if (!(scale > 0.0f) || !isfinite(scale)) problem |= 2;
        float value = decode_e4m3fn(code)*scale;
        if (!isfinite(value)) problem |= 4;
        if (dtype == 0) {
            ((float*)output)[index] = value;
        } else if (dtype == 1) {
            __half result = __float2half_rn(value);
            if (!isfinite(__half2float(result))) problem |= 4;
            ((__half*)output)[index] = result;
        } else {
            unsigned bits = __float_as_uint(value);
            uint16_t result = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
            if ((result & 0x7fff) == 0x7f80) problem |= 4;
            ((uint16_t*)output)[index] = result;
        }
        if (problem) atomicOr(errors, problem);
    }
}

extern "C" int venturi_fp8_decode_abi() { return 1; }

extern "C" int venturi_fp8_decode(const void* codes, const void* scales, void* output,
                                  void* errors, long long m, long long n, int dtype,
                                  void* stream) {
    if (!codes || !scales || !output || !errors || m <= 0 || n <= 0 || dtype < 0 || dtype > 2)
        return (int)cudaErrorInvalidValue;
    long long blocks = (m*n+255)/256;
    if (blocks > 65535) blocks = 65535;
    decode_blocks<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)codes, (const uint16_t*)scales, output, (unsigned*)errors, m, n, dtype);
    return (int)cudaGetLastError();
}
