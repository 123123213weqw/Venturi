// Same byte/rounding contract as linear v1; row tiles remove per-element64-bit division.
#define venturi_fp8_decode venturi_fp8_decode_linear
#include "flash_next_fp8_decode.cu"
#undef venturi_fp8_decode

__global__ void decode_blocks_2d(const uint8_t* codes, const uint16_t* scales, void* output,
                               unsigned* errors, long long m, long long n, int dtype) {
    long long row = blockIdx.y;
    long long col = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= n) return;
    long long index = row*n+col;
    uint8_t code = codes[index];
    unsigned problem = 0;
    if ((code & 127) == 127) problem |= 1;
    float scale = __uint_as_float((unsigned)scales[(row/128)*((n+127)/128)+(col/128)] << 16);
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
        uint16_t result = (uint16_t)((bits+0x7fffu+((bits >> 16) & 1u)) >> 16);
        if ((result & 0x7fff) == 0x7f80) problem |= 4;
        ((uint16_t*)output)[index] = result;
    }
    if (problem) atomicOr(errors, problem);
}

extern "C" int venturi_fp8_decode(const void* codes, const void* scales, void* output,
                                  void* errors, long long m, long long n, int dtype,
                                  void* stream) {
    if (!codes || !scales || !output || !errors || m <= 0 || n <= 0 || dtype < 0 || dtype > 2)
        return (int)cudaErrorInvalidValue;
    if (m > 65535) return venturi_fp8_decode_linear(codes, scales, output, errors, m, n, dtype, stream);
    dim3 grid((unsigned)((n+255)/256), (unsigned)m);
    decode_blocks_2d<<<grid, 256, 0, (cudaStream_t)stream>>>((const uint8_t*)codes,
        (const uint16_t*)scales, output, (unsigned*)errors, m, n, dtype);
    return (int)cudaGetLastError();
}
