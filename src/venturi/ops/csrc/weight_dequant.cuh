// Venturi row-codec weight dequantization kernels (q4/q5/q6_g64 family).
// Included inside the private operator namespace vrow by blocks.cu. Contract:
// Signed row-split group decode: per row, groups = padded/64;
// group codes are 32 packed bytes (even lane = low nibble, odd = high nibble);
// q5/q6 add a high-bit plane ((bits-4) bits per lane, LSB-first little-endian
// stream, (bits-4)*64/8 bytes per group); values are bits-wide two's
// complement scaled by one little-endian FP16 scale per group; padding lanes
// (col >= k) are never written. Output is a contiguous FP16 [rows, k] matrix
// with ld = k, ready for venturi_dense_run.
#pragma once


// Adjacent lanes decode adjacent octets, rather than one lane serially writing
// a whole k64 group. Coalesced uint32 reads/uint4 stores change IO geometry only:
// each signed integer * FP16 scale still rounds once to FP16 exactly as before.
__device__ __forceinline__ void row_dequant_octet(
    const unsigned char* codes,const unsigned char* high,const __half* scales,
    __half* out,std::uint64_t row,int local,int col,int k,int padded,int bits) {
    const int groups=padded>>6,hb=bits-4;
    const auto* c=codes+row*(padded>>1)+(col>>1);
    const std::uint32_t packed=(reinterpret_cast<std::uintptr_t>(c)&3u)==0
        ?*reinterpret_cast<const std::uint32_t*>(c)
        :std::uint32_t(c[0])|(std::uint32_t(c[1])<<8)|(std::uint32_t(c[2])<<16)|(std::uint32_t(c[3])<<24);
    unsigned hi=0;
    if(hb) {
        const auto* h=high+row*groups*hb*8+(col*hb>>3);
        hi=h[0];if(hb==2)hi|=unsigned(h[1])<<8;
    }
    const float scale=__half2float(scales[row*groups+(col>>6)]);
    const int sign=1<<(bits-1),mask=(1<<hb)-1;
    __half value[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        const int raw=int((packed>>(j*4))&15u)|(int((hi>>(j*hb))&mask)<<4);
        const int q=(raw^sign)-sign;
        value[j]=__float2half_rn(float(q)*scale);
    }
    auto* d=out+std::uint64_t(local)*k+col;
    if(col+7<k && (reinterpret_cast<std::uintptr_t>(d)&15u)==0) {
        const uint4 words=make_uint4(
            std::uint32_t(__half_as_ushort(value[0]))|(std::uint32_t(__half_as_ushort(value[1]))<<16),
            std::uint32_t(__half_as_ushort(value[2]))|(std::uint32_t(__half_as_ushort(value[3]))<<16),
            std::uint32_t(__half_as_ushort(value[4]))|(std::uint32_t(__half_as_ushort(value[5]))<<16),
            std::uint32_t(__half_as_ushort(value[6]))|(std::uint32_t(__half_as_ushort(value[7]))<<16));
        *reinterpret_cast<uint4*>(d)=words;
    } else {
        // Odd K, offset contiguous output views and final partial octets.
        #pragma unroll
        for(int j=0;j<8;++j)if(col+j<k)d[j]=value[j];
    }
}

__global__ inline void row_dequant_rows_kernel(
    const unsigned char* codes,const unsigned char* high,const __half* scales,
    __half* out,int row_begin,int rows,int k,int padded,int bits) {
    const int local=blockIdx.y,col=(blockIdx.x*blockDim.x+threadIdx.x)*8;
    if(local>=rows || col>=k)return;
    row_dequant_octet(codes,high,scales,out,std::uint64_t(row_begin)+local,local,col,k,padded,bits);
}

__global__ inline void row_dequant_indexed_kernel(
    const unsigned char* codes,const unsigned char* high,const __half* scales,
    const std::int64_t* ids,__half* out,int count,int k,int padded,int bits) {
    const int slot=blockIdx.y,col=(blockIdx.x*blockDim.x+threadIdx.x)*8;
    if(slot>=count || col>=k)return;
    row_dequant_octet(codes,high,scales,out,static_cast<std::uint64_t>(ids[slot]),slot,col,k,padded,bits);
}
