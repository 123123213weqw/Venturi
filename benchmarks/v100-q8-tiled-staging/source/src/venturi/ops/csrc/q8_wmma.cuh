// Explicit research operator: original Q8 codes and FP16 activations, but new
// groupwise FP32/WMMA reduction. Never selected by ordinary Q8 model execution.
#pragma once
#include <mma.h>

#if VENTURI_BLOCKS_ARCH == 70
namespace vwt {
namespace wm=nvcuda::wmma;
struct __align__(32) Tile {
    __half a[16*48],b[16*48];
    float c[16*16];
};

template<int Warps>
__global__ void group32(const __half* x,const std::int8_t* codes,const __half* scales,
    __half* out,int tokens,int n,int k,int padded) {
    __shared__ Tile tiles[Warps];
    const int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const int first_row=blockIdx.x*16,first_token=blockIdx.y*16;
    const int row=first_row+lane/2,token=first_token+lane/2,packet=(lane&1)*16;
    const int groups=(k+31)/32;
    auto& tile=tiles[warp];float sum[8]{};
    wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;
    wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> b;
    wm::fragment<wm::accumulator,16,16,16,float> c;
    for(int group=warp;group<groups;group+=Warps) {
        const int col=group*32+packet;
        int4 word{};
        if(row<n)word=*reinterpret_cast<const int4*>(codes+static_cast<std::int64_t>(row)*padded+col);
        const auto* code=reinterpret_cast<const signed char*>(&word);
        #pragma unroll
        for(int j=0;j<16;++j) {
            // Integer codes in [-127,127] are exactly representable as FP16.
            // Apply the ORIGINAL weight scale only after the FP32 group dot;
            // do not round a scaled/dequantized weight into FP16 storage.
            tile.a[(lane/2)*48+packet+j]=__int2half_rn(static_cast<int>(code[j]));
            tile.b[(lane/2)*48+packet+j]=(token<tokens && col+j<k)?x[static_cast<std::int64_t>(token)*k+col+j]:__float2half_rn(0.0f);
        }
        __syncwarp();
        wm::fill_fragment(c,0.0f);
        wm::load_matrix_sync(a,tile.a,48);wm::load_matrix_sync(b,tile.b,48);
        wm::mma_sync(c,a,b,c);
        wm::load_matrix_sync(a,tile.a+16,48);wm::load_matrix_sync(b,tile.b+16,48);
        wm::mma_sync(c,a,b,c);
        wm::store_matrix_sync(tile.c,c,16,wm::mem_row_major);
        __syncwarp();
        const float scale=(lane<16 && first_row+lane<n)?__half2float(scales[static_cast<std::int64_t>(first_row+lane)*(padded/32)+group]):0.0f;
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int index=lane+32*i;
            // WMMA fragment element mapping is unspecified: scale explicit
            // row-major shared output, not guessed fragment register indices.
            sum[i]=__fmaf_rn(tile.c[index],__shfl_sync(0xffffffff,scale,index/16),sum[i]);
        }
        __syncwarp();
    }
    #pragma unroll
    for(int i=0;i<8;++i)tile.c[lane+32*i]=sum[i];
    __syncthreads();
    for(int index=threadIdx.x;index<256;index+=Warps*32) {
        float value=0.0f;
        #pragma unroll
        for(int w=0;w<Warps;++w)value+=tiles[w].c[index];
        const int r=first_row+index/16,t=first_token+index%16;
        if(r<n && t<tokens)out[static_cast<std::int64_t>(t)*n+r]=__float2half_rn(value);
    }
}
}
#endif

extern "C" int venturi_q8_wmma_run(const void* x,const std::int8_t* codes,const __half* scales,
    void* out,int tokens,int n,int k,int padded,int warps,void* stream_ptr) {
    try {
        if(!x || !codes || !scales || !out || tokens<=0 || tokens>262140 || n<=0 || n>INT32_MAX-15 || k<=0 || k>INT32_MAX-127 ||
            padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX ||
            static_cast<std::uint64_t>(tokens)*k>=INT32_MAX || reinterpret_cast<std::uintptr_t>(codes)%16 ||
            (warps!=1 && warps!=2 && warps!=4 && warps!=8))throw std::invalid_argument("invalid WMMA geometry/buffers/warp count");
#if VENTURI_BLOCKS_ARCH == 70
        const dim3 grid((n+15)/16,(tokens+15)/16);auto stream=static_cast<cudaStream_t>(stream_ptr);
        switch(warps) {
            case 1:vwt::group32<1><<<grid,32,0,stream>>>(static_cast<const __half*>(x),codes,scales,static_cast<__half*>(out),tokens,n,k,padded);break;
            case 2:vwt::group32<2><<<grid,64,0,stream>>>(static_cast<const __half*>(x),codes,scales,static_cast<__half*>(out),tokens,n,k,padded);break;
            case 4:vwt::group32<4><<<grid,128,0,stream>>>(static_cast<const __half*>(x),codes,scales,static_cast<__half*>(out),tokens,n,k,padded);break;
            case 8:vwt::group32<8><<<grid,256,0,stream>>>(static_cast<const __half*>(x),codes,scales,static_cast<__half*>(out),tokens,n,k,padded);break;
        }
        check(cudaGetLastError());return 0;
#else
        throw std::invalid_argument("WMMA group32 research operator is explicit SM70 only");
#endif
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
