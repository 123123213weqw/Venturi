// SM70/SM89 common row-codec G64 / FP16 projection. The packed-plane reader
// is parameterized by bit width; staging, MMA and output handling are shared.
#pragma once
#include <mma.h>
namespace vrowtile {
using namespace nvcuda;

__device__ __forceinline__ uint32_t packed_mul2(const __half2 x,const __half2 y) {
    uint32_t xx, yy, zz;
    __builtin_memcpy(&xx,&x,sizeof(xx));
    __builtin_memcpy(&yy,&y,sizeof(yy));
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(zz) : "r"(xx), "r"(yy));
    return zz;
}

// Values are two's-complement integers. Every format has a packed low
// nibble; Q5/Q6 add 1/2 bits per lane in an optional little-endian plane.
template<int Bits> __device__ __forceinline__ void decode_pair(
    unsigned low,unsigned high,int& left,int& right) {
    constexpr unsigned extra=Bits-4, mask=(1u<<extra)-1, sign=1u<<(Bits-1);
    left=int(((low&15u)|((high&mask)<<4))^sign)-int(sign);
    right=int((((low>>4)&15u)|(((high>>extra)&mask)<<4))^sign)-int(sign);
}
template<int Bits> __device__ __forceinline__ unsigned read_high(const uint8_t* h,size_t octet) {
    if constexpr(Bits==4)return 0;
    if constexpr(Bits==5)return h[octet];
    if constexpr(Bits==6)return unsigned(h[2*octet])|(unsigned(h[2*octet+1])<<8);
    return 0;
}

// One warp per output channel; a round of decoded weights is consumed by
// multiple independent accumulators. The body is the same for all codecs.
template<int Bits,int MT> __global__ __launch_bounds__(256) void row_grouped_gemv(
    const half* __restrict__ x,const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ high,const half* __restrict__ scale,
    half* __restrict__ y,int m,int n,int k,int padded) {
    const int lane=threadIdx.x&31,r=blockIdx.x*8+(threadIdx.x>>5);
    if(r>=n)return;
    const int groups=(k-1)/64+1;
    float acc[MT]{};
    for(int g=0;g<groups;++g) {
        const int col=g*64+2*lane;
        const auto octet=size_t(r)*(padded/8)+g*8+lane/4;
        const unsigned p=codes[size_t(r)*(padded/2)+g*32+lane];
        const unsigned h=read_high<Bits>(high,octet)>>((lane&3)*2*(Bits-4));
        int lo,hi;
        decode_pair<Bits>(p,h,lo,hi);
        const float s=__half2float(scale[size_t(r)*(padded/64)+g]);
        const float v0=__half2float(__float2half_rn(float(lo)*s)),v1=__half2float(__float2half_rn(float(hi)*s));
        #pragma unroll
        for(int t=0;t<MT;++t)if(t<m) {
            const auto* a=x+size_t(t)*k;
            const float u0=col<k?__half2float(a[col]):0.0f,u1=col+1<k?__half2float(a[col+1]):0.0f;
            acc[t]+=v0*u0+v1*u1;
        }
    }
    #pragma unroll
    for(int t=0;t<MT;++t) {
        #pragma unroll
        for(int delta=16;delta;delta>>=1)acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],delta);
        if(lane==0 && t<m)y[size_t(t)*n+r]=__float2half_rn(acc[t]);
    }
}
template<int Bits,int W,int M_TILE,bool Masked>
__global__ __launch_bounds__(W*32) void row_tensor_tile(
    const half* __restrict__ x,
    const uint32_t* __restrict__ codes, const uint8_t* __restrict__ high,
    const half* __restrict__ scale, half* __restrict__ y, int m_total, int n, int k, int padded) {
    constexpr int stride = 72, threads=32*W, m_frags=M_TILE/16;
    __shared__ __align__(32) half a[M_TILE][stride];
    __shared__ __align__(32) half b[W][16][stride];
    __shared__ __align__(32) float out[W][16][16];
    const int tx=threadIdx.x, warp=tx/32, lane=tx%32;
    const int m0=blockIdx.y*M_TILE, n0=(blockIdx.x*W+warp)*16;
    using CF=wmma::fragment<wmma::accumulator,16,16,16,float>;
    CF acc[m_frags];
    #pragma unroll
    for(int m=0;m<m_frags;++m)wmma::fill_fragment(acc[m],0.f);
    const int groups=Masked?(k-1)/64+1:k/64;
    for(int g=0;g<groups;++g) {
        for(int i=tx;i<M_TILE*8;i+=threads) {
            const int row=i/8, col=i%8*8;
            if constexpr(Masked) {
                const int c=g*64+col;
                if(m0+row>=m_total)
                    *reinterpret_cast<uint4*>(a[row]+col)=make_uint4(0,0,0,0);
                else {
                    const auto* xp=x+size_t(m0+row)*k;
                    if(c+7<k && !(reinterpret_cast<uintptr_t>(xp+c)&15u))
                        *reinterpret_cast<uint4*>(a[row]+col)=*reinterpret_cast<const uint4*>(xp+c);
                    else for(int j=0;j<8;++j)
                        a[row][col+j]=c+j<k?xp[c+j]:__float2half_rn(0.f);
                }
            } else {
                *reinterpret_cast<uint4*>(a[row]+col)=
                    *reinterpret_cast<const uint4*>(x+size_t(m0+row)*k+g*64+col);
            }
        }
        #pragma unroll
        for(int i=lane;i<128;i+=32) {
            const int row=i/8, c8=i%8;
            if constexpr(Masked) {
                if(n0+row>=n) {
                    *reinterpret_cast<uint4*>(b[warp][row]+c8*8)=make_uint4(0,0,0,0);
                    continue;
                }
            }
            const int pitch=Masked?padded:k;
            const size_t base=size_t(n0+row)*(pitch/8)+g*8+c8;
            const auto low=codes[base], hb=read_high<Bits>(high,base);
            const half2 s=__half2half2(scale[size_t(n0+row)*(pitch/64)+g]);
            #pragma unroll
            for(int j=0;j<4;++j) {
                int lo,hi;
                decode_pair<Bits>(low>>(j*8),hb>>(j*2*(Bits-4)),lo,hi);
                const half2 q=__floats2half2_rn(float(lo),float(hi));
                *reinterpret_cast<uint32_t*>(b[warp][row]+c8*8+2*j)=packed_mul2(q,s);
            }
        }
        __syncthreads();
        #pragma unroll
        for(int step=0;step<4;++step) {
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> wb;
            wmma::load_matrix_sync(wb,&b[warp][0][16*step],stride);
            #pragma unroll
            for(int m=0;m<m_frags;++m) {
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aa;
                wmma::load_matrix_sync(aa,&a[m*16][16*step],stride);
                wmma::mma_sync(acc[m],aa,wb,acc[m]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<m_frags;++m) {
        wmma::store_matrix_sync(&out[warp][0][0],acc[m],16,wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for(int i=lane;i<256;i+=32) {
            const int r=m0+m*16+i/16,c=n0+i%16;
            if(!Masked || (r<m_total && c<n))
                y[size_t(r)*n+c]=__float2half_rn(out[warp][i/16][i%16]);
        }
        __syncwarp();
    }
}

template<int Bits,int W,int TM>
void launch_wmma(const half* x,const uint8_t* c,const uint8_t* h,const half* s,
                 half* y,int m,int n,int k,int padded,cudaStream_t st) {
    const dim3 grid((n-1)/(16*W)+1,(m-1)/TM+1);
    const auto* low=reinterpret_cast<const uint32_t*>(c);
    if(m%TM==0 && n%(16*W)==0 && k==padded && !(reinterpret_cast<uintptr_t>(x)&15u))
        row_tensor_tile<Bits,W,TM,false><<<grid,W*32,0,st>>>(x,low,h,s,y,m,n,k,padded);
    else
        row_tensor_tile<Bits,W,TM,true><<<grid,W*32,0,st>>>(x,low,h,s,y,m,n,k,padded);
}

template<int Bits>
void launch(const half* x,const uint8_t* c,const uint8_t* h,const half* s,
            half* y,int m,int n,int k,int padded,bool grouped,cudaStream_t st) {
    if(grouped) {
        const dim3 grid((n-1)/8+1);
        if(m==1)row_grouped_gemv<Bits,1><<<grid,256,0,st>>>(x,c,h,s,y,m,n,k,padded);
        else if(m==2)row_grouped_gemv<Bits,2><<<grid,256,0,st>>>(x,c,h,s,y,m,n,k,padded);
        else if(m<=4)row_grouped_gemv<Bits,4><<<grid,256,0,st>>>(x,c,h,s,y,m,n,k,padded);
        else row_grouped_gemv<Bits,8><<<grid,256,0,st>>>(x,c,h,s,y,m,n,k,padded);
    } else if(m<=16)launch_wmma<Bits,4,16>(x,c,h,s,y,m,n,k,padded,st);
    else if(m<=32)launch_wmma<Bits,2,32>(x,c,h,s,y,m,n,k,padded,st);
    else launch_wmma<Bits,4,64>(x,c,h,s,y,m,n,k,padded,st);
}
} // namespace vrowtile
