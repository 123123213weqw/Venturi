// Experimental SM70 activation staging. Same quantization/reduction bytes;
// no model dispatch until matched operator and trained performance gates pass.
#pragma once
namespace vds {
template<int Tokens,int Passes,int Rows>
__global__ void dot(const std::int8_t* activations,const float* activation_scales,
    const std::int8_t* weights,const __half* weight_scales,__half* out,
    int tokens,int n,int k,int padded,int groups) {
    const int row=blockIdx.x*Rows+threadIdx.x/32,lane=threadIdx.x&31,token=blockIdx.y*Tokens;
    extern __shared__ __align__(16) char shared[];
    const int all=tokens*groups*Passes;
    auto* a0s=reinterpret_cast<int4*>(shared);
    auto* a1s=a0s+all;
    auto* scales=reinterpret_cast<float*>(a1s+all);
    for(int group=threadIdx.x;group<all;group+=blockDim.x) {
        const auto* source=reinterpret_cast<const int4*>(activations+static_cast<std::int64_t>(group)*32);
        a0s[group]=source[0];a1s[group]=source[1];scales[group]=activation_scales[group];
    }
    __syncthreads();
    if(row>=n)return;
    float sum[Tokens]{};
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    for(int col=lane*32;col<k;col+=1024) {
        // Vector transport plus one integer dot over a full scale group avoids
        // eight repeated FP32 scale/FMA operations and short column-loop trips.
        const auto* wp=reinterpret_cast<const int4*>(weights+static_cast<std::int64_t>(row)*padded+col);
        const int4 w0=wp[0],w1=wp[1];
        const float ws=__half2float(weight_scales[static_cast<std::int64_t>(row)*(padded/32)+col/32]);
        #pragma unroll
        for(int t=0;t<Tokens;++t)if(token+t<tokens) {
            #pragma unroll
            for(int p=0;p<Passes;++p) {
                const auto group=p*count+static_cast<std::int64_t>(token+t)*groups+col/32;
                const int4 a0=a0s[group],a1=a1s[group];
                int d0=__dp4a(w0.x,a0.x,0),d1=__dp4a(w1.x,a1.x,0);
                d0=__dp4a(w0.y,a0.y,d0);d1=__dp4a(w1.y,a1.y,d1);
                d0=__dp4a(w0.z,a0.z,d0);d1=__dp4a(w1.z,a1.z,d1);
                d0=__dp4a(w0.w,a0.w,d0);d1=__dp4a(w1.w,a1.w,d1);
                sum[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(ws,scales[group]),sum[t]);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        for(int delta=16;delta;delta>>=1)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],delta);
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*n+row]=__float2half_rn(sum[t]);
    }
}


template<int Tokens,int Passes,int Rows>void launch(const void* workspace,const std::int8_t* weights,const __half* scales,void* out,int n,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;
    auto* codes=static_cast<const std::int8_t*>(workspace);
    const auto* as=reinterpret_cast<const float*>(codes+static_cast<std::uint64_t>(Tokens)*groups*Passes*32);
    dot<Tokens,Passes,Rows><<<dim3((n+Rows-1)/Rows,1),Rows*32,Tokens*groups*Passes*36,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),Tokens,n,k,padded,groups);
}
template<int Tokens,int Passes>void rows(const void* workspace,const std::int8_t* w,const __half* s,void* out,int n,int k,int padded,int rows,cudaStream_t stream) {
    if(rows==4)launch<Tokens,Passes,4>(workspace,w,s,out,n,k,padded,stream);
    else if(rows==8)launch<Tokens,Passes,8>(workspace,w,s,out,n,k,padded,stream);
    else launch<Tokens,Passes,16>(workspace,w,s,out,n,k,padded,stream);
}
}
extern "C" int venturi_q8_shared_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(VENTURI_BLOCKS_ARCH!=70 || (tokens!=1 && tokens!=2) || (rows!=4 && rows!=8 && rows!=16) || needed>48*1024 || !workspace || available<needed || !codes || !scales || !out || n<=0 || n>INT32_MAX-16 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(codes)%16)throw std::invalid_argument("invalid shared DP4A geometry/buffers (SM70 T1/T2, <=48KiB)");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(tokens==1) {
            if(passes==1)vds::rows<1,1>(workspace,codes,scales,out,n,k,padded,rows,stream);else vds::rows<1,2>(workspace,codes,scales,out,n,k,padded,rows,stream);
        } else {
            if(passes==1)vds::rows<2,1>(workspace,codes,scales,out,n,k,padded,rows,stream);else vds::rows<2,2>(workspace,codes,scales,out,n,k,padded,rows,stream);
        }
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}

// Experimental K-tiled staging. Every warp participates in both barriers,
// including invalid-row and token tails. The old lane's K/FMA order is intact.
namespace vdt {
// Four low/high word-pair planes keep each dot's activation pair short-lived.
// The original two integer chains and final FP32 FMA order remain intact.
template<int All>__device__ __forceinline__ void put_pairs(int2* dst,int i,int4 lo,int4 hi) {
    dst[i]=make_int2(lo.x,hi.x);dst[All+i]=make_int2(lo.y,hi.y);
    dst[2*All+i]=make_int2(lo.z,hi.z);dst[3*All+i]=make_int2(lo.w,hi.w);
}
template<int All>__device__ __forceinline__ int paired_dot(const int2* words,int group,int4 w0,int4 w1) {
    int2 a=words[group];int d0=__dp4a(w0.x,a.x,0),d1=__dp4a(w1.x,a.y,0);
    a=words[All+group];d0=__dp4a(w0.y,a.x,d0);d1=__dp4a(w1.y,a.y,d1);
    a=words[2*All+group];d0=__dp4a(w0.z,a.x,d0);d1=__dp4a(w1.z,a.y,d1);
    a=words[3*All+group];d0=__dp4a(w0.w,a.x,d0);d1=__dp4a(w1.w,a.y,d1);
    return d0+d1;
}
template<int Tokens,int Passes,int Rows>
__global__ __launch_bounds__(Rows*32,1) void dot(const std::int8_t* activations,const float* activation_scales,
    const std::int8_t* weights,const __half* weight_scales,__half* out,
    int tokens,int n,int k,int padded,int groups) {
    const int row=blockIdx.x*Rows+threadIdx.x/32,lane=threadIdx.x&31,token=blockIdx.y*Tokens;
    extern __shared__ __align__(16) char shared[];
    constexpr int all=Tokens*Passes*32;
    auto* a0s=reinterpret_cast<int4*>(shared);auto* a1s=a0s+all;
    auto* shared_scales=reinterpret_cast<float*>(a1s+all);
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    float sum[Tokens]{};
    for(int base=0;base<k;base+=1024) {
        for(int i=threadIdx.x;i<all;i+=blockDim.x) {
            const int p=i/(Tokens*32),t=(i/32)%Tokens,g=base/32+i%32;
            if(token+t<tokens && g<groups) {
                const auto group=p*count+static_cast<std::int64_t>(token+t)*groups+g;
                const auto* src=reinterpret_cast<const int4*>(activations+group*32);
                a0s[i]=src[0];a1s[i]=src[1];shared_scales[i]=activation_scales[group];
            } else {a0s[i]=make_int4(0,0,0,0);a1s[i]=make_int4(0,0,0,0);shared_scales[i]=0;}
        }
        __syncthreads();
        const int col=base+lane*32;
        if(row<n && col<k) {
            const auto* wp=reinterpret_cast<const int4*>(weights+static_cast<std::int64_t>(row)*padded+col);
            const int4 w0=wp[0],w1=wp[1];
            const float ws=__half2float(weight_scales[static_cast<std::int64_t>(row)*(padded/32)+col/32]);
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(token+t<tokens) {
                #pragma unroll
                for(int p=0;p<Passes;++p) {
                    const int group=(p*Tokens+t)*32+lane;
                    const int4 a0=a0s[group],a1=a1s[group];
                    int d0=__dp4a(w0.x,a0.x,0),d1=__dp4a(w1.x,a1.x,0);
                    d0=__dp4a(w0.y,a0.y,d0);d1=__dp4a(w1.y,a1.y,d1);
                    d0=__dp4a(w0.z,a0.z,d0);d1=__dp4a(w1.z,a1.z,d1);
                    d0=__dp4a(w0.w,a0.w,d0);d1=__dp4a(w1.w,a1.w,d1);
                    sum[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(ws,shared_scales[group]),sum[t]);
                }
            }

        }
        __syncthreads();
    }
    if(row>=n)return;
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        for(int delta=16;delta;delta>>=1)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],delta);
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*n+row]=__float2half_rn(sum[t]);
    }
}
template<int Tokens,int Passes,int Rows>void launch(const void* workspace,const std::int8_t* weights,const __half* scales,void* out,int tokens,int n,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;auto* codes=static_cast<const std::int8_t*>(workspace);
    auto* as=reinterpret_cast<const float*>(codes+static_cast<std::uint64_t>(tokens)*groups*Passes*32);
    dot<Tokens,Passes,Rows><<<dim3((n+Rows-1)/Rows,(tokens+Tokens-1)/Tokens),Rows*32,Tokens*Passes*32*36,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
}
template<int Tokens,int Passes>void rows(const void* workspace,const std::int8_t* w,const __half* s,void* out,int tokens,int n,int k,int padded,int rows,cudaStream_t stream) {
    if(rows==4)launch<Tokens,Passes,4>(workspace,w,s,out,tokens,n,k,padded,stream);
    else if(rows==8)launch<Tokens,Passes,8>(workspace,w,s,out,tokens,n,k,padded,stream);
    else launch<Tokens,Passes,16>(workspace,w,s,out,tokens,n,k,padded,stream);
}
}
extern "C" int venturi_q8_tiled_shared_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,int tile,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(VENTURI_BLOCKS_ARCH!=70 || (tile!=4 && tile!=8) || (rows!=4 && rows!=8 && rows!=16) || !workspace || available<needed || !codes || !scales || !out || n<=0 || n>INT32_MAX-16 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(codes)%16)throw std::invalid_argument("invalid K-tiled shared DP4A geometry/buffers (SM70 tile4/8)");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(tile==4) {
            if(passes==1)vdt::rows<4,1>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);else vdt::rows<4,2>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);
        } else {
            if(passes==1)vdt::rows<8,1>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);else vdt::rows<8,2>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);
        }
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}

// Explicit T3 research entry: existing model dispatch remains unchanged until
// checkpoint-shape and trained-model gates select a measured geometry.
extern "C" int venturi_q8_t3_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,int staging,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(VENTURI_BLOCKS_ARCH!=70 || tokens!=3 || (staging!=0 && staging!=1) || (rows!=4 && rows!=8 && rows!=16) || (staging==0 && needed>48*1024) || !workspace || available<needed || !codes || !scales || !out || n<=0 || n>INT32_MAX-16 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(codes)%16)throw std::invalid_argument("invalid T3 research geometry/buffers (SM70 full<=48KiB or K-tiled)");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(staging==0) {
            if(passes==1)vds::rows<3,1>(workspace,codes,scales,out,n,k,padded,rows,stream);else vds::rows<3,2>(workspace,codes,scales,out,n,k,padded,rows,stream);
        } else {
            if(passes==1)vdt::rows<3,1>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);else vdt::rows<3,2>(workspace,codes,scales,out,tokens,n,k,padded,rows,stream);
        }
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
