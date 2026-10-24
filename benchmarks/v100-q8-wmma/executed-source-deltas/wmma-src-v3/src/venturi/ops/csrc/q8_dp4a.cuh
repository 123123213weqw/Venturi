// Explicit experimental activation quantization. Never selected by launch_q8.
// Q8 weights/scales are unchanged. FP32 group scales and optional residual Q8
// compensation define NEW arithmetic, not bit-equivalent FP16 activations.
#pragma once

namespace vdq {
inline std::uint64_t bytes(int tokens,int k,int passes) {
    if(tokens<=0 || tokens>262140 || k<=0 || k>INT32_MAX-1024 || (passes!=1 && passes!=2))throw std::invalid_argument("invalid DP4A geometry/profile");
    const auto groups=(static_cast<std::uint64_t>(k)+31)/32;
    const auto n=static_cast<std::uint64_t>(tokens)*groups;
    if(n>INT32_MAX/32)throw std::invalid_argument("DP4A activation workspace exceeds int32");
    return n*36*passes;
}

__global__ void quantize(const __half* x,std::int8_t* codes,float* scales,int tokens,int k,int groups,int passes) {
    const int lane=threadIdx.x&31,index=blockIdx.x*8+threadIdx.x/32;
    if(index>=tokens*groups)return;
    const int token=index/groups,col=(index%groups)*32+lane;
    const float original=col<k?__half2float(x[static_cast<std::int64_t>(token)*k+col]):0.0f;
    float residual=original;
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    const bool invalid=__any_sync(0xffffffff,!isfinite(original));
    for(int p=0;p<passes;++p) {
        float maximum=fabsf(residual);
        for(int delta=16;delta;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,delta));
        maximum=__shfl_sync(0xffffffff,maximum,0);
        const float scale=maximum/127.0f;
        const float inverse=maximum>0?127.0f/maximum:0.0f;
        const int code=invalid?0:max(-127,min(127,__float2int_rn(__fmul_rn(residual,inverse))));
        codes[(p*count+index)*32+lane]=static_cast<std::int8_t>(code);
        if(lane==0)scales[p*count+index]=invalid?CUDART_NAN_F:scale;
        residual=__fmaf_rn(-static_cast<float>(code),scale,residual);
    }
}

template<int Tokens,int Passes>
__global__ void dot(const std::int8_t* activations,const float* activation_scales,
    const std::int8_t* weights,const __half* weight_scales,__half* out,
    int tokens,int n,int k,int padded,int groups) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31,token=blockIdx.y*Tokens;
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
                const auto* ap=reinterpret_cast<const int4*>(activations+group*32);
                const int4 a0=ap[0],a1=ap[1];
                int d0=__dp4a(w0.x,a0.x,0),d1=__dp4a(w1.x,a1.x,0);
                d0=__dp4a(w0.y,a0.y,d0);d1=__dp4a(w1.y,a1.y,d1);
                d0=__dp4a(w0.z,a0.z,d0);d1=__dp4a(w1.z,a1.z,d1);
                d0=__dp4a(w0.w,a0.w,d0);d1=__dp4a(w1.w,a1.w,d1);
                sum[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(ws,activation_scales[group]),sum[t]);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        for(int delta=16;delta;delta>>=1)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],delta);
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*n+row]=__float2half_rn(sum[t]);
    }
}

template<int Passes>void launch(const void* workspace,const std::int8_t* weights,const __half* scales,void* out,int tokens,int n,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto count=static_cast<std::uint64_t>(tokens)*groups;
    auto* codes=static_cast<const std::int8_t*>(workspace);
    auto* as=reinterpret_cast<const float*>(codes+count*32*Passes);
    const dim3 grid((n+3)/4,(tokens+3)/4);
    if(tokens==1)dot<1,Passes><<<dim3((n+3)/4,1),128,0,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
    else if(tokens==2)dot<2,Passes><<<dim3((n+3)/4,1),128,0,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
    else if(tokens>=8)dot<8,Passes><<<dim3((n+3)/4,(tokens+7)/8),128,0,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
    else dot<4,Passes><<<grid,128,0,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
}
}

extern "C" std::uint64_t venturi_q8_dp4a_bytes(int tokens,int k,int passes) {
    try{return vdq::bytes(tokens,k,passes);}catch(const std::exception& e){error_text=e.what();return 0;}
}
extern "C" int venturi_q8_dp4a_quantize(const void* x,void* workspace,std::uint64_t available,int tokens,int k,int passes,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(!x || !workspace || available<needed || reinterpret_cast<std::uintptr_t>(workspace)%16)throw std::invalid_argument("invalid DP4A workspace/input");
        const int groups=(k+31)/32,count=tokens*groups;
        auto* codes=static_cast<std::int8_t*>(workspace);auto* scales=reinterpret_cast<float*>(codes+static_cast<std::uint64_t>(count)*32*passes);
        vdq::quantize<<<(count+7)/8,256,0,static_cast<cudaStream_t>(stream_ptr)>>>(static_cast<const __half*>(x),codes,scales,tokens,k,groups,passes);check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_q8_dp4a_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(!workspace || available<needed || !codes || !scales || !out || n<=0 || n>INT32_MAX-3 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(codes)%16)throw std::invalid_argument("invalid DP4A projection geometry/buffers");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(passes==1)vdq::launch<1>(workspace,codes,scales,out,tokens,n,k,padded,stream);else vdq::launch<2>(workspace,codes,scales,out,tokens,n,k,padded,stream);check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
