// Bit-preserving post-attention RMS/activation packing; native routing opt-in.
#pragma once
#ifndef VENTURI_Q8_RMS_QUANT
#define VENTURI_Q8_RMS_QUANT 0
#endif
static_assert(VENTURI_Q8_RMS_QUANT==0 || VENTURI_Q8_RMS_QUANT==1);
namespace vrq {
template<int Chunk,bool Debug>
__global__ void fused(const __half* x,const float* weight,std::int8_t* codes,float* scales,
    __half* debug,int k,int groups,int tokens,float epsilon) {
    // Replicate the original 256-thread reduction without changing association.
    // Several chunks can independently recompute statistics for one token, trading
    // redundant reads/arithmetic for enough CTAs at small decode batches.
    __shared__ float partial[8];
    const int token=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto offset=static_cast<std::int64_t>(token)*k;
    float sum=0.0f;
    for(std::int64_t col=threadIdx.x;col<k;col+=256) {
        const float v=__half2float(x[offset+col]);sum=fmaf(v,v,sum);
    }
    for(int delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x<32) {
        sum=threadIdx.x<8?partial[threadIdx.x]:0.0f;
        for(int delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffff,sum,delta);
        if(threadIdx.x==0)partial[0]=rsqrtf(sum/k+epsilon);
    }
    __syncthreads();
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    for(int g=blockIdx.x*Chunk+warp;g<groups && g<(blockIdx.x+1)*Chunk;g+=8) {
        const int col=g*32+lane,index=token*groups+g;
        float original=0.0f;
        if(col<k) {
            const __half rounded=__float2half_rn((__half2float(x[offset+col])*partial[0])*weight[col]);
            if constexpr(Debug)debug[offset+col]=rounded;
            original=__half2float(rounded);
        }
        float residual=original;
        const bool invalid=__any_sync(0xffffffff,!isfinite(original));
        #pragma unroll
        for(int p=0;p<2;++p) {
            float maximum=fabsf(residual);
            for(int delta=16;delta;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,delta));
            maximum=__shfl_sync(0xffffffff,maximum,0);
            const float scale=maximum/127.0f,inverse=maximum>0?127.0f/maximum:0.0f;
            const int code=invalid?0:max(-127,min(127,__float2int_rn(__fmul_rn(residual,inverse))));
            codes[(p*count+index)*32+lane]=static_cast<std::int8_t>(code);
            if(lane==0)scales[p*count+index]=invalid?CUDART_NAN_F:scale;
            residual=__fmaf_rn(-static_cast<float>(code),scale,residual);
        }
    }
}
inline bool overlap(const void* a,std::uint64_t an,const void* b,std::uint64_t bn) {
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(an>UINTPTR_MAX-x || bn>UINTPTR_MAX-y)throw std::invalid_argument("range overflow");
    return x<y+bn && y<x+an;
}
template<int Chunk>void launch(const __half* x,const float* w,void* work,__half* debug,int t,int k,float eps,cudaStream_t stream) {
    const int groups=(k+31)/32;const auto count=static_cast<std::uint64_t>(t)*groups;
    auto* codes=static_cast<std::int8_t*>(work);auto* scales=reinterpret_cast<float*>(codes+count*64);
    const dim3 grid((groups+Chunk-1)/Chunk,t);
    if(debug)fused<Chunk,true><<<grid,256,0,stream>>>(x,w,codes,scales,debug,k,groups,t,eps);
    else fused<Chunk,false><<<grid,256,0,stream>>>(x,w,codes,scales,nullptr,k,groups,t,eps);
}
}
extern "C" int venturi_rms_q8_quantize(const void* x,const float* w,void* workspace,std::uint64_t available,
    int t,int k,float epsilon,int chunk,void* intermediate,void* raw_stream) {
    try {
        const auto needed=vdq::bytes(t,k,2),xb=static_cast<std::uint64_t>(t)*k*2,wb=static_cast<std::uint64_t>(k)*4;
        if(VENTURI_BLOCKS_ARCH!=70 || !x || !w || !workspace || available<needed || !(epsilon>0) || !isfinite(epsilon) ||
            (chunk!=8 && chunk!=32 && chunk!=128) || t>65535 ||
            reinterpret_cast<std::uintptr_t>(x)%2 || reinterpret_cast<std::uintptr_t>(w)%4 ||
            reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(intermediate)%2 ||
            vrq::overlap(workspace,needed,x,xb) || vrq::overlap(workspace,needed,w,wb) ||
            (intermediate && (vrq::overlap(intermediate,xb,x,xb) || vrq::overlap(intermediate,xb,w,wb) || vrq::overlap(intermediate,xb,workspace,needed))))
            throw std::invalid_argument("invalid RMS/two-pass Q8 operands");
        const auto* input=static_cast<const __half*>(x);auto* debug=static_cast<__half*>(intermediate);auto stream=static_cast<cudaStream_t>(raw_stream);
        if(chunk==8)vrq::launch<8>(input,w,workspace,debug,t,k,epsilon,stream);
        else if(chunk==32)vrq::launch<32>(input,w,workspace,debug,t,k,epsilon,stream);
        else vrq::launch<128>(input,w,workspace,debug,t,k,epsilon,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}

namespace vrq {
inline bool eligible(int tokens,int k) {
    return VENTURI_Q8_RMS_QUANT && VENTURI_BLOCKS_ARCH==70 && tokens>=1 && tokens<=32 && k==5120;
}
}
extern "C" int venturi_q8_rms_quant_enabled(){return VENTURI_Q8_RMS_QUANT;}
extern "C" int venturi_q8_rms_quant_eligible(int t,int k){return vrq::eligible(t,k);}
