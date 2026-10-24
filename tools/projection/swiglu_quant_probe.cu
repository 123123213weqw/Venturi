// Research producer fusion; not included by blocks.cu or selected by a model.
#include "prefetch_probe.cu"
#include <cuda_profiler_api.h>

namespace vsq {
// Keep exactly the stable sigmoid, multiplication order and F16 boundary used
// by elementwise<__half,true> in blocks.cu. No fast-math/reassociation.
__device__ __forceinline__ __half value(__half a,__half b) {
    const float x=__half2float(a),y=__half2float(b);
    const float sigmoid=x>=0 ? 1.0f/(1.0f+expf(-x)) : expf(x)/(1.0f+expf(x));
    return __float2half_rn((x*sigmoid)*y);
}
__global__ void clone(const __half* a,const __half* b,__half* out,std::int64_t count) {
    for(std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<count;i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x)out[i]=value(a[i],b[i]);
}
template<int Threads,bool Materialize>
__global__ __launch_bounds__(Threads) void fused(const __half* gate,const __half* up,
    std::int8_t* codes,float* scales,__half* intermediate,int tokens,int k,int groups) {
    const int lane=threadIdx.x&31,index=blockIdx.x*(Threads/32)+threadIdx.x/32;
    if(index>=tokens*groups)return; // Whole warps return together.
    const int token=index/groups,col=(index%groups)*32+lane;
    float original=0.0f;
    if(col<k) {
        const auto offset=static_cast<std::int64_t>(token)*k+col;
        const __half rounded=value(gate[offset],up[offset]);
        if constexpr(Materialize)intermediate[offset]=rounded;
        original=__half2float(rounded); // Preserve the old observable F16 cast.
    }
    float residual=original;
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    const bool invalid=__any_sync(0xffffffff,!isfinite(original));
    #pragma unroll
    for(int p=0;p<2;++p) {
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
inline bool overlaps(const void* a,std::uint64_t an,const void* b,std::uint64_t bn) {
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(an>UINTPTR_MAX-x || bn>UINTPTR_MAX-y)throw std::invalid_argument("operand range overflow");
    return x<y+bn && y<x+an;
}
template<int Threads>void launch(const __half* gate,const __half* up,void* workspace,
    __half* debug,int tokens,int k,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto count=static_cast<std::uint64_t>(tokens)*groups;
    auto* codes=static_cast<std::int8_t*>(workspace);
    auto* scales=reinterpret_cast<float*>(codes+count*64);
    const int blocks=(count+Threads/32-1)/(Threads/32);
    if(debug)fused<Threads,true><<<blocks,Threads,0,stream>>>(gate,up,codes,scales,debug,tokens,k,groups);
    else fused<Threads,false><<<blocks,Threads,0,stream>>>(gate,up,codes,scales,nullptr,tokens,k,groups);
}
}
extern "C" int venturi_swiglu_clone_run(const void* a,const void* b,void* out,
    std::int64_t count,int dtype,void* raw_stream) {
    try {
        if(!a || !b || !out || count<=0 || dtype!=0)throw std::invalid_argument("invalid F16 clone operands");
        const int grid=static_cast<int>(std::min<std::int64_t>((count+255)/256,65535));
        vsq::clone<<<grid,256,0,static_cast<cudaStream_t>(raw_stream)>>>(
            static_cast<const __half*>(a),static_cast<const __half*>(b),static_cast<__half*>(out),count);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_swiglu_q8_quantize(const void* gate,const void* up,void* workspace,
    std::uint64_t available,int tokens,int k,int threads,void* intermediate_debug,void* raw_stream) {
    try {
        const auto needed=vdq::bytes(tokens,k,2);
        const auto input_bytes=static_cast<std::uint64_t>(tokens)*k*2;
        if(VENTURI_BLOCKS_ARCH!=70 || !gate || !up || !workspace || available<needed ||
            (threads!=128 && threads!=256) || reinterpret_cast<std::uintptr_t>(workspace)%16 ||
            reinterpret_cast<std::uintptr_t>(gate)%2 || reinterpret_cast<std::uintptr_t>(up)%2 ||
            reinterpret_cast<std::uintptr_t>(intermediate_debug)%2 ||
            vsq::overlaps(workspace,needed,gate,input_bytes) || vsq::overlaps(workspace,needed,up,input_bytes))
            throw std::invalid_argument("invalid fused SwiGLU/two-pass Q8 operands");
        if(intermediate_debug && (vsq::overlaps(workspace,needed,intermediate_debug,input_bytes) ||
            (intermediate_debug!=gate && vsq::overlaps(intermediate_debug,input_bytes,gate,input_bytes)) ||
            (intermediate_debug!=up && vsq::overlaps(intermediate_debug,input_bytes,up,input_bytes))))
            throw std::invalid_argument("invalid fused debug output overlap");
        const auto* g=static_cast<const __half*>(gate);const auto* u=static_cast<const __half*>(up);
        auto* debug=static_cast<__half*>(intermediate_debug);const auto stream=static_cast<cudaStream_t>(raw_stream);
        if(threads==128)vsq::launch<128>(g,u,workspace,debug,tokens,k,stream);
        else vsq::launch<256>(g,u,workspace,debug,tokens,k,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_sq_profile_start(){return static_cast<int>(cudaProfilerStart());}
extern "C" int venturi_sq_profile_stop(){return static_cast<int>(cudaProfilerStop());}
