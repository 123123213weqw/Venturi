// Experimental complete-row projection fusion. No partial-sum reduction.
#pragma once
struct VQFOutput {const std::int8_t* codes;const __half* scales;__half* out;int n,stride;};
static_assert(sizeof(VQFOutput)==32);
namespace vqf {
struct Group {VQFOutput item[3]{};int count=0;};
template<int Rows>__device__ __forceinline__ VQFOutput select(Group g,int& row) {
    int block=blockIdx.x;
    #pragma unroll
    for(int i=0;i<3;++i) {
        if(i>=g.count)break;
        const int blocks=(g.item[i].n+Rows-1)/Rows;
        if(block<blocks) {row=block*Rows+threadIdx.x/32;return g.item[i];}
        block-=blocks;
    }
    row=0;return {};
}
template<int Tokens,int Passes,int Rows>
__global__ __launch_bounds__(Rows*32,1) void global_dot(const std::int8_t* activations,const float* activation_scales,Group group,
    int tokens,int k,int padded,int groups) {
    int row;const auto plan=select<Rows>(group,row);
    const int n=plan.n;const auto* weights=plan.codes;const auto* weight_scales=plan.scales;auto* out=plan.out;
    const int lane=threadIdx.x&31,token=blockIdx.y*Tokens;
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
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*plan.stride+row]=__float2half_rn(sum[t]);
    }
}


template<int Tokens,int Passes,int Rows>
__global__ __launch_bounds__(Rows*32,1) void full_dot(const std::int8_t* activations,const float* activation_scales,Group group,
    int tokens,int k,int padded,int groups) {
    int row;const auto plan=select<Rows>(group,row);
    const int n=plan.n;const auto* weights=plan.codes;const auto* weight_scales=plan.scales;auto* out=plan.out;
    const int lane=threadIdx.x&31,token=blockIdx.y*Tokens;
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
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*plan.stride+row]=__float2half_rn(sum[t]);
    }
}



template<int Tokens,int Passes,int Rows>
__global__ __launch_bounds__(Rows*32,1) void tiled_dot(const std::int8_t* activations,const float* activation_scales,Group group,
    int tokens,int k,int padded,int groups) {
    int row;const auto plan=select<Rows>(group,row);
    const int n=plan.n;const auto* weights=plan.codes;const auto* weight_scales=plan.scales;auto* out=plan.out;
    const int lane=threadIdx.x&31,token=blockIdx.y*Tokens;
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
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*plan.stride+row]=__float2half_rn(sum[t]);
    }
}

template<int Tokens,int Passes,int Rows,int Kind>void launch(const void* workspace,Group g,int tokens,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;const auto count=static_cast<std::uint64_t>(tokens)*groups;
    auto* codes=static_cast<const std::int8_t*>(workspace);auto* scales=reinterpret_cast<const float*>(codes+count*32*Passes);
    std::uint64_t blocks=0;for(int i=0;i<g.count;++i)blocks+=(g.item[i].n+Rows-1)/Rows;
    const dim3 grid(static_cast<unsigned>(blocks),(tokens+Tokens-1)/Tokens);
    if constexpr(Kind==0)global_dot<Tokens,Passes,Rows><<<grid,Rows*32,0,stream>>>(codes,scales,g,tokens,k,padded,groups);
    else if constexpr(Kind==1)full_dot<Tokens,Passes,Rows><<<grid,Rows*32,tokens*groups*Passes*36,stream>>>(codes,scales,g,tokens,k,padded,groups);
    else tiled_dot<Tokens,Passes,Rows><<<grid,Rows*32,Tokens*Passes*32*36,stream>>>(codes,scales,g,tokens,k,padded,groups);
}
template<int Passes>void run(const void* w,Group g,int tokens,int k,int padded,cudaStream_t stream) {
    int maxn=0;for(int i=0;i<g.count;++i)maxn=std::max(maxn,g.item[i].n);
    if(tokens==1)launch<1,Passes,4,0>(w,g,tokens,k,padded,stream);
    else if(tokens==2 && vdq::bytes(tokens,k,Passes)<=48*1024)launch<2,Passes,16,1>(w,g,tokens,k,padded,stream);
    else if(tokens==2)launch<2,Passes,4,0>(w,g,tokens,k,padded,stream);
    else if(tokens>=8 && maxn>=1024)launch<8,Passes,16,2>(w,g,tokens,k,padded,stream);
    else launch<4,Passes,8,2>(w,g,tokens,k,padded,stream);
}
}
extern "C" int venturi_q8_fused_run(const void* workspace,std::uint64_t available,const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(VENTURI_BLOCKS_ARCH!=70 || !workspace || available<needed || !outputs || (count!=2 && count!=3) || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || reinterpret_cast<std::uintptr_t>(workspace)%16)throw std::invalid_argument("invalid fused DP4A input/geometry");
        vqf::Group g{};g.count=count;std::uint64_t total=0;
        for(int i=0;i<count;++i) {
            const auto& p=outputs[i];
            if(!p.codes || !p.scales || !p.out || p.n<=0 || p.n>INT32_MAX-16 || p.stride<p.n || static_cast<std::uint64_t>(tokens-1)*p.stride+p.n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(p.codes)%16)throw std::invalid_argument("invalid fused DP4A descriptor");
            total+=p.n;g.item[i]=p;
        }
        if(total>=INT32_MAX-48)throw std::invalid_argument("fused row sum exceeds int32");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        if(passes==1)vqf::run<1>(workspace,g,tokens,k,padded,stream);else vqf::run<2>(workspace,g,tokens,k,padded,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}

extern "C" int venturi_q8_t3_fused_run(const void* workspace,std::uint64_t available,const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,int rows,int staging,void* stream_ptr) {
    try {
        const auto needed=vdq::bytes(tokens,k,passes);
        if(VENTURI_BLOCKS_ARCH!=70 || tokens!=3 || (staging!=0 && staging!=1) || (rows!=4 && rows!=8 && rows!=16) || (staging==0 && needed>48*1024) || !workspace || available<needed || !outputs || (count!=2 && count!=3) || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || reinterpret_cast<std::uintptr_t>(workspace)%16)throw std::invalid_argument("invalid T3 fused research geometry/buffers");
        vqf::Group g{};g.count=count;std::uint64_t total=0;
        for(int i=0;i<count;++i) {
            const auto& p=outputs[i];
            if(!p.codes || !p.scales || !p.out || p.n<=0 || p.n>INT32_MAX-16 || p.stride<p.n || static_cast<std::uint64_t>(tokens-1)*p.stride+p.n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(p.codes)%16)throw std::invalid_argument("invalid T3 fused descriptor");
            total+=p.n;g.item[i]=p;
        }
        if(total>=INT32_MAX-48)throw std::invalid_argument("T3 fused row sum exceeds int32");
        auto stream=static_cast<cudaStream_t>(stream_ptr);
#define VENTURI_T3(P,R) if(staging==0)vqf::launch<3,P,R,1>(workspace,g,tokens,k,padded,stream);else vqf::launch<3,P,R,2>(workspace,g,tokens,k,padded,stream)
        if(passes==1) {if(rows==4){VENTURI_T3(1,4);}else if(rows==8){VENTURI_T3(1,8);}else{VENTURI_T3(1,16);}}
        else {if(rows==4){VENTURI_T3(2,4);}else if(rows==8){VENTURI_T3(2,8);}else{VENTURI_T3(2,16);}}
#undef VENTURI_T3
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
