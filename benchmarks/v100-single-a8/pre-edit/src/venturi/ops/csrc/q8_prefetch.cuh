// Explicit SM70 research operators. Default/native model dispatch is unchanged.
// Hoist weight loads (mode 1), or keep one K tile ahead in registers (mode 2).
// Preserve group/pass/FMA/reduction order, scales, and output rounding.
#pragma once

namespace vpf {
struct WeightTile {int4 a,b;float scale;};
__device__ __forceinline__ WeightTile load(VQFOutput plan,int row,int col,int k,int padded) {
    if(row>=plan.n || col>=k)return {make_int4(0,0,0,0),make_int4(0,0,0,0),0};
    const auto* w=reinterpret_cast<const int4*>(plan.codes+static_cast<std::int64_t>(row)*padded+col);
    return {w[0],w[1],__half2float(plan.scales[static_cast<std::int64_t>(row)*(padded/32)+col/32])};
}
template<int Mode,bool Partial>
__global__ __launch_bounds__(512,1) void dot(const std::int8_t* activations,const float* scales,
    vqf::Group group,int tokens,int k,int padded,int groups) {
    constexpr int T=8,P=2,R=16,all=T*P*32;
    int row;const auto plan=vqf::select<R>(group,row);
    const int lane=threadIdx.x&31,token=blockIdx.y*T;
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    extern __shared__ __align__(16) char shared[];
    auto* a0s=reinterpret_cast<int4*>(shared);auto* a1s=a0s+all;
    auto* ss=reinterpret_cast<float*>(a1s+all);
    float sums[T]{};
    WeightTile current{};
    if constexpr(Mode==2)current=load(plan,row,lane*32,k,padded);
    for(int base=0;base<k;base+=1024) {
        WeightTile next{};
        if constexpr(Mode==1)current=load(plan,row,base+lane*32,k,padded);
        else next=load(plan,row,base+1024+lane*32,k,padded);
        for(int i=threadIdx.x;i<all;i+=blockDim.x) {
            const int pass=i/(T*32),t=(i/32)%T,g=base/32+i%32;
            if(token+t<tokens && g<groups) {
                const auto index=pass*count+static_cast<std::int64_t>(token+t)*groups+g;
                const auto* x=reinterpret_cast<const int4*>(activations+index*32);
                a0s[i]=x[0];a1s[i]=x[1];ss[i]=scales[index];
            }else{a0s[i]=make_int4(0,0,0,0);a1s[i]=make_int4(0,0,0,0);ss[i]=0;}
        }
        __syncthreads();
        if(row<plan.n && base+lane*32<k) {
            #pragma unroll
            for(int t=0;t<T;++t)if(token+t<tokens) {
                #pragma unroll
                for(int pass=0;pass<P;++pass) {
                    const int i=(pass*T+t)*32+lane;
                    const int4 a=a0s[i],b=a1s[i];
                    int d0=__dp4a(current.a.x,a.x,0),d1=__dp4a(current.b.x,b.x,0);
                    d0=__dp4a(current.a.y,a.y,d0);d1=__dp4a(current.b.y,b.y,d1);
                    d0=__dp4a(current.a.z,a.z,d0);d1=__dp4a(current.b.z,b.z,d1);
                    d0=__dp4a(current.a.w,a.w,d0);d1=__dp4a(current.b.w,b.w,d1);
                    sums[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(current.scale,ss[i]),sums[t]);
                }
            }
        }
        __syncthreads();
        if constexpr(Mode==2)current=next;
    }
    if(row>=plan.n)return;
    #pragma unroll
    for(int t=0;t<T;++t) {
        for(int delta=16;delta;delta>>=1)sums[t]+=__shfl_down_sync(0xffffffff,sums[t],delta);
        if(lane==0 && token+t<tokens) {
            const auto index=static_cast<std::int64_t>(token+t)*plan.stride+row;
            if constexpr(Partial)reinterpret_cast<float*>(plan.out)[index]=sums[t];
            else plan.out[index]=__float2half_rn(sums[t]);
        }
    }
}

// One warp owns two output rows. Reuse each activation/scalar load for both,
// preserving each row's independent pass/FMA/reduction order.
template<bool Partial>
__global__ __launch_bounds__(256,1) void row_pair(const std::int8_t* activations,const float* scales,
    vqf::Group group,int tokens,int k,int padded,int groups) {
    constexpr int T=8,P=2,R=16,all=T*P*32;
    int row;const auto plan=vqf::select<R>(group,row);
    const int other=row+8,lane=threadIdx.x&31,token=blockIdx.y*T;
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    extern __shared__ __align__(16) char shared[];
    auto* a0s=reinterpret_cast<int4*>(shared);auto* a1s=a0s+all;
    auto* ss=reinterpret_cast<float*>(a1s+all);
    float first[T]{},second[T]{};
    for(int base=0;base<k;base+=1024) {
        for(int i=threadIdx.x;i<all;i+=blockDim.x) {
            const int pass=i/(T*32),t=(i/32)%T,g=base/32+i%32;
            if(token+t<tokens && g<groups) {
                const auto index=pass*count+static_cast<std::int64_t>(token+t)*groups+g;
                const auto* x=reinterpret_cast<const int4*>(activations+index*32);
                a0s[i]=x[0];a1s[i]=x[1];ss[i]=scales[index];
            }else{a0s[i]=make_int4(0,0,0,0);a1s[i]=make_int4(0,0,0,0);ss[i]=0;}
        }
        __syncthreads();
        const int col=base+lane*32;
        if(row<plan.n && col<k) {
            const auto w=load(plan,row,col,k,padded),v=load(plan,other,col,k,padded);
            #pragma unroll
            for(int t=0;t<T;++t)if(token+t<tokens) {
                #pragma unroll
                for(int pass=0;pass<P;++pass) {
                    const int i=(pass*T+t)*32+lane;
                    const int4 a=a0s[i],b=a1s[i];const float scale=ss[i];
                    int d0=__dp4a(w.a.x,a.x,0),d1=__dp4a(w.b.x,b.x,0);
                    d0=__dp4a(w.a.y,a.y,d0);d1=__dp4a(w.b.y,b.y,d1);
                    d0=__dp4a(w.a.z,a.z,d0);d1=__dp4a(w.b.z,b.z,d1);
                    d0=__dp4a(w.a.w,a.w,d0);d1=__dp4a(w.b.w,b.w,d1);
                    first[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(w.scale,scale),first[t]);
                    if(other<plan.n) {
                        int e0=__dp4a(v.a.x,a.x,0),e1=__dp4a(v.b.x,b.x,0);
                        e0=__dp4a(v.a.y,a.y,e0);e1=__dp4a(v.b.y,b.y,e1);
                        e0=__dp4a(v.a.z,a.z,e0);e1=__dp4a(v.b.z,b.z,e1);
                        e0=__dp4a(v.a.w,a.w,e0);e1=__dp4a(v.b.w,b.w,e1);
                        second[t]=__fmaf_rn(static_cast<float>(e0+e1),__fmul_rn(v.scale,scale),second[t]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if(row>=plan.n)return;
    #pragma unroll
    for(int t=0;t<T;++t) {
        for(int delta=16;delta;delta>>=1) {
            first[t]+=__shfl_down_sync(0xffffffff,first[t],delta);
            second[t]+=__shfl_down_sync(0xffffffff,second[t],delta);
        }
        if(lane==0 && token+t<tokens) {
            const auto index=static_cast<std::int64_t>(token+t)*plan.stride+row;
            if constexpr(Partial) {
                reinterpret_cast<float*>(plan.out)[index]=first[t];
                if(other<plan.n)reinterpret_cast<float*>(plan.out)[index+8]=second[t];
            }else {
                plan.out[index]=__float2half_rn(first[t]);
                if(other<plan.n)plan.out[index+8]=__float2half_rn(second[t]);
            }
        }
    }
}
template<bool Partial>void launch(const void* workspace,vqf::Group g,int tokens,int k,int padded,int mode,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto count=static_cast<std::uint64_t>(tokens)*groups;
    auto* x=static_cast<const std::int8_t*>(workspace);
    auto* scales=reinterpret_cast<const float*>(x+count*32*2);
    int blocks=0;for(int i=0;i<g.count;++i)blocks+=(g.item[i].n+15)/16;
    const dim3 grid(blocks,(tokens+7)/8);
    if(mode==1)dot<1,Partial><<<grid,512,18432,stream>>>(x,scales,g,tokens,k,padded,groups);
    else if(mode==2)dot<2,Partial><<<grid,512,18432,stream>>>(x,scales,g,tokens,k,padded,groups);
    else row_pair<Partial><<<grid,256,18432,stream>>>(x,scales,g,tokens,k,padded,groups);
}
}

extern "C" int venturi_q8_prefetch_fused_run(const void* workspace,std::uint64_t available,
    const VQFOutput* outputs,int count,int tokens,int k,int padded,int mode,void* raw_stream) {
    try {
        if(VENTURI_BLOCKS_ARCH!=70 || !workspace || available<vdq::bytes(tokens,k,2) || !outputs ||
            (count!=2 && count!=3) || mode<1 || mode>3 ||
            padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || reinterpret_cast<std::uintptr_t>(workspace)%16)
            throw std::invalid_argument("invalid explicit prefetch fused geometry");
        vqf::Group g{};g.count=count;std::uint64_t total=0;
        for(int i=0;i<count;++i) {
            const auto& p=outputs[i];
            if(!p.codes || !p.scales || !p.out || p.n<=0 || p.n>INT32_MAX-16 || p.stride<p.n ||
                static_cast<std::uint64_t>(tokens-1)*p.stride+p.n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(p.codes)%16)
                throw std::invalid_argument("invalid explicit prefetch fused descriptor");
            g.item[i]=p;total+=p.n;
        }
        if(total>=INT32_MAX-48)throw std::invalid_argument("prefetch fused row sum exceeds int32");
        vpf::launch<false>(workspace,g,tokens,k,padded,mode,static_cast<cudaStream_t>(raw_stream));
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_q8_prefetch_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,int mode,void* raw_stream) {
    try {
        if(VENTURI_BLOCKS_ARCH!=70 || !workspace || available<vdq::bytes(tokens,k,2) || !codes || !scales || !out ||
            n<=0 || n>INT32_MAX-16 || mode<1 || mode>3 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 ||
            static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 ||
            reinterpret_cast<std::uintptr_t>(codes)%16 || reinterpret_cast<std::uintptr_t>(out)%4)
            throw std::invalid_argument("invalid explicit prefetch partial geometry");
        vqf::Group g{};g.count=1;g.item[0]={codes,scales,reinterpret_cast<__half*>(out),n,n};
        vpf::launch<true>(workspace,g,tokens,k,padded,mode,static_cast<cudaStream_t>(raw_stream));
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
