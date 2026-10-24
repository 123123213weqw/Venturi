// Research-only input-column TP primitive. Never selected by model dispatch.
// Stored Q8 weights and two-pass activation quantization are unchanged. Keep
// partials in FP32 across transport; sum once before the final FP16 cast.
#pragma once
namespace vct {
template<int Tokens,int Rows,bool Staged>
__global__ void partial(const std::int8_t* activations,const float* activation_scales,
    const std::int8_t* weights,const __half* weight_scales,float* out,
    int tokens,int n,int k,int padded,int groups) {
    const int row=blockIdx.x*Rows+threadIdx.x/32,lane=threadIdx.x&31;
    const int token=blockIdx.y*Tokens;
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    float sums[Tokens]{};
    extern __shared__ __align__(16) char shared[];
    constexpr int all=Tokens*2*32;
    auto* a0s=reinterpret_cast<int4*>(shared);auto* a1s=a0s+all;
    auto* ss=reinterpret_cast<float*>(a1s+all);
    if constexpr (!Staged) {if(row>=n)return;}
    for(int base=0;base<k;base+=1024) {
        if constexpr (Staged) {
            for(int i=threadIdx.x;i<all;i+=blockDim.x) {
                const int pass=i/(Tokens*32),t=(i/32)%Tokens,g=base/32+i%32;
                if(token+t<tokens && g<groups) {
                    const auto group=pass*count+static_cast<std::int64_t>(token+t)*groups+g;
                    const auto* a=reinterpret_cast<const int4*>(activations+group*32);
                    a0s[i]=a[0];a1s[i]=a[1];ss[i]=activation_scales[group];
                } else {a0s[i]=make_int4(0,0,0,0);a1s[i]=make_int4(0,0,0,0);ss[i]=0;}
            }
            __syncthreads();
        }
        const int col=base+lane*32;
        if(row<n && col<k) {
            const auto* w=reinterpret_cast<const int4*>(weights+static_cast<std::int64_t>(row)*padded+col);
            const int4 w0=w[0],w1=w[1];
            const float ws=__half2float(weight_scales[static_cast<std::int64_t>(row)*(padded/32)+col/32]);
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(token+t<tokens) {
                #pragma unroll
                for(int p=0;p<2;++p) {
                    const auto group=p*count+static_cast<std::int64_t>(token+t)*groups+col/32;
                    int4 a0,a1;float scale;
                    if constexpr (Staged) {const int i=(p*Tokens+t)*32+lane;a0=a0s[i];a1=a1s[i];scale=ss[i];}
                    else {const auto* a=reinterpret_cast<const int4*>(activations+group*32);a0=a[0];a1=a[1];scale=activation_scales[group];}
                    int d0=__dp4a(w0.x,a0.x,0),d1=__dp4a(w1.x,a1.x,0);
                    d0=__dp4a(w0.y,a0.y,d0);d1=__dp4a(w1.y,a1.y,d1);
                    d0=__dp4a(w0.z,a0.z,d0);d1=__dp4a(w1.z,a1.z,d1);
                    d0=__dp4a(w0.w,a0.w,d0);d1=__dp4a(w1.w,a1.w,d1);
                    sums[t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(ws,scale),sums[t]);
                }
            }
        }
        if constexpr (Staged) {__syncthreads();}
    }
    if(row>=n)return;
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        for(int delta=16;delta;delta>>=1)sums[t]+=__shfl_down_sync(0xffffffff,sums[t],delta);
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*n+row]=sums[t];
    }
}
__global__ void sum(const float* a,const float* b,__half* out,int count) {
    for(std::int64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        out[i]=__float2half_rn(__fadd_rn(a[i],b[i]));
}
template<int Tokens,int Rows,bool Staged>
void launch(const void* workspace,const std::int8_t* w,const __half* s,float* out,
    int tokens,int n,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto* codes=static_cast<const std::int8_t*>(workspace);
    const auto* scales=reinterpret_cast<const float*>(codes+static_cast<std::uint64_t>(tokens)*groups*2*32);
    partial<Tokens,Rows,Staged><<<dim3((n+Rows-1)/Rows,(tokens+Tokens-1)/Tokens),Rows*32,Staged?Tokens*2*32*36:0,stream>>>(codes,scales,w,s,out,tokens,n,k,padded,groups);
}
}
extern "C" int venturi_q8_column_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,void* raw_stream) {
    try {
        const auto required=vdq::bytes(tokens,k,2);
        if(VENTURI_BLOCKS_ARCH!=70 || !workspace || available<required || !codes || !scales || !out ||
            n<=0 || n>INT32_MAX-16 || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 ||
            static_cast<std::uint64_t>(tokens)*n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(workspace)%16 ||
            reinterpret_cast<std::uintptr_t>(codes)%16 || reinterpret_cast<std::uintptr_t>(out)%4)
            throw std::invalid_argument("invalid SM70 FP32 column-partial geometry/buffers");
        const auto stream=static_cast<cudaStream_t>(raw_stream);
        if(tokens==1)vct::launch<1,4,false>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        else if(tokens>=8 && n>=1024)vct::launch<8,16,true>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        else vct::launch<4,8,true>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_q8_column_sum(const float* a,const float* b,void* out,int count,void* raw_stream) {
    try {
        if(VENTURI_BLOCKS_ARCH!=70 || !a || !b || !out || count<=0 || count>INT32_MAX-256 ||
            reinterpret_cast<std::uintptr_t>(a)%4 || reinterpret_cast<std::uintptr_t>(b)%4 || reinterpret_cast<std::uintptr_t>(out)%2)
            throw std::invalid_argument("invalid FP32 column-partial sum buffers/count");
        vct::sum<<<std::min((count+255)/256,65535),256,0,static_cast<cudaStream_t>(raw_stream)>>>(a,b,static_cast<__half*>(out),count);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
