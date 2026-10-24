// Research-only INT16 activations x stored Q8 weights. Not in model dispatch.
// One group32 INT16 pass/F32 scale changes arithmetic; no FP16 parity claim.
#pragma once
namespace vi16 {
struct Output {const std::int8_t* codes;const __half* scales;void* out;int n,stride;};
static_assert(sizeof(Output)==32);
struct Group {Output item[3]{};int count=0;};
inline std::uint64_t bytes(int tokens,int k) {
    if(tokens<=0 || tokens>262140 || k<=0 || k>INT32_MAX-1024)
        throw std::invalid_argument("invalid DP2A geometry");
    const auto count=static_cast<std::uint64_t>(tokens)*((static_cast<std::uint64_t>(k)+31)/32);
    if(count>INT32_MAX/64)throw std::invalid_argument("DP2A workspace exceeds bounded indexing");
    return count*68;
}
__global__ void quantize(const __half* x,std::int16_t* codes,float* scales,int tokens,int k,int groups) {
    const int lane=threadIdx.x&31,index=blockIdx.x*8+threadIdx.x/32;
    if(index>=tokens*groups)return;
    const int token=index/groups,col=(index%groups)*32+lane;
    const float value=col<k?__half2float(x[static_cast<std::int64_t>(token)*k+col]):0.0f;
    const bool invalid=__any_sync(0xffffffff,!isfinite(value));
    float maximum=fabsf(value);
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
    maximum=__shfl_sync(0xffffffff,maximum,0);
    const float scale=maximum/32767.0f,inverse=maximum>0?32767.0f/maximum:0.0f;
    const int code=invalid?0:max(-32767,min(32767,__float2int_rn(__fmul_rn(value,inverse))));
    codes[static_cast<std::int64_t>(index)*32+lane]=static_cast<std::int16_t>(code);
    if(lane==0)scales[index]=invalid?CUDART_NAN_F:scale;
}
__device__ __forceinline__ int dot32(int4 a0,int4 a1,int4 a2,int4 a3,int4 w0,int4 w1) {
    // DP2A srcA packs two signed INT16 values; srcB packs four signed INT8.
    // Four independent chains cover 32 products with 16 integer instructions.
    int d0=__dp2a_lo(a0.x,w0.x,0),d1=__dp2a_lo(a1.x,w0.z,0);
    int d2=__dp2a_lo(a2.x,w1.x,0),d3=__dp2a_lo(a3.x,w1.z,0);
    d0=__dp2a_hi(a0.y,w0.x,d0);d1=__dp2a_hi(a1.y,w0.z,d1);
    d2=__dp2a_hi(a2.y,w1.x,d2);d3=__dp2a_hi(a3.y,w1.z,d3);
    d0=__dp2a_lo(a0.z,w0.y,d0);d1=__dp2a_lo(a1.z,w0.w,d1);
    d2=__dp2a_lo(a2.z,w1.y,d2);d3=__dp2a_lo(a3.z,w1.w,d3);
    d0=__dp2a_hi(a0.w,w0.y,d0);d1=__dp2a_hi(a1.w,w0.w,d1);
    d2=__dp2a_hi(a2.w,w1.y,d2);d3=__dp2a_hi(a3.w,w1.w,d3);
    return (d0+d1)+(d2+d3);
}
__global__ void integer_probe(const std::int16_t* a,const std::int8_t* w,int* out,int groups) {
    const int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=groups)return;
    const auto* ap=reinterpret_cast<const int4*>(a+static_cast<std::int64_t>(g)*32);
    const auto* wp=reinterpret_cast<const int4*>(w+static_cast<std::int64_t>(g)*32);
    out[g]=dot32(ap[0],ap[1],ap[2],ap[3],wp[0],wp[1]);
}
template<int Rows>__device__ __forceinline__ Output select(Group g,int& row) {
    int block=blockIdx.x;
    #pragma unroll
    for(int i=0;i<3;++i) {
        if(i>=g.count)break;
        const int blocks=(g.item[i].n+Rows-1)/Rows;
        if(block<blocks){row=block*Rows+threadIdx.x/32;return g.item[i];}
        block-=blocks;
    }
    row=0;return {};
}
template<int Tokens,int Rows,bool Staged,bool FP32>
__global__ __launch_bounds__(Rows*32,1) void project(const std::int16_t* a,const float* as,Group g,
    int tokens,int k,int padded,int groups) {
    int row;const auto plan=select<Rows>(g,row);
    const int lane=threadIdx.x&31,token=blockIdx.y*Tokens;
    extern __shared__ __align__(16) char shared[];
    constexpr int all=Tokens*32;
    auto* s0=reinterpret_cast<int4*>(shared);auto* s1=s0+all;auto* s2=s1+all;auto* s3=s2+all;
    auto* ss=reinterpret_cast<float*>(s3+all);
    if constexpr(!Staged){if(row>=plan.n)return;}
    float sums[Tokens]{};
    for(int base=0;base<k;base+=1024) {
        if constexpr(Staged) {
            for(int i=threadIdx.x;i<all;i+=blockDim.x) {
                const int t=i/32,group=base/32+i%32;
                if(token+t<tokens && group<groups) {
                    const auto idx=static_cast<std::int64_t>(token+t)*groups+group;
                    const auto* p=reinterpret_cast<const int4*>(a+idx*32);
                    s0[i]=p[0];s1[i]=p[1];s2[i]=p[2];s3[i]=p[3];ss[i]=as[idx];
                }else{s0[i]=s1[i]=s2[i]=s3[i]=make_int4(0,0,0,0);ss[i]=0;}
            }
            __syncthreads();
        }
        const int col=base+lane*32;
        if(row<plan.n && col<k) {
            const auto* w=reinterpret_cast<const int4*>(plan.codes+static_cast<std::int64_t>(row)*padded+col);
            const int4 w0=w[0],w1=w[1];
            const float scale=__half2float(plan.scales[static_cast<std::int64_t>(row)*(padded/32)+col/32]);
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(token+t<tokens) {
                int4 a0,a1,a2,a3;float sa;
                if constexpr(Staged){const int i=t*32+lane;a0=s0[i];a1=s1[i];a2=s2[i];a3=s3[i];sa=ss[i];}
                else{const auto idx=static_cast<std::int64_t>(token+t)*groups+col/32;const auto* p=reinterpret_cast<const int4*>(a+idx*32);a0=p[0];a1=p[1];a2=p[2];a3=p[3];sa=as[idx];}
                sums[t]=__fmaf_rn(static_cast<float>(dot32(a0,a1,a2,a3,w0,w1)),__fmul_rn(scale,sa),sums[t]);
            }
        }
        if constexpr(Staged){__syncthreads();}
    }
    if(row>=plan.n)return;
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        for(int d=16;d;d>>=1)sums[t]+=__shfl_down_sync(0xffffffff,sums[t],d);
        if(lane==0 && token+t<tokens) {
            const auto idx=static_cast<std::int64_t>(token+t)*plan.stride+row;
            if constexpr(FP32)static_cast<float*>(plan.out)[idx]=sums[t];
            else static_cast<__half*>(plan.out)[idx]=__float2half_rn(sums[t]);
        }
    }
}
template<int T,int R,bool Staged,bool FP32>
void launch(const void* workspace,Group g,int tokens,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto* codes=static_cast<const std::int16_t*>(workspace);
    const auto* scales=reinterpret_cast<const float*>(codes+static_cast<std::uint64_t>(tokens)*groups*32);
    unsigned blocks=0;for(int i=0;i<g.count;++i)blocks+=(g.item[i].n+R-1)/R;
    project<T,R,Staged,FP32><<<dim3(blocks,(tokens+T-1)/T),R*32,Staged?T*32*68:0,stream>>>(codes,scales,g,tokens,k,padded,groups);
}
template<bool FP32>void run(const void* w,Group g,int t,int k,int padded,cudaStream_t s) {
    int n=0;for(int i=0;i<g.count;++i)n=std::max(n,g.item[i].n);
    if(t==1)launch<1,4,false,FP32>(w,g,t,k,padded,s);
    else if(t==2)launch<2,4,false,FP32>(w,g,t,k,padded,s);
    else if(t>=8 && n>=1024)launch<8,16,true,FP32>(w,g,t,k,padded,s);
    else launch<4,8,true,FP32>(w,g,t,k,padded,s);
}
}
extern "C" std::uint64_t venturi_dp2a_bytes(int tokens,int k) {
    try{return vi16::bytes(tokens,k);}catch(const std::exception& e){error_text=e.what();return 0;}
}
extern "C" int venturi_dp2a_quantize(const void* x,void* workspace,std::uint64_t available,int t,int k,void* raw) {
    try {
        const auto needed=vi16::bytes(t,k);
        if(!x || !workspace || available<needed || reinterpret_cast<std::uintptr_t>(x)%2 || reinterpret_cast<std::uintptr_t>(workspace)%16)
            throw std::invalid_argument("invalid DP2A quantizer buffers");
        const int groups=(k+31)/32,count=t*groups;
        auto* codes=static_cast<std::int16_t*>(workspace);auto* scales=reinterpret_cast<float*>(codes+static_cast<std::uint64_t>(count)*32);
        vi16::quantize<<<(count+7)/8,256,0,static_cast<cudaStream_t>(raw)>>>(static_cast<const __half*>(x),codes,scales,t,k,groups);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_dp2a_run(const void* w,std::uint64_t available,const vi16::Output* outputs,int count,int t,int k,int padded,int fp32,void* raw) {
    try {
        const auto needed=vi16::bytes(t,k);
        if(!w || available<needed || !outputs || count<1 || count>3 || (fp32!=0 && fp32!=1) ||
            padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || reinterpret_cast<std::uintptr_t>(w)%16)
            throw std::invalid_argument("invalid DP2A projection geometry/buffers");
        vi16::Group g{};g.count=count;std::uint64_t total=0;
        for(int i=0;i<count;++i) {
            const auto& p=outputs[i];
            if(!p.codes || !p.scales || !p.out || p.n<=0 || p.n>INT32_MAX-16 || p.stride<p.n ||
                static_cast<std::uint64_t>(t-1)*p.stride+p.n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(p.codes)%16 ||
                reinterpret_cast<std::uintptr_t>(p.scales)%2 || reinterpret_cast<std::uintptr_t>(p.out)%(fp32?4:2))
                throw std::invalid_argument("invalid DP2A output descriptor");
            total+=p.n;g.item[i]=p;
        }
        if(total>=INT32_MAX-48)throw std::invalid_argument("DP2A fused row sum exceeds int32");
        const auto stream=static_cast<cudaStream_t>(raw);
        if(fp32)vi16::run<true>(w,g,t,k,padded,stream);else vi16::run<false>(w,g,t,k,padded,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_dp2a_integer_dot(const void* a,const void* w,void* out,int groups,void* raw) {
    try {
        if(!a || !w || !out || groups<=0 || groups>INT32_MAX/64 || reinterpret_cast<std::uintptr_t>(a)%16 ||
            reinterpret_cast<std::uintptr_t>(w)%16 || reinterpret_cast<std::uintptr_t>(out)%4)
            throw std::invalid_argument("invalid integer DP2A probe");
        vi16::integer_probe<<<(groups+127)/128,128,0,static_cast<cudaStream_t>(raw)>>>(static_cast<const std::int16_t*>(a),static_cast<const std::int8_t*>(w),static_cast<int*>(out),groups);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
