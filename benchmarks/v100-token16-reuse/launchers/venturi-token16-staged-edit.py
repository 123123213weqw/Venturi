from pathlib import Path
shared=Path('src/venturi/ops/csrc/q8_shared.cuh');fused=Path('src/venturi/ops/csrc/q8_fused.cuh')
# Both kernels use exactly the original per-output dot math, but stage only
# eight activation rows at a time while retaining weights for sixteen outputs.
def kernel(group):
 pre='''    int row;const auto plan=select<Rows>(group,row);
    const int n=plan.n;const auto* weights=plan.codes;const auto* weight_scales=plan.scales;auto* out=plan.out;
    const int lane=threadIdx.x&31,token=blockIdx.y*16;
''' if group else '''    const int row=blockIdx.x*Rows+threadIdx.x/32,lane=threadIdx.x&31,token=blockIdx.y*16;
'''
 args='''const std::int8_t* activations,const float* activation_scales,Group group,
    int tokens,int k,int padded,int groups''' if group else '''const std::int8_t* activations,const float* activation_scales,
    const std::int8_t* weights,const __half* weight_scales,__half* out,
    int tokens,int n,int k,int padded,int groups'''
 stride='plan.stride' if group else 'n'
 return '''template<int Passes,int Rows>
__global__ __launch_bounds__(Rows*32,1) void staged16_dot('''+args+''') {
'''+pre+'''    extern __shared__ __align__(16) char shared[];
    constexpr int all=8*Passes*32;
    auto* a0s=reinterpret_cast<int4*>(shared);auto* a1s=a0s+all;
    auto* shared_scales=reinterpret_cast<float*>(a1s+all);
    const auto count=static_cast<std::int64_t>(tokens)*groups;
    float sum[16]{};
    for(int base=0;base<k;base+=1024) {
        const int col=base+lane*32;const bool valid=row<n && col<k;
        int4 w0=make_int4(0,0,0,0),w1=make_int4(0,0,0,0);float ws=0;
        if(valid) {
            const auto* wp=reinterpret_cast<const int4*>(weights+static_cast<std::int64_t>(row)*padded+col);
            w0=wp[0];w1=wp[1];
            ws=__half2float(weight_scales[static_cast<std::int64_t>(row)*(padded/32)+col/32]);
        }
        #pragma unroll
        for(int chunk=0;chunk<16;chunk+=8) {
            for(int i=threadIdx.x;i<all;i+=blockDim.x) {
                const int p=i/(8*32),t=(i/32)%8,g=base/32+i%32;
                if(token+chunk+t<tokens && g<groups) {
                    const auto group=p*count+static_cast<std::int64_t>(token+chunk+t)*groups+g;
                    const auto* src=reinterpret_cast<const int4*>(activations+group*32);
                    a0s[i]=src[0];a1s[i]=src[1];shared_scales[i]=activation_scales[group];
                } else {a0s[i]=make_int4(0,0,0,0);a1s[i]=make_int4(0,0,0,0);shared_scales[i]=0;}
            }
            __syncthreads();
            if(valid) {
                #pragma unroll
                for(int t=0;t<8;++t)if(token+chunk+t<tokens) {
                    #pragma unroll
                    for(int p=0;p<Passes;++p) {
                        const int group=(p*8+t)*32+lane;
                        const int4 a0=a0s[group],a1=a1s[group];
                        int d0=__dp4a(w0.x,a0.x,0),d1=__dp4a(w1.x,a1.x,0);
                        d0=__dp4a(w0.y,a0.y,d0);d1=__dp4a(w1.y,a1.y,d1);
                        d0=__dp4a(w0.z,a0.z,d0);d1=__dp4a(w1.z,a1.z,d1);
                        d0=__dp4a(w0.w,a0.w,d0);d1=__dp4a(w1.w,a1.w,d1);
                        sum[chunk+t]=__fmaf_rn(static_cast<float>(d0+d1),__fmul_rn(ws,shared_scales[group]),sum[chunk+t]);
                    }
                }
            }
            __syncthreads();
        }
    }
    if(row>=n)return;
    #pragma unroll
    for(int t=0;t<16;++t) {
        for(int delta=16;delta;delta>>=1)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],delta);
        if(lane==0 && token+t<tokens)out[static_cast<std::int64_t>(token+t)*'''+stride+'''+row]=__float2half_rn(sum[t]);
    }
}
'''
s=shared.read_text();pos=s.index('template<int Tokens,int Passes,int Rows>void launch(',s.index('namespace vdt {'))
s=s[:pos]+'''// Reuse weights for sixteen outputs with two eight-token shared stages.
'''+kernel(False)+'''template<int Passes>void staged16_launch(const void* workspace,const std::int8_t* weights,const __half* scales,void* out,int tokens,int n,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;auto* codes=static_cast<const std::int8_t*>(workspace);
    auto* as=reinterpret_cast<const float*>(codes+static_cast<std::uint64_t>(tokens)*groups*Passes*32);
    staged16_dot<Passes,8><<<dim3((n+7)/8,(tokens+15)/16),256,8*Passes*32*36,stream>>>(codes,as,weights,scales,static_cast<__half*>(out),tokens,n,k,padded,groups);
}
'''+s[pos:]
s=s.replace('vdt::launch<16,1,8>','vdt::staged16_launch<1>').replace('vdt::launch<16,2,8>','vdt::staged16_launch<2>');shared.write_text(s)
s=fused.read_text();pos=s.index('template<int Tokens,int Passes,int Rows,int Kind>void launch(');s=s[:pos]+'''// Shared-memory footprint stays at eight rows; each weight spans both stages.
'''+kernel(True)+'''template<int Passes>void staged16_launch(const void* workspace,Group g,int tokens,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;const auto count=static_cast<std::uint64_t>(tokens)*groups;
    auto* codes=static_cast<const std::int8_t*>(workspace);auto* scales=reinterpret_cast<const float*>(codes+count*32*Passes);
    std::uint64_t blocks=0;for(int i=0;i<g.count;++i)blocks+=(g.item[i].n+7)/8;
    staged16_dot<Passes,8><<<dim3(static_cast<unsigned>(blocks),(tokens+15)/16),256,8*Passes*32*36,stream>>>(codes,scales,g,tokens,k,padded,groups);
}
'''+s[pos:];s=s.replace('launch<16,Passes,8,2>(w,g,tokens,k,padded,stream)','staged16_launch<Passes>(w,g,tokens,k,padded,stream)');fused.write_text(s)
p=Path('tools/inspect_cuda_occupancy.py');s=p.read_text();a='''    match = re.search(
        r"3(?:vdt3dot|vqf9tiled_dot|vqf16paired_tiled_dot)''';assert a in s;s=s.replace(a,'''    staged = re.search(r"3(?:vdt|vqf)12staged16_dotILi([12])ELi8E", symbol)
    if staged:
        return 256, 8 * int(staged[1]) * 32 * 36
    match = re.search(
        r"3(?:vdt3dot|vqf9tiled_dot|vqf16paired_tiled_dot)''',1);p.write_text(s)
p=Path('tests/test_cuda_occupancy_tool.py');s=p.read_text()+'''

def test_staged16_dynamic_memory_is_eight_tokens(tool):
    for family in ["vdt", "vqf"]:
        for passes in [1, 2]:
            symbol=f"_ZN3{family}12staged16_dotILi{passes}ELi8EEEv"
            assert tool.compiled_launch_geometry(symbol)==(256,8*passes*32*36)
''';p.write_text(s)
