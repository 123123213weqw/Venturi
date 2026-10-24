// Research-only single-pass arithmetic. Not included by blocks.cu/default build.
#include "prefetch_probe.cu"
#include <cuda_profiler_api.h>
namespace vpa {
template<bool Partial>void pair(const void* workspace,vqf::Group group,int tokens,int k,int padded,cudaStream_t stream) {
    const int groups=(k+31)/32;
    const auto count=static_cast<std::uint64_t>(tokens)*groups;
    const auto* codes=static_cast<const std::int8_t*>(workspace);
    const auto* scales=reinterpret_cast<const float*>(codes+count*32);
    int blocks=0;for(int i=0;i<group.count;++i)blocks+=(group.item[i].n+15)/16;
    vpf::row_pair<Partial,1><<<dim3(blocks,(tokens+7)/8),256,9216,stream>>>(codes,scales,group,tokens,k,padded,groups);
}
}
extern "C" int venturi_q8_a8_fused_run(const void* workspace,std::uint64_t available,
    const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,void* raw_stream) {
    if(passes==2)return vprojection::fused(workspace,available,outputs,count,tokens,k,padded,2,raw_stream);
    try {
        if(passes!=1 || VENTURI_BLOCKS_ARCH!=70 || !workspace || available<vdq::bytes(tokens,k,1) ||
            !outputs || (count!=2 && count!=3) || padded!=(static_cast<std::uint64_t>(k)+127)/128*128 ||
            reinterpret_cast<std::uintptr_t>(workspace)%16)
            throw std::invalid_argument("invalid single-pass fused geometry");
        vqf::Group group{};group.count=count;std::uint64_t total=0;
        for(int i=0;i<count;++i) {
            const auto& p=outputs[i];
            if(!p.codes || !p.scales || !p.out || p.n<=0 || p.n>INT32_MAX-16 || p.stride<p.n ||
                static_cast<std::uint64_t>(tokens-1)*p.stride+p.n>=INT32_MAX || reinterpret_cast<std::uintptr_t>(p.codes)%16)
                throw std::invalid_argument("invalid single-pass fused descriptor");
            group.item[i]=p;total+=p.n;
        }
        if(total>=INT32_MAX-48)throw std::invalid_argument("single-pass fused row sum exceeds int32");
        auto stream=static_cast<cudaStream_t>(raw_stream);
        // Compare pass count with the same measured row-pair selection policy.
        if(vprojection::fused_eligible(outputs,count,tokens,k,2))vpa::pair<false>(workspace,group,tokens,k,padded,stream);
        else vqf::run<1>(workspace,group,tokens,k,padded,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_q8_a8_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,int passes,void* raw_stream) {
    if(passes==2)return vprojection::partial(workspace,available,codes,scales,out,tokens,n,k,padded,raw_stream);
    try {
        if(passes!=1 || VENTURI_BLOCKS_ARCH!=70 || !workspace || available<vdq::bytes(tokens,k,1) ||
            !codes || !scales || !out || n<=0 || n>INT32_MAX-16 ||
            padded!=(static_cast<std::uint64_t>(k)+127)/128*128 || static_cast<std::uint64_t>(tokens)*n>=INT32_MAX ||
            reinterpret_cast<std::uintptr_t>(workspace)%16 || reinterpret_cast<std::uintptr_t>(codes)%16 || reinterpret_cast<std::uintptr_t>(out)%4)
            throw std::invalid_argument("invalid single-pass partial geometry");
        auto stream=static_cast<cudaStream_t>(raw_stream);
        if(vprojection::partial_eligible(tokens,n,k)) {
            vqf::Group group{};group.count=1;group.item[0]={codes,scales,reinterpret_cast<__half*>(out),n,n};
            vpa::pair<true>(workspace,group,tokens,k,padded,stream);
        }else if(tokens==1)vct::launch<1,4,false,1>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        else if(tokens>=8 && n>=1024)vct::launch<8,16,true,1>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        else vct::launch<4,8,true,1>(workspace,codes,scales,out,tokens,n,k,padded,stream);
        check(cudaGetLastError());return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_a8_profile_start(){return static_cast<int>(cudaProfilerStart());}
extern "C" int venturi_a8_profile_stop(){return static_cast<int>(cudaProfilerStop());}
