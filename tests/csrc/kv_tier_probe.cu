// Direct production-tier unit test: cross physical pages without O(N^2) attention.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace vr {
inline void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
inline void check(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
inline std::uint64_t mul(std::uint64_t a,std::uint64_t b){require(!b || a<=UINT64_MAX/b,"overflow");return a*b;}
inline std::uint64_t add(std::uint64_t a,std::uint64_t b){require(a<=UINT64_MAX-b,"overflow");return a+b;}
struct Device{int old;Device(int d){check(cudaGetDevice(&old));check(cudaSetDevice(d));}~Device(){cudaSetDevice(old);}};
// Test-only allocation seam: exercise genuine CUDA_ERROR_OUT_OF_MEMORY handling
// without exhausting shared GPUs. The production build contains no injection.
int allocation_allowance=-1,access_allowance=-1;
inline CUresult probe_mem_create(CUmemGenericAllocationHandle* h,std::size_t n,const CUmemAllocationProp* p,unsigned long long f){
    if(allocation_allowance==0)return CUDA_ERROR_OUT_OF_MEMORY;
    if(allocation_allowance>0)--allocation_allowance;
    return ::cuMemCreate(h,n,p,f);
}
inline CUresult probe_mem_access(CUdeviceptr p,std::size_t n,const CUmemAccessDesc* d,std::size_t count){
    if(access_allowance==0)return CUDA_ERROR_INVALID_VALUE;
    if(access_allowance>0)--access_allowance;
    return ::cuMemSetAccess(p,n,d,count);
}
#define cuMemSetAccess probe_mem_access
#define cuMemCreate probe_mem_create
#include "qwen_kv_tier.cuh"
#undef cuMemCreate
#undef cuMemSetAccess
}
int main(){try{
    using namespace vr;check(cudaSetDevice(0));check(cudaFree(nullptr));cudaStream_t s;check(cudaStreamCreate(&s));void* stage=nullptr;check(cudaHostAlloc(&stage,1048576,cudaHostAllocPortable));
    {
        KVTier t(0,s,stage,1048576,0,0,0);const auto g=t.granule;t.gpu_limit=4*g;t.ram_limit=64*g;
        auto a=t.create(3*g,1),b=t.create(3*g,1),c=t.create(3*g,1);
        const auto n=g-17;t.prepare({{a.get(),n},{b.get(),n},{c.get(),n}});
        std::vector<unsigned char> expected[3][2];KVTier::Entry* e[]={a.get(),b.get(),c.get()};
        for(int j=0;j<3;++j){t.ensure(*e[j],n);for(int k=0;k<2;++k){auto& v=expected[j][k];v.resize(n);for(std::size_t i=0;i<v.size();++i)v[i]=(i*73+j*31+k*119)%251;check(cudaMemcpyAsync(e[j]->ptr(k),v.data(),n,cudaMemcpyHostToDevice,s));}t.written(*e[j],n);}
        for(int j=0;j<3;++j)for(int k=0;k<2;++k){std::vector<unsigned char> out(n);t.snapshot(*e[j],k,out.data(),n);require(out==expected[j][k],"spill snapshot mismatch");}
        auto before=t.info();for(int k=0;k<2;++k){std::vector<unsigned char> out(n);t.snapshot(*a,k,out.data(),n);}require(t.info()==before,"snapshot changed residency/counters");
        const auto grown=g+37;t.prepare({{a.get(),grown}});t.ensure(*a,grown);
        for(int k=0;k<2;++k){auto& v=expected[0][k];v.resize(grown);for(auto i=n;i<grown;++i)v[i]=(i*41+k)%251;check(cudaMemcpyAsync(a->ptr(k,n),v.data()+n,grown-n,cudaMemcpyHostToDevice,s));}t.written(*a,grown);
        for(int k=0;k<2;++k){std::vector<unsigned char> out(grown);t.snapshot(*a,k,out.data(),grown);require(out==expected[0][k],"page growth lost prefix");}
        bool rejected=false;try{t.prepare({{a.get(),2*g+1}});}catch(const std::exception&){rejected=true;}require(rejected && a->valid==grown,"largest entry admission failed");
        t.reset(*b);require(!b->valid && !b->host,"reset retained valid CPU data");
        auto stats=t.info();require(stats[7]>0 && stats[8]>0 && stats[9]>0 && stats[10]>0,"spill/reload was not exercised");
        // Non-OOM driver failures are fatal to the tier, not a RAM fallback or
        // a retry of an already-mutated recurrent forward. No real GPU OOM here.
        bool fatal=false;try{t.driver(CUDA_ERROR_INVALID_VALUE);}catch(const std::exception&){fatal=true;}
        require(fatal && t.poisoned,"driver failure was not fail-closed");
        rejected=false;try{t.prepare({{a.get(),grown}});}catch(const std::exception&){rejected=true;}
        require(rejected,"poisoned model admitted another forward");
        a.reset();b.reset();c.reset();auto end=t.info();require(end[17]==0 && end[5]==0 && end[3]==0,"tier teardown leaked mappings/backing");
        std::cout<<"{\"passed\":true,\"page_bytes\":"<<g<<",\"evictions\":"<<stats[7]<<",\"reloads\":"<<stats[8]<<",\"d2h_bytes\":"<<stats[9]<<",\"h2d_bytes\":"<<stats[10]<<"}"<<std::endl;
    }
    {
        KVTier t(0,s,stage,1048576,0,0,0);const auto g=t.granule;t.gpu_limit=8*g;t.ram_limit=32*g;
        auto a=t.create(2*g,1),b=t.create(2*g,1);
        allocation_allowance=0;bool rejected=false;
        try{t.prepare({{a.get(),123},{b.get(),123}});}catch(const std::exception&){rejected=true;}
        require(rejected && !t.poisoned && a->valid==0 && t.physical()==0,"physical OOM did not reject before forward");
        allocation_allowance=2;t.prepare({{a.get(),123},{b.get(),123}});
        require(!t.poisoned && t.physical()==2*g && t.ram_bytes>0,"physical OOM did not admit bounded RAM fallback");
        std::vector<unsigned char> expected(123,91),out(123);
        t.ensure(*a,123);for(int k=0;k<2;++k)check(cudaMemcpyAsync(a->ptr(k),expected.data(),123,cudaMemcpyHostToDevice,s));t.written(*a,123);
        t.ensure(*b,123);t.snapshot(*a,0,out.data(),123);require(out==expected,"allocation OOM fallback lost KV bytes");
        t.ensure(*a,123);t.snapshot(*a,1,out.data(),123);require(out==expected,"allocation OOM reload lost bytes");
        require(t.ooms==2 && t.reloads>0,"OOM branch/reload was not exercised");allocation_allowance=-1;
        std::cout<<"{\"allocation_oom_injection_passed\":true,\"ooms\":"<<t.ooms<<",\"tier_not_poisoned\":true}"<<std::endl;
    }
    {
        KVTier t(0,s,stage,1048576,0,0,0);const auto g=t.granule;t.gpu_limit=2*g;
        auto a=t.create(g,1);t.prepare({{a.get(),123}});access_allowance=0;bool rejected=false;
        try{t.ensure(*a,123);}catch(const std::exception&){rejected=true;}
        require(rejected && t.poisoned && a->pages[0][0] && a->mapped==0,"partial map ownership/fatal guard lost");
        a.reset();require(t.entries.empty() && t.free_pages.size()==t.pool.size(),"partial map teardown leaked physical page");access_allowance=-1;
        std::cout<<"{\"partial_mapping_failure_passed\":true}"<<std::endl;
    }
    check(cudaFreeHost(stage));check(cudaStreamDestroy(s));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
