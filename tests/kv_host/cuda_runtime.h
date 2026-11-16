#pragma once
// HOST-ONLY CUDA API emulator for control-flow review, NOT GPU validation.
#include <cstdint>
#include <unordered_map>
inline std::unordered_map<void*,std::size_t> mock_allocations;
inline std::size_t mock_live_bytes(){std::size_t n=0;for(auto p:mock_allocations)n+=p.second;return n;}
inline int mock_copy_fail_after=-1;
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
using cudaError_t=int; using cudaStream_t=void*;using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaEventDisableTiming=2,cudaHostAllocPortable=1,cudaStreamNonBlocking=1;
constexpr int cudaMemcpyDeviceToDevice=0,cudaMemcpyDeviceToHost=1,cudaMemcpyHostToDevice=2;
inline const char* cudaGetErrorString(int){return "host emulator failure";}
inline int cudaGetDevice(int*p){*p=0;return 0;} inline int cudaSetDevice(int){return 0;}
inline int cudaEventCreateWithFlags(void**p,unsigned){*p=nullptr;return 0;}
inline int cudaEventDestroy(void*){return 0;} inline int cudaEventSynchronize(void*){return 0;}
inline int cudaEventRecord(void*,void*){return 0;} inline int cudaStreamSynchronize(void*){return 0;}
inline int cudaStreamWaitEvent(void*,void*){return 0;} inline int cudaStreamDestroy(void*){return 0;}
inline int cudaStreamCreateWithFlags(void**p,unsigned){*p=nullptr;return 0;}
inline int cudaHostAlloc(void**p,std::size_t n,unsigned){*p=std::malloc(n);return *p?0:1;}
inline int cudaFreeHost(void*p){std::free(p);return 0;}
inline int cudaMemGetInfo(std::size_t*f,std::size_t*t){*f=*t=1ull<<30;return 0;}
inline int cudaMalloc(void**p,std::size_t n){*p=std::malloc(n);if(*p)mock_allocations[*p]=n;return *p?0:1;}
inline int cudaFree(void*p){mock_allocations.erase(p);std::free(p);return 0;}
inline int cudaMemcpy(void*d,const void*s,std::size_t n,int){
 if(mock_copy_fail_after==0)return 1;
 if(mock_copy_fail_after>0)--mock_copy_fail_after;
 std::memcpy(d,s,n);return 0;
}
inline int cudaMemcpyAsync(void*d,const void*s,std::size_t n,int k,void*){return cudaMemcpy(d,s,n,k);}
