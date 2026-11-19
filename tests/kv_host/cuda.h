#pragma once
// POSIX shared mappings model page aliasing; streams are synchronous stubs.
#include "cuda_runtime.h"
#include <string>
using CUdeviceptr=std::uintptr_t;using CUmemGenericAllocationHandle=std::uint64_t;using CUresult=int;
constexpr int CUDA_SUCCESS=0,CUDA_ERROR_OUT_OF_MEMORY=2,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED=1,CU_MEM_LOCATION_TYPE_DEVICE=1,CU_MEM_ALLOC_GRANULARITY_MINIMUM=1;
constexpr int CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3,CU_MEM_ACCESS_FLAGS_PROT_READ=1;
struct CUmemLocation {int type=0,id=0;};
struct CUmemAllocationProp {int type=0;CUmemLocation location;};
struct CUmemAccessDesc {CUmemLocation location;int flags=0;};
inline int cuDeviceGetAttribute(int*p,int,int){*p=1;return 0;}
inline int cuMemGetAllocationGranularity(std::size_t*p,const CUmemAllocationProp*,int){*p=65536;return 0;}
inline int cuGetErrorString(int,const char**s){*s="host emulator VMM failure";return 0;}
inline int cuMemAddressReserve(CUdeviceptr*p,std::size_t n,std::size_t,CUdeviceptr,unsigned){
 void*x=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);if(x==MAP_FAILED)return 1;*p=reinterpret_cast<CUdeviceptr>(x);return 0;}
inline int cuMemAddressFree(CUdeviceptr p,std::size_t n){return munmap(reinterpret_cast<void*>(p),n)?1:0;}
inline int cuMemCreate(CUmemGenericAllocationHandle*p,std::size_t n,const CUmemAllocationProp*,unsigned){
 char name[]="/tmp/venturi-review-vmm-XXXXXX";int fd=mkstemp(name);if(fd<0)return 1;unlink(name);
 if(ftruncate(fd,n)){close(fd);return 1;}*p=static_cast<std::uint64_t>(fd);return 0;}
inline int cuMemRelease(CUmemGenericAllocationHandle p){return close(static_cast<int>(p))?1:0;}
inline int cuMemMap(CUdeviceptr p,std::size_t n,std::size_t off,CUmemGenericAllocationHandle h,unsigned){
 return mmap(reinterpret_cast<void*>(p),n,PROT_READ|PROT_WRITE,MAP_FIXED|MAP_SHARED,static_cast<int>(h),off)==MAP_FAILED?1:0;}
inline int cuMemUnmap(CUdeviceptr p,std::size_t n){
 return mmap(reinterpret_cast<void*>(p),n,PROT_NONE,MAP_FIXED|MAP_ANON|MAP_PRIVATE,-1,0)==MAP_FAILED?1:0;}
inline int cuMemSetAccess(CUdeviceptr p,std::size_t n,const CUmemAccessDesc*a,std::size_t){
 return mprotect(reinterpret_cast<void*>(p),n,a->flags==CU_MEM_ACCESS_FLAGS_PROT_READ?PROT_READ:PROT_READ|PROT_WRITE)?1:0;}
