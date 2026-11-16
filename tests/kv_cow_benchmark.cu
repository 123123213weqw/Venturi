// KV append mechanism only: not attention/model generation or TPS.
#include "venturi/ops/csrc/kv/kv_pool.cuh"
#include "venturi/ops/csrc/kv/kv_prefix.cuh"
#include <chrono>
#include <ctime>
#include <string>
static std::string error;
extern "C" {
__attribute__((visibility("default"))) const char* cow_bench_error(){return error.c_str();}
__attribute__((visibility("default"))) int cow_bench_run(double* seconds,std::uint64_t* stats){
 try{
  vkv::Pool pool(0,nullptr,0,512ull<<20,1ull<<30,1<<20);
  const auto g=pool.granule,n=8*g+32,end=n+8;
  auto donor=pool.create(12*g,8),borrower=pool.create(12*g,8);
  vkv::PrefixManager manager(0);
  pool.prepare({{donor.get(),n}});pool.ensure(*donor,n);pool.attach(*donor);
  for(int k=0;k<2;++k)vkv::check(cudaMemsetAsync(donor->ptr(k),0x31,n,0));
  vkv::check(cudaStreamSynchronize(0));pool.written(*donor,n);
  manager.publish(pool,1,{donor.get()},n,{});
#ifdef VENTURI_PREFIX_EXPECTED_BYTES_ABI
  vkv::require(manager.borrow(pool,1,{borrower.get()},{},n),"benchmark prefix miss");
#else
  vkv::require(manager.borrow(pool,1,{borrower.get()},{}),"benchmark prefix miss");
#endif
  vkv::check(cudaStreamSynchronize(0));
  const auto wall=std::chrono::steady_clock::now();const auto cpu=std::clock();
  pool.prepare({{borrower.get(),end}});pool.ensure(*borrower,end);pool.attach(*borrower);
  for(int k=0;k<2;++k)vkv::check(cudaMemsetAsync(borrower->ptr(k,n),0x55,8,0));
  pool.written(*borrower,end);vkv::check(cudaStreamSynchronize(0));
  seconds[0]=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();
  seconds[1]=double(std::clock()-cpu)/CLOCKS_PER_SEC;
  std::uint64_t changed=0;
  for(int k=0;k<2;++k)for(std::uint64_t p=0;p<pool.round(n)/g;++p)changed+=donor->pages[k][p]!=borrower->pages[k][p];
  stats[0]=g;stats[1]=n;stats[2]=changed;stats[3]=pool.physical();stats[4]=pool.d2h;stats[5]=pool.h2d;
  // Correctness outside timing; check BOTH donor and borrower, BOTH planes.
  std::vector<unsigned char> data(end);
  for(int k=0;k<2;++k){
   pool.snapshot(*donor,k,data.data(),n);
   for(std::uint64_t i=0;i<n;++i)vkv::require(data[i]==0x31,"donor data changed");
   pool.snapshot(*borrower,k,data.data(),end);
   for(std::uint64_t i=0;i<end;++i)vkv::require(data[i]==(i<n?0x31:0x55),"borrower data mismatch");
  }
  return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
}
