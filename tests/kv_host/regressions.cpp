// HOST-ONLY control flow. No GPU math, async scheduling or CUDA validation.
#include <cassert>
#include <iostream>
#include <string>
#include "venturi/ops/csrc/kv/kv_pool.cuh"
#include "venturi/ops/csrc/kv/kv_prefix.cuh"
constexpr std::uint64_t G=65536;
void fill(vkv::Pool& pool,vkv::Pool::Entry& e,unsigned char value,std::uint64_t n=32){
 for(int k=0;k<2;++k)std::memset(e.ptr(k),value,n);pool.written(e,n);
}
using KVTier=vkv::Pool;using vkv::require;using vkv::mul;
struct MockBuffer {int value=0;void*ptr=&value;std::uint64_t bytes=sizeof(int);};
struct MockModel {
 struct {int layers=2;}c;struct Layer{int kind;};std::vector<Layer>layers{{0},{1}};
 bool head_gdn_enabled=false;
 std::unique_ptr<KVTier>kv_tier=std::make_unique<KVTier>(0,nullptr,0,16*G,64*G,4096);
 std::unique_ptr<vkv::PrefixManager>kv_prefix=std::make_unique<vkv::PrefixManager>(0);
};
struct MockSession {
 std::shared_ptr<MockModel>model;int capacity=64,position=0;bool failed=false;std::mutex mutex;
 std::vector<std::unique_ptr<KVTier::Entry>>tiered_kv;
 std::vector<std::unique_ptr<MockBuffer>>history,state;
 explicit MockSession(std::shared_ptr<MockModel>m):model(std::move(m)),tiered_kv(2),history(2),state(2){
  tiered_kv[1]=model->kv_tier->create(2*G,8);history[0]=std::make_unique<MockBuffer>();state[0]=std::make_unique<MockBuffer>();
 }
 #include "session_methods.inc"
};
int main(int argc,char**argv){
 assert(argc==2);std::string which=argv[1];
 vkv::Pool pool(0,nullptr,0,16*G,64*G,4096);vkv::PrefixManager manager(0);
 auto donor=pool.create(4*G,8);auto borrower=pool.create(4*G,8);
 pool.prepare({{donor.get(),32}});pool.ensure(*donor,32);fill(pool,*donor,0x11);
 int state=111,out=0;
 if(which=="reset"){
  manager.publish(pool,1,{donor.get()},32,{{&state,4}});
  assert(manager.borrow(pool,1,{borrower.get()},{&out},32));
  pool.reset(*donor);pool.prepare({{donor.get(),32}});pool.ensure(*donor,32);fill(pool,*donor,0x22);
  assert(static_cast<unsigned char*>(borrower->ptr())[0]==0x11);
  auto later=pool.create(4*G,8);assert(!manager.borrow(pool,1,{later.get()},{&out},32));
 }else if(which=="private_reset"){
  auto old_page=donor->pages[0][0];auto old_generation=donor->incarnation;
  manager.publish(pool,1,{donor.get()},32,{});
  pool.reset(*donor);
  assert(donor->mapped==1&&donor->pages[0][0]==old_page&&donor->incarnation!=old_generation);
  pool.prepare({{donor.get(),32}});pool.ensure(*donor,32);fill(pool,*donor,0x62);
  assert(donor->pages[0][0]==old_page&&!manager.borrow(pool,1,{borrower.get()},{},32));
 }else if(which=="entry_aba"){
  alignas(vkv::Pool::Entry) unsigned char storage[sizeof(vkv::Pool::Entry)];
  auto* old=new(storage)vkv::Pool::Entry(pool,2*G,8);
  pool.prepare({{old,32}});pool.ensure(*old,32);fill(pool,*old,0x51);
  auto generation=old->incarnation;manager.publish(pool,7,{old},32,{});
  old->~Entry();
  auto* fresh=new(storage)vkv::Pool::Entry(pool,2*G,8);
  pool.prepare({{fresh,32}});pool.ensure(*fresh,32);fill(pool,*fresh,0x62);
  assert(fresh==old&&fresh->incarnation!=generation);
  assert(!manager.borrow(pool,7,{borrower.get()},{},32));
  fresh->~Entry();
 }else if(which=="leases"){
  manager.capacity=1;manager.publish(pool,1,{donor.get()},32,{});
  assert(manager.borrow(pool,1,{borrower.get()},{},32));assert(manager.records[0].borrowed());
  bool caught=false;try{manager.publish(pool,2,{donor.get()},32,{});}catch(...){caught=true;}assert(caught);
  pool.reset(*borrower);assert(!manager.records[0].borrowed());
  manager.publish(pool,2,{donor.get()},32,{});assert(manager.records[0].key==2);
  assert(manager.borrow(pool,2,{borrower.get()},{},32));borrower.reset();
  manager.publish(pool,3,{donor.get()},32,{});assert(manager.records[0].key==3);
 }else if(which=="capacity64"){
  for(int i=0;i<96;++i){manager.publish(pool,i,{donor.get()},32,{});auto b=pool.create(4*G,8);assert(manager.borrow(pool,i,{b.get()},{},32));}
  assert(manager.records.size()==64);
 }else if(which=="cow_tail"){
  pool.reset(*donor);pool.prepare({{donor.get(),2*G+32}});pool.ensure(*donor,2*G+32);fill(pool,*donor,0x33,2*G+32);
  manager.publish(pool,1,{donor.get()},2*G+32,{});assert(manager.borrow(pool,1,{borrower.get()},{},2*G+32));
  pool.prepare({{borrower.get(),2*G+40}});assert(pool.physical()==8*G);
  auto before=pool.d2h;pool.ensure(*borrower,2*G+40);
  int shared=0;for(int k=0;k<2;++k)for(int p=0;p<3;++p)shared+=donor->pages[k][p]==borrower->pages[k][p];
  assert(shared==4&&pool.d2h-before==2*G);
  for(int k=0;k<2;++k)std::memset(borrower->ptr(k,2*G+32),0x44,8);
  assert(static_cast<unsigned char*>(donor->ptr(0,2*G+31))[0]==0x33);
 }else if(which=="cow_pressure"){
  // A separate exact-two-page pool; admission must not require a third page.
  vkv::Pool tiny(0,nullptr,0,2*G,64*G,4096);vkv::PrefixManager reg(0);
  auto d=tiny.create(2*G,8);auto b=tiny.create(2*G,8);
  tiny.prepare({{d.get(),32}});tiny.ensure(*d,32);fill(tiny,*d,0x55);
  reg.publish(tiny,1,{d.get()},32,{});assert(reg.borrow(tiny,1,{b.get()},{},32));
  tiny.prepare({{b.get(),40}});tiny.ensure(*b,40);
  for(int k=0;k<2;++k)std::memset(b->ptr(k,32),0x66,8);tiny.written(*b,40);
  tiny.prepare({{d.get(),32}});tiny.ensure(*d,32);tiny.attach(*d);
  assert(static_cast<unsigned char*>(d->ptr())[0]==0x55);
  tiny.prepare({{b.get(),40}});tiny.ensure(*b,40);tiny.attach(*b);
  assert(static_cast<unsigned char*>(b->ptr(0,32))[0]==0x66&&!b->shared[0][0]);
 }else if(which=="rejection_ownership"){
  manager.capacity=1;manager.publish(pool,1,{donor.get()},32,{{&state,4}});
  assert(manager.borrow(pool,1,{borrower.get()},{&out},32));auto before=mock_live_bytes();
  for(int i=0;i<8;++i){bool caught=false;try{manager.publish(pool,2+i,{donor.get()},32,{{&state,4}});}catch(...){caught=true;}assert(caught);}
  assert(mock_live_bytes()==before);manager.forget(1);assert(mock_live_bytes()==before-4);
 }else if(which=="snapshot_failure"){
  manager.publish(pool,1,{donor.get()},32,{{&state,4}});auto before=mock_live_bytes();
  for(int fail_after: {0,1}) {
   mock_copy_fail_after=fail_after;bool caught=false;
   try{manager.publish(pool,1,{donor.get()},32,{{&state,4},{&state,4}});}catch(...){caught=true;}
   mock_copy_fail_after=-1;
   assert(caught&&mock_live_bytes()==before&&manager.records[0].snapshots.size()==1);
  }
  assert(manager.borrow(pool,1,{borrower.get()},{&out},32)&&out==111);
 }else if(which=="validation"){
  manager.publish(pool,1,{donor.get()},32,{{&state,4}});
  for(int i=0;i<3;++i){bool caught=false;try{
   if(i==0)manager.borrow(pool,1,{borrower.get()},{&out},24);
   if(i==1)manager.borrow(pool,1,{borrower.get()},{nullptr},32);
   if(i==2)manager.borrow(pool,1,{borrower.get(),borrower.get()},{&out},32);
  }catch(...){caught=true;}assert(caught&&!borrower->mapped&&!borrower->read_lease&&!pool.poisoned);}
  assert(manager.borrow(pool,1,{borrower.get()},{&out},32));assert(out==111);
 }else if(which=="session_boundaries"){
  auto m=std::make_shared<MockModel>();MockSession d(m);auto&e=*d.tiered_kv[1];
  m->kv_tier->prepare({{&e,64}});m->kv_tier->ensure(e,64);fill(*m->kv_tier,e,0x77,64);
  d.position=8;d.state[0]->value=8;bool caught=false;
  try{d.publish_prefix(1,5);}catch(...){caught=true;}assert(caught&&m->kv_prefix->records.empty());
  d.position=5;d.state[0]->value=5;d.publish_prefix(1,5);MockSession b(m);caught=false;
  try{b.attach_prefix(1,3);}catch(...){caught=true;}assert(caught&&b.position==0&&!b.tiered_kv[1]->mapped);
  d.failed=true;caught=false;try{d.publish_prefix(2,5);}catch(...){caught=true;}assert(caught);
  b.attach_prefix(1,5);assert(b.position==5&&b.state[0]->value==5);
 }else if(which=="donor_append"){
  manager.publish(pool,1,{donor.get()},32,{});assert(manager.borrow(pool,1,{borrower.get()},{},32));
  pool.prepare({{donor.get(),40}});pool.ensure(*donor,40);
  for(int k=0;k<2;++k)std::memset(donor->ptr(k,32),0x88,8);pool.written(*donor,40);
  assert(donor->pages[0][0]!=borrower->pages[0][0]);
  assert(static_cast<unsigned char*>(borrower->ptr())[0]==0x11);
 }else {return 2;}
 std::cout<<"PASS "<<which<<" (host-only, NOT GPU validation)\n";
}
