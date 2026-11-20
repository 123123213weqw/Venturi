// Research host orchestrator only. Reuses a hash-pinned production blocks SO;
// absent from blocks.cu and from model/session/serving dispatch.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
void require(bool ok,const char* text){if(!ok)throw std::invalid_argument(text);}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Device {int old;Device(){check(cudaGetDevice(&old));}~Device(){(void)cudaSetDevice(old);}};
struct Output {const void* codes;const void* scales;void* out;int n,stride;};
static_assert(sizeof(Output)==32);
struct Config {int mode,hidden,qk,heads,dk,dv,kernel,sequences,steps,prefixes,gamma_fp32;float epsilon,scale;void* input;void* output;const void* norm;void* projected;};
struct Rank {
    const void* codes[3];const void* scales[3];
    const void *a,*b,*conv,*rate,*bias,*gamma;
    void *normalized,*qkv,*z,*av,*bv,*convolved,*decay,*beta,*recurrent,*mixed,*local_output,*partial,*remote_partial,*quant;
    std::uint64_t quant_bytes;
    void* history[8];void* state[8];void* prefix_history[16];void* prefix_state[16];
};
struct Library {
    void* handle=nullptr;
    explicit Library(const char* path){require(path && *path,"empty blocks library");handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());}
    ~Library(){if(handle)dlclose(handle);}
    template<class T>T load(const char* name){auto p=dlsym(handle,name);if(!p)throw std::runtime_error(std::string("missing pinned export: ")+name);return reinterpret_cast<T>(p);}
};
struct Functions {
    const char*(*last_error)();
    int(*rms)(const void*,const float*,void*,int,int,int,float,void*);
    int(*quantize)(const void*,void*,std::uint64_t,int,int,int,void*);
    int(*fused)(const void*,std::uint64_t,const Output*,int,int,int,int,int,void*);
    int(*dot)(const void*,std::uint64_t,const void*,const void*,void*,int,int,int,int,int,void*);
    int(*tiled)(const void*,std::uint64_t,const void*,const void*,void*,int,int,int,int,int,int,int,void*);
    int(*partial)(const void*,std::uint64_t,const void*,const void*,float*,int,int,int,int,void*);
    int(*sum)(const float*,const float*,void*,int,void*);
    int(*add)(const void*,const void*,void*,std::int64_t,int,void*);
    int(*linear)(const void*,const float*,void*,int,int,int,int,void*);
    int(*conv)(const void*,const void*,const void*,void*,void*,int,int,int,int,int,void*);
    int(*gates)(const void*,const void*,const float*,const float*,float*,float*,int,int,int,int,void*);
    int(*gdn)(const void*,const void*,const void*,const float*,const float*,const float*,void*,float*,int,int,int,int,int,int,int,int,float,void*);
    int(*prefix)(const void*,const void*,const void*,const float*,const float*,const float*,void*,float*,int,int,int,int,int,int,int,int,float,void*,void*,int,void*);
    int(*history)(const void*,const void*,void*,void*,int,int,int,int,void*);
    int(*gated_norm)(const void*,const void*,const void*,void*,int,int,int,int,int,float,void*);
    explicit Functions(Library& l) {
#define BIND(member,name) member=l.load<decltype(member)>(name)
        BIND(last_error,"venturi_blocks_error");BIND(rms,"venturi_rms_run");BIND(quantize,"venturi_q8_dp4a_quantize");
        BIND(fused,"venturi_q8_fused_run");BIND(dot,"venturi_q8_dp4a_run");BIND(tiled,"venturi_q8_tiled_shared_run");
        BIND(partial,"venturi_q8_column_partial_run");BIND(sum,"venturi_q8_column_sum");BIND(linear,"venturi_fp32_linear_run");
        BIND(add,"venturi_add_run");BIND(conv,"venturi_conv_run");BIND(gates,"venturi_gates_run");BIND(gdn,"venturi_gdn_compact_run");
        BIND(prefix,"venturi_gdn_prefix_run");BIND(history,"venturi_history_prefix_run");BIND(gated_norm,"venturi_gated_rms_run");
#undef BIND
    }
    void ok(int rc){if(rc<0)throw std::runtime_error(last_error());}
};
struct Counters {std::uint64_t h2d_bytes=0,d2h_bytes=0,h2d_calls=0,d2h_calls=0,d2d_calls=0,iterations=0;};
struct Plan {
    Library library;Functions f;Config c;std::array<Rank,2> r;
    cudaStream_t streams[2]{};cudaEvent_t ready[2]{};
    void *host_input=nullptr,*host_group=nullptr,*host_output=nullptr;
    std::uint64_t pinned=0;Counters counters;std::mutex mutex;
    int rows()const{return c.sequences*c.steps;}
    int channels(int rank)const{return (2*c.qk*c.dk+c.heads*c.dv)/((c.mode && rank>=0)?2:1);}
    int values(int rank)const{return c.heads*c.dv/((c.mode && rank>=0)?2:1);}
    void device(int rank){check(cudaSetDevice(rank));}
    void pointer(const void* p,int rank){require(p && reinterpret_cast<std::uintptr_t>(p)%16==0,"null/unaligned pilot GPU pointer");cudaPointerAttributes a{};check(cudaPointerGetAttributes(&a,p));require(a.type==cudaMemoryTypeDevice && a.device==rank,"pilot pointer device mismatch");}
    Plan(const char* path,const Config& config,const Rank* ranks):library(path),f(library),c(config),r{ranks[0],ranks[1]} {
        try {
            require((c.mode==0 || c.mode==1) && c.hidden>0 && c.hidden<=32768 && c.hidden%128==0,"invalid pilot mode/hidden geometry");
            require(c.qk>=2 && c.qk%2==0 && c.heads>0 && c.heads% c.qk==0 && c.heads<=128,"invalid semantic GDN heads");
            require((c.dk==64 || c.dk==128 || c.dk==256) && c.dv>0 && c.dv<=256 && c.dv%64==0 && c.kernel>=2 && c.kernel<=8,"invalid GDN dimensions");
            require(c.sequences>=1 && c.sequences<=8 && c.steps>=1 && c.steps<=3 && rows()<=24 && (c.prefixes==0 || (c.prefixes==2 && c.steps==3 && c.sequences==1)),"invalid pilot independent-sequence/prefix geometry");
            require(c.epsilon>0 && c.scale>0 && std::isfinite(c.epsilon) && std::isfinite(c.scale),"invalid pilot numerical parameters");
            require(c.gamma_fp32==0 || c.gamma_fp32==1,"invalid gated norm source precision");
            int count=0;check(cudaGetDeviceCount(&count));require(count==2,"pilot requires exactly two explicitly selected devices");
            pointer(c.input,0);pointer(c.output,0);pointer(c.norm,0);pointer(c.projected,0);require(c.input!=c.output && c.input!=c.projected && c.output!=c.projected,"pilot input/output alias");
            for(int rank=0;rank<2;++rank) {
                device(rank);cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,rank));require(prop.major==7 && prop.minor==0,"pilot requires SM70");
                for(int j=0;j<3;++j){pointer(r[rank].codes[j],rank);pointer(r[rank].scales[j],rank);}
                for(auto p:{r[rank].normalized,r[rank].qkv,r[rank].z,r[rank].local_output,r[rank].quant})pointer(p,rank);
                require(r[rank].quant_bytes>=static_cast<std::uint64_t>(rows())*((std::max(c.hidden,values(-1))+31)/32)*72,"pilot quantization workspace too small");
                if(c.mode || rank==0) {
                    for(auto p:{r[rank].a,r[rank].b,r[rank].conv,r[rank].rate,r[rank].bias,r[rank].gamma})pointer(p,rank);
                    for(auto p:{r[rank].av,r[rank].bv,r[rank].convolved,r[rank].decay,r[rank].beta,r[rank].recurrent,r[rank].mixed})pointer(p,rank);
                    for(int s=0;s<c.sequences;++s){pointer(r[rank].history[s],rank);pointer(r[rank].state[s],rank);
                        if(c.prefixes)for(int j=0;j<2;++j){pointer(r[rank].prefix_history[j*8+s],rank);pointer(r[rank].prefix_state[j*8+s],rank);}}
                }
                if(c.mode){pointer(r[rank].partial,rank);if(!rank)pointer(r[rank].remote_partial,rank);}
                check(cudaStreamCreateWithFlags(&streams[rank],cudaStreamNonBlocking));check(cudaEventCreateWithFlags(&ready[rank],cudaEventDisableTiming));
            }
            const std::uint64_t input=static_cast<std::uint64_t>(rows())*std::max(c.hidden,values(-1))*2;
            const std::uint64_t group=static_cast<std::uint64_t>(rows())*(channels(-1)+values(-1));
            const std::uint64_t output=static_cast<std::uint64_t>(rows())*c.hidden*4;
            check(cudaHostAlloc(&host_input,input,cudaHostAllocPortable));check(cudaHostAlloc(&host_group,group,cudaHostAllocPortable));check(cudaHostAlloc(&host_output,output,cudaHostAllocPortable));pinned=input+group+output;
        }catch(...){cleanup();throw;}
    }
    ~Plan(){cleanup();}
    void cleanup()noexcept{int old=0;(void)cudaGetDevice(&old);for(int rank=0;rank<2;++rank){(void)cudaSetDevice(rank);if(streams[rank])(void)cudaStreamSynchronize(streams[rank]);if(ready[rank])(void)cudaEventDestroy(ready[rank]);if(streams[rank])(void)cudaStreamDestroy(streams[rank]);ready[rank]=nullptr;streams[rank]=nullptr;}
        for(auto p:{host_input,host_group,host_output})if(p)(void)cudaFreeHost(p);host_input=host_group=host_output=nullptr;(void)cudaSetDevice(old);}
    void sync(){for(int rank=0;rank<2;++rank){device(rank);check(cudaStreamSynchronize(streams[rank]));}}
    void copy(void* dst,const void* src,std::uint64_t bytes,cudaMemcpyKind kind,int rank){device(rank);check(cudaMemcpyAsync(dst,src,bytes,kind,streams[rank]));if(kind==cudaMemcpyHostToDevice){counters.h2d_bytes+=bytes;++counters.h2d_calls;}else if(kind==cudaMemcpyDeviceToHost){counters.d2h_bytes+=bytes;++counters.d2h_calls;}else ++counters.d2d_calls;}
    void copy2(void* dst,std::uint64_t dp,const void* src,std::uint64_t sp,std::uint64_t width,cudaMemcpyKind kind){device(0);check(cudaMemcpy2DAsync(dst,dp,src,sp,width,rows(),kind,streams[0]));if(kind==cudaMemcpyHostToDevice){counters.h2d_bytes+=width*rows();++counters.h2d_calls;}else ++counters.d2d_calls;}
    void record(int rank){device(rank);check(cudaEventRecord(ready[rank],streams[rank]));}
    void wait(int rank,int producer){device(rank);check(cudaStreamWaitEvent(streams[rank],ready[producer],0));}
    void broadcast(const void* input,int k){copy(host_input,input,static_cast<std::uint64_t>(rows())*k*2,cudaMemcpyDeviceToHost,0);record(0);wait(1,0);copy(r[1].normalized,host_input,static_cast<std::uint64_t>(rows())*k*2,cudaMemcpyHostToDevice,1);}
    void fused(int rank){device(rank);const int q=channels(rank),v=values(rank);
        Output out[2]={{r[rank].codes[0],r[rank].scales[0],r[rank].qkv,c.mode?q:q/2,q},{r[rank].codes[1],r[rank].scales[1],r[rank].z,c.mode?v:v/2,v}};
        if(!c.mode && rank==1){out[0].stride=out[0].n;out[1].stride=out[1].n;}
        f.ok(f.quantize(r[rank].normalized,r[rank].quant,r[rank].quant_bytes,rows(),c.hidden,2,streams[rank]));
        f.ok(f.fused(r[rank].quant,r[rank].quant_bytes,out,2,rows(),c.hidden,c.hidden,2,streams[rank]));
    }
    void core(int rank){device(rank);auto& b=r[rank];const int split=c.mode?2:1,h=c.heads/split,qk=c.qk/split,ch=channels(rank),v=values(rank),stride=qk*c.dk;
        f.ok(f.linear(b.normalized,static_cast<const float*>(b.a),b.av,rows(),h,c.hidden,0,streams[rank]));
        f.ok(f.linear(b.normalized,static_cast<const float*>(b.b),b.bv,rows(),h,c.hidden,0,streams[rank]));
        for(int s=0;s<c.sequences;++s){const int start=s*c.steps;auto off=[&](void* p,int width,int bytes=2)->void*{return static_cast<char*>(p)+static_cast<std::uint64_t>(start)*width*bytes;};
            if(c.prefixes)f.ok(f.history(off(b.qkv,ch),b.history[s],b.prefix_history[s],b.prefix_history[8+s],ch,c.kernel,2,0,streams[rank]));
            f.ok(f.conv(off(b.qkv,ch),b.conv,b.history[s],off(b.convolved,ch),b.history[s],c.steps,ch,c.kernel,0,1,streams[rank]));
            f.ok(f.gates(off(b.av,h),off(b.bv,h),static_cast<const float*>(b.rate),static_cast<const float*>(b.bias),static_cast<float*>(off(b.decay,h,4)),static_cast<float*>(off(b.beta,h,4)),c.steps,h,0,1,streams[rank]));
            auto* q=off(b.convolved,ch);auto* k=static_cast<char*>(q)+stride*2ULL;auto* value=static_cast<char*>(q)+stride*4ULL;
            if(c.prefixes)f.ok(f.prefix(q,k,value,static_cast<const float*>(off(b.decay,h,4)),static_cast<const float*>(off(b.beta,h,4)),static_cast<const float*>(b.state[s]),off(b.recurrent,v),static_cast<float*>(b.state[s]),c.steps,qk,h,c.dk,c.dv,1,1,0,c.scale,b.prefix_state[s],b.prefix_state[8+s],2,streams[rank]));
            else f.ok(f.gdn(q,k,value,static_cast<const float*>(off(b.decay,h,4)),static_cast<const float*>(off(b.beta,h,4)),static_cast<const float*>(b.state[s]),off(b.recurrent,v),static_cast<float*>(b.state[s]),c.steps,qk,h,c.dk,c.dv,1,1,0,c.scale,streams[rank]));
        }
        f.ok(f.gated_norm(b.recurrent,b.z,b.gamma,b.mixed,rows(),h,c.dv,0,c.gamma_fp32,c.epsilon,streams[rank]));
    }
    void row_output(int rank,const void* input,void* output){device(rank);auto& b=r[rank];const int k=values(-1),n=c.hidden/2,padded=(k+127)/128*128;
        f.ok(f.quantize(input,b.quant,b.quant_bytes,rows(),k,2,streams[rank]));
        if(rows()==1)f.ok(f.dot(b.quant,b.quant_bytes,b.codes[2],b.scales[2],output,rows(),n,k,padded,2,streams[rank]));
        else f.ok(f.tiled(b.quant,b.quant_bytes,b.codes[2],b.scales[2],output,rows(),n,k,padded,2,rows()>=8 && n>=1024?16:8,rows()>=8 && n>=1024?8:4,streams[rank]));
    }
    void column_output(int rank){device(rank);auto& b=r[rank];const int k=values(rank);
        f.ok(f.quantize(b.mixed,b.quant,b.quant_bytes,rows(),k,2,streams[rank]));
        f.ok(f.partial(b.quant,b.quant_bytes,b.codes[2],b.scales[2],static_cast<float*>(b.partial),rows(),c.hidden,k,(k+127)/128*128,streams[rank]));
    }
    void flow(){device(0);f.ok(f.rms(c.input,static_cast<const float*>(c.norm),r[0].normalized,rows(),c.hidden,0,c.epsilon,streams[0]));broadcast(r[0].normalized,c.hidden);
        if(c.mode){fused(1);core(1);column_output(1);copy(host_output,r[1].partial,static_cast<std::uint64_t>(rows())*c.hidden*4,cudaMemcpyDeviceToHost,1);record(1);
            fused(0);core(0);column_output(0);wait(0,1);copy(r[0].remote_partial,host_output,static_cast<std::uint64_t>(rows())*c.hidden*4,cudaMemcpyHostToDevice,0);
            f.ok(f.sum(static_cast<const float*>(r[0].partial),static_cast<const float*>(r[0].remote_partial),c.projected,rows()*c.hidden,streams[0]));
        }else{fused(1);const auto qbytes=static_cast<std::uint64_t>(rows())*channels(-1);const auto bytes=static_cast<std::uint64_t>(rows())*(channels(-1)+values(-1));copy(host_group,r[1].qkv,bytes,cudaMemcpyDeviceToHost,1);record(1);fused(0);wait(0,1);
            copy2(static_cast<char*>(r[0].qkv)+channels(-1),channels(-1)*2ULL,host_group,channels(-1),channels(-1),cudaMemcpyHostToDevice);
            copy2(static_cast<char*>(r[0].z)+values(-1),values(-1)*2ULL,static_cast<char*>(host_group)+qbytes,values(-1),values(-1),cudaMemcpyHostToDevice);
            core(0);broadcast(r[0].mixed,values(-1));row_output(1,r[1].normalized,r[1].local_output);copy(host_output,r[1].local_output,static_cast<std::uint64_t>(rows())*c.hidden,cudaMemcpyDeviceToHost,1);record(1);
            row_output(0,r[0].mixed,rows()==1?c.projected:r[0].local_output);
            if(rows()!=1)copy2(c.projected,c.hidden*2ULL,r[0].local_output,c.hidden,c.hidden,cudaMemcpyDeviceToDevice);
            wait(0,1);copy2(static_cast<char*>(c.projected)+c.hidden,c.hidden*2ULL,host_output,c.hidden,c.hidden,cudaMemcpyHostToDevice);
        }
        f.ok(f.add(c.input,c.projected,c.output,static_cast<std::int64_t>(rows())*c.hidden,0,streams[0]));
    }
};
}
extern "C" const char* venturi_gdn_head_error(){return error.c_str();}
extern "C" void venturi_gdn_head_sizes(std::uint64_t* out){if(out){out[0]=sizeof(Config);out[1]=sizeof(Rank);}}
extern "C" void* venturi_gdn_head_create(const char* path,const Config* c,const Rank* r){try{Device device;require(c && r,"null pilot descriptors");return new Plan(path,*c,r);}catch(const std::exception& e){error=e.what();return nullptr;}}
extern "C" void venturi_gdn_head_destroy(void* h){try{Device device;delete static_cast<Plan*>(h);}catch(const std::exception& e){error=e.what();}}
extern "C" int venturi_gdn_head_run(void* h,int count,double* seconds){try{Device device;require(h && count>0 && count<=10000 && seconds,"invalid pilot invocation");auto& p=*static_cast<Plan*>(h);std::lock_guard<std::mutex> lock(p.mutex);p.sync();p.counters={};const auto start=std::chrono::steady_clock::now();for(int i=0;i<count;++i)p.flow();p.sync();*seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();p.counters.iterations=count;return 0;}catch(const std::exception& e){error=e.what();return -1;}}
extern "C" int venturi_gdn_head_info(void* h,std::uint64_t* out){try{require(h && out,"null pilot info");auto& p=*static_cast<Plan*>(h);std::lock_guard<std::mutex> lock(p.mutex);const std::uint64_t v[]={p.counters.h2d_bytes,p.counters.d2h_bytes,p.counters.h2d_calls,p.counters.d2h_calls,p.counters.d2d_calls,p.counters.iterations,p.pinned};std::copy(v,v+7,out);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
