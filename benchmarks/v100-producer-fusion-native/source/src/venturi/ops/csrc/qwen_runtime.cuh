// Venturi-owned synchronous native Qwen host runtime. Included once by blocks.cu.
// No Python callbacks, Tensor ABI, external model forward, or dtype guessing.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <initializer_list>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <vector>
#include <array>
#include <span>

struct VRConfig {
    int device, dtype, layers, hidden, intermediate, vocab, hq, hkv, dim, rotary;
    int gqk, gvh, gkd, gvd, convolution, max_tokens, max_position, resident;
    float epsilon, gdn_scale, attention_scale;
    int second_device,peer_transport;
    std::uint64_t reserve;
};
struct VRSpan {
    const char* path;
    std::uint64_t logical, file_offset, bytes, file_bytes;
};
struct VRMatrix { std::uint64_t offset, scales; int n, k; };
struct VRLayer {
    int kind, conv_fp32, gamma_fp32, ab_fp32;
    // GDN: QKV,z,output,unused. Attention: Q/gate,K,V,O. Then MLP gate/up/down.
    VRMatrix matrix[7];
    const void *norm, *post, *conv, *rate, *bias, *gamma, *a, *b, *qgamma, *kgamma;
};

namespace vr {
inline double native_clock() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (b && a>UINT64_MAX/b) throw std::invalid_argument("native Qwen size overflow");
    return a*b;
}
inline std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if (a>UINT64_MAX-b) throw std::invalid_argument("native Qwen size overflow");
    return a+b;
}
inline void require(bool value,const char* message) {
    if (!value) throw std::invalid_argument(message);
}
inline void op(int result) {
    if (result<0) throw std::runtime_error(error_text);
}
struct Device {
    int old=-1;
    explicit Device(int device) { check(cudaGetDevice(&old));check(cudaSetDevice(device)); }
    ~Device() { if (old>=0) (void)cudaSetDevice(old); }
};
struct Buffer {
    void* ptr=nullptr; std::uint64_t bytes=0; int device=0;
    Buffer()=default;
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    void allocate(std::uint64_t n) {
        require(!ptr,"native Qwen buffer already allocated");
        check(cudaGetDevice(&device));bytes=n;
        if (n) check(cudaMalloc(&ptr,n));
    }
    ~Buffer() {
        if (ptr) { int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);
            (void)cudaFree(ptr);(void)cudaSetDevice(old); }
    }
    char* at(std::uint64_t offset=0) const { return static_cast<char*>(ptr)+offset; }
};
struct Stream {
    cudaStream_t ptr=nullptr; int device=0;
    void create() { check(cudaGetDevice(&device));check(cudaStreamCreateWithFlags(&ptr,cudaStreamNonBlocking)); }
    ~Stream() { if (ptr) { int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);
        (void)cudaStreamSynchronize(ptr);(void)cudaStreamDestroy(ptr);(void)cudaSetDevice(old); } }
};
struct Pinned {
    void* ptr=nullptr;
    std::uint64_t bytes=0;
    void allocate(std::uint64_t n) { bytes=n;check(cudaHostAlloc(&ptr,n,cudaHostAllocPortable)); }
    ~Pinned() { if (ptr) (void)cudaFreeHost(ptr); }
};
struct Event {
    cudaEvent_t ptr=nullptr;int device=0;
    void create() { check(cudaGetDevice(&device));check(cudaEventCreateWithFlags(&ptr,cudaEventDisableTiming)); }
    ~Event() { if (ptr) { int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);
        (void)cudaEventDestroy(ptr);(void)cudaSetDevice(old); } }
};
struct Dense {
    void* ptr=nullptr;int device=0;
    void allocate(int n,int k,int dtype) { check(cudaGetDevice(&device));ptr=venturi_dense_create(1,n,k,dtype);if (!ptr) throw std::runtime_error(error_text); }
    ~Dense() { if (ptr) { int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);venturi_dense_destroy(ptr);(void)cudaSetDevice(old); } }
};
struct Mapping {
    std::string path; struct stat stamp{}; int fd=-1; void* ptr=MAP_FAILED;
    std::uint64_t logical,begin,bytes;
    explicit Mapping(const VRSpan& s):path(s.path?s.path:""),logical(s.logical),begin(s.file_offset),bytes(s.bytes) {
        require(!path.empty() && bytes && add(begin,bytes)==s.file_bytes,"invalid native Qwen file span");
        fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if (fd<0) throw std::runtime_error("native Qwen cannot open source file");
        if (::fstat(fd,&stamp) || !S_ISREG(stamp.st_mode) || static_cast<std::uint64_t>(stamp.st_size)!=s.file_bytes) {
            ::close(fd);fd=-1;throw std::runtime_error("native Qwen source size/type changed");
        }
        ptr=::mmap(nullptr,stamp.st_size,PROT_READ,MAP_PRIVATE,fd,0);
        if (ptr==MAP_FAILED) { ::close(fd);fd=-1;throw std::runtime_error("native Qwen mmap failed"); }
    }
    bool same(const struct stat& s) const {
        return s.st_dev==stamp.st_dev && s.st_ino==stamp.st_ino && s.st_size==stamp.st_size &&
            s.st_mtim.tv_sec==stamp.st_mtim.tv_sec && s.st_mtim.tv_nsec==stamp.st_mtim.tv_nsec &&
            s.st_ctim.tv_sec==stamp.st_ctim.tv_sec && s.st_ctim.tv_nsec==stamp.st_ctim.tv_nsec;
    }
    void verify() const {
        struct stat a{},b{};
        if (::fstat(fd,&a) || ::stat(path.c_str(),&b) || !same(a) || !same(b))
            throw std::runtime_error("native Qwen source changed; recreate and verify model");
    }
    ~Mapping() { if (ptr!=MAP_FAILED) ::munmap(ptr,stamp.st_size);if (fd>=0) ::close(fd); }
};
inline int padded(const VRMatrix& m) { return ((m.k+127)/128)*128; }
inline std::uint64_t codes(const VRMatrix& m) { return mul(m.n,padded(m)); }
inline std::uint64_t scales(const VRMatrix& m) { return mul(m.n,padded(m)/32*2); }

template<class T> __global__ void tables(const float* inv,float* cos,float* sin,int half,int count) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        const float phase=__fmul_rn(static_cast<float>(i/half),inv[i%half]);
        cos[i]=read(cast<T>(cosf(phase)));sin[i]=read(cast<T>(sinf(phase)));
    }
}

#include "qwen_head_gdn.cuh"

// Explicit source ablation: 0=legacy, 1=grouped transport, 2=fused compute.
#ifndef VENTURI_Q8_GROUP_MODE
#define VENTURI_Q8_GROUP_MODE 2
#endif
struct Model {
    VRConfig c; std::vector<VRLayer> layers; VRMatrix embedding,head;
    int begin=0,end=0;bool stage=false;
    bool owns(int layer) const { return begin<=layer && layer<end; }
    std::vector<std::unique_ptr<Mapping>> files;
    std::vector<std::unique_ptr<Buffer>> constants;
    std::map<std::uint64_t,VRMatrix> matrices;
    std::map<std::uint64_t,std::unique_ptr<Buffer>> resident_weights;
    std::map<std::uint64_t,std::unique_ptr<Buffer>> second_weights;
    Buffer arena, inverse, blas_workspace;
    Buffer first_output,second_input,second_output;
    // Explicit alternate arithmetic only. Scratch is shared by serialized model
    // calls, not by independent request caches; no quantized weights are copied.
    int activation_passes=0;
    bool column_mlp_enabled=false,head_gdn_enabled=false;
    std::set<std::uint64_t> semantic_qkv,column_gdn;
    std::vector<VRLayer> head_peer_layers;HeadGDNWork head_work[2];
    bool column_weight(std::uint64_t offset) const {return column_down.count(offset) || column_gdn.count(offset);}
    std::set<std::uint64_t> column_down;
    Buffer column_local,column_remote,column_peer,column_gu,column_activation;
    Pinned column_host;
    Buffer first_quant,second_quant;
    Stream stream;
    Stream second_stream;
    Event first_ready,second_ready;
    Dense dense;
    const float* norm=nullptr;
    Pinned staging,host_input,host_output;
    std::uint64_t payload=0,device_bytes=0,uploaded=0,embedding_uploaded=0;
    std::uint64_t second_device_bytes=0,peer_input_bytes=0,peer_output_bytes=0;
    std::uint64_t stage_bytes=8*1024*1024;
    std::mutex mutex;
    void sync() {
        check(cudaStreamSynchronize(stream.ptr));
        if (second_stream.ptr) { Device d(c.second_device);check(cudaStreamSynchronize(second_stream.ptr)); }
    }
    bool parallel() const { return c.second_device>=0; }
    int first_rows(const VRMatrix& m) const { return parallel() && m.offset!=embedding.offset ? 1+(m.n-1)/2 : m.n; }
    int second_rows(const VRMatrix& m) const {
        return parallel() && (m.offset!=embedding.offset || m.offset==head.offset) ? m.n-(1+(m.n-1)/2) : 0;
    }
    std::uint64_t plane(const VRMatrix& m,int rows) const { return mul(rows,padded(m)); }
    std::uint64_t encoded(const VRMatrix& m,int rows) const { return add(plane(m,rows),mul(rows,padded(m)/32*2)); }
    void probe_peer() {
        Buffer first,second;
        const std::uint64_t limit=128*1024;
        first.allocate(limit);{ Device d(c.second_device);second.allocate(limit); }
        std::vector<unsigned char> expected(limit),actual(limit);
        for (std::uint64_t i=0;i<limit;++i) expected[i]=static_cast<unsigned char>((i*73)^(i/127)^0x39);
        const std::uint64_t sizes[]={64,4096,limit};
        for (const auto n:sizes) {
            check(cudaMemcpy(first.ptr,expected.data(),n,cudaMemcpyHostToDevice));
            { Device d(c.second_device);
                check(cudaMemsetAsync(second.ptr,0xa5,n,second_stream.ptr));
                check(cudaMemcpyPeerAsync(second.ptr,c.second_device,first.ptr,c.device,n,second_stream.ptr));
                check(cudaStreamSynchronize(second_stream.ptr));
                check(cudaMemcpy(actual.data(),second.ptr,n,cudaMemcpyDeviceToHost));
            }
            require(std::equal(expected.begin(),expected.begin()+n,actual.begin()),"CUDA primary-to-peer byte probe failed; select explicit host transport");
            check(cudaMemsetAsync(first.ptr,0xa5,n,stream.ptr));
            check(cudaMemcpyPeerAsync(first.ptr,c.device,second.ptr,c.second_device,n,stream.ptr));sync();
            check(cudaMemcpy(actual.data(),first.ptr,n,cudaMemcpyDeviceToHost));
            require(std::equal(expected.begin(),expected.begin()+n,actual.begin()),"CUDA peer-to-primary byte probe failed; select explicit host transport");
        }
    }
    void source() { for (const auto& f:files) f->verify(); }
    void guard(std::uint64_t n) {
        std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
        if (add(n,c.reserve)>free) throw std::runtime_error("native Qwen memory admission failed; no implicit offload/precision fallback");
    }
    void read_bytes(void* target,std::uint64_t offset,std::uint64_t bytes) const {
        require(offset<=payload && bytes<=payload-offset,"native Qwen read exceeds mapped payload");
        auto* out=static_cast<char*>(target);
        for (const auto& f:files) {
            const auto first=std::max(offset,f->logical),last=std::min(add(offset,bytes),add(f->logical,f->bytes));
            if (first<last) std::memcpy(out+first-offset,static_cast<const char*>(f->ptr)+f->begin+first-f->logical,last-first);
        }
    }
    void upload(void* target,std::uint64_t offset,std::uint64_t bytes) {
        for (std::uint64_t start=0;start<bytes;) {
            const auto n=std::min(stage_bytes,bytes-start);
            read_bytes(staging.ptr,offset+start,n);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging.ptr,n,cudaMemcpyHostToDevice,stream.ptr));
            sync();start+=n;
        }
        uploaded=add(uploaded,bytes);
    }
    void upload_other(void* target,std::uint64_t offset,std::uint64_t bytes) {
        Device d(c.second_device);
        for (std::uint64_t start=0;start<bytes;) {
            const auto n=std::min(stage_bytes,bytes-start);read_bytes(staging.ptr,offset+start,n);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging.ptr,n,cudaMemcpyHostToDevice,second_stream.ptr));
            check(cudaStreamSynchronize(second_stream.ptr));start+=n;
        }
        uploaded=add(uploaded,bytes);
    }
    void validate_matrix(const VRMatrix& m,int n,int k) {
        require(m.n==n && m.k==k && n>0 && k>0 && k<=INT32_MAX-127,"native Qwen matrix geometry mismatch");
        const auto a=codes(m),b=scales(m);
        require(m.offset<=payload && a<=payload-m.offset && m.scales<=payload && b<=payload-m.scales &&
            m.scales>=add(m.offset,a),"native Qwen matrix planes out of bounds");
        auto found=matrices.find(m.offset);
        if (found!=matrices.end()) require(found->second.n==m.n && found->second.k==m.k && found->second.scales==m.scales,"inconsistent native Qwen tying");
        else matrices.emplace(m.offset,m);
    }
    void validate_codes(const VRMatrix& m) {
        // One bounded CPU validation at load, not a repeated per-token scan.
        for (std::uint64_t start=0;start<codes(m);) {
            auto n=std::min(stage_bytes,codes(m)-start);read_bytes(staging.ptr,m.offset+start,n);
            for (std::uint64_t i=0;i<n;++i) if (static_cast<const unsigned char*>(staging.ptr)[i]==128)
                throw std::runtime_error("native Qwen forbidden Q8 code -128");
            start+=n;
        }
        for (std::uint64_t start=0;start<scales(m);) {
            auto n=std::min(stage_bytes,scales(m)-start);read_bytes(staging.ptr,m.scales+start,n);
            const auto* p=static_cast<const std::uint16_t*>(staging.ptr);
            for (std::uint64_t i=0;i<n/2;++i) if ((p[i]&0x7c00)==0x7c00 || ((p[i]&0x8000) && (p[i]&0x7fff)))
                throw std::runtime_error("native Qwen invalid Q8 scale");
            start+=n;
        }
    }
    const void* copy_constant(const void* p,std::uint64_t bytes) {
        require(p && bytes,"null native Qwen constant");cudaPointerAttributes attr{};check(cudaPointerGetAttributes(&attr,p));
        require(attr.type==cudaMemoryTypeDevice && attr.device==c.device,"native Qwen constant device mismatch");
        auto out=std::make_unique<Buffer>();out->allocate(bytes);
        check(cudaMemcpyAsync(out->ptr,p,bytes,cudaMemcpyDeviceToDevice,stream.ptr));
        auto result=out->ptr;device_bytes=add(device_bytes,bytes);constants.push_back(std::move(out));return result;
    }
    std::pair<const std::int8_t*,const __half*> weight(const VRMatrix& m) {
        require(!column_weight(m.offset) && !semantic_qkv.count(m.offset),"special-packed weight requires its explicit MLP/GDN path");
        source();const Buffer* b=nullptr;
        if (c.resident) b=resident_weights.at(m.offset).get();
        else { upload(arena.ptr,m.offset,codes(m));upload(arena.at(codes(m)),m.scales,scales(m));b=&arena; }
        return {static_cast<const std::int8_t*>(b->ptr),reinterpret_cast<const __half*>(b->at(c.resident?plane(m,first_rows(m)):codes(m)))};
    }
    void project(const VRMatrix& m,const void* x,const std::int8_t* codes,const __half* scales,void* out,int t,int rows,Buffer& quant,cudaStream_t s,bool packed=false) {
        if(!activation_passes) {op(venturi_q8_head_run(x,codes,scales,out,t,rows,m.k,padded(m),c.dtype,s));return;}
        if(!packed)op(venturi_q8_dp4a_quantize(x,quant.ptr,quant.bytes,t,m.k,activation_passes,s));
#if VENTURI_BLOCKS_ARCH == 70
        // Only the measured two-token residual path uses shared staging.
        // T1 regresses; larger activation planes retain the qualified kernel.
        if(t==2 && activation_passes==2 && vdq::bytes(t,m.k,activation_passes)<=48*1024) {
            op(venturi_q8_shared_run(quant.ptr,quant.bytes,codes,scales,out,t,rows,m.k,padded(m),activation_passes,16,s));return;
        }
        if(t>=3 && activation_passes==2) {
            // Small row grids retain the measured tile-4 geometry; tile-8
            // would underfill them even though it wins on larger row shards.
            const bool wide=t>=8 && rows>=1024;
            op(venturi_q8_tiled_shared_run(quant.ptr,quant.bytes,codes,scales,out,t,rows,m.k,padded(m),activation_passes,wide?16:8,wide?8:4,s));return;
        }
#endif
        op(venturi_q8_dp4a_run(quant.ptr,quant.bytes,codes,scales,out,t,rows,m.k,padded(m),activation_passes,s));
    }
    void linear(const VRMatrix& m,const void* x,void* out,int t,bool reuse_first=false,bool reuse_second=false) {
        auto w=weight(m);
        const int nr=second_rows(m),nf=m.n-nr;
        if (!nr) { project(m,x,w.first,w.second,out,t,m.n,first_quant,stream.ptr,reuse_first);return; }
        // Row shards evaluate complete dot products; gather FP16/BF16 results,
        // never rounded partial sums. The next producer event fences scratch
        // reuse until the previous peer output copy has completed.
        const auto xb=mul(mul(t,m.k),2);
        if (!reuse_second && c.peer_transport==1) check(cudaMemcpyAsync(host_input.ptr,x,xb,cudaMemcpyDeviceToHost,stream.ptr));
        // Retain both cross-device fences even when the immutable input is
        // shared: the previous output gather must finish before scratch reuse.
        check(cudaEventRecord(first_ready.ptr,stream.ptr));
        {
            Device d(c.second_device);
            check(cudaStreamWaitEvent(second_stream.ptr,first_ready.ptr,0));
            if (!reuse_second) {
                if (c.peer_transport==1) check(cudaMemcpyAsync(second_input.ptr,host_input.ptr,xb,cudaMemcpyHostToDevice,second_stream.ptr));
                else check(cudaMemcpyPeerAsync(second_input.ptr,c.second_device,x,c.device,xb,second_stream.ptr));
                peer_input_bytes=add(peer_input_bytes,xb);
            }
            auto& b=*second_weights.at(m.offset);
            project(m,second_input.ptr,static_cast<const std::int8_t*>(b.ptr),
                reinterpret_cast<const __half*>(b.at(plane(m,nr))),second_output.ptr,t,nr,second_quant,second_stream.ptr,reuse_second);
            if (c.peer_transport==1) check(cudaMemcpyAsync(host_output.ptr,second_output.ptr,mul(mul(t,nr),2),cudaMemcpyDeviceToHost,second_stream.ptr));
            check(cudaEventRecord(second_ready.ptr,second_stream.ptr));
        }
        void* first=t==1?out:first_output.ptr;
        project(m,x,w.first,w.second,first,t,nf,first_quant,stream.ptr,reuse_first);
        if (t!=1) check(cudaMemcpy2DAsync(out,m.n*2ULL,first,nf*2ULL,nf*2ULL,t,cudaMemcpyDeviceToDevice,stream.ptr));
        check(cudaStreamWaitEvent(stream.ptr,second_ready.ptr,0));
        if (c.peer_transport==1) check(cudaMemcpy2DAsync(static_cast<char*>(out)+nf*2ULL,m.n*2ULL,host_output.ptr,nr*2ULL,nr*2ULL,t,cudaMemcpyHostToDevice,stream.ptr));
        else if (t==1) check(cudaMemcpyPeerAsync(static_cast<char*>(out)+nf*2ULL,c.device,second_output.ptr,c.second_device,nr*2ULL,stream.ptr));
        else {
            cudaMemcpy3DPeerParms p{};
            p.srcDevice=c.second_device;p.dstDevice=c.device;
            p.srcPtr=make_cudaPitchedPtr(second_output.ptr,nr*2ULL,nr*2ULL,t);
            p.dstPtr=make_cudaPitchedPtr(static_cast<char*>(out)+nf*2ULL,m.n*2ULL,nr*2ULL,t);
            p.extent=make_cudaExtent(nr*2ULL,t,1);
            check(cudaMemcpy3DPeerAsync(&p,stream.ptr));
        }
        peer_output_bytes=add(peer_output_bytes,mul(mul(t,nr),2));
    }
    bool grouped_linears(const void* x,int t,std::initializer_list<std::pair<const VRMatrix*,void*>> projections) {
#if VENTURI_BLOCKS_ARCH == 70 && VENTURI_Q8_GROUP_MODE != 0
        if(activation_passes!=2 || projections.size()>3 || (parallel() && c.peer_transport!=1))return false;
        const int count=static_cast<int>(projections.size()),k=projections.begin()->first->k;
        VQFOutput first[3]{},peer[3]{};const VRMatrix* matrix[3]{};void* destination[3]{};
        std::uint64_t offsets[3]{},bytes=0;int i=0;
        for(const auto& p:projections) {
            for(int j=0;j<i;++j) {
                const auto a=reinterpret_cast<std::uintptr_t>(p.second),b=reinterpret_cast<std::uintptr_t>(destination[j]);
                if(a<b+mul(mul(t,matrix[j]->n),2) && b<a+mul(mul(t,p.first->n),2))return false;
            }
            const auto& m=*p.first;auto w=weight(m);const int nr=second_rows(m),nf=m.n-nr;
            if(parallel() && !nr)return false;
            matrix[i]=&m;destination[i]=p.second;first[i]={w.first,w.second,static_cast<__half*>(p.second),nf,m.n};
            if(parallel()) {
                auto& b=*second_weights.at(m.offset);offsets[i]=bytes;
                peer[i]={static_cast<const std::int8_t*>(b.ptr),reinterpret_cast<const __half*>(b.at(plane(m,nr))),reinterpret_cast<__half*>(second_output.at(bytes)),nr,nr};
                bytes=add(bytes,mul(mul(t,nr),2));
            }
            ++i;
        }
        if(parallel())require(bytes<=second_output.bytes && bytes<=host_output.bytes,"fused group exceeds admitted peer scratch");
        const auto xb=mul(mul(t,k),2);
        if(parallel()) {
            check(cudaMemcpyAsync(host_input.ptr,x,xb,cudaMemcpyDeviceToHost,stream.ptr));
            check(cudaEventRecord(first_ready.ptr,stream.ptr));
            Device d(c.second_device);check(cudaStreamWaitEvent(second_stream.ptr,first_ready.ptr,0));
            check(cudaMemcpyAsync(second_input.ptr,host_input.ptr,xb,cudaMemcpyHostToDevice,second_stream.ptr));peer_input_bytes=add(peer_input_bytes,xb);
#if VENTURI_Q8_GROUP_MODE == 2
            op(venturi_q8_dp4a_quantize(second_input.ptr,second_quant.ptr,second_quant.bytes,t,k,activation_passes,second_stream.ptr));
            op(vprojection::fused(second_quant.ptr,second_quant.bytes,peer,count,t,k,padded(*matrix[0]),activation_passes,second_stream.ptr));
#else
            for(int j=0;j<count;++j)project(*matrix[j],second_input.ptr,peer[j].codes,peer[j].scales,peer[j].out,t,peer[j].n,second_quant,second_stream.ptr,j!=0);
#endif
            check(cudaMemcpyAsync(host_output.ptr,second_output.ptr,bytes,cudaMemcpyDeviceToHost,second_stream.ptr));
            check(cudaEventRecord(second_ready.ptr,second_stream.ptr));
        }
#if VENTURI_Q8_GROUP_MODE == 2
        op(venturi_q8_dp4a_quantize(x,first_quant.ptr,first_quant.bytes,t,k,activation_passes,stream.ptr));
        op(vprojection::fused(first_quant.ptr,first_quant.bytes,first,count,t,k,padded(*matrix[0]),activation_passes,stream.ptr));
#else
        for(int j=0;j<count;++j) {
            void* out=(parallel() && t!=1)?first_output.ptr:destination[j];
            project(*matrix[j],x,first[j].codes,first[j].scales,out,t,first[j].n,first_quant,stream.ptr,j!=0);
            if(parallel() && t!=1)check(cudaMemcpy2DAsync(destination[j],matrix[j]->n*2ULL,out,first[j].n*2ULL,first[j].n*2ULL,t,cudaMemcpyDeviceToDevice,stream.ptr));
        }
#endif
        if(parallel()) {
            check(cudaStreamWaitEvent(stream.ptr,second_ready.ptr,0));
            for(int j=0;j<count;++j)check(cudaMemcpy2DAsync(static_cast<char*>(destination[j])+first[j].n*2ULL,matrix[j]->n*2ULL,static_cast<char*>(host_output.ptr)+offsets[j],peer[j].n*2ULL,peer[j].n*2ULL,t,cudaMemcpyHostToDevice,stream.ptr));
            peer_output_bytes=add(peer_output_bytes,bytes);
        }
        return true;
#else
        return false;
#endif
    }
    void shared_linears(const void* x,int t,std::initializer_list<std::pair<const VRMatrix*,void*>> projections) {
        require(projections.size()>1 && projections.begin()->first,"invalid shared projection group");
        const int k=projections.begin()->first->k;
        for(const auto& p:projections)require(p.first && p.first->k==k && p.second && p.second!=x,"shared projection input/geometry mismatch");
        if(grouped_linears(x,t,projections))return;
        // Local, explicit scope only: no pointer-based cache survives a group,
        // layer, forward chunk, head, failed call, request switch or reset.
        bool first=false,second=false;
        for(const auto& p:projections) {
            linear(*p.first,x,p.second,t,first,second);
            first=true;second=second || second_rows(*p.first)>0;
        }
    }
    VRMatrix column_descriptor(const VRMatrix& m) const {
        require(column_mlp_enabled && column_weight(m.offset),"matrix is not an explicit column-packed projection");
        return VRMatrix{m.offset,m.scales,m.n,m.k/2};
    }
    void upload_columns(Buffer& target,const VRMatrix& m,int rank) {
        source();const auto local=column_descriptor(m);
        for(bool scale:{false,true}) {
            const auto stride=scale?mul(padded(local)/32,2):static_cast<std::uint64_t>(padded(local));
            const auto copied=scale?mul(local.k/32,2):static_cast<std::uint64_t>(local.k);
            const auto source_stride=scale?mul(padded(m)/32,2):static_cast<std::uint64_t>(padded(m));
            require(stride<=stage_bytes,"column-packed row exceeds bounded staging");
            for(std::uint64_t row=0;row<static_cast<std::uint64_t>(m.n);) {
                const auto rows=std::min(static_cast<std::uint64_t>(m.n)-row,stage_bytes/stride);
                std::memset(staging.ptr,0,mul(rows,stride));
                for(std::uint64_t i=0;i<rows;++i)
                    read_bytes(static_cast<char*>(staging.ptr)+mul(i,stride),add(scale?m.scales:m.offset,add(mul(row+i,source_stride),mul(rank,copied))),copied);
                const auto offset=add(scale?plane(local,local.n):0,mul(row,stride));
                if(rank) {
                    Device device(c.second_device);
                    check(cudaMemcpyAsync(target.at(offset),staging.ptr,mul(rows,stride),cudaMemcpyHostToDevice,second_stream.ptr));
                    check(cudaStreamSynchronize(second_stream.ptr));
                } else {
                    check(cudaMemcpyAsync(target.at(offset),staging.ptr,mul(rows,stride),cudaMemcpyHostToDevice,stream.ptr));
                    check(cudaStreamSynchronize(stream.ptr));
                }
                uploaded=add(uploaded,mul(rows,stride));row+=rows;
            }
        }
        source();
    }
    void mlp(const VRLayer& l,const void* x,void* gate,void* up,void* out,int t,bool materialize=false) {
        if(!column_mlp_enabled) {
            shared_linears(x,t,{{&l.matrix[4],gate},{&l.matrix[5],up}});
            op(venturi_swiglu_run(gate,up,gate,mul(t,c.intermediate),c.dtype,stream.ptr));
            linear(l.matrix[6],gate,out,t);return;
        }
        const int half=c.intermediate/2;const auto xb=mul(mul(t,c.hidden),2),partial_bytes=mul(xb,2);
        require(t>0 && t<=c.max_tokens && x && gate && up && out,"invalid column MLP operands");
        auto gw=weight(l.matrix[4]),uw=weight(l.matrix[5]);const auto local=column_descriptor(l.matrix[6]);source();
        const auto& down=*resident_weights.at(local.offset);
        const auto half_bytes=mul(mul(t,half),2);
        VQFOutput first[2]={{gw.first,gw.second,static_cast<__half*>(gate),half,half},{uw.first,uw.second,static_cast<__half*>(up),half,half}};
        // The primary producer follows the prior gather before permitting peer
        // scratch reuse. The peer consumer fences its FP32 partial transport.
        check(cudaMemcpyAsync(host_input.ptr,x,xb,cudaMemcpyDeviceToHost,stream.ptr));
        check(cudaEventRecord(first_ready.ptr,stream.ptr));
        {
            Device device(c.second_device);check(cudaStreamWaitEvent(second_stream.ptr,first_ready.ptr,0));
            check(cudaMemcpyAsync(second_input.ptr,host_input.ptr,xb,cudaMemcpyHostToDevice,second_stream.ptr));
            VQFOutput peer[2]{};
            for(int i=0;i<2;++i) {
                const auto& matrix=l.matrix[4+i];const auto& b=*second_weights.at(matrix.offset);
                peer[i]={static_cast<const std::int8_t*>(b.ptr),reinterpret_cast<const __half*>(b.at(plane(matrix,half))),reinterpret_cast<__half*>(column_gu.at(mul(i,half_bytes))),half,half};
            }
            op(venturi_q8_dp4a_quantize(second_input.ptr,second_quant.ptr,second_quant.bytes,t,c.hidden,2,second_stream.ptr));
            op(vprojection::fused(second_quant.ptr,second_quant.bytes,peer,2,t,c.hidden,padded(l.matrix[4]),2,second_stream.ptr));
            op(vproducer::run(column_gu.ptr,column_gu.at(half_bytes),column_activation.ptr,second_quant.ptr,second_quant.bytes,t,half,second_stream.ptr,materialize));
            const auto& b=*second_weights.at(local.offset);
            op(vprojection::partial(second_quant.ptr,second_quant.bytes,static_cast<const std::int8_t*>(b.ptr),reinterpret_cast<const __half*>(b.at(plane(local,local.n))),static_cast<float*>(column_peer.ptr),t,c.hidden,half,padded(local),second_stream.ptr));
            check(cudaMemcpyAsync(column_host.ptr,column_peer.ptr,partial_bytes,cudaMemcpyDeviceToHost,second_stream.ptr));
            check(cudaEventRecord(second_ready.ptr,second_stream.ptr));
        }
        op(venturi_q8_dp4a_quantize(x,first_quant.ptr,first_quant.bytes,t,c.hidden,2,stream.ptr));
        op(vprojection::fused(first_quant.ptr,first_quant.bytes,first,2,t,c.hidden,padded(l.matrix[4]),2,stream.ptr));
        op(vproducer::run(gate,up,gate,first_quant.ptr,first_quant.bytes,t,half,stream.ptr,materialize));
        op(vprojection::partial(first_quant.ptr,first_quant.bytes,static_cast<const std::int8_t*>(down.ptr),reinterpret_cast<const __half*>(down.at(plane(local,local.n))),static_cast<float*>(column_local.ptr),t,c.hidden,half,padded(local),stream.ptr));
        check(cudaStreamWaitEvent(stream.ptr,second_ready.ptr,0));
        check(cudaMemcpyAsync(column_remote.ptr,column_host.ptr,partial_bytes,cudaMemcpyHostToDevice,stream.ptr));
        op(venturi_q8_column_sum(static_cast<const float*>(column_local.ptr),static_cast<const float*>(column_remote.ptr),out,static_cast<int>(mul(t,c.hidden)),stream.ptr));
        peer_input_bytes=add(peer_input_bytes,xb);peer_output_bytes=add(peer_output_bytes,partial_bytes);
    }
    Model(const VRConfig& cfg,const VRSpan* spans,int count,const VRLayer* ls,const VRMatrix& emb,const VRMatrix& h,const void* final_norm,const float* inv,int first=0,int last=-1,bool staged=false,const VRMatrix* mtp_input=nullptr,int profile=0,bool columns=false,bool head_gdn=false):c(cfg),embedding(emb),head(h),begin(first),end(last<0?cfg.layers:last),stage(staged),activation_passes(profile),column_mlp_enabled(columns),head_gdn_enabled(head_gdn) {
        require(begin>=0 && begin<end && end<=c.layers,"invalid native stage range");
        require(!stage || (c.resident==1 && c.second_device==-1),"pipeline stages require resident non-TP weights");
        require(profile==0 || (profile==2 && c.dtype==0 && c.resident==1 && !stage &&
            (!mtp_input || (VENTURI_BLOCKS_ARCH==70 && c.layers==1 && c.second_device==-1))),"residual activation profile requires an explicit complete resident FP16 target or SM70 single-device MTP body");
        require(c.dtype==0 || c.dtype==1,"native Qwen needs explicit FP16/BF16");(void)data_type(c.dtype);
        require(c.second_device>=-1 && c.second_device!=c.device,"invalid native Qwen peer device");
        require(c.peer_transport==0 || c.peer_transport==1,"invalid native Qwen peer transport");
        require(!columns || (VENTURI_BLOCKS_ARCH==70 && profile==2 && parallel() && c.peer_transport==1 && c.intermediate>0 && c.intermediate%64==0 && mul(c.max_tokens,c.hidden)<INT32_MAX-256),"column MLP requires resident SM70 FP16 residual host TP and group-aligned halves");
        require(!head_gdn || (columns && !mtp_input && c.gqk>=2 && c.gqk%2==0 && c.gvh<=128 && c.hidden<=32768 && c.hidden%32==0 && c.gvd>0 && c.gvd<=256 && c.gvd%64==0 && c.convolution>=2 && c.convolution<=8),"head GDN requires an explicit aligned resident SM70 FP16 host-TP column profile");
        if (parallel()) {
            require(c.resident==1,"tensor parallel requires explicit resident weights");
            int count=0;check(cudaGetDeviceCount(&count));require(c.second_device<count,"peer device out of range");
            cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,c.second_device));
            require(p.major*10+p.minor==VENTURI_BLOCKS_ARCH,"peer architecture mismatch");
            if (c.peer_transport==0) {
                int access=0;check(cudaDeviceCanAccessPeer(&access,c.device,c.second_device));require(access,"CUDA peer access unavailable");
                check(cudaDeviceCanAccessPeer(&access,c.second_device,c.device));require(access,"reverse CUDA peer access unavailable");
            }
        }
        require(c.layers>0 && c.layers<=1024 && c.hidden>0 && c.intermediate>0 && c.vocab>0 &&
            c.hq>0 && c.hkv>0 && c.hq%c.hkv==0 && (c.dim==64 || c.dim==128 || c.dim==256) &&
            c.rotary>=2 && c.rotary<=c.dim && c.rotary%2==0 && c.gqk>0 && c.gvh>0 && c.gvh%c.gqk==0 &&
            (c.gkd==64 || c.gkd==128 || c.gkd==256) && c.gvd>0 && c.convolution>0 && c.convolution<=32 &&
            c.max_tokens>0 && c.max_tokens<=1024 && c.max_position>0 && (c.resident==0 || c.resident==1) &&
            c.epsilon>0 && std::isfinite(c.epsilon) && c.gdn_scale>0 && std::isfinite(c.gdn_scale) &&
            c.attention_scale>0 && std::isfinite(c.attention_scale) && spans && count>0 && ls && inv,"invalid native Qwen config");
        const auto packed=add(mul(2,mul(c.gqk,c.gkd)),mul(c.gvh,c.gvd));
        require(mul(c.max_tokens,std::max({static_cast<std::uint64_t>(c.vocab),packed,mul(2,mul(c.hq,c.dim)),static_cast<std::uint64_t>(c.intermediate)}))<INT32_MAX &&
            mul(c.gvh,c.gvd)<INT32_MAX,"native Qwen launch geometry exceeds int32");
        for (int i=0;i<count;++i) { require(spans[i].logical==payload,"noncontiguous native Qwen file spans");files.push_back(std::make_unique<Mapping>(spans[i]));payload=add(payload,spans[i].bytes); }
        if (mtp_input) {
            require(!stage && !parallel() && c.layers==1 && c.resident==1,"MTP base requires one resident unsharded layer");
            validate_matrix(*mtp_input,c.hidden,2*c.hidden);
        } else {validate_matrix(embedding,c.vocab,c.hidden);validate_matrix(head,c.vocab,c.hidden);}
        require(static_cast<std::uint64_t>(padded(embedding))<=stage_bytes,"native Qwen embedding row exceeds staging bound");
        for (int i=0;i<c.rotary/2;++i) require(std::isfinite(inv[i]),"native Qwen inverse frequency must be finite");
        layers.assign(ls,ls+c.layers);
        std::uint64_t constant_bytes=mul(c.hidden,4);
        for (const auto& l:layers) {
            require((l.kind==0 || l.kind==1) && (l.conv_fp32==0 || l.conv_fp32==1) &&
                (l.gamma_fp32==0 || l.gamma_fp32==1) && (l.ab_fp32==0 || l.ab_fp32==1),"invalid native Qwen layer flags");
            constant_bytes=add(constant_bytes,mul(c.hidden,8));
            if (l.kind==0) {
                validate_matrix(l.matrix[0],packed,c.hidden);validate_matrix(l.matrix[1],c.gvh*c.gvd,c.hidden);
                validate_matrix(l.matrix[2],c.hidden,c.gvh*c.gvd);
                constant_bytes=add(constant_bytes,add(mul(c.convolution*packed,l.conv_fp32?4:2),
                    add(mul(c.gvh,8),add(mul(c.gvd,l.gamma_fp32?4:2),mul(mul(2,mul(c.gvh,c.hidden)),l.ab_fp32?4:2)))));
            } else {
                validate_matrix(l.matrix[0],2*c.hq*c.dim,c.hidden);validate_matrix(l.matrix[1],c.hkv*c.dim,c.hidden);
                validate_matrix(l.matrix[2],c.hkv*c.dim,c.hidden);validate_matrix(l.matrix[3],c.hidden,c.hq*c.dim);
                constant_bytes=add(constant_bytes,mul(c.dim,8));
            }
            validate_matrix(l.matrix[4],c.intermediate,c.hidden);validate_matrix(l.matrix[5],c.intermediate,c.hidden);validate_matrix(l.matrix[6],c.hidden,c.intermediate);
        }
        if(head_gdn) {
            std::set<std::uint64_t> ordinary{embedding.offset,head.offset};
            for(const auto& l:layers) {
                if(l.kind==0){require(l.ab_fp32==1,"head GDN requires source-preserved FP32 A/B projections");semantic_qkv.insert(l.matrix[0].offset);column_gdn.insert(l.matrix[2].offset);ordinary.insert(l.matrix[1].offset);}
                else for(int j=0;j<4;++j)ordinary.insert(l.matrix[j].offset);
                for(int j=4;j<7;++j)ordinary.insert(l.matrix[j].offset);
            }
            for(auto offset:semantic_qkv)require(!ordinary.count(offset) && !column_gdn.count(offset),"semantic QKV aliases an incompatible role");
            for(auto offset:column_gdn)require(!ordinary.count(offset),"column GDN output aliases an incompatible role");
            head_peer_layers=layers;
        }
        if(columns) {
            std::set<std::uint64_t> other_roles{embedding.offset,head.offset};
            for(const auto& l:layers) {
                column_down.insert(l.matrix[6].offset);
                for(int j=0;j<6;++j)if(l.kind==1 || j!=3)other_roles.insert(l.matrix[j].offset);
            }
            for(const auto offset:column_down)require(!other_roles.count(offset),"down projection aliases a non-column role");
        }
        if (stage) {
            std::set<std::uint64_t> used;
            if (begin==0) used.insert(embedding.offset);
            if (end==c.layers) used.insert(head.offset);
            for (int i=begin;i<end;++i) for (int j=0;j<7;++j)
                if (layers[i].kind==1 || j!=3) used.insert(layers[i].matrix[j].offset);
            for (auto it=matrices.begin();it!=matrices.end();)
                if (!used.count(it->first)) it=matrices.erase(it);else ++it;
        }
        std::uint64_t all=0,other=0,maxk=0,maxn=0,peak=mul(c.max_tokens,add(padded(embedding),padded(embedding)/32*2));
        for (const auto& [_,m]:matrices) {
            if(column_weight(m.offset)) {
                const auto local=column_descriptor(m);all=add(all,encoded(local,local.n));other=add(other,encoded(local,local.n));
            } else {all=add(all,encoded(m,first_rows(m)));other=add(other,encoded(m,second_rows(m)));}
            maxk=std::max(maxk,static_cast<std::uint64_t>(m.k));maxn=std::max(maxn,static_cast<std::uint64_t>(1+(m.n-1)/2));
            peak=std::max(peak,add(codes(m),scales(m)));
        }
#if VENTURI_BLOCKS_ARCH == 70 && VENTURI_Q8_GROUP_MODE != 0
        if(parallel() && activation_passes==2 && c.peer_transport==1) {
            for(const auto& l:layers) {
                for(bool second:{false,true}) {
                    const auto rows=[&](const VRMatrix& m){return static_cast<std::uint64_t>(second?second_rows(m):first_rows(m));};
                    auto grouped=add(rows(l.matrix[0]),rows(l.matrix[1]));
                    if(l.kind==1)grouped=add(grouped,rows(l.matrix[2]));
                    maxn=std::max(maxn,std::max(grouped,add(rows(l.matrix[4]),rows(l.matrix[5]))));
                }
            }
        }
#endif
        const auto scratch=parallel()?mul(mul(c.max_tokens,maxn),2):0;
        const auto quant=profile?vdq::bytes(c.max_tokens,static_cast<int>(maxk),profile):0;
        const auto column_partial=columns?mul(mul(c.max_tokens,c.hidden),4):0;
        const auto column_gu_bytes=columns?mul(mul(c.max_tokens,c.intermediate),2):0;
        const auto column_activation_bytes=columns?mul(c.max_tokens,c.intermediate):0;
        const auto head_scratch=head_gdn?HeadGDNWork::estimate(c):0;
        guard(add(head_scratch,add(add(add(add(add(c.resident?all:peak,constant_bytes),scratch),4*1024*1024+mul(c.rotary/2,4)),quant),mul(column_partial,2))));
        if (parallel()) {
            Device d(c.second_device);guard(add(add(head_scratch,head_gdn?constant_bytes:0),add(add(add(other,add(scratch,mul(mul(c.max_tokens,maxk),2))),quant),add(column_partial,add(column_gu_bytes,column_activation_bytes)))));
            if (c.peer_transport==0) {
                auto status=cudaDeviceEnablePeerAccess(c.device,0);
                if (status==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();else check(status);
            }
            second_stream.create();second_ready.create();
            second_input.allocate(mul(mul(c.max_tokens,maxk),2));second_output.allocate(scratch);
            second_device_bytes=add(second_input.bytes,second_output.bytes);
            if(quant){second_quant.allocate(quant);second_device_bytes=add(second_device_bytes,quant);}
            if(columns) {column_peer.allocate(column_partial);column_gu.allocate(column_gu_bytes);column_activation.allocate(column_activation_bytes);second_device_bytes=add(second_device_bytes,add(column_partial,add(column_gu_bytes,column_activation_bytes)));}
        }
        if(quant){first_quant.allocate(quant);device_bytes=add(device_bytes,quant);}
        if (parallel()) {
            if (c.peer_transport==0) {
                auto status=cudaDeviceEnablePeerAccess(c.second_device,0);
                if (status==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();else check(status);
            }
            first_ready.create();first_output.allocate(scratch);device_bytes=add(device_bytes,scratch);
            if (c.peer_transport==1) { host_input.allocate(second_input.bytes);host_output.allocate(scratch); }
        }
        if(columns) {column_local.allocate(column_partial);column_remote.allocate(column_partial);column_host.allocate(column_partial);device_bytes=add(device_bytes,mul(column_partial,2));}
        stream.create();staging.allocate(stage_bytes);
        if(head_gdn){head_work[0].allocate(c);device_bytes=add(device_bytes,head_work[0].bytes);{Device d(c.second_device);head_work[1].allocate(c);second_device_bytes=add(second_device_bytes,head_work[1].bytes);}}
        if (parallel() && c.peer_transport==0) probe_peer();
        // No large CPU expansion: validation walks the mapped encoded planes.
        for (const auto& [_,m]:matrices) validate_codes(m);
        if (c.resident) {
            for (const auto& [offset,m]:matrices) {
                if(column_weight(offset)) {
                    const auto local=column_descriptor(m);auto b=std::make_unique<Buffer>();b->allocate(encoded(local,local.n));
                    upload_columns(*b,m,0);device_bytes=add(device_bytes,b->bytes);resident_weights.emplace(offset,std::move(b));
                    {Device d(c.second_device);b=std::make_unique<Buffer>();b->allocate(encoded(local,local.n));}
                    upload_columns(*b,m,1);second_device_bytes=add(second_device_bytes,b->bytes);second_weights.emplace(offset,std::move(b));continue;
                }
                const int nf=first_rows(m),ns=second_rows(m);auto b=std::make_unique<Buffer>();b->allocate(encoded(m,nf));
                if(semantic_qkv.count(offset))head_gdn_upload_rows(*this,*b,m,0);
                else {upload(b->ptr,m.offset,plane(m,nf));upload(b->at(plane(m,nf)),m.scales,mul(nf,padded(m)/32*2));}
                device_bytes=add(device_bytes,b->bytes);resident_weights.emplace(offset,std::move(b));
                if (ns) {
                    { Device d(c.second_device);b=std::make_unique<Buffer>();b->allocate(encoded(m,ns)); }
                    const auto skip=static_cast<std::uint64_t>(m.n-ns);
                    if(semantic_qkv.count(offset))head_gdn_upload_rows(*this,*b,m,1);
                    else {upload_other(b->ptr,add(m.offset,mul(skip,padded(m))),plane(m,ns));
                        upload_other(b->at(plane(m,ns)),add(m.scales,mul(skip,padded(m)/32*2)),mul(ns,padded(m)/32*2));}
                    second_device_bytes=add(second_device_bytes,b->bytes);second_weights.emplace(offset,std::move(b));
                }
            }
        } else { arena.allocate(peak);device_bytes=add(device_bytes,peak); }
        inverse.allocate(mul(c.rotary/2,4));check(cudaMemcpyAsync(inverse.ptr,inv,inverse.bytes,cudaMemcpyHostToDevice,stream.ptr));device_bytes=add(device_bytes,inverse.bytes);
        if (end==c.layers) norm=static_cast<const float*>(copy_constant(final_norm,mul(c.hidden,4)));
        for (int index=begin;index<end;++index) {auto& l=layers[index];
            l.norm=copy_constant(l.norm,mul(c.hidden,4));l.post=copy_constant(l.post,mul(c.hidden,4));
            if (l.kind==0 && head_gdn) {
                const auto original=l;head_gdn_constants(*this,original,l,0);head_gdn_constants(*this,original,head_peer_layers[index],1);
            } else if (l.kind==0) {
                l.conv=copy_constant(l.conv,mul(c.convolution*packed,l.conv_fp32?4:2));
                l.rate=copy_constant(l.rate,mul(c.gvh,4));l.bias=copy_constant(l.bias,mul(c.gvh,4));
                l.gamma=copy_constant(l.gamma,mul(c.gvd,l.gamma_fp32?4:2));
                l.a=copy_constant(l.a,mul(mul(c.gvh,c.hidden),l.ab_fp32?4:2));l.b=copy_constant(l.b,mul(mul(c.gvh,c.hidden),l.ab_fp32?4:2));
            } else { l.qgamma=copy_constant(l.qgamma,mul(c.dim,4));l.kgamma=copy_constant(l.kgamma,mul(c.dim,4)); }
        }
        blas_workspace.allocate(4*1024*1024);device_bytes=add(device_bytes,blas_workspace.bytes);
        dense.allocate(c.gvh,c.hidden,c.dtype);
        sync();source();
    }
    ~Model() { int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(c.device);
        if (stream.ptr) (void)cudaStreamSynchronize(stream.ptr);
        if (second_stream.ptr) { (void)cudaSetDevice(c.second_device);(void)cudaStreamSynchronize(second_stream.ptr); }
        (void)cudaSetDevice(old); }
};

// Optional request-owned verification traces. No model-global pointer cache.
struct PrefixTrace {
    int prefixes=0;std::vector<void*> history[2],state[2],peer_history[2],peer_state[2];
    std::vector<void*>& slots(int rank,int kind,int prefix){return rank?(kind?peer_state[prefix]:peer_history[prefix]):(kind?state[prefix]:history[prefix]);}
    void resize(int layers){for(int rank=0;rank<2;++rank)for(int kind=0;kind<2;++kind)for(int prefix=0;prefix<2;++prefix)slots(rank,kind,prefix).resize(layers);}
};
struct Session {
    PrefixTrace* verification_trace=nullptr;
    std::shared_ptr<Model> model;
    int capacity,position=0; bool failed=false;
    double first_token_seconds=0,generation_seconds=0;
    int fault_layer=-1,fault_position=-1;
    std::mutex mutex;
    std::map<std::string,std::unique_ptr<Buffer>> w;
    std::vector<std::unique_ptr<Buffer>> history,state,keys,values,peer_history,peer_state;
    std::vector<std::uint16_t> host_logits;
    std::vector<std::int64_t> ids;
    std::uint64_t device_bytes=0,peer_device_bytes=0;
    void* alloc(const std::string& name,std::uint64_t n) {
        auto b=std::make_unique<Buffer>();b->allocate(n);auto p=b->ptr;device_bytes=add(device_bytes,n);w.emplace(name,std::move(b));return p;
    }
    void* p(const std::string& name) { return w.at(name)->ptr; }
    static std::uint64_t estimate(const Model& m,int capacity,bool caches=true) {
        const auto& c=m.c;require(capacity>0 && capacity<=c.max_position && mul(capacity,c.rotary/2)<INT32_MAX,"native Qwen capacity exceeds context/table bounds");
        const std::uint64_t packed=2*mul(c.gqk,c.gkd)+mul(c.gvh,c.gvd),qw=mul(c.hq,c.dim),kw=mul(c.hkv,c.dim),t=c.max_tokens;
        std::uint64_t widths=mul(c.hidden,7); // x,out,normalized,projected,residual,MLP normalized + spare.
        if(!m.head_gdn_enabled)widths=add(widths,add(mul(packed,2),add(mul(c.gvh*c.gvd,3),mul(c.gvh,2))));
        widths=add(widths,add(mul(c.intermediate,2),add(mul(qw,4),kw)));
        std::uint64_t n=add(mul(mul(widths,t),2),m.head_gdn_enabled?0:mul(mul(t,c.gvh),8));
        n=add(n,add(mul(mul(t,c.vocab),2),mul(t,8)));
        n=add(n,add(mul(mul(capacity,c.rotary/2),8),16+min_tiles(c.vocab)*16));
        for (int i=m.begin;caches && i<m.end;++i) {const auto& l=m.layers[i];n=add(n,l.kind==0 ? (add(mul(mul(c.convolution-1,packed),2),mul(mul(c.gvh*c.gvd,c.gkd),4))/(m.head_gdn_enabled?2:1)) : mul(mul(capacity,kw),4));}
        return n;
    }
    static std::uint64_t estimate_peer(const Model& m,bool caches=true) {
        if(!m.head_gdn_enabled || !caches)return 0;const auto& c=m.c;std::uint64_t n=0;
        const auto packed=add(mul(2,mul(c.gqk,c.gkd)),mul(c.gvh,c.gvd));
        for(int i=m.begin;i<m.end;++i)if(m.layers[i].kind==0)n=add(n,add(mul(mul(c.convolution-1,packed),2),mul(mul(c.gvh*c.gvd,c.gkd),4))/2);return n;
    }
    static int min_tiles(int n) { return std::min((n-1)/4096+1,128); }
    explicit Session(std::shared_ptr<Model> m,int cap,bool caches=true):model(std::move(m)),capacity(cap) {
        auto& md=*model;const auto& c=md.c;md.guard(estimate(md,cap,caches));if(md.head_gdn_enabled){Device d(c.second_device);md.guard(estimate_peer(md,caches));}
        const std::uint64_t packed=2*mul(c.gqk,c.gkd)+mul(c.gvh,c.gvd),qw=mul(c.hq,c.dim),kw=mul(c.hkv,c.dim),t=c.max_tokens;
        for (auto name: {"x","out","normalized","projected","residual","mlp_norm","spare"}) alloc(name,mul(mul(t,c.hidden),2));
        if(!md.head_gdn_enabled) {
        for (auto name: {"qkv","convolved"}) alloc(name,mul(mul(t,packed),2));
        for (auto name: {"z","mixed","recurrent"}) alloc(name,mul(mul(t,c.gvh*c.gvd),2));
        for (auto name: {"a","b"}) alloc(name,mul(mul(t,c.gvh),2));
        for (auto name: {"decay","beta"}) alloc(name,mul(mul(t,c.gvh),4));
        }
        for (auto name: {"mlp_gate","mlp_up"}) alloc(name,mul(mul(t,c.intermediate),2));
        alloc("qg",mul(mul(t,qw),4));alloc("query",mul(mul(t,qw),2));alloc("attended",mul(mul(t,qw),2));alloc("key",mul(mul(t,kw),2));
        alloc("logits",mul(mul(t,c.vocab),2));alloc("ids",mul(t,8));alloc("selection",16);alloc("select_scratch",min_tiles(c.vocab)*16);
        alloc("cos",mul(mul(cap,c.rotary/2),4));alloc("sin",mul(mul(cap,c.rotary/2),4));
        history.resize(c.layers);state.resize(c.layers);keys.resize(c.layers);values.resize(c.layers);peer_history.resize(c.layers);peer_state.resize(c.layers);
        auto statebuf=[&](std::unique_ptr<Buffer>& dst,std::uint64_t bytes,bool zero) { dst=std::make_unique<Buffer>();dst->allocate(bytes);device_bytes=add(device_bytes,bytes);if (zero && bytes) check(cudaMemsetAsync(dst->ptr,0,bytes,md.stream.ptr)); };
        for (int i=md.begin;caches && i<md.end;++i) {
            if (md.layers[i].kind==0) {
                const auto hb=mul(mul(c.convolution-1,packed),2)/(md.head_gdn_enabled?2:1),sb=mul(mul(c.gvh*c.gvd,c.gkd),4)/(md.head_gdn_enabled?2:1);
                statebuf(history[i],hb,true);statebuf(state[i],sb,true);
                if(md.head_gdn_enabled){Device d(c.second_device);for(auto pair:{std::pair<std::unique_ptr<Buffer>*,std::uint64_t>{&peer_history[i],hb},{&peer_state[i],sb}}){auto b=std::make_unique<Buffer>();b->allocate(pair.second);check(cudaMemsetAsync(b->ptr,0,b->bytes,md.second_stream.ptr));peer_device_bytes=add(peer_device_bytes,b->bytes);*pair.first=std::move(b);}}
            }
            else { statebuf(keys[i],mul(mul(cap,kw),2),false);statebuf(values[i],mul(mul(cap,kw),2),false); }
        }
        host_logits.resize(mul(t,c.vocab));ids.resize(t);
        const int count=static_cast<int>(mul(cap,c.rotary/2));require(mul(cap,c.rotary/2)<INT32_MAX,"native Qwen RoPE table exceeds int32");
        const int grid=std::min((count-1)/256+1,65535);
        if (c.dtype==0) tables<__half><<<grid,256,0,md.stream.ptr>>>(static_cast<const float*>(md.inverse.ptr),static_cast<float*>(p("cos")),static_cast<float*>(p("sin")),c.rotary/2,count);
        else tables<__nv_bfloat16><<<grid,256,0,md.stream.ptr>>>(static_cast<const float*>(md.inverse.ptr),static_cast<float*>(p("cos")),static_cast<float*>(p("sin")),c.rotary/2,count);
        check(cudaGetLastError());md.sync();require(device_bytes==estimate(md,cap,caches) && peer_device_bytes==estimate_peer(md,caches),"native Qwen memory estimate mismatch");
    }
    ~Session(){try{Device d(model->c.device);model->sync();}catch(...){}}
    std::array<Buffer*,4> recurrent_buffers(int layer){return {history[layer].get(),state[layer].get(),peer_history[layer].get(),peer_state[layer].get()};}
    HeadGDNSegment head_segment(int layer,int row,int count) {
        HeadGDNSegment result;result.row=row;result.count=count;result.prefixes=verification_trace?verification_trace->prefixes:0;
        for(int rank=0;rank<2;++rank){auto& part=result.rank[rank];part.history=(rank?peer_history:history)[layer]->ptr;part.state=(rank?peer_state:state)[layer]->ptr;
            for(int prefix=0;prefix<result.prefixes;++prefix){part.prefix_history[prefix]=verification_trace->slots(rank,0,prefix)[layer];part.prefix_state[prefix]=verification_trace->slots(rank,1,prefix)[layer];}}
        return result;
    }
    void snapshot(int layer,int kind,void* target,std::uint64_t bytes) {
        auto& m=*model;const auto& c=m.c;const auto* src=kind==0?history[layer].get():kind==1?state[layer].get():kind==2?keys[layer].get():values[layer].get();require(src,"native Qwen snapshot slot absent");
        const bool split=m.head_gdn_enabled && kind<2;const auto n=kind<2?mul(src->bytes,split?2:1):mul(mul(position,c.hkv*c.dim),2);require(bytes==n && (target || !n),"native Qwen snapshot byte size mismatch");
        m.sync();if(!split){if(n)check(cudaMemcpyAsync(target,src->ptr,n,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();return;}
        for(int rank=0;rank<2;++rank){Device d(rank?c.second_device:c.device);auto stream=rank?m.second_stream.ptr:m.stream.ptr;auto* buffer=rank?(kind?peer_state[layer].get():peer_history[layer].get()):src;
            if(kind==1)check(cudaMemcpyAsync(static_cast<char*>(target)+mul(rank,src->bytes),buffer->ptr,src->bytes,cudaMemcpyDeviceToHost,stream));
            else {const auto indices=head_gdn_rows(c,rank);const int full=2*c.gqk*c.gkd+c.gvh*c.gvd,local=full/2;const int widths[]={c.gqk*c.gkd/2,c.gqk*c.gkd/2,c.gvh*c.gvd/2};std::uint64_t offset=0;
                for(int width:widths){check(cudaMemcpy2DAsync(static_cast<char*>(target)+mul(indices[offset],2),mul(full,2),buffer->at(mul(offset,2)),mul(local,2),mul(width,2),c.convolution-1,cudaMemcpyDeviceToHost,stream));offset+=width;}}
        }m.sync();
    }
    void input(const std::int64_t* tokens,int n) {
        require(!failed,"native Qwen request failed; reset required");
        require(tokens && n>0 && n<=capacity-position,"native Qwen input exceeds capacity");
        for (int i=0;i<n;++i) require(tokens[i]>=0 && tokens[i]<model->c.vocab,"native Qwen token ID out of range");
        model->source();
    }
    void embedding(const std::int64_t* tokens,int t,void* out) {
        auto& m=*model;const auto& c=m.c;const auto& e=m.embedding;
        if (c.resident) {
            auto w=m.weight(e);check(cudaMemcpyAsync(p("ids"),tokens,t*8,cudaMemcpyHostToDevice,m.stream.ptr));
            op(venturi_q8_embedding_run(w.first,w.second,static_cast<const std::int64_t*>(p("ids")),out,t,c.vocab,c.hidden,padded(e),c.dtype,m.stream.ptr));
        } else {
            m.source();const auto a=mul(t,padded(e)),b=mul(t,padded(e)/32*2);
            // Selected rows only; source code/scale values are copied unchanged.
            for (int i=0;i<t;++i) {
                m.read_bytes(m.staging.ptr,e.offset+mul(tokens[i],padded(e)),padded(e));
                check(cudaMemcpyAsync(m.arena.at(mul(i,padded(e))),m.staging.ptr,padded(e),cudaMemcpyHostToDevice,m.stream.ptr));m.sync();
                m.read_bytes(m.staging.ptr,e.scales+mul(tokens[i],padded(e)/32*2),padded(e)/32*2);
                check(cudaMemcpyAsync(m.arena.at(a+mul(i,padded(e)/32*2)),m.staging.ptr,padded(e)/32*2,cudaMemcpyHostToDevice,m.stream.ptr));m.sync();
                ids[i]=i;
            }
            check(cudaMemcpyAsync(p("ids"),ids.data(),t*8,cudaMemcpyHostToDevice,m.stream.ptr));
            op(venturi_q8_embedding_run(static_cast<const std::int8_t*>(m.arena.ptr),reinterpret_cast<const __half*>(m.arena.at(a)),static_cast<const std::int64_t*>(p("ids")),out,t,t,c.hidden,padded(e),c.dtype,m.stream.ptr));
            m.uploaded=add(m.uploaded,add(a,b));m.embedding_uploaded=add(m.embedding_uploaded,add(a,b));
        }
    }
    void layer(int index,int pos,int t,void* x,void* out) {
        auto& m=*model;const auto& c=m.c;const auto& l=m.layers[index];auto s=m.stream.ptr;
        const auto count=mul(t,c.hidden);
        op(venturi_rms_run(x,static_cast<const float*>(l.norm),p("normalized"),t,c.hidden,c.dtype,c.epsilon,s));
        if(l.kind==0 && m.head_gdn_enabled){const auto segment=head_segment(index,0,t);head_gdn_forward(m,index,p("normalized"),p("projected"),t,std::span<const HeadGDNSegment>(&segment,1));}
        else if (l.kind==0) {
            m.shared_linears(p("normalized"),t,{{&l.matrix[0],p("qkv")},{&l.matrix[1],p("z")}});
            for (int i=0;i<2;++i) {
                auto w=i?l.b:l.a;auto target=p(i?"b":"a");
                if (l.ab_fp32) op(venturi_fp32_linear_run(p("normalized"),static_cast<const float*>(w),target,t,c.gvh,c.hidden,c.dtype,s));
                else { static_cast<Linear*>(m.dense.ptr)->tokens=t;op(venturi_dense_run(m.dense.ptr,p("normalized"),w,target,m.blas_workspace.ptr,m.blas_workspace.bytes,s)); }
            }
            const int width=2*c.gqk*c.gkd+c.gvh*c.gvd;
            if(verification_trace) {
                require(t==verification_trace->prefixes+1,"verification/history trace width mismatch");
                op(venturi_history_prefix_run(p("qkv"),history[index]->ptr,verification_trace->history[0][index],verification_trace->history[1][index],width,c.convolution,verification_trace->prefixes,c.dtype,s));
            }
            op(venturi_conv_run(p("qkv"),l.conv,history[index]->ptr,p("convolved"),history[index]->ptr,t,width,c.convolution,c.dtype,l.conv_fp32,s));
            op(venturi_gates_run(p("a"),p("b"),static_cast<const float*>(l.rate),static_cast<const float*>(l.bias),static_cast<float*>(p("decay")),static_cast<float*>(p("beta")),t,c.gvh,c.dtype,1,s));
            auto* q=static_cast<char*>(p("convolved"));const auto stride=mul(mul(c.gqk,c.gkd),2);
            if(verification_trace)op(venturi_gdn_prefix_run(q,q+stride,q+2*stride,static_cast<const float*>(p("decay")),static_cast<const float*>(p("beta")),static_cast<const float*>(state[index]->ptr),p("recurrent"),static_cast<float*>(state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,verification_trace->state[0][index],verification_trace->state[1][index],verification_trace->prefixes,s));
            else op((m.activation_passes==2?venturi_gdn_compact_run:venturi_gdn_run)(q,q+stride,q+2*stride,static_cast<const float*>(p("decay")),static_cast<const float*>(p("beta")),static_cast<const float*>(state[index]->ptr),p("recurrent"),static_cast<float*>(state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,s));
            op(venturi_gated_rms_run(p("recurrent"),p("z"),l.gamma,p("mixed"),t,c.gvh,c.gvd,c.dtype,l.gamma_fp32,c.epsilon,s));
            m.linear(l.matrix[2],p("mixed"),p("projected"),t);
        } else {
            auto* kt=keys[index]->at(mul(mul(pos,c.hkv*c.dim),2));auto* vt=values[index]->at(mul(mul(pos,c.hkv*c.dim),2));
            m.shared_linears(p("normalized"),t,{{&l.matrix[0],p("qg")},{&l.matrix[1],p("key")},{&l.matrix[2],vt}});
            auto* cos=static_cast<const float*>(p("cos"))+mul(pos,c.rotary/2);auto* sin=static_cast<const float*>(p("sin"))+mul(pos,c.rotary/2);
            op(venturi_qwen_qk_run(p("qg"),p("key"),static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),cos,sin,p("query"),kt,t,c.hq,c.hkv,c.dim,c.rotary,c.dtype,c.epsilon,s));
            op(venturi_attention_run(p("query"),keys[index]->ptr,values[index]->ptr,p("attended"),t,pos+t,c.hq,c.hkv,c.dim,c.dtype,c.attention_scale,s));
            op(venturi_qwen_gate_run(p("attended"),p("qg"),p("attended"),t,c.hq,c.dim,c.dtype,s));
            m.linear(l.matrix[3],p("attended"),p("projected"),t);
        }
        op(venturi_add_run(x,p("projected"),p("residual"),count,c.dtype,s));
        op(venturi_rms_run(p("residual"),static_cast<const float*>(l.post),p("mlp_norm"),t,c.hidden,c.dtype,c.epsilon,s));
        m.mlp(l,p("mlp_norm"),p("mlp_gate"),p("mlp_up"),p("projected"),t);
        op(venturi_add_run(p("residual"),p("projected"),out,count,c.dtype,s));
    }
    void* run_layers(int pos,int t,void* x,void* trace,int n,int start) {
        auto& m=*model;const auto& c=m.c;void* out=p("out");
        for (int i=m.begin;i<m.end;++i) {
            layer(i,pos,t,x,out);std::swap(x,out);
            if (trace) check(cudaMemcpyAsync(static_cast<char*>(trace)+mul(add(mul(i,n),start),mul(c.hidden,2)),x,mul(mul(t,c.hidden),2),cudaMemcpyDeviceToHost,m.stream.ptr));
            if (i==fault_layer && pos==fault_position) { fault_layer=-1;throw std::runtime_error("injected native Qwen layer failure"); }
        }
        return x;
    }
    void output(void* x,int t,bool all,float* logits) {
        auto& m=*model;const auto& c=m.c;
        op(venturi_rms_run(x,m.norm,p("normalized"),t,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
        const int rows=all?t:1;
        m.linear(m.head,static_cast<char*>(p("normalized"))+(all?0:mul(mul(t-1,c.hidden),2)),p("logits"),rows);
        if (logits) {
            const auto elements=mul(rows,c.vocab);
            check(cudaMemcpyAsync(host_logits.data(),p("logits"),mul(elements,2),cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
            for (std::uint64_t j=0;j<elements;++j) logits[j]=c.dtype==0 ? __half2float(reinterpret_cast<const __half*>(host_logits.data())[j]) : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(host_logits.data())[j]);
        }
    }
    void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace,void* hidden=nullptr) {
        input(tokens,n);auto& m=*model;const auto& c=m.c;const int initial=position;bool started=false;
        require(!m.stage,"internal pipeline stage cannot execute as a complete model");
        try {
            for (int start=0;start<n;start+=c.max_tokens) {
                const int t=std::min(c.max_tokens,n-start),pos=initial+start;
                started=true;embedding(tokens+start,t,p("x"));
                auto* x=run_layers(pos,t,p("x"),trace,n,start);
                if (all || start+t==n) output(x,t,all,logits?logits+(all?mul(start,c.vocab):0):nullptr);
                else if (hidden) op(venturi_rms_run(x,m.norm,p("normalized"),t,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
                if (hidden) check(cudaMemcpyAsync(static_cast<char*>(hidden)+mul(mul(start,c.hidden),2),p("normalized"),mul(mul(t,c.hidden),2),cudaMemcpyDeviceToDevice,m.stream.ptr));
            }
            m.sync();m.source();position=initial+n;
        } catch (...) { if (started) { failed=true;try { m.sync(); } catch (...) {} }throw; }
    }
    void reset() {
        auto& m=*model;
        try { m.sync();for (int i=0;i<m.c.layers;++i) if (state[i]) {
            if (history[i]->bytes) check(cudaMemsetAsync(history[i]->ptr,0,history[i]->bytes,m.stream.ptr));
            check(cudaMemsetAsync(state[i]->ptr,0,state[i]->bytes,m.stream.ptr)); }
            if(m.head_gdn_enabled){Device d(m.c.second_device);for(int i=0;i<m.c.layers;++i)if(peer_state[i]){if(peer_history[i]->bytes)check(cudaMemsetAsync(peer_history[i]->ptr,0,peer_history[i]->bytes,m.second_stream.ptr));check(cudaMemsetAsync(peer_state[i]->ptr,0,peer_state[i]->bytes,m.second_stream.ptr));}}
            m.sync();position=0;failed=false;fault_layer=-1;
        } catch (...) { failed=true;throw; }
    }
};
using Handle=std::shared_ptr<Model>;
}

extern "C" int venturi_qwen_sizes(std::uint64_t* out) {
    if (!out) return -1;out[0]=sizeof(VRConfig);out[1]=sizeof(VRSpan);out[2]=sizeof(VRMatrix);out[3]=sizeof(VRLayer);return 0;
}
extern "C" void* venturi_qwen_create(const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layers,const VRMatrix* embedding,const VRMatrix* head,const void* norm,const float* inverse) {
    try { vr::require(c && embedding && head,"null native Qwen model descriptor");vr::Device d(c->device);
        int device=0;check(cudaGetDevice(&device));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device));
        vr::require(prop.major*10+prop.minor==VENTURI_BLOCKS_ARCH,"native Qwen architecture mismatch");
        return new vr::Handle(std::make_shared<vr::Model>(*c,spans,count,layers,*embedding,*head,norm,inverse));
    } catch (const std::exception& e) { error_text=e.what();return nullptr; }
}
extern "C" void venturi_qwen_destroy(void* h) { delete static_cast<vr::Handle*>(h); }
// Additive ABI-16 entry point: the old constructor retains original arithmetic.
extern "C" void* venturi_qwen_create_profiled(const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layers,const VRMatrix* embedding,const VRMatrix* head,const void* norm,const float* inverse,int profile) {
    try {vr::require(c && embedding && head && profile==2,"invalid explicit projection profile");vr::Device d(c->device);
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,c->device));vr::require(prop.major*10+prop.minor==VENTURI_BLOCKS_ARCH,"native Qwen architecture mismatch");
        return new vr::Handle(std::make_shared<vr::Model>(*c,spans,count,layers,*embedding,*head,norm,inverse,0,-1,false,nullptr,profile));
    }catch(const std::exception& e){error_text=e.what();return nullptr;}
}
// Additive opt-in constructor; existing profile IDs/layouts stay unchanged.
extern "C" void* venturi_qwen_create_column_mlp(const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layers,const VRMatrix* embedding,const VRMatrix* head,const void* norm,const float* inverse) {
    try {vr::require(c && embedding && head,"null column MLP model descriptor");vr::Device device(c->device);
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,c->device));vr::require(prop.major*10+prop.minor==VENTURI_BLOCKS_ARCH,"native Qwen architecture mismatch");
        return new vr::Handle(std::make_shared<vr::Model>(*c,spans,count,layers,*embedding,*head,norm,inverse,0,-1,false,nullptr,2,true));
    }catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" int venturi_qwen_column_mlp_info(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null column MLP info");auto m=*static_cast<vr::Handle*>(h);std::lock_guard<std::mutex> lock(m->mutex);
        const std::uint64_t values[]={m->column_mlp_enabled?1ULL:0ULL,m->column_local.bytes,m->column_remote.bytes,m->column_peer.bytes,m->column_gu.bytes,m->column_activation.bytes,m->column_host.bytes,m->column_down.size()};
        std::copy(values,values+8,out);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_projection_info(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null projection profile info");auto m=*static_cast<vr::Handle*>(h);std::lock_guard<std::mutex> lock(m->mutex);
        out[0]=m->activation_passes;out[1]=m->first_quant.bytes;out[2]=m->second_quant.bytes;return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" void* venturi_qwen_session(void* h,int capacity) {
    try { vr::require(h,"null native Qwen model");auto m=*static_cast<vr::Handle*>(h);vr::Device d(m->c.device);std::lock_guard<std::mutex> lock(m->mutex);
        m->source();return new vr::Session(m,capacity);
    } catch (const std::exception& e) { error_text=e.what();return nullptr; }
}
extern "C" void venturi_qwen_session_destroy(void* h) { delete static_cast<vr::Session*>(h); }
extern "C" int venturi_qwen_forward(void* h,const std::int64_t* ids,int n,float* logits,std::uint64_t elements,int all,void* trace,std::uint64_t trace_elements) {
    try { vr::require(h,"null native Qwen session");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;vr::Device d(m.c.device);
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
        vr::require((all==0 || all==1) && logits && n>0 && elements==vr::mul(all?n:1,m.c.vocab) &&
            ((!trace && !trace_elements) || (trace && trace_elements==vr::mul(vr::mul(n,m.c.hidden),m.c.layers))),"native Qwen output/trace shape mismatch");
        s.forward(ids,n,logits,all,trace);return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
// Untimed host diagnostic for independent full-MLP tests; no private caches or
// cursors are advanced. Production generation never calls this entry point.
extern "C" int venturi_qwen_column_mlp_debug(void* h,int layer,const void* input,int t,void* output,void* activation,float* partials) {
    try {vr::require(h && input && output && activation && partials,"null column MLP diagnostic operand");
        auto& r=*static_cast<vr::Session*>(h);auto& m=*r.model;vr::Device device(m.c.device);
        std::lock_guard<std::mutex> a(r.mutex),b(m.mutex);
        vr::require(m.column_mlp_enabled && !r.failed && layer>=0 && layer<m.c.layers && t>0 && t<=m.c.max_tokens,"invalid column MLP diagnostic state/shape");
        m.source();const auto half=m.c.intermediate/2,hidden=m.c.hidden;const auto bytes=vr::mul(vr::mul(t,hidden),2);
        try {
            check(cudaMemcpyAsync(r.p("mlp_norm"),input,bytes,cudaMemcpyHostToDevice,m.stream.ptr));
            m.mlp(m.layers[layer],r.p("mlp_norm"),r.p("mlp_gate"),r.p("mlp_up"),r.p("projected"),t,true);
            check(cudaMemcpyAsync(output,r.p("projected"),bytes,cudaMemcpyDeviceToHost,m.stream.ptr));
            check(cudaMemcpy2DAsync(activation,m.c.intermediate*2ULL,r.p("mlp_gate"),half*2ULL,half*2ULL,t,cudaMemcpyDeviceToHost,m.stream.ptr));
            check(cudaMemcpyAsync(partials,m.column_local.ptr,bytes*2,cudaMemcpyDeviceToHost,m.stream.ptr));
            check(cudaMemcpyAsync(partials+vr::mul(t,hidden),m.column_remote.ptr,bytes*2,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
            {vr::Device peer(m.c.second_device);check(cudaMemcpy2DAsync(static_cast<char*>(activation)+half*2ULL,m.c.intermediate*2ULL,m.column_activation.ptr,half*2ULL,half*2ULL,t,cudaMemcpyDeviceToHost,m.second_stream.ptr));}m.sync();m.source();
        }catch(...){r.failed=true;try{m.sync();}catch(...){}throw;}
        return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_generate(void* h,const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* tokens,int* count) {
    const double begun=vr::native_clock();
    try { vr::require(h,"null native Qwen session");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;vr::Device d(m.c.device);
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
        s.first_token_seconds=s.generation_seconds=0;
        vr::require(maximum>0 && tokens && count && eos_count>=0 && (eos || !eos_count),"invalid native Qwen generation arguments");*count=0;s.input(prompt,n);
        vr::require(vr::add(n,maximum)<=static_cast<std::uint64_t>(s.capacity-s.position),"native Qwen generation exceeds capacity");
        for (int i=0;i<eos_count;++i) vr::require(eos[i]>=0 && eos[i]<m.c.vocab,"native Qwen EOS out of range");
        try {
            for (int step=0;step<maximum;++step) {
                s.forward(step?tokens+step-1:prompt,step?1:n,nullptr,false,nullptr);
                vr::op(venturi_greedy_run(s.p("logits"),static_cast<std::int64_t*>(s.p("selection")),s.p("select_scratch"),vr::Session::min_tiles(m.c.vocab)*16,1,m.c.vocab,m.c.dtype,m.stream.ptr));
                std::int64_t selected[2];check(cudaMemcpyAsync(selected,s.p("selection"),16,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
                if (selected[1] || selected[0]<0 || selected[0]>=m.c.vocab) throw std::runtime_error("native Qwen greedy rejected nonfinite logits");
                tokens[step]=selected[0];*count=step+1;
                if(step==0)s.first_token_seconds=vr::native_clock()-begun;
                if (eos_count && std::find(eos,eos+eos_count,selected[0])!=eos+eos_count) break;
            }
            m.source();s.generation_seconds=vr::native_clock()-begun;return 0;
        } catch (...) { s.failed=true;throw; }
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_info(void* h,std::uint64_t* out) {
    try { vr::require(h && out,"null native Qwen info");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(s.position),s.failed?1ULL:0ULL,m.device_bytes,s.device_bytes,m.stage_bytes+m.host_input.bytes+m.host_output.bytes+m.column_host.bytes,m.payload,m.uploaded,m.embedding_uploaded};
        std::copy(v,v+8,out);return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_parallel_info(void* h,std::uint64_t* out) {
    try { vr::require(h && out,"null native Qwen parallel info");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(m.c.second_device+1),m.second_device_bytes,m.peer_input_bytes,m.peer_output_bytes};
        std::copy(v,v+4,out);return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_reset(void* h) {
    try { vr::require(h,"null native Qwen reset");auto& s=*static_cast<vr::Session*>(h);vr::Device d(s.model->c.device);
        std::lock_guard<std::mutex> a(s.mutex),b(s.model->mutex);s.reset();return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_snapshot(void* h,int layer,int kind,void* target,std::uint64_t bytes) {
    try { vr::require(h,"null native Qwen snapshot");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;vr::Device d(m.c.device);
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);vr::require(layer>=0 && layer<m.c.layers && kind>=0 && kind<=3,"invalid native Qwen snapshot slot");
        s.snapshot(layer,kind,target,bytes);return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_debug_failure(void* h,int layer,int position) {
    try { vr::require(h,"null native Qwen debug session");auto& s=*static_cast<vr::Session*>(h);std::lock_guard<std::mutex> a(s.mutex);
        vr::require(layer>=-1 && layer<s.model->c.layers && position>=0,"invalid native Qwen debug failure");s.fault_layer=layer;s.fault_position=position;return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}

extern "C" int venturi_qwen_timings(void* h,double* out) {try{vr::require(h && out,"null native Qwen timings");auto& s=*static_cast<vr::Session*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}

extern "C" void* venturi_qwen_create_head_gdn(const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layers,const VRMatrix* embedding,const VRMatrix* head,const void* norm,const float* inverse) {
    try{vr::require(c && embedding && head,"null head GDN descriptor");vr::Device d(c->device);cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,c->device));vr::require(p.major*10+p.minor==VENTURI_BLOCKS_ARCH,"head GDN architecture mismatch");return new vr::Handle(std::make_shared<vr::Model>(*c,spans,count,layers,*embedding,*head,norm,inverse,0,-1,false,nullptr,2,true,true));}catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" int venturi_qwen_head_gdn_info(void* h,std::uint64_t* out) {
    try{vr::require(h && out,"null head GDN model info");auto m=*static_cast<vr::Handle*>(h);std::lock_guard<std::mutex> lock(m->mutex);const std::uint64_t values[]={m->head_gdn_enabled?1ULL:0ULL,m->head_work[0].bytes,m->head_work[1].bytes,m->semantic_qkv.size(),m->column_gdn.size()};std::copy(values,values+5,out);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_head_gdn_session_info(void* h,std::uint64_t* out) {
    try{vr::require(h && out,"null head GDN session info");auto& s=*static_cast<vr::Session*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.peer_device_bytes;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}
}
