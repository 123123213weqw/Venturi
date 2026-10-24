// Venturi-owned synchronous native Qwen host runtime. Included once by blocks.cu.
// No Python callbacks, Tensor ABI, external model forward, or dtype guessing.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

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

struct Model {
    VRConfig c; std::vector<VRLayer> layers; VRMatrix embedding,head;
    std::vector<std::unique_ptr<Mapping>> files;
    std::vector<std::unique_ptr<Buffer>> constants;
    std::map<std::uint64_t,VRMatrix> matrices;
    std::map<std::uint64_t,std::unique_ptr<Buffer>> resident_weights;
    std::map<std::uint64_t,std::unique_ptr<Buffer>> second_weights;
    Buffer arena, inverse, blas_workspace;
    Buffer first_output,second_input,second_output;
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
        source();const Buffer* b=nullptr;
        if (c.resident) b=resident_weights.at(m.offset).get();
        else { upload(arena.ptr,m.offset,codes(m));upload(arena.at(codes(m)),m.scales,scales(m));b=&arena; }
        return {static_cast<const std::int8_t*>(b->ptr),reinterpret_cast<const __half*>(b->at(c.resident?plane(m,first_rows(m)):codes(m)))};
    }
    void linear(const VRMatrix& m,const void* x,void* out,int t) {
        auto w=weight(m);
        const int nr=second_rows(m),nf=m.n-nr;
        if (!nr) { op(venturi_q8_head_run(x,w.first,w.second,out,t,m.n,m.k,padded(m),c.dtype,stream.ptr));return; }
        // Row shards evaluate complete dot products; gather FP16/BF16 results,
        // never rounded partial sums. The next producer event fences scratch
        // reuse until the previous peer output copy has completed.
        const auto xb=mul(mul(t,m.k),2);
        if (c.peer_transport==1) check(cudaMemcpyAsync(host_input.ptr,x,xb,cudaMemcpyDeviceToHost,stream.ptr));
        check(cudaEventRecord(first_ready.ptr,stream.ptr));
        {
            Device d(c.second_device);
            check(cudaStreamWaitEvent(second_stream.ptr,first_ready.ptr,0));
            if (c.peer_transport==1) check(cudaMemcpyAsync(second_input.ptr,host_input.ptr,xb,cudaMemcpyHostToDevice,second_stream.ptr));
            else check(cudaMemcpyPeerAsync(second_input.ptr,c.second_device,x,c.device,xb,second_stream.ptr));
            peer_input_bytes=add(peer_input_bytes,xb);
            auto& b=*second_weights.at(m.offset);
            op(venturi_q8_head_run(second_input.ptr,static_cast<const std::int8_t*>(b.ptr),
                reinterpret_cast<const __half*>(b.at(plane(m,nr))),second_output.ptr,t,nr,m.k,padded(m),c.dtype,second_stream.ptr));
            if (c.peer_transport==1) check(cudaMemcpyAsync(host_output.ptr,second_output.ptr,mul(mul(t,nr),2),cudaMemcpyDeviceToHost,second_stream.ptr));
            check(cudaEventRecord(second_ready.ptr,second_stream.ptr));
        }
        void* first=t==1?out:first_output.ptr;
        op(venturi_q8_head_run(x,w.first,w.second,first,t,nf,m.k,padded(m),c.dtype,stream.ptr));
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
    Model(const VRConfig& cfg,const VRSpan* spans,int count,const VRLayer* ls,const VRMatrix& emb,const VRMatrix& h,const void* final_norm,const float* inv):c(cfg),embedding(emb),head(h) {
        require(c.dtype==0 || c.dtype==1,"native Qwen needs explicit FP16/BF16");(void)data_type(c.dtype);
        require(c.second_device>=-1 && c.second_device!=c.device,"invalid native Qwen peer device");
        require(c.peer_transport==0 || c.peer_transport==1,"invalid native Qwen peer transport");
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
        validate_matrix(embedding,c.vocab,c.hidden);validate_matrix(head,c.vocab,c.hidden);
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
        std::uint64_t all=0,other=0,maxk=0,maxn=0,peak=mul(c.max_tokens,add(padded(embedding),padded(embedding)/32*2));
        for (const auto& [_,m]:matrices) {
            all=add(all,encoded(m,first_rows(m)));other=add(other,encoded(m,second_rows(m)));
            maxk=std::max(maxk,static_cast<std::uint64_t>(m.k));maxn=std::max(maxn,static_cast<std::uint64_t>(1+(m.n-1)/2));
            peak=std::max(peak,add(codes(m),scales(m)));
        }
        const auto scratch=parallel()?mul(mul(c.max_tokens,maxn),2):0;
        guard(add(add(add(c.resident?all:peak,constant_bytes),scratch),4*1024*1024+mul(c.rotary/2,4)));
        if (parallel()) {
            Device d(c.second_device);guard(add(other,add(scratch,mul(mul(c.max_tokens,maxk),2))));
            if (c.peer_transport==0) {
                auto status=cudaDeviceEnablePeerAccess(c.device,0);
                if (status==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();else check(status);
            }
            second_stream.create();second_ready.create();
            second_input.allocate(mul(mul(c.max_tokens,maxk),2));second_output.allocate(scratch);
            second_device_bytes=add(second_input.bytes,second_output.bytes);
        }
        if (parallel()) {
            if (c.peer_transport==0) {
                auto status=cudaDeviceEnablePeerAccess(c.second_device,0);
                if (status==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();else check(status);
            }
            first_ready.create();first_output.allocate(scratch);device_bytes=add(device_bytes,scratch);
            if (c.peer_transport==1) { host_input.allocate(second_input.bytes);host_output.allocate(scratch); }
        }
        stream.create();staging.allocate(stage_bytes);
        if (parallel() && c.peer_transport==0) probe_peer();
        // No large CPU expansion: validation walks the mapped encoded planes.
        for (const auto& [_,m]:matrices) validate_codes(m);
        if (c.resident) {
            for (const auto& [offset,m]:matrices) {
                const int nf=first_rows(m),ns=second_rows(m);auto b=std::make_unique<Buffer>();b->allocate(encoded(m,nf));
                upload(b->ptr,m.offset,plane(m,nf));upload(b->at(plane(m,nf)),m.scales,mul(nf,padded(m)/32*2));
                device_bytes=add(device_bytes,b->bytes);resident_weights.emplace(offset,std::move(b));
                if (ns) {
                    { Device d(c.second_device);b=std::make_unique<Buffer>();b->allocate(encoded(m,ns)); }
                    const auto skip=static_cast<std::uint64_t>(m.n-ns);
                    upload_other(b->ptr,add(m.offset,mul(skip,padded(m))),plane(m,ns));
                    upload_other(b->at(plane(m,ns)),add(m.scales,mul(skip,padded(m)/32*2)),mul(ns,padded(m)/32*2));
                    second_device_bytes=add(second_device_bytes,b->bytes);second_weights.emplace(offset,std::move(b));
                }
            }
        } else { arena.allocate(peak);device_bytes=add(device_bytes,peak); }
        inverse.allocate(mul(c.rotary/2,4));check(cudaMemcpyAsync(inverse.ptr,inv,inverse.bytes,cudaMemcpyHostToDevice,stream.ptr));device_bytes=add(device_bytes,inverse.bytes);
        norm=static_cast<const float*>(copy_constant(final_norm,mul(c.hidden,4)));
        for (auto& l:layers) {
            l.norm=copy_constant(l.norm,mul(c.hidden,4));l.post=copy_constant(l.post,mul(c.hidden,4));
            if (l.kind==0) {
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

struct Session {
    std::shared_ptr<Model> model;
    int capacity,position=0; bool failed=false;
    int fault_layer=-1,fault_position=-1;
    std::mutex mutex;
    std::map<std::string,std::unique_ptr<Buffer>> w;
    std::vector<std::unique_ptr<Buffer>> history,state,keys,values;
    std::vector<std::uint16_t> host_logits;
    std::vector<std::int64_t> ids;
    std::uint64_t device_bytes=0;
    void* alloc(const std::string& name,std::uint64_t n) {
        auto b=std::make_unique<Buffer>();b->allocate(n);auto p=b->ptr;device_bytes=add(device_bytes,n);w.emplace(name,std::move(b));return p;
    }
    void* p(const std::string& name) { return w.at(name)->ptr; }
    static std::uint64_t estimate(const Model& m,int capacity) {
        const auto& c=m.c;require(capacity>0 && capacity<=c.max_position && mul(capacity,c.rotary/2)<INT32_MAX,"native Qwen capacity exceeds context/table bounds");
        const std::uint64_t packed=2*mul(c.gqk,c.gkd)+mul(c.gvh,c.gvd),qw=mul(c.hq,c.dim),kw=mul(c.hkv,c.dim),t=c.max_tokens;
        std::uint64_t widths=mul(c.hidden,7); // x,out,normalized,projected,residual,MLP normalized + spare.
        widths=add(widths,add(mul(packed,2),add(mul(c.gvh*c.gvd,3),mul(c.gvh,2))));
        widths=add(widths,add(mul(c.intermediate,2),add(mul(qw,4),kw)));
        std::uint64_t n=add(mul(mul(widths,t),2),mul(mul(t,c.gvh),8));
        n=add(n,add(mul(mul(t,c.vocab),2),mul(t,8)));
        n=add(n,add(mul(mul(capacity,c.rotary/2),8),16+min_tiles(c.vocab)*16));
        for (const auto& l:m.layers) n=add(n,l.kind==0 ? add(mul(mul(c.convolution-1,packed),2),mul(mul(c.gvh*c.gvd,c.gkd),4)) : mul(mul(capacity,kw),4));
        return n;
    }
    static int min_tiles(int n) { return std::min((n-1)/4096+1,128); }
    explicit Session(std::shared_ptr<Model> m,int cap):model(std::move(m)),capacity(cap) {
        auto& md=*model;const auto& c=md.c;md.guard(estimate(md,cap));
        const std::uint64_t packed=2*mul(c.gqk,c.gkd)+mul(c.gvh,c.gvd),qw=mul(c.hq,c.dim),kw=mul(c.hkv,c.dim),t=c.max_tokens;
        for (auto name: {"x","out","normalized","projected","residual","mlp_norm","spare"}) alloc(name,mul(mul(t,c.hidden),2));
        for (auto name: {"qkv","convolved"}) alloc(name,mul(mul(t,packed),2));
        for (auto name: {"z","mixed","recurrent"}) alloc(name,mul(mul(t,c.gvh*c.gvd),2));
        for (auto name: {"a","b"}) alloc(name,mul(mul(t,c.gvh),2));
        for (auto name: {"decay","beta"}) alloc(name,mul(mul(t,c.gvh),4));
        for (auto name: {"mlp_gate","mlp_up"}) alloc(name,mul(mul(t,c.intermediate),2));
        alloc("qg",mul(mul(t,qw),4));alloc("query",mul(mul(t,qw),2));alloc("attended",mul(mul(t,qw),2));alloc("key",mul(mul(t,kw),2));
        alloc("logits",mul(mul(t,c.vocab),2));alloc("ids",mul(t,8));alloc("selection",16);alloc("select_scratch",min_tiles(c.vocab)*16);
        alloc("cos",mul(mul(cap,c.rotary/2),4));alloc("sin",mul(mul(cap,c.rotary/2),4));
        history.resize(c.layers);state.resize(c.layers);keys.resize(c.layers);values.resize(c.layers);
        auto statebuf=[&](std::unique_ptr<Buffer>& dst,std::uint64_t bytes,bool zero) { dst=std::make_unique<Buffer>();dst->allocate(bytes);device_bytes=add(device_bytes,bytes);if (zero && bytes) check(cudaMemsetAsync(dst->ptr,0,bytes,md.stream.ptr)); };
        for (int i=0;i<c.layers;++i) {
            if (md.layers[i].kind==0) { statebuf(history[i],mul(mul(c.convolution-1,packed),2),true);statebuf(state[i],mul(mul(c.gvh*c.gvd,c.gkd),4),true); }
            else { statebuf(keys[i],mul(mul(cap,kw),2),false);statebuf(values[i],mul(mul(cap,kw),2),false); }
        }
        host_logits.resize(mul(t,c.vocab));ids.resize(t);
        const int count=static_cast<int>(mul(cap,c.rotary/2));require(mul(cap,c.rotary/2)<INT32_MAX,"native Qwen RoPE table exceeds int32");
        const int grid=std::min((count-1)/256+1,65535);
        if (c.dtype==0) tables<__half><<<grid,256,0,md.stream.ptr>>>(static_cast<const float*>(md.inverse.ptr),static_cast<float*>(p("cos")),static_cast<float*>(p("sin")),c.rotary/2,count);
        else tables<__nv_bfloat16><<<grid,256,0,md.stream.ptr>>>(static_cast<const float*>(md.inverse.ptr),static_cast<float*>(p("cos")),static_cast<float*>(p("sin")),c.rotary/2,count);
        check(cudaGetLastError());md.sync();require(device_bytes==estimate(md,cap),"native Qwen memory estimate mismatch");
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
        if (l.kind==0) {
            m.linear(l.matrix[0],p("normalized"),p("qkv"),t);m.linear(l.matrix[1],p("normalized"),p("z"),t);
            for (int i=0;i<2;++i) {
                auto w=i?l.b:l.a;auto target=p(i?"b":"a");
                if (l.ab_fp32) op(venturi_fp32_linear_run(p("normalized"),static_cast<const float*>(w),target,t,c.gvh,c.hidden,c.dtype,s));
                else { static_cast<Linear*>(m.dense.ptr)->tokens=t;op(venturi_dense_run(m.dense.ptr,p("normalized"),w,target,m.blas_workspace.ptr,m.blas_workspace.bytes,s)); }
            }
            const int width=2*c.gqk*c.gkd+c.gvh*c.gvd;
            op(venturi_conv_run(p("qkv"),l.conv,history[index]->ptr,p("convolved"),history[index]->ptr,t,width,c.convolution,c.dtype,l.conv_fp32,s));
            op(venturi_gates_run(p("a"),p("b"),static_cast<const float*>(l.rate),static_cast<const float*>(l.bias),static_cast<float*>(p("decay")),static_cast<float*>(p("beta")),t,c.gvh,c.dtype,1,s));
            auto* q=static_cast<char*>(p("convolved"));const auto stride=mul(mul(c.gqk,c.gkd),2);
            op(venturi_gdn_run(q,q+stride,q+2*stride,static_cast<const float*>(p("decay")),static_cast<const float*>(p("beta")),static_cast<const float*>(state[index]->ptr),p("recurrent"),static_cast<float*>(state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,s));
            op(venturi_gated_rms_run(p("recurrent"),p("z"),l.gamma,p("mixed"),t,c.gvh,c.gvd,c.dtype,l.gamma_fp32,c.epsilon,s));
            m.linear(l.matrix[2],p("mixed"),p("projected"),t);
        } else {
            auto* kt=keys[index]->at(mul(mul(pos,c.hkv*c.dim),2));auto* vt=values[index]->at(mul(mul(pos,c.hkv*c.dim),2));
            m.linear(l.matrix[0],p("normalized"),p("qg"),t);m.linear(l.matrix[1],p("normalized"),p("key"),t);m.linear(l.matrix[2],p("normalized"),vt,t);
            auto* cos=static_cast<const float*>(p("cos"))+mul(pos,c.rotary/2);auto* sin=static_cast<const float*>(p("sin"))+mul(pos,c.rotary/2);
            op(venturi_qwen_qk_run(p("qg"),p("key"),static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),cos,sin,p("query"),kt,t,c.hq,c.hkv,c.dim,c.rotary,c.dtype,c.epsilon,s));
            op(venturi_attention_run(p("query"),keys[index]->ptr,values[index]->ptr,p("attended"),t,pos+t,c.hq,c.hkv,c.dim,c.dtype,c.attention_scale,s));
            op(venturi_qwen_gate_run(p("attended"),p("qg"),p("attended"),t,c.hq,c.dim,c.dtype,s));
            m.linear(l.matrix[3],p("attended"),p("projected"),t);
        }
        op(venturi_add_run(x,p("projected"),p("residual"),count,c.dtype,s));
        op(venturi_rms_run(p("residual"),static_cast<const float*>(l.post),p("mlp_norm"),t,c.hidden,c.dtype,c.epsilon,s));
        m.linear(l.matrix[4],p("mlp_norm"),p("mlp_gate"),t);m.linear(l.matrix[5],p("mlp_norm"),p("mlp_up"),t);
        op(venturi_swiglu_run(p("mlp_gate"),p("mlp_up"),p("mlp_gate"),mul(t,c.intermediate),c.dtype,s));
        m.linear(l.matrix[6],p("mlp_gate"),p("projected"),t);op(venturi_add_run(p("residual"),p("projected"),out,count,c.dtype,s));
    }
    void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace) {
        input(tokens,n);auto& m=*model;const auto& c=m.c;const int initial=position;bool started=false;
        try {
            for (int start=0;start<n;start+=c.max_tokens) {
                const int t=std::min(c.max_tokens,n-start),pos=initial+start;void* x=p("x");void* out=p("out");
                started=true;embedding(tokens+start,t,x);
                for (int i=0;i<c.layers;++i) {
                    layer(i,pos,t,x,out);std::swap(x,out);
                    if (trace) check(cudaMemcpyAsync(static_cast<char*>(trace)+mul(add(mul(i,n),start),mul(c.hidden,2)),x,mul(mul(t,c.hidden),2),cudaMemcpyDeviceToHost,m.stream.ptr));
                    if (i==fault_layer && pos==fault_position) { fault_layer=-1;throw std::runtime_error("injected native Qwen layer failure"); }
                }
                if (all || start+t==n) {
                    op(venturi_rms_run(x,m.norm,p("normalized"),t,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
                    const int rows=all?t:1;
                    m.linear(m.head,static_cast<char*>(p("normalized"))+(all?0:mul(mul(t-1,c.hidden),2)),p("logits"),rows);
                    if (logits) {
                        const auto elements=mul(rows,c.vocab);
                        check(cudaMemcpyAsync(host_logits.data(),p("logits"),mul(elements,2),cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
                        auto* target=logits+(all?mul(start,c.vocab):0);
                        for (std::uint64_t j=0;j<elements;++j) target[j]=c.dtype==0 ? __half2float(reinterpret_cast<const __half*>(host_logits.data())[j]) : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(host_logits.data())[j]);
                    }
                }
            }
            m.sync();m.source();position=initial+n;
        } catch (...) { if (started) { failed=true;try { m.sync(); } catch (...) {} }throw; }
    }
    void reset() {
        auto& m=*model;
        try { m.sync();for (int i=0;i<m.c.layers;++i) if (state[i]) {
            if (history[i]->bytes) check(cudaMemsetAsync(history[i]->ptr,0,history[i]->bytes,m.stream.ptr));
            check(cudaMemsetAsync(state[i]->ptr,0,state[i]->bytes,m.stream.ptr)); }
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
extern "C" int venturi_qwen_generate(void* h,const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* tokens,int* count) {
    try { vr::require(h,"null native Qwen session");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;vr::Device d(m.c.device);
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
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
                if (eos_count && std::find(eos,eos+eos_count,selected[0])!=eos+eos_count) break;
            }
            m.source();return 0;
        } catch (...) { s.failed=true;throw; }
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_info(void* h,std::uint64_t* out) {
    try { vr::require(h && out,"null native Qwen info");auto& s=*static_cast<vr::Session*>(h);auto& m=*s.model;
        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(s.position),s.failed?1ULL:0ULL,m.device_bytes,s.device_bytes,m.stage_bytes+m.host_input.bytes+m.host_output.bytes,m.payload,m.uploaded,m.embedding_uploaded};
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
        const auto* src=kind==0?s.history[layer].get():kind==1?s.state[layer].get():kind==2?s.keys[layer].get():s.values[layer].get();
        vr::require(src,"native Qwen snapshot slot absent");const auto n=kind<2?src->bytes:vr::mul(vr::mul(s.position,m.c.hkv*m.c.dim),2);
        vr::require(bytes==n && (target || !n),"native Qwen snapshot byte size mismatch");if (n) check(cudaMemcpyAsync(target,src->ptr,n,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
extern "C" int venturi_qwen_debug_failure(void* h,int layer,int position) {
    try { vr::require(h,"null native Qwen debug session");auto& s=*static_cast<vr::Session*>(h);std::lock_guard<std::mutex> a(s.mutex);
        vr::require(layer>=-1 && layer<s.model->c.layers && position>=0,"invalid native Qwen debug failure");s.fault_layer=layer;s.fault_position=position;return 0;
    } catch (const std::exception& e) { error_text=e.what();return -1; }
}
