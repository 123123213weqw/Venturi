// Venturi weight byte source: file mapping and bounded logical reads.
// Pure byte-range provider over explicit safetensors data spans; no CUDA, no model semantics.
// Included inside namespace vr by qwen_runtime.cuh, after the overflow/require
// helpers and the VRSpan ABI struct; the enclosing TU (blocks.cu via
// qwen_runtime.cuh) supplies the POSIX/STL headers and require/add helpers.
// Read-only mappings live for the model's lifetime and are re-verified
// (source()) before every consumer use.
#pragma once

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

struct WeightSource {
    std::vector<std::unique_ptr<Mapping>> files;
    std::uint64_t payload=0;

    void open(const VRSpan* spans,int count) {
        for (int i=0;i<count;++i) {
            require(spans[i].logical==payload,"noncontiguous native Qwen file spans");
            files.push_back(std::make_unique<Mapping>(spans[i]));
            payload=add(payload,spans[i].bytes);
        }
    }
    void read_bytes(void* target,std::uint64_t offset,std::uint64_t bytes) const {
        require(offset<=payload && bytes<=payload-offset,"native Qwen read exceeds mapped payload");
        auto* out=static_cast<char*>(target);
        for (const auto& f:files) {
            const auto first=std::max(offset,f->logical),last=std::min(add(offset,bytes),add(f->logical,f->bytes));
            if (first<last) std::memcpy(out+first-offset,static_cast<const char*>(f->ptr)+f->begin+first-f->logical,last-first);
        }
    }
    void source() const { for (const auto& f:files) f->verify(); }
};
