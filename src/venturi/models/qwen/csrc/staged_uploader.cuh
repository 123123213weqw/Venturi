// Bounded synchronous host-to-device transport; no model or file-format policy.
// Included inside namespace vr after WeightSource and CUDA resource primitives.
#pragma once

struct StagedUploader {
    WeightSource* source=nullptr;
    Stream* primary=nullptr;
    Stream* secondary=nullptr;
    int second_device=-1;
    Pinned staging;
    std::uint64_t stage_bytes=32*1024*1024;
    std::uint64_t uploaded=0;

    void attach(WeightSource& s,Stream& first,Stream& second,int device) {
        source=&s;primary=&first;secondary=&second;second_device=device;
        staging.allocate(stage_bytes);
    }
    void read_bytes(void* target,std::uint64_t offset,std::uint64_t bytes) const {
        source->read_bytes(target,offset,bytes);
    }
    void sync() {
        check(cudaStreamSynchronize(primary->ptr));
        if (secondary && secondary->ptr) {
            Device device(second_device);
            check(cudaStreamSynchronize(secondary->ptr));
        }
    }
    void transfer(void* target,std::uint64_t offset,std::uint64_t bytes,cudaStream_t stream) {
        source->source();
        for (std::uint64_t start=0;start<bytes;) {
            const auto count=std::min(stage_bytes,bytes-start);
            source->read_bytes(staging.ptr,add(offset,start),count);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging.ptr,count,
                                  cudaMemcpyHostToDevice,stream));
            // No staging overwrite while DMA is reading the same buffer.
            check(cudaStreamSynchronize(stream));
            start+=count;
        }
        source->source();
        uploaded=add(uploaded,bytes);
    }
    void upload(void* target,std::uint64_t offset,std::uint64_t bytes) {
        transfer(target,offset,bytes,primary->ptr);
    }
    void upload_other(void* target,std::uint64_t offset,std::uint64_t bytes) {
        Device device(second_device);
        transfer(target,offset,bytes,secondary->ptr);
    }
};
