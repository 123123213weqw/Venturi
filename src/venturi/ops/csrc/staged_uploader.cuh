// Venturi staged host-to-device uploader for mapped weight bytes.
// Included inside namespace vr by qwen_runtime.cuh, after Pinned/Stream/Device
// and weight_source.cuh. Owns the single bounded pinned staging buffer and the
// serial chunked copy loops (upload/upload_other); algorithms are unchanged
// from the former Model members. Model keeps facade references so the
// head-GDN template, pipeline accounting and Session see identical names.
#pragma once

struct StagedUploader {
    WeightSource* source=nullptr;
    Stream* primary=nullptr;
    Stream* secondary=nullptr;
    int second_device=-1;
    Pinned staging;
    std::uint64_t stage_bytes=8*1024*1024;
    std::uint64_t uploaded=0;

    // Attach happens where the former constructor allocated staging: after the
    // streams exist, in the same order, so allocation failures surface identically.
    void attach(WeightSource& s,Stream& first,Stream& second,int device) {
        source=&s;primary=&first;secondary=&second;second_device=device;
        staging.allocate(stage_bytes);
    }
    void read_bytes(void* target,std::uint64_t offset,std::uint64_t bytes) const {
        source->read_bytes(target,offset,bytes);
    }
    void sync() {
        check(cudaStreamSynchronize(primary->ptr));
        if (secondary && secondary->ptr) { Device d(second_device);check(cudaStreamSynchronize(secondary->ptr)); }
    }
    void upload(void* target,std::uint64_t offset,std::uint64_t bytes) {
        for (std::uint64_t start=0;start<bytes;) {
            const auto n=std::min(stage_bytes,bytes-start);
            read_bytes(staging.ptr,offset+start,n);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging.ptr,n,cudaMemcpyHostToDevice,primary->ptr));
            sync();start+=n;
        }
        uploaded=add(uploaded,bytes);
    }
    void upload_other(void* target,std::uint64_t offset,std::uint64_t bytes) {
        Device d(second_device);
        for (std::uint64_t start=0;start<bytes;) {
            const auto n=std::min(stage_bytes,bytes-start);read_bytes(staging.ptr,offset+start,n);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging.ptr,n,cudaMemcpyHostToDevice,secondary->ptr));
            check(cudaStreamSynchronize(secondary->ptr));start+=n;
        }
        uploaded=add(uploaded,bytes);
    }
};
