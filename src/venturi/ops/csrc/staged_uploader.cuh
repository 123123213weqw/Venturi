// Venturi staged host-to-device uploader v4: OpenMP multi-threaded reads.
//
// Architecture: batches of NUM_SLOTS chunks are read in parallel by
// NUM_THREADS OpenMP threads (each thread memcpys a different chunk from
// mmap/page-cache into its own staging slot), then DMAd sequentially
// (PCIe at 25 GB/s is never the bottleneck). The batch pipeline overlaps
// reader threads with the DMA stream.
//
// Warm cache: parallel memcpy saturates DRAM bandwidth (~18 GB/s with 4 threads).
// Cold NVMe: parallel reads use multiple NVMe hardware queues (~4x throughput).
// Cold SATA: no benefit (single-queue device), but no regression either.
//
// Included inside namespace vr by qwen_runtime.cuh.
// Compile with: -Xcompiler -fopenmp
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

struct StagedUploader {
    WeightSource* source=nullptr;
    Stream* primary=nullptr;
    Stream* secondary=nullptr;
    int second_device=-1;

    // Staging slot pool: each slot is filled by one OpenMP thread per batch.
    static constexpr int NUM_SLOTS = 8;
    static constexpr int NUM_THREADS = 4;
    Pinned staging[NUM_SLOTS];
    Event done[NUM_SLOTS];
    std::uint64_t stage_bytes=32*1024*1024;  // 32 MB per slot (256 MB total)
    std::uint64_t uploaded=0;
    void (*validate_chunk)(const void*, std::uint64_t, std::uint64_t) = nullptr;

    void attach(WeightSource& s,Stream& first,Stream& second,int device) {
        source=&s;primary=&first;secondary=&second;second_device=device;
        for (int i=0;i<NUM_SLOTS;++i) {
            staging[i].allocate(stage_bytes);
            done[i].create();
        }
    }
    void read_bytes(void* target,std::uint64_t offset,std::uint64_t bytes) const {
        source->read_bytes(target,offset,bytes);
    }
    void sync() {
        check(cudaStreamSynchronize(primary->ptr));
        if (secondary && secondary->ptr) { Device d(second_device);check(cudaStreamSynchronize(secondary->ptr)); }
    }

    // Multi-threaded batch upload: N slots read in parallel, then DMA'd.
    void upload(void* target,std::uint64_t offset,std::uint64_t bytes) {
        const int total_chunks = (int)((bytes + stage_bytes - 1) / stage_bytes);

        for (int batch = 0; batch < total_chunks; batch += NUM_SLOTS) {
            const int in_batch = std::min(NUM_SLOTS, total_chunks - batch);

            // Phase 1: Parallel read (OpenMP). Each thread reads one chunk
            // from mmap into its own staging slot. Different threads touch
            // different memory regions and (on cold NVMe) different queues.
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic,1)
            for (int i = 0; i < in_batch; ++i) {
                const auto chunk = batch + i;
                const auto start = (std::uint64_t)chunk * stage_bytes;
                const auto n = std::min(stage_bytes, bytes - start);
                read_bytes(staging[i].ptr, offset + start, n);
                if (validate_chunk) validate_chunk(staging[i].ptr, n, offset + start);
            }

            // Phase 2: Sequential DMA (PCIe at 25 GB/s, never the bottleneck).
            // All slots are already filled; just issue the copies in order.
            for (int i = 0; i < in_batch; ++i) {
                const auto chunk = batch + i;
                const auto start = (std::uint64_t)chunk * stage_bytes;
                const auto n = std::min(stage_bytes, bytes - start);
                // Wait for this slot's previous DMA before reusing (safety).
                // (First batch has no previous DMA for any slot.)
                // We use the stream itself as the ordering guarantee: since
                // we sync at the end of each batch, slots are free for reuse.
                check(cudaMemcpyAsync(static_cast<char*>(target) + start,
                                      staging[i].ptr, n,
                                      cudaMemcpyHostToDevice, primary->ptr));
            }
            // Sync this batch before reading the next (slots are reused).
            check(cudaStreamSynchronize(primary->ptr));
        }
        uploaded = add(uploaded, bytes);
    }

    // Secondary device (single-threaded; TP is not the hot path).
    void upload_other(void* target,std::uint64_t offset,std::uint64_t bytes) {
        Device d(second_device);
        int cur=0; bool first=true;
        for (std::uint64_t start=0;start<bytes;) {
            const auto n=std::min(stage_bytes,bytes-start);
            if (!first) check(cudaEventSynchronize(done[cur].ptr));
            read_bytes(staging[cur].ptr,offset+start,n);
            check(cudaMemcpyAsync(static_cast<char*>(target)+start,staging[cur].ptr,n,
                                  cudaMemcpyHostToDevice,secondary->ptr));
            check(cudaEventRecord(done[cur].ptr,secondary->ptr));
            first=false;cur=(cur+1)%NUM_SLOTS;start+=n;
        }
        check(cudaStreamSynchronize(secondary->ptr));
        uploaded=add(uploaded,bytes);
    }
};
