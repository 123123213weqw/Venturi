// Unified model-agnostic KV storage pool: CUDA VMM pages, prefix-tiered
// residency and an asynchronous copy engine. Entries are flat K/V byte planes
// (row width, plane capacity); no attention, layer or model semantics live
// here. Included by model runtimes after their own CUDA primitives; this
// header is self-contained inside namespace vkv.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkv {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > ~std::uint64_t(0) / a) throw std::overflow_error("byte count overflow");
    return a * b;
}
inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    const auto s = a + b;
    if (s < a) throw std::overflow_error("byte count overflow");
    return s;
}
struct Device {  // VMM/driver calls are device-scoped; restore the caller's.
    int saved = 0;
    explicit Device(int target) { check(cudaGetDevice(&saved)); if (saved != target) check(cudaSetDevice(target)); }
    ~Device() { if (saved >= 0) (void)cudaSetDevice(saved); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

struct Pool {
    struct Entry;
    struct Page {
        CUmemGenericAllocationHandle handle = 0;
        unsigned refs = 0, writers = 0;  // refs = entries mapping this page
        std::uint64_t counted = 0;      // admission-only unique-page stamp
    };
    int device; cudaStream_t compute, copy; void* ring; std::uint64_t ring_bytes, ring_slot;
    std::uint64_t reserve, gpu_limit, ram_limit, granule = 0;
    CUmemAllocationProp prop{}; CUmemAccessDesc access{}, readonly{};
    std::mutex mutex; bool poisoned = false;
    std::vector<std::unique_ptr<Page>> pool;
    std::vector<Page*> free_pages;
    std::vector<Entry*> entries;
    std::uint64_t aliased_pages = 0;
    std::uint64_t ram_bytes = 0, serial = 0, evictions = 0, reloads = 0, d2h = 0, h2d = 0,
                  ooms = 0, preflights = 0, peak_gpu = 0, peak_ram = 0, async_copies = 0;
    cudaEvent_t slot_done[2]; bool slot_used[2] = {false, false}; unsigned ring_cursor = 0;

    void driver(CUresult r) {
        if (r != CUDA_SUCCESS) { poisoned = true; const char* s = nullptr; cuGetErrorString(r, &s);
            throw std::runtime_error(std::string("KV VMM: ") + (s ? s : "driver error")); }
    }
    std::uint64_t round(std::uint64_t n) const { return n ? mul(1 + (n - 1) / granule, granule) : 0; }

    struct Entry {
        Pool& owner; CUdeviceptr base = 0; std::uint64_t plane = 0, row = 0, valid = 0, planned = 0, last = 0, host_plane = 0;
        bool active = false; std::uint64_t mapped = 0, incarnation = 0;
        cudaEvent_t ready = nullptr;  // signals the latest async reload completion on the copy stream
        std::array<std::vector<Page*>, 2> pages;
        // Per-plane per-page flag: this slot maps a SHARED read-only page.
        // The first write attempt copy-on-writes it into a private RW page.
        std::array<std::vector<bool>, 2> shared;
        // Type-erased publication lease. Releasing it never calls back into a
        // manager (and therefore cannot invert manager -> pool lock ordering).
        std::shared_ptr<const void> read_lease;
        std::unique_ptr<unsigned char[]> host;
        Entry(Pool& o, std::uint64_t bytes, std::uint64_t row_bytes) : owner(o), plane(o.round(bytes)), row(row_bytes) {
            for (auto& p : pages) p.resize(plane / o.granule, nullptr);
            for (auto& f : shared) f.assign(plane / o.granule, false);
            check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            std::lock_guard<std::mutex> lock(o.mutex);
            incarnation = ++o.serial;
            o.driver(cuMemAddressReserve(&base, mul(plane, 2), o.granule, 0, 0));
            try { o.entries.push_back(this); }
            catch (...) { cuMemAddressFree(base, mul(plane, 2)); throw; }
        }
        void* ptr(int kind = 0, std::uint64_t offset = 0) const { return reinterpret_cast<void*>(base + mul(kind, plane) + offset); }
        ~Entry() { owner.remove(*this); if (ready) (void)cudaEventDestroy(ready); }
    };

    Pool(int d, cudaStream_t compute_stream, std::uint64_t reserve_bytes,
         std::uint64_t gpu, std::uint64_t ram, std::uint64_t slot_bytes = 32u << 20) :
        device(d), compute(compute_stream), ring_slot(slot_bytes) {
        Device guard(d);
        int supported = 0;
        driver(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d));
        require(supported, "VRAM-first KV requires CUDA virtual memory management");
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = d;
        std::size_t g = 0;
        driver(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        granule = g;
        access.location = prop.location; access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        readonly.location = prop.location; readonly.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        require(granule && slot_bytes && slot_bytes % 4096 == 0, "invalid KV pool granularity/slot size");
        reserve = reserve_bytes; gpu_limit = gpu; ram_limit = ram; ring_bytes = mul(2, slot_bytes);
        check(cudaHostAlloc(&ring, ring_bytes, cudaHostAllocPortable));
        try {
            for (auto& e : slot_done) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        } catch (...) { cudaFreeHost(ring); throw; }
        check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    }
    ~Pool() {
        Device guard(device);
        (void)cudaStreamSynchronize(copy);
        (void)cudaStreamDestroy(copy);
        for (auto& e : slot_done) (void)cudaEventDestroy(e);
        if (ring) (void)cudaFreeHost(ring);
        for (const auto& p : pool) if (p->handle) (void)cuMemRelease(p->handle);
    }
    std::unique_ptr<Entry> create(std::uint64_t bytes, std::uint64_t row) { return std::make_unique<Entry>(*this, bytes, row); }
    std::uint64_t physical() const { return mul(pool.size(), granule); }

    // One ring slot at a time: the host waits for a slot's previous transfer,
    // then issues the next. Uploads (reloads) stay asynchronous on the copy
    // stream — completion reaches the compute stream via the entry's ready
    // event, never by blocking the host. Downloads (spill) complete each
    // chunk before touching the host backing, on the copy stream only.
    void chunk(void* gpu, void* cpu, std::uint64_t n, bool upload) {
        for (std::uint64_t start = 0; start < n;) {
            const auto count = std::min(ring_slot, n - start);
            const unsigned index = ring_cursor++ & 1u;
            if (slot_used[index]) check(cudaEventSynchronize(slot_done[index]));
            slot_used[index] = true;
            auto* slot = static_cast<unsigned char*>(ring) + mul(index, ring_slot);
            if (upload) std::memcpy(slot, static_cast<unsigned char*>(cpu) + start, count);
            check(cudaMemcpyAsync(upload ? static_cast<unsigned char*>(gpu) + start : slot,
                                  upload ? slot : static_cast<unsigned char*>(gpu) + start,
                                  count, upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, copy));
            check(cudaEventRecord(slot_done[index], copy));
            if (!upload) {
                check(cudaEventSynchronize(slot_done[index]));
                std::memcpy(static_cast<unsigned char*>(cpu) + start, slot, count);
            }
            start += count;
        }
        ++async_copies;
        if (upload) h2d = add(h2d, n); else d2h = add(d2h, n);
    }

    void unmap(Entry& e) {
        for (int k = 0; k < 2; ++k)
            for (std::uint64_t p = 0; p < e.pages[k].size(); ++p)
                if (e.pages[k][p]) {
                    driver(cuMemUnmap(e.base + mul(k, e.plane) + mul(p, granule), granule));
                    release(*e.pages[k][p]); e.pages[k][p] = nullptr;
                    e.shared[k][p] = false;
                }
        e.mapped = 0;
    }
    void release(Page& page) {
        require(page.refs > 0, "KV page reference underflow");
        if (page.refs == 2) --aliased_pages;
        if (--page.refs == 0) free_pages.push_back(&page);
    }

    void evict(Entry& e) {
        require(e.mapped > 0, "KV victim not resident");
        check(cudaStreamSynchronize(compute));
        if (e.valid) {
            require(e.host && e.host_plane >= e.valid, "KV spill backing was not admitted before forward");
            for (int k = 0; k < 2; ++k)
                chunk(e.ptr(k), e.host.get() + mul(k, e.host_plane), e.valid, false);
            check(cudaStreamSynchronize(copy));  // host backing is complete before unmapping
        }
        unmap(e); ++evictions;
    }

    // Admission occurs before embedding/GDN mutation. No implicit forward retry.
    void prepare(const std::vector<std::pair<Entry*, std::uint64_t>>& goals) {
        std::lock_guard<std::mutex> lock(mutex);
        require(!poisoned, "KV VMM driver failure: close and recreate model; do not retry recurrent forward");
        ++preflights;
        for (auto* e : entries) { e->planned = e->valid; e->active = false; }
        for (const auto& g : goals) {
            require(g.first && std::find(entries.begin(),entries.end(),g.first)!=entries.end() &&
                    g.second <= g.first->plane, "KV planned prefix exceeds virtual capacity/ownership");
            g.first->planned = std::max(g.first->planned, g.second); g.first->active = true;
        }
        std::uint64_t largest = 0;
        for (const auto* e : entries) largest = std::max(largest, mul(round(e->planned) / granule, 2));
        std::size_t free = 0, total = 0; check(cudaMemGetInfo(&free, &total));
        const auto available = add(physical(), free > reserve ? free - reserve : 0);
        const auto budget = (gpu_limit ? std::min(gpu_limit, available) : available) / granule;
        require(largest <= budget, "VRAM-first KV budget cannot fit one request/layer prefix; reduce context or batch/offload scope");
        bool trim = false;
        for (auto* e : entries) if (e->mapped > round(e->planned) / granule) trim = true;
        if (trim) {
            check(cudaStreamSynchronize(compute)); check(cudaStreamSynchronize(copy));
            for (auto* e : entries) {
                const auto n = round(e->planned) / granule;
                for (int k = 0; k < 2; ++k)
                    for (auto p = n; p < e->mapped; ++p) {
                        driver(cuMemUnmap(e->base + mul(k, e->plane) + mul(p, granule), granule));
                        release(*e->pages[k][p]); e->pages[k][p] = nullptr;
                        e->shared[k][p] = false;
                    }
                e->mapped = std::min(e->mapped, n);
            }
        }
        // Count physical aliases once, plus unmapped growth/reload pages and
        // private copies required by the active APPEND write sets. If every
        // alias writes, the last writer can reuse the original physical page.
        // Page stamps avoid a per-token hash table/full-context allocation.
        const auto stamp = ++serial;
        std::uint64_t needed = 0;
        if (!aliased_pages) {
            // Preserve the ordinary non-sharing O(entries) admission path.
            for (const auto* e : entries) needed = add(needed,mul(round(e->planned)/granule,2));
        } else {
          for (const auto* e : entries) {
            const auto n = round(e->planned) / granule;
            for (int k = 0; k < 2; ++k) for (std::uint64_t p = 0; p < n; ++p) {
                auto* page = e->pages[k][p];
                if (!page) { needed = add(needed, 1); continue; }
                if (page->counted != stamp) {
                    page->counted = stamp; page->writers = 0; needed = add(needed, 1);
                }
            }
          }
          for (const auto* e : entries) if (e->active && e->planned > e->valid) {
            const auto first = e->valid / granule, end = std::min(e->mapped, round(e->planned) / granule);
            for (int k = 0; k < 2; ++k) for (auto p = first; p < end; ++p)
                if (auto* page = e->pages[k][p]) ++page->writers;
          }
          for (const auto& page : pool) if (page->counted == stamp && page->writers)
            needed = add(needed, std::min(page->writers, page->refs - 1));
        }
        const auto target = std::min(needed, budget);
        pool.reserve(std::max<std::uint64_t>(pool.size(), target));
        free_pages.reserve(std::max<std::uint64_t>(pool.size(), target));
        while (pool.size() < target) {
            auto p = std::make_unique<Page>();
            const auto r = cuMemCreate(&p->handle, granule, &prop, 0);
            if (r == CUDA_ERROR_OUT_OF_MEMORY) { ++ooms; break; }
            driver(r); free_pages.push_back(p.get()); pool.push_back(std::move(p));
        }
        peak_gpu = std::max(peak_gpu, physical());
        require(pool.size() >= largest, "KV physical allocation OOM before forward; state unchanged");
        if (needed > pool.size()) {
            std::uint64_t final = ram_bytes;
            for (const auto* e : entries) { const auto n = round(e->planned); if (n > e->host_plane) final = add(final, mul(n - e->host_plane, 2)); }
            require(final <= ram_limit, "VRAM-first KV RAM budget exhausted before forward; state unchanged");
            for (auto* e : entries) {
                const auto n = round(e->planned);
                if (n <= e->host_plane) continue;
                require(add(ram_bytes, mul(n, 2)) <= ram_limit, "KV RAM growth peak exceeds budget before forward");
                std::unique_ptr<unsigned char[]> h(new unsigned char[mul(n, 2)]);
                if (e->host && !e->mapped && e->valid)
                    for (int k = 0; k < 2; ++k) std::memcpy(h.get() + mul(k, n), e->host.get() + mul(k, e->host_plane), e->valid);
                ram_bytes = add(ram_bytes, mul(n - e->host_plane, 2)); e->host = std::move(h); e->host_plane = n;
                peak_ram = std::max(peak_ram, ram_bytes);
            }
        }
    }

    // Ensures [0, bytes) of an entry is resident and ISSUES any reload DMA on
    // the copy stream without blocking the host or the compute stream. Callers
    // chain the compute stream onto the completion with attach() right before
    // their first read/write of the entry — issuing ensure() for a LATER
    // layer between two kernels of the current layer cannot serialize them.
    // Replace one shared read-only page with a private read-write copy of
    // the same bytes. Runs before any write may touch the page, so the old
    // mapping is still readable for the copy.
    void copy_on_write(Entry& e, int k, std::uint64_t p) {
        auto* original = e.pages[k][p];
        require(original && original->refs, "KV COW page absent");
        while (free_pages.empty() && original->refs > 1) {
            Entry* victim = nullptr;
            for (auto* other : entries)
                if (other != &e && other->mapped && (!victim || (other->active != victim->active ? !other->active : other->last < victim->last))) victim = other;
            require(victim, "KV copy-on-write found no eviction victim");
            evict(*victim);
        }
        const auto addr = e.base + mul(k, e.plane) + mul(p, granule);
        // Evicting a donor may leave this as the last alias. No extra page or
        // copy is needed, including when the admitted pool has ZERO headroom.
        if (original->refs == 1) {
            if (e.shared[k][p]) driver(cuMemSetAccess(addr, granule, &access, 1));
            e.shared[k][p] = false;
            return;
        }
        std::vector<unsigned char> image(granule);
        check(cudaStreamSynchronize(copy));
        check(cudaMemcpy(image.data(), reinterpret_cast<void*>(addr), granule, cudaMemcpyDeviceToHost));
        d2h = add(d2h, granule);
        driver(cuMemUnmap(addr, granule));
        release(*e.pages[k][p]);
        e.pages[k][p] = nullptr;
        auto* page = free_pages.back(); free_pages.pop_back(); ++page->refs;
        e.pages[k][p] = page;
        driver(cuMemMap(addr, granule, 0, page->handle, 0));
        driver(cuMemSetAccess(addr, granule, &access, 1));  // read-write
        check(cudaMemcpy(reinterpret_cast<void*>(addr), image.data(), granule, cudaMemcpyHostToDevice));
        h2d = add(h2d, granule);
        e.shared[k][p] = false;
    }

    void ensure(Entry& e, std::uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        require(!poisoned, "KV VMM driver failure: recreate model");
        const auto n = round(bytes) / granule; e.last = ++serial;
        require(bytes <= e.plane && n <= pool.size() / 2, "KV prefix not admitted");
        // ensure is an append/read-prefix operation. Only [valid, bytes) is
        // writable: untouched complete prefix pages must remain shared. Check
        // physical refs, NOT just borrower RO flags: donors can be aliases too.
        if (bytes > e.valid) for (int k = 0; k < 2; ++k)
            for (auto p = e.valid / granule; p < std::min(n, e.mapped); ++p)
                if (e.shared[k][p] || e.pages[k][p]->refs > 1) copy_on_write(e, k, p);
        if (e.mapped >= n) return;
        // Growth only maps NEW pages (no prior readers); eviction synchronizes
        // the compute stream itself before touching a victim's mappings.
        const bool reload = !e.mapped && e.valid;
        const auto extra = mul(n - e.mapped, 2);
        while (free_pages.size() < extra) {
            Entry* victim = nullptr;
            for (auto* other : entries)
                if (other != &e && other->mapped && (!victim || (other->active != victim->active ? !other->active : other->last < victim->last))) victim = other;
            require(victim, "KV residency admission invariant failed"); evict(*victim);
        }
        for (int k = 0; k < 2; ++k)
            for (auto p = e.mapped; p < n; ++p) {
                auto* page = free_pages.back();
                const auto addr = e.base + mul(k, e.plane) + mul(p, granule);
                driver(cuMemMap(addr, granule, 0, page->handle, 0));
                free_pages.pop_back(); ++page->refs; e.pages[k][p] = page;  // owner-tracked even if access setup fails
                e.shared[k][p] = false;
                driver(cuMemSetAccess(addr, granule, &access, 1));
            }
        e.mapped = n;
        if (reload) {
            require(e.host && e.host_plane >= e.valid, "KV reload backing absent");
            for (int k = 0; k < 2; ++k) chunk(e.ptr(k), e.host.get() + mul(k, e.host_plane), e.valid, true);
            check(cudaEventRecord(e.ready, copy));
            ++reloads;
        }
    }
    // Order the compute stream after the entry's latest reload. Waiting on a
    // never-recorded event is a no-op, so attach() is safe for fresh entries.
    void attach(Entry& e) { check(cudaStreamWaitEvent(compute, e.ready)); }
    void written(Entry& e, std::uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        require(bytes <= mul(e.mapped, granule), "KV write outside committed prefix"); e.valid = bytes;
    }
    void snapshot(Entry& e, int k, void* target, std::uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        require(!poisoned, "KV VMM driver failure: recreate model");
        require(bytes <= e.valid, "KV snapshot exceeds valid prefix");
        check(cudaStreamSynchronize(compute)); check(cudaStreamSynchronize(copy));
        if (bytes) {
            if (e.mapped) { check(cudaMemcpyAsync(target, e.ptr(k), bytes, cudaMemcpyDeviceToHost, compute)); check(cudaStreamSynchronize(compute)); }
            else { require(e.host && bytes <= e.host_plane, "KV snapshot backing absent"); std::memcpy(target, e.host.get() + mul(k, e.host_plane), bytes); }
        }
    }
    void reset(Entry& e) {
        std::lock_guard<std::mutex> lock(mutex);
        require(!poisoned, "KV VMM driver failure: recreate model");
        check(cudaStreamSynchronize(compute)); check(cudaStreamSynchronize(copy));
        // Detach aliases/RO mappings before request reuse. Retain ordinary
        // private RW mappings, preserving the no-sharing reset fast path.
        bool detach = false;
        for (int k = 0; k < 2; ++k) for (std::uint64_t p = 0; p < e.mapped; ++p)
            if (e.shared[k][p] || e.pages[k][p]->refs > 1) detach = true;
        if (detach) unmap(e);
        e.read_lease.reset(); e.incarnation = ++serial;
        e.valid = e.planned = 0; e.host.reset(); ram_bytes -= mul(e.host_plane, 2); e.host_plane = 0; e.active = false;
    }
    void remove(Entry& e) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            entries.erase(std::remove(entries.begin(), entries.end(), &e), entries.end());
            ram_bytes -= mul(e.host_plane, 2);
            Device d(device);
            check(cudaStreamSynchronize(compute)); check(cudaStreamSynchronize(copy));
            unmap(e);
            e.read_lease.reset();
            driver(cuMemAddressFree(e.base, mul(e.plane, 2)));
        } catch (...) { /* CUDA teardown failure must not throw from a destructor. */ }
    }
    std::array<std::uint64_t, 18> info() {
        std::lock_guard<std::mutex> lock(mutex);
        std::uint64_t mapped = 0, virt = 0, valid = 0;
        for (const auto* e : entries) { mapped = add(mapped, mul(mul(e->mapped, granule), 2)); virt = add(virt, mul(e->plane, 2)); valid = add(valid, mul(e->valid, 2)); }
        return {1, granule, physical(), mapped, virt, ram_bytes, valid, evictions, reloads, d2h, h2d, ooms, preflights, peak_gpu, peak_ram, gpu_limit, ram_limit, entries.size()};
    }
};

}  // namespace vkv
