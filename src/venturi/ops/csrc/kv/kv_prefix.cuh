// Model-agnostic prefix orchestration: immutable auxiliary snapshots, borrowed
// page mappings and bounded LRU metadata. Include after kv_pool.cuh. Lock order
// is manager -> pool; entry lease release NEVER calls back into the manager.
#pragma once

namespace vkv {

struct PrefixState { const void* src = nullptr; std::uint64_t bytes = 0; };

struct PrefixSnapshot {
    int device;
    void* data = nullptr;
    std::uint64_t bytes;
    PrefixSnapshot(int d, std::uint64_t n) : device(d), bytes(n) {
        if (n) check(cudaMalloc(&data, n));
    }
    ~PrefixSnapshot() noexcept {
        if (data) try { Device guard(device); (void)cudaFree(data); } catch (...) {}
    }
    PrefixSnapshot(const PrefixSnapshot&) = delete;
    PrefixSnapshot& operator=(const PrefixSnapshot&) = delete;
    PrefixSnapshot(PrefixSnapshot&& other) noexcept :
        device(other.device), data(std::exchange(other.data, nullptr)), bytes(other.bytes) {}
    PrefixSnapshot& operator=(PrefixSnapshot&& other) noexcept {
        std::swap(device, other.device); std::swap(data, other.data); std::swap(bytes, other.bytes);
        return *this;
    }
};

struct PrefixRecord {
    std::uint64_t key = 0, bytes = 0, serial = 0;
    std::vector<Pool::Entry*> entries;
    std::vector<std::uint64_t> incarnations;
    std::vector<PrefixSnapshot> snapshots;
    // One reference in the record, additional references in live borrowed
    // entries. Reset/remove releases these automatically, without lock callbacks.
    std::shared_ptr<const void> lease;
    bool borrowed() const { return lease.use_count() > 1; }
};

struct PrefixManager {
    int device;
    std::mutex mutex;
    std::vector<PrefixRecord> records;
    std::uint64_t serial = 0;
    std::size_t capacity = 64;

    explicit PrefixManager(int d) : device(d) {}
    PrefixManager(const PrefixManager&) = delete;
    PrefixManager& operator=(const PrefixManager&) = delete;

    static bool owned(const Pool& pool, const Pool::Entry* entry) {
        return std::find(pool.entries.begin(), pool.entries.end(), entry) != pool.entries.end();
    }
    static bool healthy(const Pool& pool, const PrefixRecord& record) {
        const auto pages = pool.round(record.bytes) / pool.granule;
        if (!pages) return false;
        for (std::size_t i = 0; i < record.entries.size(); ++i) {
            const auto* entry = record.entries[i];
            // Ownership BEFORE dereference; incarnation rejects both reset and
            // a new entry allocated at an old donor's address (pointer ABA).
            if (!owned(pool, entry) || entry->incarnation != record.incarnations[i] ||
                entry->mapped < pages || entry->valid < record.bytes) return false;
        }
        return true;
    }
    static void unique_entries(const std::vector<Pool::Entry*>& entries) {
        for (std::size_t i = 0; i < entries.size(); ++i)
            require(std::find(entries.begin(), entries.begin() + i, entries[i]) == entries.begin() + i,
                    "prefix entries must be unique");
    }

    // Caller schedules publication only at a committed model-state boundary.
    // A capacity/validation/allocation failure does not discard the old record
    // or leak the new snapshots; ownership stays RAII until atomic replacement.
    void publish(Pool& pool, std::uint64_t key, const std::vector<Pool::Entry*>& entries,
                 std::uint64_t bytes, const std::vector<PrefixState>& extra) {
        Device guard(device);
        std::lock_guard<std::mutex> lock(mutex);
        std::lock_guard<std::mutex> pool_lock(pool.mutex);
        require(device == pool.device && !pool.poisoned, "prefix pool device/health mismatch");
        require(bytes > 0 && !entries.empty() && capacity > 0, "prefix requires length, entries and capacity");
        unique_entries(entries);
        const auto pages = pool.round(bytes) / pool.granule;
        for (const auto* entry : entries)
            require(owned(pool, entry) && entry->mapped >= pages && entry->valid >= bytes,
                    "prefix donor entries must be resident over the published length");
        for (const auto& state : extra)
            require((state.src != nullptr) == (state.bytes > 0), "prefix state requires both source and bytes");

        std::size_t slot = records.size();
        for (std::size_t i = 0; i < records.size(); ++i) if (records[i].key == key) { slot = i; break; }
        if (slot == records.size() && records.size() >= capacity) {
            for (std::size_t i = 0; i < records.size(); ++i)
                if (!records[i].borrowed() && (slot == records.size() || records[i].serial < records[slot].serial)) slot = i;
            require(slot != records.size(), "prefix registry capacity exhausted by active borrowers");
        }
        PrefixRecord next;
        next.key = key; next.bytes = bytes; next.serial = serial + 1;
        next.entries = entries; next.lease = std::make_shared<const int>(0);
        for (const auto* entry : entries) next.incarnations.push_back(entry->incarnation);
        next.snapshots.reserve(extra.size());
        for (const auto& state : extra) {
            next.snapshots.emplace_back(device, state.bytes);
            if (state.bytes) {
                check(cudaMemcpyAsync(next.snapshots.back().data, state.src, state.bytes, cudaMemcpyDeviceToDevice, 0));
                check(cudaStreamSynchronize(0));
            }
        }
        if (slot == records.size()) records.push_back(std::move(next));
        else records[slot] = std::move(next);
        ++serial;
    }

    void map_shared(Pool& pool, Pool::Entry& into, const Pool::Entry& donor, std::uint64_t bytes) {
        const auto pages = pool.round(bytes) / pool.granule;
        for (int k = 0; k < 2; ++k) for (std::uint64_t p = 0; p < pages; ++p) {
            const auto addr = into.base + mul(k, into.plane) + mul(p, pool.granule);
            pool.driver(cuMemMap(addr, pool.granule, 0, donor.pages[k][p]->handle, 0));
            if (donor.pages[k][p]->refs++ == 1) ++pool.aliased_pages;
            into.pages[k][p] = donor.pages[k][p]; into.shared[k][p] = true;
            pool.driver(cuMemSetAccess(addr, pool.granule, &pool.readonly, 1));
        }
        into.mapped = pages; into.valid = bytes; into.planned = bytes;
    }

    // Length/layout/freshness/destination validation precedes ANY mapping or
    // state copy. The model passes expected byte length, never trusts key alone.
    bool borrow(Pool& pool, std::uint64_t key, const std::vector<Pool::Entry*>& entries,
                const std::vector<void*>& extra_dests, std::uint64_t expected_bytes) {
        Device guard(device);
        std::lock_guard<std::mutex> lock(mutex);
        std::lock_guard<std::mutex> pool_lock(pool.mutex);
        require(device == pool.device && !pool.poisoned, "prefix pool device/health mismatch");
        require(expected_bytes > 0, "prefix borrower expected length must be positive");
        unique_entries(entries);
        for (auto& record : records) {
            if (record.key != key || !healthy(pool, record)) continue;
            require(record.bytes == expected_bytes, "prefix length differs from the published record");
            require(entries.size() == record.entries.size(), "prefix borrower entry count differs from the record");
            require(extra_dests.size() == record.snapshots.size(), "prefix state destination count differs from the record");
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const auto* entry = entries[i];
                require(owned(pool, entry) && !entry->mapped && !entry->valid && !entry->read_lease,
                        "prefix borrower entries must be fresh");
                require(entry->plane >= pool.round(record.bytes) && entry->row == record.entries[i]->row,
                        "prefix borrower capacity/layout differs from the record");
            }
            for (std::size_t i = 0; i < record.snapshots.size(); ++i)
                require(!record.snapshots[i].bytes || extra_dests[i], "prefix state destination missing");
            for (std::size_t i = 0; i < entries.size(); ++i)
                map_shared(pool, *entries[i], *record.entries[i], record.bytes);
            for (std::size_t i = 0; i < record.snapshots.size(); ++i) if (record.snapshots[i].bytes)
                check(cudaMemcpyAsync(extra_dests[i], record.snapshots[i].data, record.snapshots[i].bytes,
                                      cudaMemcpyDeviceToDevice, 0));
            check(cudaStreamSynchronize(0));
            for (auto* entry : entries) entry->read_lease = record.lease;
            record.serial = ++serial;
            return true;
        }
        return false;
    }
    void forget(std::uint64_t key) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = records.begin(); it != records.end(); ++it) if (it->key == key) {
            records.erase(it); return;
        }
    }
};

}  // namespace vkv
