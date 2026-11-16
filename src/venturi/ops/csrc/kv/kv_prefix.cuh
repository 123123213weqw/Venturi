// Prefix sharing over vkv::Pool, orchestrated entirely inside the unified KV
// layer: publishing registers a donor's KV entries and freezes caller-supplied
// auxiliary recurrent state into registry-owned device snapshots; borrowing
// maps the shared physical pages read-only into fresh entries and restores the
// auxiliary state. Models only decide WHEN to publish/borrow and WHICH
// entries/state participate — no model semantics live here. Include after
// kv_pool.cuh.
#pragma once

namespace vkv {

// Caller-described auxiliary state to snapshot alongside a prefix (e.g. a
// hybrid model's recurrent per-layer state at the prefix boundary).
struct PrefixState {
    const void* src = nullptr;
    std::uint64_t bytes = 0;
};

struct PrefixRecord {
    std::uint64_t key = 0, bytes = 0, serial = 0;
    std::vector<Pool::Entry*> entries;  // donor entries; validated against pool ownership on use
    std::vector<void*> snapshots;       // device buffers owned by this record
    std::vector<std::uint64_t> snapshot_bytes;
    unsigned borrowers = 0;             // active borrows; records with borrowers are never evicted
};

struct PrefixManager {
    int device;
    std::mutex mutex;
    std::vector<PrefixRecord> records;
    std::uint64_t serial = 0;
    std::size_t capacity = 64;

    explicit PrefixManager(int device) : device(device) {}

    PrefixManager(const PrefixManager&) = delete;
    PrefixManager& operator=(const PrefixManager&) = delete;

    ~PrefixManager() {
        Device guard(device);
        for (auto& record : records)
            for (auto* snapshot : record.snapshots)
                if (snapshot) (void)cudaFree(snapshot);
    }

    static bool owned(const Pool& pool, const Pool::Entry* entry) {
        return std::find(pool.entries.begin(), pool.entries.end(), entry) != pool.entries.end();
    }

    // A record stays usable only while every donor entry still exists, still
    // covers the published prefix, and still holds those exact pages resident.
    static bool healthy(const Pool& pool, const PrefixRecord& record) {
        const auto pages = pool.round(record.bytes) / pool.granule;
        if (!pages) return false;
        for (const auto* entry : record.entries)
            if (!owned(pool, entry) || entry->mapped < pages || entry->valid < record.bytes) return false;
        return true;
    }

    void release_snapshots(PrefixRecord& record) {
        Device guard(device);
        for (auto* snapshot : record.snapshots)
            if (snapshot) (void)cudaFree(snapshot);
        record.snapshots.clear();
        record.snapshot_bytes.clear();
    }

    // Map one donor's shared pages read-only into a fresh borrower entry and
    // advance the borrower's committed prefix. No pool lock: callers hold the
    // manager lock; the pool's own locking rules are respected by construction
    // (borrow runs between forwards, like ensure/prepare).
    void map_shared(Pool& pool, Pool::Entry& into, const Pool::Entry& donor, std::uint64_t bytes) {
        const auto pages = pool.round(bytes) / pool.granule;
        for (int k = 0; k < 2; ++k)
            for (std::uint64_t p = 0; p < pages; ++p) {
                const auto addr = into.base + mul(k, into.plane) + mul(p, pool.granule);
                pool.driver(cuMemMap(addr, pool.granule, 0, donor.pages[k][p]->handle, 0));
                ++donor.pages[k][p]->refs;  // owner-tracked before access setup
                into.pages[k][p] = donor.pages[k][p];
                into.shared[k][p] = true;
                pool.driver(cuMemSetAccess(addr, pool.granule, &pool.readonly, 1));
            }
        into.mapped = pages;
        into.valid = bytes;
        into.planned = std::max(into.planned, bytes);
    }

    // Register a donor prefix under `key`. Every entry must be pool-owned,
    // resident and valid over `bytes`; auxiliary state is copied into
    // registry-owned device memory so the donor may continue or close freely.
    // Republishing a key replaces the record; pages already mapped by earlier
    // borrowers stay alive through their reference counts.
    void publish(Pool& pool, std::uint64_t key, const std::vector<Pool::Entry*>& entries,
                 std::uint64_t bytes, const std::vector<PrefixState>& extra) {
        Device guard(device);
        std::lock_guard<std::mutex> lock(mutex);
        // Lock order is always manager -> pool; Pool never calls the manager.
        std::lock_guard<std::mutex> pool_lock(pool.mutex);
        require(bytes > 0 && !entries.empty(), "prefix requires a positive length and entries");
        const auto pages = pool.round(bytes) / pool.granule;
        require(pages > 0, "prefix shorter than one page");
        for (const auto* entry : entries)
            require(owned(pool, entry) && entry->mapped >= pages && entry->valid >= bytes,
                    "prefix donor entries must be resident over the published length");
        for (const auto& state : extra)
            require((state.src != nullptr) == (state.bytes > 0), "prefix state requires both source and bytes");

        std::vector<void*> snapshots(extra.size(), nullptr);
        std::vector<std::uint64_t> snapshot_bytes;
        try {
            for (std::size_t i = 0; i < extra.size(); ++i) {
                if (!extra[i].bytes) { snapshot_bytes.push_back(0); continue; }
                void* target = nullptr;
                check(cudaMalloc(&target, extra[i].bytes));
                snapshots[i] = target;
                check(cudaMemcpyAsync(target, extra[i].src, extra[i].bytes, cudaMemcpyDeviceToDevice, 0));
                check(cudaStreamSynchronize(0));
                snapshot_bytes.push_back(extra[i].bytes);
            }
        } catch (...) {
            for (auto* snapshot : snapshots)
                if (snapshot) (void)cudaFree(snapshot);
            throw;
        }
        replace_or_insert(pool, key, bytes, entries, std::move(snapshots), std::move(snapshot_bytes));
    }

    void replace_or_insert(Pool& pool, std::uint64_t key, std::uint64_t bytes,
                           const std::vector<Pool::Entry*>& entries,
                           std::vector<void*>&& snapshots, std::vector<std::uint64_t>&& snapshot_bytes) {
        for (auto& record : records)
            if (record.key == key) {
                release_snapshots(record);
                record.bytes = bytes;
                record.serial = ++serial;
                record.entries = entries;
                record.snapshots = std::move(snapshots);
                record.snapshot_bytes = std::move(snapshot_bytes);
                record.borrowers = 0;
                return;
            }
        if (records.size() >= capacity) {
            auto victim = records.end();
            for (auto it = records.begin(); it != records.end(); ++it)
                if (!it->borrowers && (victim == records.end() || it->serial < victim->serial)) victim = it;
            require(victim != records.end(), "prefix registry capacity exhausted by active borrowers");
            release_snapshots(*victim);
            records.erase(victim);
        }
        records.push_back({key, bytes, ++serial, entries, std::move(snapshots), std::move(snapshot_bytes), 0});
    }

    // Assemble a borrower: shared pages map read-only into fresh entries, the
    // frozen auxiliary state is restored into the caller's buffers (device to
    // device, same order as publish). Returns false when no usable record for
    // `key` exists; malformed requests (entry count mismatch, non-fresh
    // entries, missing destinations) fail closed with an exception.
    bool borrow(Pool& pool, std::uint64_t key, const std::vector<Pool::Entry*>& entries,
                const std::vector<void*>& extra_dests) {
        Device guard(device);
        std::lock_guard<std::mutex> lock(mutex);
        // Lock order is always manager -> pool; Pool never calls the manager.
        std::lock_guard<std::mutex> pool_lock(pool.mutex);
        for (auto& record : records) {
            if (record.key != key || !healthy(pool, record)) continue;
            require(entries.size() == record.entries.size(), "prefix borrower entry count differs from the record");
            require(extra_dests.size() == record.snapshots.size(), "prefix state destination count differs from the record");
            for (const auto* entry : entries)
                require(owned(pool, entry) && !entry->mapped && !entry->valid,
                        "prefix borrower entries must be fresh");
            for (std::size_t i = 0; i < entries.size(); ++i)
                map_shared(pool, *entries[i], *record.entries[i], record.bytes);
            for (std::size_t i = 0; i < record.snapshots.size(); ++i)
                if (record.snapshot_bytes[i]) {
                    require(extra_dests[i], "prefix state destination missing");
                    check(cudaMemcpyAsync(extra_dests[i], record.snapshots[i], record.snapshot_bytes[i],
                                          cudaMemcpyDeviceToDevice, 0));
                }
            check(cudaStreamSynchronize(0));
            record.serial = ++serial;
            ++record.borrowers;
            return true;
        }
        return false;
    }

    // Drop a record explicitly. Already-shared pages stay alive through their
    // reference counts; borrowers keep generating correctly.
    void forget(std::uint64_t key) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = records.begin(); it != records.end(); ++it)
            if (it->key == key) {
                release_snapshots(*it);
                records.erase(it);
                return;
            }
    }
};

}  // namespace vkv
