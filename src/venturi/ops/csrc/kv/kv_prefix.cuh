// Page-level KV prefix sharing over vkv::Pool: identical token prefixes map
// the SAME physical pages read-only into multiple entries, so a shared system
// prompt is stored once per (layer, plane) instead of once per request.
// Include after kv_pool.cuh. No model semantics: callers derive the prefix
// key from their own token stream; this layer only guarantees byte identity
// for whatever the donor published.
#pragma once

namespace vkv {

struct PrefixRegistration {
    std::uint64_t key = 0, bytes = 0, serial = 0;
    Pool::Entry* donor = nullptr;
};

struct PrefixIndex {
    // Bounded LRU of published prefixes. Registrations never pin device
    // memory by themselves: pages stay alive only while referenced by entries.
    std::vector<PrefixRegistration> entries;
    std::uint64_t serial = 0;
    std::size_t capacity = 64;

    // Publish `bytes` of a resident donor prefix under `key`. Refreshing an
    // existing key replaces the donor; the previous donor loses nothing
    // (its mappings and refcounts are untouched).
    void publish(std::uint64_t key, Pool::Entry& donor, std::uint64_t bytes) {
        for (auto& r : entries)
            if (r.key == key) { r.donor = &donor; r.bytes = bytes; r.serial = ++serial; return; }
        if (entries.size() >= capacity) {
            auto oldest = std::min_element(entries.begin(), entries.end(),
                                           [](const auto& a, const auto& b) { return a.serial < b.serial; });
            *oldest = {key, bytes, ++serial, &donor};
        } else {
            entries.push_back({key, bytes, ++serial, &donor});
        }
    }

    bool owned(const Pool& pool, const Pool::Entry* e) const {
        return std::find(pool.entries.begin(), pool.entries.end(), e) != pool.entries.end();
    }

    // Map a published prefix into a freshly created borrower entry. The
    // shared region is READ-ONLY in the borrower's address space, so an
    // accidental write faults instead of corrupting the donor. The borrower's
    // own growth (mapping beyond the shared region) proceeds through the
    // normal Pool::ensure path with read-write access.
    bool borrow(Pool& pool, std::uint64_t key, Pool::Entry& into) {
        std::lock_guard<std::mutex> lock(pool.mutex);
        require(!pool.poisoned, "KV VMM driver failure: recreate model");
        for (auto& r : entries) {
            if (r.key != key || !owned(pool, r.donor)) continue;
            auto* donor = r.donor;
            const auto pages = pool.round(r.bytes) / pool.granule;
            if (!pages || donor->mapped < pages || donor->valid < r.bytes || into.mapped || into.valid) continue;
            for (int k = 0; k < 2; ++k)
                for (std::uint64_t p = 0; p < pages; ++p) {
                    const auto addr = into.base + mul(k, into.plane) + mul(p, pool.granule);
                    pool.driver(cuMemMap(addr, pool.granule, 0, donor->pages[k][p]->handle, 0));
                    ++donor->pages[k][p]->refs;  // owner-tracked before access setup
                    into.pages[k][p] = donor->pages[k][p];
                    pool.driver(cuMemSetAccess(addr, pool.granule, &pool.readonly, 1));
                }
            into.mapped = pages;
            into.valid = r.bytes;
            into.planned = std::max(into.planned, r.bytes);
            r.serial = ++serial;
            return true;
        }
        return false;
    }
};

// Drop a borrower's shared mappings without releasing borrowed pages:
// refcounts keep donor pages alive; Pool::unmap already routes every page
// through Pool::release. Sharing needs no separate teardown entry point.

}  // namespace vkv
