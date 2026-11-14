// Minimal ctypes harness around the unified KV pool (ops/csrc/kv). Built only
// by CUDA-gated tests; never linked into any product library.
#include "../src/venturi/ops/csrc/kv/kv_pool.cuh"
#include "../src/venturi/ops/csrc/kv/kv_prefix.cuh"

#include <cstdint>
#include <cstring>
#include <new>
#include <string>

static std::string error_text;
#define KVH_TRY try {
#define KVH_CATCH } catch (const std::exception& e) { error_text = e.what(); return -1; } \
                   catch (...) { error_text = "unknown kv harness failure"; return -1; }

extern "C" {
__attribute__((visibility("default"))) const char* kvh_error() { return error_text.c_str(); }

__attribute__((visibility("default"))) void* kvh_create(int device, std::uint64_t gpu,
                                                        std::uint64_t ram, std::uint64_t slot) {
    try { return new vkv::Pool(device, nullptr, 0, gpu, ram, slot); }
    catch (...) { return nullptr; }
}
__attribute__((visibility("default"))) void kvh_destroy(void* p) { delete static_cast<vkv::Pool*>(p); }
__attribute__((visibility("default"))) void* kvh_entry_create(void* p, std::uint64_t bytes, std::uint64_t row) {
    try { return static_cast<vkv::Pool*>(p)->create(bytes, row).release(); } catch (...) { return nullptr; }
}
__attribute__((visibility("default"))) void kvh_entry_destroy(void* p, void* e) {
    delete static_cast<vkv::Pool::Entry*>(e); (void)p;
}
__attribute__((visibility("default"))) int kvh_prepare(void* p, void** entries,
                                                       std::uint64_t* goals, std::uint64_t count) {
    KVH_TRY
    std::vector<std::pair<vkv::Pool::Entry*, std::uint64_t>> plan;
    for (std::uint64_t i = 0; i < count; ++i)
        plan.push_back({static_cast<vkv::Pool::Entry*>(entries[i]), goals[i]});
    static_cast<vkv::Pool*>(p)->prepare(plan);
    return 0;
    KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_ensure(void* p, void* e, std::uint64_t bytes) {
    KVH_TRY static_cast<vkv::Pool*>(p)->ensure(*static_cast<vkv::Pool::Entry*>(e), bytes); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_attach(void* p, void* e) {
    KVH_TRY static_cast<vkv::Pool*>(p)->attach(*static_cast<vkv::Pool::Entry*>(e)); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_written(void* p, void* e, std::uint64_t bytes) {
    KVH_TRY static_cast<vkv::Pool*>(p)->written(*static_cast<vkv::Pool::Entry*>(e), bytes); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_snapshot(void* p, void* e, int kind,
                                                        void* out, std::uint64_t bytes) {
    KVH_TRY static_cast<vkv::Pool*>(p)->snapshot(*static_cast<vkv::Pool::Entry*>(e), kind, out, bytes); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_reset(void* p, void* e) {
    KVH_TRY static_cast<vkv::Pool*>(p)->reset(*static_cast<vkv::Pool::Entry*>(e)); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_info(void* p, std::uint64_t* out) {
    KVH_TRY const auto a = static_cast<vkv::Pool*>(p)->info(); std::copy(a.begin(), a.end(), out); return 0; KVH_CATCH
}
// Test helper standing in for the model's K/V kernels: host pattern -> entry.
__attribute__((visibility("default"))) int kvh_fill(void* p, void* e, int kind,
                                                    std::uint64_t bytes, unsigned char seed) {
    KVH_TRY
    auto* pool = static_cast<vkv::Pool*>(p);
    auto* entry = static_cast<vkv::Pool::Entry*>(e);
    std::vector<unsigned char> pattern(bytes);
    for (std::uint64_t i = 0; i < bytes; ++i) pattern[i] = static_cast<unsigned char>(seed + i);
    vkv::check(cudaMemcpyAsync(entry->ptr(kind, 0), pattern.data(), bytes, cudaMemcpyHostToDevice, 0));
    vkv::check(cudaStreamSynchronize(0));
    pool->written(*entry, std::max(bytes, entry->valid));
    return 0;
    KVH_CATCH
}
__attribute__((visibility("default"))) void* kvh_index_create() {
    return new (std::nothrow) vkv::PrefixIndex();
}
__attribute__((visibility("default"))) void kvh_index_destroy(void* index) {
    delete static_cast<vkv::PrefixIndex*>(index);
}
__attribute__((visibility("default"))) int kvh_publish(void* index, std::uint64_t key,
                                                       void* entry, std::uint64_t bytes) {
    KVH_TRY static_cast<vkv::PrefixIndex*>(index)->publish(key, *static_cast<vkv::Pool::Entry*>(entry), bytes); return 0; KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_borrow(void* p, void* index,
                                                      std::uint64_t key, void* entry) {
    KVH_TRY return static_cast<vkv::PrefixIndex*>(index)->borrow(*static_cast<vkv::Pool*>(p), key,
                                                                 *static_cast<vkv::Pool::Entry*>(entry)) ? 1 : 0; KVH_CATCH
}
}
