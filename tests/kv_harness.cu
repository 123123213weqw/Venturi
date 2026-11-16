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
// ---- PrefixManager surface ----
struct KvhState { const void* src; std::uint64_t bytes; };  // layout-compatible with vkv::PrefixState
__attribute__((visibility("default"))) void* kvh_manager_create(int device) {
    return new (std::nothrow) vkv::PrefixManager(device);
}
__attribute__((visibility("default"))) void kvh_manager_destroy(void* manager) {
    delete static_cast<vkv::PrefixManager*>(manager);
}
__attribute__((visibility("default"))) int kvh_publish(void* manager, void* pool, std::uint64_t key,
                                                       void** entries, std::uint64_t entry_count,
                                                       std::uint64_t bytes,
                                                       KvhState* states, std::uint64_t state_count) {
    KVH_TRY
    std::vector<vkv::Pool::Entry*> owned;
    for (std::uint64_t i = 0; i < entry_count; ++i) owned.push_back(static_cast<vkv::Pool::Entry*>(entries[i]));
    std::vector<vkv::PrefixState> extra;
    for (std::uint64_t i = 0; i < state_count; ++i) extra.push_back({states[i].src, states[i].bytes});
    static_cast<vkv::PrefixManager*>(manager)->publish(*static_cast<vkv::Pool*>(pool), key, owned, bytes, extra);
    return 0;
    KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_borrow(void* manager, void* pool, std::uint64_t key,
                                                      void** entries, std::uint64_t entry_count,
                                                      void** dests, std::uint64_t dest_count) {
    KVH_TRY
    std::vector<vkv::Pool::Entry*> owned;
    for (std::uint64_t i = 0; i < entry_count; ++i) owned.push_back(static_cast<vkv::Pool::Entry*>(entries[i]));
    std::vector<void*> targets;
    for (std::uint64_t i = 0; i < dest_count; ++i) targets.push_back(dests[i]);
    return static_cast<vkv::PrefixManager*>(manager)->borrow(*static_cast<vkv::Pool*>(pool), key, owned, targets) ? 1 : 0;
    KVH_CATCH
}
__attribute__((visibility("default"))) int kvh_forget(void* manager, std::uint64_t key) {
    KVH_TRY static_cast<vkv::PrefixManager*>(manager)->forget(key); return 0; KVH_CATCH
}
// Test helper: a patterned device buffer standing in for model recurrent state.
__attribute__((visibility("default"))) void* kvh_state_alloc(std::uint64_t bytes, unsigned char seed) {
    void* buffer = nullptr;
    if (cudaMalloc(&buffer, bytes) != cudaSuccess) return nullptr;
    std::vector<unsigned char> pattern(bytes);
    for (std::uint64_t i = 0; i < bytes; ++i) pattern[i] = static_cast<unsigned char>(seed + i);
    if (cudaMemcpy(buffer, pattern.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(buffer); return nullptr;
    }
    return buffer;
}
__attribute__((visibility("default"))) int kvh_state_free(void* buffer) {
    return cudaFree(buffer) == cudaSuccess ? 0 : -1;
}
__attribute__((visibility("default"))) int kvh_state_read(void* buffer, void* out, std::uint64_t bytes) {
    return cudaMemcpy(out, buffer, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}
__attribute__((visibility("default"))) int kvh_state_overwrite(void* buffer, std::uint64_t bytes, unsigned char seed) {
    std::vector<unsigned char> pattern(bytes);
    for (std::uint64_t i = 0; i < bytes; ++i) pattern[i] = static_cast<unsigned char>(seed + i);
    return cudaMemcpy(buffer, pattern.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}
}
