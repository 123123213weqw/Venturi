// Research transport only. All model, scheduling, cache and CUDA work is external NInfer.
#include "ninfer/engine.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static json stats(const ninfer::Engine& engine) {
    const auto s = engine.runtime_stats();
    const auto& h = s.host_work;
    json j;
#define S(field) j[#field] = s.field
    S(computed_prefill_tokens); S(committed_decode_tokens); S(decode_rounds); S(decode_row_rounds);
    S(running_requests); S(waiting_requests); S(prefilling_requests); S(decode_ready_requests);
    S(reused_prompt_tokens); S(root_selections); S(shared_stable_prefix_selections);
    S(private_endpoint_selections); S(active_captures_completed); S(active_captures_aborted);
    S(state_moves); S(state_forks); S(state_restores); S(state_d2d_bytes); S(state_h2d_bytes);
    S(main_kv_d2d_bytes); S(main_kv_h2d_bytes); S(partial_tail_cow_pages);
    S(device_state_occupied_slots); S(device_main_kv_occupied_pages); S(host_kv_occupied_bytes);
    S(pressure_searches); S(pressure_checkpoints_dropped);
#undef S
    json host;
#define H(field) host[#field] = h.field
    H(engine_boundary_ns); H(program_submit_ns); H(program_post_ns); H(engine_commit_output_ns);
    H(engine_maintenance_ns); H(device_wait_ns); H(decode_host_ns); H(decode_device_wait_ns);
    H(prefill_host_ns); H(prefill_device_wait_ns); H(prefill_units); H(control_units);
    H(admission_policy_ns); H(context_progress_ns); H(stats_publication_ns);
#undef H
    j["host_work"] = host;
    return j;
}
static json memory(const ninfer::Engine& engine) {
    const auto m = engine.memory_summary();
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    json j;
#define M(field) j[#field] = m.field
    M(kv_capacity); M(kv_capacity_page_groups); M(kv_capacity_max_page_groups);
    M(runtime_reservation_bytes); M(minimum_runtime_reservation_bytes); M(kv_capacity_increment_bytes);
    M(kv_payload_bytes); M(cuda_graph_allowance_bytes); M(host_state_capacity_slots);
    M(host_kv_capacity_bytes); M(workspace_logical_peak_bytes);
#undef M
    j["weights_bytes"] = m.weights.used_bytes;
    j["workspace_capacity_bytes"] = m.workspace.capacity_bytes;
    j["sequence_capacity_bytes"] = m.sequence.capacity_bytes;
    j["device_free_bytes"] = free; j["device_total_bytes"] = total;
    return j;
}
static json result_json(const ninfer::GenerationResult& r) {
    const auto& t = r.timings;
    const auto& e = r.engine_timing;
    return {{"generated_ids", r.generated_token_ids}, {"text", r.content},
      {"prompt_tokens", r.prompt.prompt_tokens}, {"reused_prompt_tokens", r.reused_prompt_tokens},
      {"prefix_reuse_path", static_cast<int>(r.prefix_reuse_path)},
      {"finish_reason", static_cast<int>(r.finish_reason)},
      {"timings", {{"first_token_seconds", t.first_token_seconds}, {"prefill_seconds", t.prefill_seconds},
        {"decode_seconds", t.decode_seconds}, {"prompt_wall_seconds", t.prompt_wall_seconds},
        {"generation_wall_seconds", t.generation_wall_seconds}, {"total_seconds", t.total_seconds}}},
      {"engine_timing", {{"queue_wait_seconds", e.queue_wait_seconds},
        {"decode_host_exposed_seconds", e.decode_host_exposed_seconds},
        {"decode_device_wait_exposed_seconds", e.decode_device_wait_exposed_seconds},
        {"prefill_units", e.prefill_units}, {"decode_rounds", e.decode_rounds}}}};
}
class TimingSink final : public ninfer::OutputSink {
public:
    json events = json::array();
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation t) override {
        events.push_back({{"tokens", t.generated_tokens}, {"prompt_ns", t.prompt_elapsed_ns},
                          {"generation_ns", t.generation_elapsed_ns}});
    }
};
static void emit(const json& j) { std::cout << j.dump() << std::endl; }
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("probe requires an explicit artifact path");
        spdlog::set_default_logger(spdlog::stderr_color_mt("venturi-probe"));
        spdlog::set_level(spdlog::level::warn);
        std::string line;
        if (!std::getline(std::cin, line)) throw std::invalid_argument("missing configuration");
        const json c = json::parse(line);
        ninfer::EngineOptions o;
        o.artifact_path = argv[1];
        o.max_context = c.value("max_context", 2048U);
        o.max_concurrency = c.value("concurrency", 1U);
        if (o.max_concurrency < 1 || o.max_concurrency > 8) throw std::invalid_argument("concurrency must be 1..8");
        o.max_pending_requests = 32;
        o.pending_timeout_ms = 300000;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(c.value("kv_capacity", o.max_context));
        o.prefill_chunk = c.value("prefill_chunk", 512U);
        o.use_cuda_graph = c.value("cuda_graph", true);
        const auto kv = c.value("kv_dtype", std::string("bf16"));
        if (kv == "bf16") o.kv_cache = ninfer::KvCacheStorage::BFloat16;
        else if (kv == "k8v4") o.kv_cache = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
        else throw std::invalid_argument("unsupported KV profile");
        if (c.value("speculative", std::string("none")) == "dflash2") {
            o.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
            o.speculative.draft_tokens = 8;
        } else if (c.value("speculative", std::string("none")) != "none") {
            throw std::invalid_argument("unsupported speculation");
        }
        o.context_cache.enabled = c.value("cache", false);
        o.context_cache.device_state_slots = c.value("checkpoint_slots", 1U);
        o.context_cache.host_state_slots = 0;
        o.context_cache.host_kv_capacity_bytes = 0;
        const auto startup = Clock::now();
        ninfer::Engine engine(std::move(o));
        emit({{"ok", true}, {"event", "ready"}, {"protocol", 1}, {"config", c},
              {"startup_seconds", std::chrono::duration<double>(Clock::now()-startup).count()},
              {"memory", memory(engine)}, {"stats", stats(engine)}});
        while (std::getline(std::cin, line)) {
            bool profiling = false;
            try {
                const auto request = json::parse(line);
                if (request.value("op", std::string()) == "close") break;
                if (request.value("op", std::string()) == "tokenize") {
                    emit({{"ok", true}, {"token_ids", engine.tokenize_text(request.at("prompt").get<std::string>())}});
                    continue;
                }
                const auto& requests = request.at("requests");
                if (!requests.is_array() || requests.empty() || requests.size() > 16)
                    throw std::invalid_argument("requests must have 1..16 elements");
                std::vector<ninfer::PreparedPrompt> prompts;
                std::vector<ninfer::RequestOptions> options;
                for (const auto& item : requests) {
                    ninfer::PreparedPrompt prompt;
                    if (item.contains("token_ids")) {
                        prompt = engine.prepare_tokens(item.at("token_ids").get<std::vector<ninfer::TokenId>>(), false);
                    } else {
                        ninfer::PromptInput input;
                        if (item.contains("system")) {
                            ninfer::ChatMessage system;
                            system.role = ninfer::ChatRole::System;
                            ninfer::MessagePart p; p.text = item.at("system");
                            system.parts.push_back(std::move(p)); input.messages.push_back(std::move(system));
                            ninfer::PromptCacheMarker marker;
                            marker.after_message_count = 1;
                            input.context_cache.markers.push_back(marker);
                        }
                        ninfer::ChatMessage user;
                        ninfer::MessagePart p; p.text = item.at("prompt");
                        user.parts.push_back(std::move(p)); input.messages.push_back(std::move(user));
                        input.options.enable_thinking = false;
                        input.context_cache.allow_engine_automatic_shared_prefixes = true;
                        prompt = engine.prepare(std::move(input));
                    }
                    ninfer::RequestOptions opt;
                    opt.execution.requested_output_tokens = item.value("max_new_tokens", 64U);
                    if (!opt.execution.requested_output_tokens ||
                        static_cast<std::uint64_t>(prompt.summary().prompt_tokens)+opt.execution.requested_output_tokens > engine.options().max_context)
                        throw std::invalid_argument("prompt plus output exceeds max_context or zero output budget");
                    opt.execution.allow_prefix_reuse = item.value("reuse", false);
                    auto& s = opt.execution.sampling;
                    s.temperature = 0.0F; s.top_k = 0; s.top_p = 1.0F; s.min_p = 0.0F;
                    s.presence_penalty = 0.0F; s.frequency_penalty = 0.0F; s.seed = 0;
                    opt.stop.include_model_defaults = !item.value("ignore_eos", true);
                    opt.stop.publish_stop_token = item.value("ignore_eos", true);
                    prompts.push_back(std::move(prompt)); options.push_back(std::move(opt));
                }
                const auto before = stats(engine);
                if (request.value("profile", false)) {
                    if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("cudaProfilerStart failed");
                    profiling = true;
                }
                const auto begin = Clock::now();
                const bool streaming = request.value("stream_timings", false);
                std::vector<ninfer::GenerationHandle> handles;
                for (std::size_t i = 0; i < prompts.size(); ++i)
                    handles.push_back(engine.submit(std::move(prompts[i]), std::move(options[i]),
                      streaming ? ninfer::OutputConsumerMode::Streaming : ninfer::OutputConsumerMode::Aggregate,
                      {.phase_timings=true, .live_timings=streaming}));
                json results = json::array();
                for (auto& h : handles) {
                    TimingSink sink;
                    auto result = result_json(h.wait(streaming ? &sink : nullptr));
                    if (streaming) result["commit_timings"] = std::move(sink.events);
                    results.push_back(std::move(result));
                }
                const auto seconds = std::chrono::duration<double>(Clock::now()-begin).count();
                if (profiling) { cudaProfilerStop(); profiling = false; }
                emit({{"ok", true}, {"wall_seconds", seconds}, {"results", results},
                      {"stats_before", before}, {"stats_after", stats(engine)}, {"memory", memory(engine)}});
            } catch (const std::exception& e) {
                if (profiling) cudaProfilerStop();
                emit({{"ok", false}, {"error", e.what()}, {"available", engine.is_available()}});
            }
        }
        return 0;
    } catch (const std::exception& e) {
        emit({{"ok", false}, {"event", "startup_error"}, {"error", e.what()}});
        return 1;
    }
}
