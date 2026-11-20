// Venturi JSON-lines transport for the external Apache-2.0 NInfer/Cinference Engine.
// Model mathematics, CUDA kernels, artifact materialization and DFlash2 are upstream.
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
#include <algorithm>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static std::uint32_t integer(const json& j, const char* key, std::uint32_t fallback,
                             std::uint32_t minimum, std::uint32_t maximum) {
    if (!j.contains(key)) return fallback;
    if (!j.at(key).is_number_integer()) throw std::invalid_argument(std::string(key) + " must be integer");
    const auto n = j.at(key).get<std::int64_t>();
    if (n < minimum || n > maximum) throw std::invalid_argument(std::string(key) + " is out of range");
    return static_cast<std::uint32_t>(n);
}

static json memory(const ninfer::Engine& engine) {
    const auto m = engine.memory_summary();
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return {{"weights_used_bytes", m.weights.used_bytes}, {"weights_capacity_bytes", m.weights.capacity_bytes},
            {"runtime_reservation_bytes", m.runtime_reservation_bytes}, {"kv_payload_bytes", m.kv_payload_bytes},
            {"kv_capacity", m.kv_capacity}, {"kv_capacity_page_groups", m.kv_capacity_page_groups},
            {"workspace_peak_bytes", m.workspace.peak_used_bytes}, {"device_free_bytes", free},
            {"device_total_bytes", total}};
}

static void emit(const json& result) { std::cout << result.dump() << std::endl; }

static std::pair<ninfer::PreparedPrompt, ninfer::RequestOptions>
prepare(ninfer::Engine& engine, const json& request) {
    ninfer::PreparedPrompt prompt;
    if (request.contains("token_ids")) {
        const auto& raw_ids = request.at("token_ids");
        if (!raw_ids.is_array() || raw_ids.empty()) throw std::invalid_argument("token_ids must be nonempty");
        std::vector<ninfer::TokenId> ids;
        for (const auto& value : raw_ids) {
            if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
                value.get<std::int64_t>() > 2147483647) throw std::invalid_argument("invalid token ID");
            ids.push_back(value.get<ninfer::TokenId>());
        }
        prompt = engine.prepare_tokens(std::move(ids), false);
    } else if (request.value("raw_prompt", false)) {
        prompt = engine.prepare_tokens(engine.tokenize_text(request.at("prompt").get<std::string>()), false);
    } else {
        ninfer::PromptInput input;
        ninfer::ChatMessage message;
        ninfer::MessagePart part;
        part.text = request.at("prompt").get<std::string>();
        message.parts.push_back(std::move(part));
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        prompt = engine.prepare(std::move(input));
    }
    ninfer::RequestOptions generation;
    generation.execution.requested_output_tokens = integer(request, "max_new_tokens", 32, 1, engine.options().max_context);
    generation.execution.allow_prefix_reuse = false;
    generation.execution.sampling.temperature = 0.0F;
    generation.execution.sampling.top_k = 0;
    generation.execution.sampling.top_p = 1.0F;
    generation.execution.sampling.min_p = 0.0F;
    generation.execution.sampling.presence_penalty = 0.0F;
    generation.execution.sampling.frequency_penalty = 0.0F;
    generation.execution.sampling.seed = 0;
    generation.stop.include_model_defaults = !request.value("ignore_eos", false);
    generation.stop.publish_stop_token = request.value("ignore_eos", false);
    const auto prompt_tokens = prompt.summary().prompt_tokens;
    if (static_cast<std::uint64_t>(prompt_tokens) + generation.execution.requested_output_tokens > engine.options().max_context)
        throw std::invalid_argument("prompt plus output exceeds max_context; no automatic truncation");
    return {std::move(prompt), std::move(generation)};
}

static json generation_json(const ninfer::GenerationResult& result) {
    const auto& t = result.timings;
    const auto& s = result.speculative;
    return {{"ok", true}, {"prompt_tokens", result.prompt.prompt_tokens},
            {"generated_ids", result.generated_token_ids},
            {"text", result.content}, {"reasoning", result.reasoning},
            {"finish_reason", static_cast<int>(result.finish_reason)},
            {"reused_prompt_tokens", result.reused_prompt_tokens},
            {"timings", {{"prepare_seconds", t.prepare_seconds}, {"first_token_seconds", t.first_token_seconds},
                {"prefill_seconds", t.prefill_seconds}, {"decode_seconds", t.decode_seconds},
                {"generation_wall_seconds", t.generation_wall_seconds}, {"total_seconds", t.total_seconds}}},
            {"speculative", {{"enabled", s.enabled}, {"rounds", s.rounds},
                {"drafted_tokens", s.drafted_tokens}, {"accepted_tokens", s.accepted_tokens},
                {"fallback_steps", s.fallback_steps}, {"draft_window", s.draft_window},
                {"accepted_per_position", s.accepted_per_position}}}};
}

static json interval_stats(const ninfer::RuntimeStats& before, const ninfer::RuntimeStats& after) {
    const auto rounds = after.decode_rounds - before.decode_rounds;
    const auto rows = after.decode_row_rounds - before.decode_row_rounds;
    return {{"computed_prefill_tokens", after.computed_prefill_tokens - before.computed_prefill_tokens},
            {"committed_decode_tokens", after.committed_decode_tokens - before.committed_decode_tokens},
            {"decode_rounds", rounds}, {"decode_row_rounds", rows},
            {"mean_decode_batch", rounds ? json(static_cast<double>(rows) / rounds) : json(nullptr)}};
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: venturi-qwen-runner MODEL.ninfer (JSON-lines stdin/stdout)");
        spdlog::set_default_logger(spdlog::stderr_color_mt("venturi-native"));
        spdlog::set_level(spdlog::level::warn);
        std::string line;
        if (!std::getline(std::cin, line)) throw std::invalid_argument("missing startup configuration");
        const json config = json::parse(line);
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device = integer(config, "device", 0, 0, 128);
        options.max_context = integer(config, "max_context", 2048, 16, 262144);
        options.max_concurrency = integer(config, "max_concurrency", 1, 1, 8);
        const auto logical_pages = (options.max_context + 63U) / 64U;
        const auto capacity = integer(config, "kv_capacity", options.max_concurrency * logical_pages * 64U,
                                      options.max_context, options.max_concurrency * logical_pages * 64U);
        if ((capacity + 63U) / 64U < std::max(logical_pages, options.max_concurrency))
            throw std::invalid_argument("kv_capacity needs at least one page per lane");
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(capacity);
        options.max_pending_requests = 16;
        options.pending_timeout_ms = integer(config, "pending_timeout_ms", 300000, 1, 4294967295U);
        options.prefill_chunk = integer(config, "prefill_chunk", 128, 1, options.max_context);
        options.use_cuda_graph = config.value("cuda_graph", true);
        options.context_cache.enabled = false;
        options.context_cache.host_state_slots = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
        options.startup_observer.callback = [](const ninfer::StartupEvent& event) {
            if (event.status == ninfer::StartupStatus::Complete || event.status == ninfer::StartupStatus::Failed)
                std::cerr << "startup phase=" << static_cast<int>(event.phase)
                          << " status=" << static_cast<int>(event.status)
                          << " elapsed_ns=" << event.elapsed_ns << '\n';
        };
        const auto kv = config.value("kv_dtype", std::string("bf16"));
        if (kv == "bf16") options.kv_cache = ninfer::KvCacheStorage::BFloat16;
        else if (kv == "k8v4") options.kv_cache = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
        else throw std::invalid_argument("kv_dtype must be bf16 or k8v4");
        const auto spec = config.value("speculative", std::string("none"));
        if (spec == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = integer(config, "draft_tokens", 8, 1, 15);
            options.speculative.verify_tree = config.value("verify_tree", false);
            if (options.speculative.verify_tree && options.max_concurrency != 1)
                throw std::invalid_argument("verify-tree requires max_concurrency=1");
            options.speculative.proposal_head = config.value("optimized_proposal", false)
                ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        } else if (spec != "none") throw std::invalid_argument("unsupported speculative backend");
        const auto cpu_wait = config.value("cpu_wait", std::string("blocking"));
        if (cpu_wait != "auto" && cpu_wait != "blocking")
            throw std::invalid_argument("cpu_wait must be auto or blocking");
        // Only this owned process's CUDA primary context is affected. Leave
        // unrelated GPU jobs alone; no clocks or system scheduler are changed.
        auto status = cudaSetDevice(options.device);
        if (status != cudaSuccess) throw std::runtime_error(std::string("cudaSetDevice: ") + cudaGetErrorString(status));
        unsigned int device_flags = 0;
        status = cudaGetDeviceFlags(&device_flags);
        if (status != cudaSuccess) throw std::runtime_error(std::string("cudaGetDeviceFlags: ") + cudaGetErrorString(status));
        const auto schedule = cpu_wait == "blocking" ? cudaDeviceScheduleBlockingSync : cudaDeviceScheduleAuto;
        status = cudaSetDeviceFlags((device_flags & ~cudaDeviceScheduleMask) | schedule);
        if (status != cudaSuccess) throw std::runtime_error(std::string("cudaSetDeviceFlags: ") + cudaGetErrorString(status));
        status = cudaGetDeviceFlags(&device_flags);
        if (status != cudaSuccess || (device_flags & cudaDeviceScheduleMask) != schedule)
            throw std::runtime_error("CUDA did not apply the requested CPU wait policy");
        const auto started = Clock::now();
        ninfer::Engine engine(std::move(options));
        const auto load = engine.load_summary();
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, config.value("device", 0)) != cudaSuccess)
            throw std::runtime_error("cudaGetDeviceProperties failed");
        emit({{"ok", true}, {"event", "ready"}, {"protocol", 3},
              {"backend", "external-ninfer-cinference"}, {"config", config},
              {"cpu_wait", cpu_wait}, {"cuda_device_flags", device_flags},
              {"limits", {{"max_concurrency", engine.options().max_concurrency},
                          {"kv_capacity", engine.memory_summary().kv_capacity}, {"max_batch_requests", 16}}},
              {"architecture", load.architecture}, {"weight_formats", load.weight_formats},
              {"load_seconds", load.load_seconds}, {"upload_seconds", load.upload_seconds},
              {"startup_seconds", std::chrono::duration<double>(Clock::now() - started).count()},
              {"artifact_bytes_read", load.artifact_bytes_read},
              {"host_to_device_bytes", load.host_to_device_bytes},
              {"gpu", properties.name}, {"compute_capability", {properties.major, properties.minor}},
              {"memory", memory(engine)}});
        while (std::getline(std::cin, line)) {
            bool profiling = false;
            try {
                const auto request = json::parse(line);
                const auto op = request.value("op", std::string("generate"));
                if (op == "close") { emit({{"ok", true}, {"event", "closed"}}); break; }
                if (op == "tokenize") {
                    emit({{"ok", true}, {"token_ids", engine.tokenize_text(request.at("prompt").get<std::string>())}});
                    continue;
                }
                if (op != "generate" && op != "generate_many") throw std::invalid_argument("unknown operation");
                const auto wall_started = Clock::now();
                const bool many = op == "generate_many";
                const auto mode = many ? request.at("mode").get<std::string>() : "single";
                if (mode != "single" && mode != "parallel") throw std::invalid_argument("mode must be single or parallel");
                if (mode == "parallel" && engine.options().max_concurrency < 2)
                    throw std::invalid_argument("parallel mode requires reserved concurrency >= 2");
                const json items = many ? request.at("requests") : json::array({request});
                if (!items.is_array() || items.empty() || items.size() > 16)
                    throw std::invalid_argument("requests must contain 1..16 elements");
                std::vector<ninfer::PreparedPrompt> prompts;
                std::vector<ninfer::RequestOptions> generations;
                bool want_profile = false;
                for (const auto& item : items) {
                    auto [prompt, generation] = prepare(engine, item);
                    prompts.push_back(std::move(prompt));
                    generations.push_back(std::move(generation));
                    want_profile = want_profile || item.value("profile", false);
                }
                if (want_profile) {
                    if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("cudaProfilerStart failed");
                    profiling = true;
                }
                engine.reset_memory_peaks();
                const auto before = engine.runtime_stats();
                json results = json::array();
                std::vector<ninfer::GenerationHandle> handles;
                // Submit before waiting to let the native scheduler form compact batches.
                // Serial dispatch instead drains each request before admitting the next.
                for (std::size_t i = 0; i < prompts.size(); ++i) {
                    auto handle = engine.submit(std::move(prompts[i]), std::move(generations[i]),
                        ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true});
                    if (mode == "parallel") handles.push_back(std::move(handle));
                    else results.push_back(generation_json(handle.wait()));
                }
                for (auto& handle : handles) results.push_back(generation_json(handle.wait()));
                const auto seconds = std::chrono::duration<double>(Clock::now() - wall_started).count();
                const auto stats = interval_stats(before, engine.runtime_stats());
                if (profiling) { cudaProfilerStop(); profiling = false; }
                if (many) {
                    for (std::size_t i = 0; i < results.size(); ++i) results[i]["index"] = i;
                    emit({{"ok", true}, {"mode", mode}, {"results", results}, {"wall_seconds", seconds},
                          {"runtime", stats}, {"memory", memory(engine)}});
                } else {
                    auto result = std::move(results[0]);
                    result["timings"]["native_request_seconds"] = seconds;
                    result["runtime"] = stats;
                    result["memory"] = memory(engine);
                    emit(result);
                }
            } catch (const std::exception& error) {
                if (profiling) cudaProfilerStop();
                emit({{"ok", false}, {"error", error.what()}, {"available", engine.is_available()}});
            }
        }
        return 0;
    } catch (const std::exception& error) {
        emit({{"ok", false}, {"event", "startup_error"}, {"error", error.what()}});
        return 1;
    }
}
