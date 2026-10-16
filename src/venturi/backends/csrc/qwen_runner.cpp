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
            {"workspace_peak_bytes", m.workspace.peak_used_bytes}, {"device_free_bytes", free},
            {"device_total_bytes", total}};
}

static void emit(const json& result) { std::cout << result.dump() << std::endl; }

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
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.prefill_chunk = integer(config, "prefill_chunk", 128, 1, options.max_context);
        options.use_cuda_graph = config.value("cuda_graph", true);
        options.max_concurrency = 1;
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
            options.speculative.proposal_head = config.value("optimized_proposal", false)
                ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        } else if (spec != "none") throw std::invalid_argument("unsupported speculative backend");
        const auto started = Clock::now();
        ninfer::Engine engine(std::move(options));
        const auto load = engine.load_summary();
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, config.value("device", 0)) != cudaSuccess)
            throw std::runtime_error("cudaGetDeviceProperties failed");
        emit({{"ok", true}, {"event", "ready"}, {"protocol", 1},
              {"backend", "external-ninfer-cinference"}, {"config", config},
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
                if (op != "generate") throw std::invalid_argument("unknown operation");
                const auto wall_started = Clock::now();
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
                if (request.value("profile", false)) {
                    if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("cudaProfilerStart failed");
                    profiling = true;
                }
                engine.reset_memory_peaks();
                auto handle = engine.submit(std::move(prompt), std::move(generation),
                    ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true});
                const auto result = handle.wait();
                if (profiling) { cudaProfilerStop(); profiling = false; }
                const auto& t = result.timings;
                const auto& s = result.speculative;
                emit({{"ok", true}, {"prompt_tokens", prompt_tokens}, {"generated_ids", result.generated_token_ids},
                      {"text", result.content}, {"reasoning", result.reasoning},
                      {"finish_reason", static_cast<int>(result.finish_reason)},
                      {"reused_prompt_tokens", result.reused_prompt_tokens},
                      {"timings", {{"prepare_seconds", t.prepare_seconds}, {"first_token_seconds", t.first_token_seconds},
                          {"prefill_seconds", t.prefill_seconds}, {"decode_seconds", t.decode_seconds},
                          {"generation_wall_seconds", t.generation_wall_seconds}, {"total_seconds", t.total_seconds},
                          {"native_request_seconds", std::chrono::duration<double>(Clock::now() - wall_started).count()}}},
                      {"speculative", {{"enabled", s.enabled}, {"rounds", s.rounds},
                          {"drafted_tokens", s.drafted_tokens}, {"accepted_tokens", s.accepted_tokens},
                          {"fallback_steps", s.fallback_steps}, {"draft_window", s.draft_window},
                          {"accepted_per_position", s.accepted_per_position}}}, {"memory", memory(engine)}});
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
