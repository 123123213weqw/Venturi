// Matched single-sequence greedy benchmark using the public llama.cpp API.
// Build against the pinned upstream checkout; no llama.cpp implementation is copied.
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static llama_token greedy(llama_context * ctx, int vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("Missing logits");
    int best = 0;
    for (int i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite logits");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static void forward(llama_context * ctx, std::vector<llama_token> ids) {
    if (llama_decode(ctx, llama_batch_get_one(ids.data(), ids.size())) != 0)
        throw std::runtime_error("llama_decode failed");
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "Usage: bench MODEL.gguf REQUEST.json REPORT.json LOGITS.f32\n";
        return 2;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        json request;
        std::ifstream(argv[2]) >> request;
        const auto prompt = request.at("prompt_ids").get<std::vector<llama_token>>();
        const auto teacher = request.at("teacher_ids").get<std::vector<llama_token>>();
        const int budget = request.at("max_new_tokens");
        const int warmup = request.at("warmup");
        const int repeats = request.at("repeats");
        if (prompt.empty() || budget < 1 || warmup < 0 || repeats < 1)
            throw std::runtime_error("Invalid request");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model load failed");
        const auto vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (const auto & ids : {prompt, teacher})
            for (int id : ids)
                if (id < 0 || id >= n_vocab) throw std::runtime_error("Invalid token ID");
        const auto text = request.at("prompt").get<std::string>();
        std::vector<llama_token> encoded(text.size() + 32);
        int n = llama_tokenize(vocab, text.data(), text.size(), encoded.data(), encoded.size(), false, true);
        if (n < 0) throw std::runtime_error("Tokenizer output buffer insufficient");
        encoded.resize(n);
        if (encoded != prompt) throw std::runtime_error("Tokenizer IDs differ across engines");
        auto cp = llama_context_default_params();
        cp.n_ctx = prompt.size() + std::max(budget, int(teacher.size()) + 1);
        cp.n_batch = std::max(1, int(prompt.size()));
        cp.n_ubatch = cp.n_batch;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = request.at("threads");
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = true;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Context initialization failed");
        json runs = json::array();
        for (int iteration = -warmup; iteration < repeats; ++iteration) {
            llama_synchronize(ctx);
            const auto started = Clock::now();
            llama_memory_clear(llama_get_memory(ctx), false);
            const auto prefill = Clock::now();
            forward(ctx, prompt);
            std::vector<llama_token> output = {greedy(ctx, n_vocab)};
            const double prefill_seconds = seconds(prefill);
            const double ttft = seconds(started);
            const auto decode = Clock::now();
            while (int(output.size()) < budget) {
                forward(ctx, {output.back()});
                output.push_back(greedy(ctx, n_vocab));
            }
            llama_synchronize(ctx);
            const double decode_seconds = seconds(decode);
            const double total_seconds = seconds(started);
            if (iteration >= 0) runs.push_back({
                {"generated_ids", output}, {"prefill_seconds", prefill_seconds},
                {"time_to_first_token_seconds", ttft}, {"decode_seconds", decode_seconds},
                {"total_seconds", total_seconds}, {"output_tokens", output.size()},
                {"decode_tokens", output.size() - 1}
            });
        }
        // Separate untimed teacher-forced pass: every logit row has identical
        // input token IDs and cache positions in both engines.
        std::ofstream logits_file(argv[4], std::ios::binary);
        if (!logits_file) throw std::runtime_error("Cannot create logits file");
        llama_memory_clear(llama_get_memory(ctx), false);
        forward(ctx, prompt);
        for (size_t i = 0; i <= teacher.size(); ++i) {
            greedy(ctx, n_vocab); // Validate all logits before saving.
            logits_file.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, -1)),
                              n_vocab * sizeof(float));
            if (i < teacher.size()) forward(ctx, {teacher[i]});
        }
        if (!logits_file) throw std::runtime_error("Logits write failed");
        json report = {
            {"engine", "llama.cpp"}, {"prompt_ids", prompt}, {"tokenized_ids", encoded},
            {"vocab_size", n_vocab}, {"logit_rows", teacher.size() + 1},
            {"actual_context_capacity", llama_n_ctx(ctx)}, {"runs", runs},
            {"threads", cp.n_threads}, {"cache_dtype", "F16"}, {"flash_attention", "on"},
            {"cuda_graphs_disabled", std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr},
            {"system_info", llama_print_system_info()}
        };
        std::ofstream report_file(argv[3]);
        report_file << report.dump(2) << '\n';
        if (!report_file) throw std::runtime_error("Report write failed");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "benchmark: " << error.what() << '\n';
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
