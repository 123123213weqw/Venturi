// Public fused LinearAdd GEMM benchmark; synthetic screening, not model TTFT.
#include "ninfer/ops/linear_add.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace ninfer;
    try {
        constexpr int matrices = 8;
        std::vector<int> tokens{128, 193, 256, 509, 512, 513, 1024};
        if (argc > 1) {
            tokens.clear();
            for (int i = 1; i < argc; ++i) {
                std::size_t consumed = 0;
                int t = std::stoi(argv[i], &consumed);
                if (consumed != std::string(argv[i]).size() || t < 1 || t > 2048)
                    throw std::invalid_argument("Expected token extents in 1..2048");
                tokens.push_back(t);
            }
        }
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("k,t,ring_bytes,calls_per_graph,graph_nodes,median_us,min_us,p95_us\n");
        for (int k : {6144, 17408}) {
            std::vector<bench::PackedQuantizedWeight> weights;
            for (int i = 0; i < matrices; ++i)
                weights.push_back(bench::make_row_split_weight(
                    QType::Q4_G64_FP16, 5120, k, k, {0x31, 0xa5, 0x3c00}));
            for (int t : tokens) {
                DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(k) * t);
                DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(5120) * t);
                Tensor x(input.p, DType::BF16, {k, t});
                Tensor r(residual.p, DType::BF16, {5120, t});
                WorkspaceArena workspace(256);
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    for (auto& packed : weights)
                        ops::linear_add(x, packed.weight, r, workspace, s);
                });
                for (int i = 0; i < 20; ++i) graph.launch(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<double> samples;
                for (int i = 0; i < 40; ++i)
                    samples.push_back(graph.launch_timed(stream) / matrices);
                std::sort(samples.begin(), samples.end());
                std::printf("%d,%d,%zu,%d,%zu,%.6f,%.6f,%.6f\n", k, t,
                    matrices * weights[0].model_weight_bytes(), matrices, graph.nodes(),
                    samples[20], samples[0], samples[37]);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Prefill GEMM screening: %s\n", error.what());
        return 1;
    }
}
