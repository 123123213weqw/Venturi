// Public fused LinearAdd screening; no private kernel calls or cache-flush writes.
#include "ninfer/ops/linear_add.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    using namespace ninfer;
    try {
        constexpr int matrices = 8, cycles = 8, launches = matrices * cycles;
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("k,matrices,ring_bytes,calls_per_graph,graph_nodes,median_us,min_us,p95_us\n");
        for (int k : {6144, 17408}) {
            DeviceBuffer input = bench::make_bf16(k);
            DeviceBuffer residual = bench::make_bf16(5120);
            std::vector<bench::PackedQuantizedWeight> weights;
            weights.reserve(matrices);
            for (int i = 0; i < matrices; ++i)
                weights.push_back(bench::make_row_split_weight(
                    QType::Q4_G64_FP16, 5120, k, k, {0x31, 0xa5, 0x3c00}));
            WorkspaceArena workspace(256);
            Tensor x(input.p, DType::BF16, {k, 1});
            Tensor r(residual.p, DType::BF16, {5120, 1});
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) {
                for (int cycle = 0; cycle < cycles; ++cycle)
                    for (auto& packed : weights)
                        ops::linear_add(x, packed.weight, r, workspace, s);
            });
            for (int i = 0; i < 20; ++i) graph.launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<double> samples;
            for (int i = 0; i < 40; ++i)
                samples.push_back(graph.launch_timed(stream) / launches);
            std::sort(samples.begin(), samples.end());
            std::printf("%d,%d,%zu,%d,%zu,%.6f,%.6f,%.6f\n", k, matrices,
                matrices * weights[0].model_weight_bytes(), launches, graph.nodes(),
                samples[20], samples[0], samples[37]);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "LinearAdd screening: %s\n", error.what());
        return 1;
    }
}
