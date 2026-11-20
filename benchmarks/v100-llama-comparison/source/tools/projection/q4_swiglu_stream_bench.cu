// Public LinearSwiGLU benchmark with a read-heavy, larger-than-L2 weight ring.
// Uses the external benchmark fixture/graph utilities with their attribution.
#include "ninfer/ops/linear_swiglu.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include <array>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

int main() {
    using namespace ninfer;
    try {
        constexpr int matrices = 8, cycles = 8, launches = matrices * cycles;
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer input = bench::make_bf16(5120);
        DeviceBuffer output(17408 * 2);
        std::vector<bench::PackedQuantizedWeight> weights;
        weights.reserve(matrices);
        for (int i = 0; i < matrices; ++i)
            weights.push_back(bench::make_row_split_weight(
                QType::Q4_G64_FP16, 34816, 5120, 5120, {0x31, 0xa5, 0x3c00}));
        WorkspaceArena workspace(256);
        Tensor x(input.p, DType::BF16, {5120, 1});
        Tensor y(output.p, DType::BF16, {17408, 1});
        bench::TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t s) {
            for (int iteration = 0; iteration < cycles; ++iteration)
                for (auto& packed : weights)
                    ops::linear_swiglu(x, packed.weight, y, workspace, s);
        });
        for (int i = 0; i < 30; ++i) graph.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> samples;
        for (int i = 0; i < 50; ++i) samples.push_back(graph.launch_timed(stream) / launches);
        std::sort(samples.begin(), samples.end());
        std::printf("matrices,ring_bytes,public_calls_per_graph,graph_nodes,median_us,min_us,p95_us\n");
        std::printf("%d,%zu,%d,%zu,%.6f,%.6f,%.6f\n",matrices,
            matrices * weights[0].model_weight_bytes(), launches,graph.nodes(),samples[25],samples[0],samples[47]);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"stream Op benchmark: %s\n",error.what());return 1;
    }
}
