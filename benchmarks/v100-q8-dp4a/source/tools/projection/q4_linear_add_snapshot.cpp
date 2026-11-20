// Supplemental bit-parity probe, not a replacement for the independent FP64 oracle.
#include "ninfer/ops/linear_add.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    using namespace ninfer;
    try {
        if (argc != 2) throw std::invalid_argument("Expected explicit output snapshot path");
        std::ofstream snapshot(argv[1], std::ios::binary);
        if (!snapshot) throw std::runtime_error("Cannot create output snapshot");
        for (int k : {6144, 17408}) {
            for (unsigned seed : {429U, 431U}) {
                auto host = test::quantized_weight::make_patterned_weight(
                    QType::Q4_G64_FP16, 5120, k, seed);
                test::GuardedDeviceBuffer weight(host.payload.size());
                weight.copy_from_host(host.payload.data(), host.payload.size());
                std::vector<std::uint16_t> activation(k), residual(5120);
                for (int i = 0; i < k; ++i)
                    activation[i] = test::f32_to_bf16((static_cast<int>((i * 13U + seed) % 127U) - 63) * 0.001F);
                for (int i = 0; i < 5120; ++i)
                    residual[i] = test::f32_to_bf16((static_cast<int>((i * 7U + seed) % 63U) - 31) * 0.002F);
                test::GuardedDeviceBuffer input(k * 2), output(5120 * 2);
                input.copy_from_host(activation.data(), k * 2);
                output.copy_from_host(residual.data(), 5120 * 2);
                Tensor x(input.data(), DType::BF16, {k, 1});
                Tensor r(output.data(), DType::BF16, {5120, 1});
                WorkspaceArena workspace(256);
                ops::linear_add(x, host.device_weight(weight.data()), r, workspace, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                output.copy_to_host(residual.data(), 5120 * 2);
                snapshot.write(reinterpret_cast<const char*>(residual.data()), 5120 * 2);
            }
        }
        if (!snapshot) throw std::runtime_error("Snapshot write failed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "LinearAdd snapshot: %s\n", error.what());
        return 1;
    }
}
