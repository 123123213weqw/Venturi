// Use the external independent FP64 fused LinearAdd oracle without relaxing its criterion.
#include "ops/linear_add/linear_add_test_common.h"
#include <array>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace ninfer::test::linear_add;
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "512")) {
        std::cerr << "Usage: test [512]\n";
        return 2;
    }
    if (!cuda_available()) return 77;
    try {
        int failures = 0;
        // Qualify all outputs at either the first changed extent or the full
        // chunk under test. The sampled-row oracle also checks tile boundaries.
        const std::array<std::int32_t, 1> full{argc == 2 ? 512 : 193};
        constexpr std::int32_t sampled[]{192, 194, 255, 256, 257, 319, 320, 321,
            383, 384, 385, 447, 448, 449, 480, 509, 511, 512, 513, 1024};
        constexpr std::int32_t graphs[]{256, 385, 511, 512, 513};
        for (int k : {6144, 17408}) {
            failures += run_shape("Venturi prefill full", WeightFormat::Q4G64F16S,
                {5120, k, 429U, {}, full, full, true});
            failures += run_shape("Venturi prefill boundaries", WeightFormat::Q4G64F16S,
                {5120, k, 431U, {}, sampled, graphs, false});
        }
        std::cout << (failures ? "FAIL" : "OK") << " prefill FP64/guards/replay\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
