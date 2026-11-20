// Independent external FP64 oracle for the represented fused residual += W*x.
#include "ops/linear_add/linear_add_test_common.h"
#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer::test::linear_add;
    if (!cuda_available()) return 77;
    try {
        int failures = 0;
        constexpr std::array<std::int32_t, 2> full_tokens{1, 2};
        constexpr std::array<std::int32_t, 1> graph_tokens{1};
        for (int k : {6144, 17408}) {
            for (unsigned seed : {429U, 431U})
                failures += run_shape("Venturi Q4 LinearAdd full", WeightFormat::Q4G64F16S,
                    {5120, k, seed, {}, full_tokens, graph_tokens, true});
            constexpr std::array<std::int32_t, 4> regression_tokens{8, 16, 17, 33};
            failures += run_shape("Venturi Q4 LinearAdd dispatch", WeightFormat::Q4G64F16S,
                {5120, k, 433U, {}, regression_tokens, {}, false});
        }
        std::cout << (failures ? "FAIL" : "OK") << " Q4 LinearAdd oracle/guards/replay\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
