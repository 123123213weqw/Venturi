// Public Op qualification using the external independent FP64 oracle and guards.
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    try {
        int failures = 0;
        // T=1 is the changed route. T=17/32 share its translation unit; other
        // cases check dispatch regression. Graph inputs are also negated/replayed.
        for (unsigned seed : {1401U, 2903U})
            failures += run_profile("Venturi projection qualification",
                {QType::Q4_G64_FP16, 34816, 5120, 17408, seed, ActivationCompute::A16},
                std::array<std::int32_t, 5>{1, 2, 17, 32, 512},
                std::array<std::int32_t, 3>{1, 2, 17});
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
