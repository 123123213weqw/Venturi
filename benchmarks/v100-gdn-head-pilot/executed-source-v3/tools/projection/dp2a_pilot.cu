// Standalone research library: deliberately absent from production blocks.cu.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string error_text;
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
#include "../../src/venturi/ops/csrc/q8_dp2a.cuh"
extern "C" const char* venturi_dp2a_error(){return error_text.c_str();}
