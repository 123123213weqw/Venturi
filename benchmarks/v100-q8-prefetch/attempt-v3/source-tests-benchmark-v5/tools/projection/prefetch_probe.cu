// Minimal research-only operator build. Does not replace the native model SO.
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#define VENTURI_BLOCKS_ARCH 70
static thread_local std::string error_text;
static void check(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
#include "q8_dp4a.cuh"
#include "q8_fused.cuh"
#include "q8_column_tp.cuh"
#include "q8_prefetch.cuh"
#include "q8_projection_dispatch.cuh"
extern "C" const char* venturi_prefetch_error(){return error_text.c_str();}
