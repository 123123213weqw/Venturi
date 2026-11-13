// Qwen model runtime. Link against the explicit operator library, never its implementation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../../../ops/csrc/cuda_common.cuh"
#include "../../../ops/csrc/ops_api.h"
namespace { thread_local std::string error_text; }
extern "C" VENTURI_EXPORT int venturi_qwen_abi() { return 1; }
extern "C" VENTURI_EXPORT int venturi_qwen_required_ops_abi() { return VENTURI_OPS_ABI; }
extern "C" VENTURI_EXPORT int venturi_qwen_arch() { return VENTURI_BLOCKS_ARCH; }
extern "C" VENTURI_EXPORT const char* venturi_qwen_error() { return error_text.c_str(); }
extern "C" VENTURI_EXPORT const void* venturi_qwen_ops_identity() { return reinterpret_cast<const void*>(&venturi_blocks_abi); }
#include "qwen_runtime.cuh"
#include "qwen_batch.cuh"
#include "qwen_pipeline.cuh"
#include "qwen_handoff.cuh"
#include "qwen_mtp.cuh"
