// Additive ABI-17 extension: bounded Q5 -> FP16 + Tensor Core cuBLAS.
// Reconstruction is part of every call; there is no persistent FP16 shadow.
__global__ void q5_cublas_decode(const unsigned char* codes, const unsigned char* high,
    const __half* scales, __half* out, int begin, int rows, int k, int padded) {
    const std::uint64_t pairs = std::uint64_t(rows) * (padded / 2);
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < pairs; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const int row = i / (padded / 2), col = (i % (padded / 2)) * 2;
        if (col >= k) continue;
        const std::uint64_t n = begin + row;
        const unsigned char packed = codes[n * (padded / 2) + col / 2];
        const int h = (high[n * (padded / 8) + col / 8] >> (col & 7)) & 3;
        const int a = ((packed & 15) | ((h & 1) << 4));
        const int b = ((packed >> 4) | ((h >> 1) << 4));
        const __half2 value = __hmul2(__halves2half2(__int2half_rn((a ^ 16) - 16),
                                                    __int2half_rn((b ^ 16) - 16)),
                                      __half2half2(scales[n * (padded / 64) + col / 64]));
        out[std::uint64_t(row) * k + col] = __low2half(value);
        if (col + 1 < k) out[std::uint64_t(row) * k + col + 1] = __high2half(value);
    }
}

// Diagnostic strong baseline: the *same decoded Q5 values* already resident.
// This is a single-operator control, not a claim that a whole FP16 model fits.
extern "C" int venturi_q5_dense_reference_run(void* opaque, const void* x,
    const void* w, void* out, int t, int n, int k, void* workspace,
    std::size_t bytes, void* raw_stream) {
    try {
        auto* p = static_cast<Linear*>(opaque);
        require(VENTURI_BLOCKS_ARCH == 70 && p && p->dtype == 0 && x && w && out &&
                workspace && bytes >= 4 * 1024 * 1024 && t > 0 && n > 0 && k > 0,
                "invalid resident Q5 cuBLAS reference");
        check(cublasSetStream(p->handle, static_cast<cudaStream_t>(raw_stream)));
        check(cublasSetWorkspace(p->handle, workspace, bytes));
        const float alpha = 1.f, beta = 0.f;
        check(cublasGemmEx(p->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k,
            &alpha, w, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, out, CUDA_R_16F, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
extern "C" int venturi_q5_backend_version() { return 1; }
extern "C" int venturi_q5_cublas_run(
    void* opaque, const void* x, const void* codes, const void* high,
    const void* scales, void* out, int tokens, int n, int k, int padded,
    void* workspace, std::size_t available, int chunk, void* raw_stream) {
    try {
        auto* p = static_cast<Linear*>(opaque);
        require(VENTURI_BLOCKS_ARCH == 70 && p && p->dtype == 0,
                "Q5 cuBLAS requires SM70 FP16");
        require(x && codes && high && scales && out && workspace && tokens > 0 &&
                tokens <= 65535 && n > 0 && k > 0 && padded >= k && padded % 128 == 0 &&
                chunk > 0 && chunk <= n && chunk <= 65535,
                "invalid Q5 cuBLAS geometry");
        const std::size_t bytes = (std::size_t(chunk) * k * 2 + 255) / 256 * 256;
        require(available >= bytes + 4 * 1024 * 1024 &&
                reinterpret_cast<std::uintptr_t>(workspace) % 256 == 0,
                "Q5 cuBLAS workspace too small or misaligned");
        auto stream = static_cast<cudaStream_t>(raw_stream);
        check(cublasSetStream(p->handle, stream));
        check(cublasSetWorkspace(p->handle, static_cast<char*>(workspace) + bytes, available - bytes));
        const float alpha = 1.f, beta = 0.f;
        for (int begin = 0; begin < n; begin += chunk) {
            const int rows = std::min(chunk, n - begin);
            const int grid = static_cast<int>(std::min<std::uint64_t>(65535,
                (std::uint64_t(rows) * (padded / 2) + 255) / 256));
            q5_cublas_decode<<<grid,256,0,stream>>>(
                static_cast<const unsigned char*>(codes), static_cast<const unsigned char*>(high),
                static_cast<const __half*>(scales), static_cast<__half*>(workspace),
                begin, rows, k, padded);
            check(cudaGetLastError());
            // FP32 accumulation; unlike PEDANTIC this permits Tensor Cores.
            // ldc=n writes into the full output with no staging/copy kernel.
            check(cublasGemmEx(p->handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, k,
                &alpha, workspace, CUDA_R_16F, k, x, CUDA_R_16F, k,
                &beta, static_cast<__half*>(out) + begin, CUDA_R_16F, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        return 0;
    } catch (const std::exception& e) { error_text = e.what(); return -1; }
}
