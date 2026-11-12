// VR engine-level GEMM dispatch: architecture-neutral.
// Included inside namespace vr by the architecture adapter's runtime header.
// Given a weight matrix in any registered row codec and FP16 activations,
// dispatches to the correct compute kernel. Zero adapter dependency — any
// model architecture can call this directly.
//
// Paths handled:
//   q4 fused      (bits==4, !resident)   → warp-coalesced fused GEMM kernel
//   FP16 shadow   (bits∈[4,6], resident) → single cuBLAS dense GEMM
//   q5/q6 chunked (bits∈[5,6], !resident) → chunked dequant + dense GEMM
//   Everything else returns 1 (caller uses its own dispatch).
#pragma once

extern "C" {
int venturi_q4_fused_run(const void* x,const void* codes,const void* scales,void* out,int t,int rows,int k,int padded,int dtype,void* stream);
int venturi_row_dequant_rows(const void* codes,const void* high,const void* scales,void* out,int row,int rows,int k,int padded,int bits,void* stream);
int venturi_dense_run(void* opaque,const void* x,const void* w,void* out,void* workspace,std::size_t workspace_bytes,void* stream);
}

// Caller-provided compute resources (not owned by this struct).
struct VRGemm {
    void* dense = nullptr;               // opaque dense-GEMM plan handle
    void* blas_workspace = nullptr;
    std::uint64_t blas_bytes = 0;
    void* dequant_scratch = nullptr;     // chunked dequant [chunk_rows, k]
    void* chunk_scratch = nullptr;       // chunked GEMM output [t, chunk_rows]
    int chunk_rows = 8192;
    int weight_bits = 8;                 // 4/5/6/8
    int dtype = 0;                       // 0=FP16, 1=BF16
    bool shadow = false;                 // weights already FP16 in VRAM?
};

// Returns 0=handled, 1=not handled (caller falls back), <0=error.
inline int gemm_project(
    const VRGemm& g,
    const void* x, const void* codes, const void* high, const void* scales,
    void* out, int t, int rows, int k, int padded, void* stream) {

    if (g.weight_bits == 4 && !g.shadow)
        return venturi_q4_fused_run(x, codes, scales, out, t, rows, k, padded,
                                    g.dtype, stream);

    if (g.shadow && g.weight_bits >= 4 && g.weight_bits <= 6) {
        int* plan = static_cast<int*>(g.dense);
        plan[2] = t; plan[3] = rows; plan[4] = k;
        return venturi_dense_run(g.dense, x, codes, out,
                                  g.blas_workspace, g.blas_bytes, stream);
    }

    if (g.weight_bits >= 5 && g.weight_bits <= 6 && !g.shadow) {
        for (int begin = 0; begin < rows; begin += g.chunk_rows) {
            const int n = std::min(g.chunk_rows, rows - begin);
            int rc = venturi_row_dequant_rows(codes, high, scales,
                                              g.dequant_scratch, begin, n, k,
                                              padded, g.weight_bits, stream);
            if (rc != 0) return rc;
            int* plan = static_cast<int*>(g.dense);
            plan[2] = t; plan[3] = n; plan[4] = k;
            rc = venturi_dense_run(g.dense, x, g.dequant_scratch, g.chunk_scratch,
                                   g.blas_workspace, g.blas_bytes, stream);
            if (rc != 0) return rc;
            for (int tok = 0; tok < t; ++tok)
                if (cudaMemcpyAsync(
                        static_cast<char*>(out) + (std::uint64_t)(tok * rows + begin) * 2,
                        static_cast<const char*>(g.chunk_scratch) + (std::uint64_t)tok * n * 2,
                        (std::uint64_t)n * 2, cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream)) != cudaSuccess)
                    return -1;
        }
        return 0;
    }

    return 1;
}

