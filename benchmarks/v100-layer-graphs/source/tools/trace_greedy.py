"""Small profiler-only greedy diagnostic. Never use its wall time as a benchmark."""

import argparse

import torch

from venturi.ops import CudaBackend, prepare_greedy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    backend = CudaBackend(args.library)
    for rows in (1, 8):
        x = torch.zeros(rows, 248320, dtype=torch.float32, device="cuda:0")
        x[:, 8192] = 1
        p = prepare_greedy(backend, rows=rows, vocab_size=x.shape[1], device=x.device)
        out = torch.empty(rows, 2, dtype=torch.int64, device=x.device)
        scratch = torch.empty(p.workspace_bytes, dtype=torch.uint8, device=x.device)
        for _ in range(5):
            p.run(x, out=out, workspace=scratch)
        torch.cuda.synchronize()
        for _ in range(10):
            with torch.cuda.nvtx.range(f"native_greedy_rows_{rows}"):
                p.run(x, out=out, workspace=scratch)
                pairs = out.cpu().tolist()
                assert pairs == [[8192, 0]] * rows
            with torch.cuda.nvtx.range(f"cpu_greedy_rows_{rows}"):
                value = x.cpu()
                assert bool(value.isfinite().all())
                assert value.argmax(1).tolist() == [8192] * rows
    print("Profiler diagnostic completed; no timing/performance gate")


if __name__ == "__main__":
    main()
