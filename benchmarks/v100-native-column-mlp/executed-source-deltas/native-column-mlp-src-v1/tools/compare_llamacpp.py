"""Compare dense Pythia generation with a pinned, separately built llama.cpp.

Only the benchmark worker imports Venturi/PyTorch. Engines run in separate,
sequential processes; no model downloads or compilation are performed here.
"""

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(2**20):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(report):
    runs = report["runs"]
    decode_tokens = sum(run["decode_tokens"] for run in runs)
    decode_time = sum(run["decode_seconds"] for run in runs)
    result = {
        "decode_tokens_per_second": decode_tokens / decode_time if decode_tokens else None,
        "request_tokens_per_second": sum(run["output_tokens"] for run in runs)
        / sum(run["total_seconds"] for run in runs),
        "identical_generated_ids": all(
            run["generated_ids"] == runs[0]["generated_ids"] for run in runs
        ),
    }
    for key in ("time_to_first_token_seconds", "total_seconds"):
        values = [run[key] for run in runs]
        result[key] = {"min": min(values), "median": median(values), "max": max(values)}
    return result


def check_gpu_processes(text, allowed):
    processes = [
        {"pid": int(row[0]), "name": row[1].strip(), "memory": row[2].strip()}
        for row in csv.reader(text.splitlines()) if row
    ]
    if any(process["name"] not in allowed for process in processes):
        raise RuntimeError("GPU has active compute processes; retry when it is idle")
    return processes


class ProcessMemory:
    """Best-effort NVML per-process resident memory samples, not allocator peaks."""

    class Info(ctypes.Structure):
        _fields_ = [
            ("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
            ("gpu_instance", ctypes.c_uint), ("compute_instance", ctypes.c_uint),
        ]

    def __init__(self):
        self.library = None
        self.error = None
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1")
            lib.nvmlInit_v2.restype = ctypes.c_int
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            lib.nvmlDeviceGetComputeRunningProcesses_v2.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(self.Info),
            ]
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit_v2 failed")
            self.device = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.device)) != 0:
                raise RuntimeError("NVML device 0 unavailable")
            self.library = lib
        except (OSError, AttributeError, RuntimeError) as error:
            self.error = str(error)

    def sample(self, pid):
        if self.library is None:
            return None
        entries = (self.Info * 128)()
        count = ctypes.c_uint(len(entries))
        result = self.library.nvmlDeviceGetComputeRunningProcesses_v2(
            self.device, ctypes.byref(count), entries
        )
        if result != 0:
            self.error = f"NVML process query returned {result}"
            return None
        for entry in entries[:count.value]:
            if entry.pid == pid and entry.used != 2**64 - 1:
                return entry.used
        return None

    def close(self):
        if self.library is not None:
            self.library.nvmlShutdown()
            self.library = None


def run_process(command, log, *, env=None, timeout=180):
    monitor = ProcessMemory()
    samples = []
    with Path(log).open("w") as file:
        child = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT, env=env)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Timed out after {timeout}s: {command[0]}")
                value = monitor.sample(child.pid)
                if value is not None:
                    samples.append(value)
                time.sleep(0.02)
            if child.returncode != 0:
                raise RuntimeError(f"Process failed ({child.returncode}); see {log}")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            monitor.close()
    return {
        "max_sampled_resident_bytes": max(samples) if samples else None,
        "samples": len(samples), "interval_seconds": 0.02, "error": monitor.error,
        "scope": "NVML per-process GPU 0 resident bytes including CUDA context; sampled maximum, not exact transient peak; includes loading and untimed teacher-forced validation",
    }


def venturi_worker(args):
    import platform

    import numpy as np
    import torch

    from venturi import load_checkpoint
    from venturi.generation import LocalTokenizer, _greedy_token
    from venturi.models import GPTNeoXModel

    request = json.loads(Path(args.request).read_text())
    torch.set_num_threads(request["threads"])
    checkpoint = load_checkpoint(args.checkpoint, require_float_dtype="F16")
    model = GPTNeoXModel(checkpoint)
    tokenizer = LocalTokenizer(
        Path(args.checkpoint) / "tokenizer.json", vocab_size=model.config.vocab_size
    )
    ids = tokenizer.encode(request["prompt"])
    if not ids:
        raise ValueError("Comparison requires a nonempty tokenized prompt")
    budget = request["max_new_tokens"]
    if args.venturi_execution == "cuda-graph":
        session = model.create_cuda_graph_session(
            len(ids) + budget, decode_start=len(ids), decode_steps=budget - 1
        )
    else:
        session = model.create_session(len(ids) + budget)
    select = getattr(session, "select_token", _greedy_token)
    runs = []
    for iteration in range(-request["warmup"], request["repeats"]):
        torch.cuda.current_stream(model.device).synchronize()
        started = time.perf_counter()
        session.reset()
        prefill = time.perf_counter()
        output = [select(session.forward(ids))]
        prefill_seconds = time.perf_counter() - prefill
        ttft = time.perf_counter() - started
        decode = time.perf_counter()
        while len(output) < budget:
            output.append(select(session.forward([output[-1]])))
        torch.cuda.current_stream(model.device).synchronize()
        decode_seconds = time.perf_counter() - decode
        total_seconds = time.perf_counter() - started
        if iteration >= 0:
            runs.append({
                "generated_ids": output, "prefill_seconds": prefill_seconds,
                "time_to_first_token_seconds": ttft, "decode_seconds": decode_seconds,
                "total_seconds": total_seconds, "output_tokens": len(output),
                "decode_tokens": len(output) - 1,
            })
    # Keep logit validation out of measured timing, using the same precision.
    teacher = runs[0]["generated_ids"][:min(31, budget - 1)]
    session.reset()
    rows = [session.forward(ids).float().cpu().numpy()]
    for token in teacher:
        rows.append(session.forward([token]).float().cpu().numpy())
    np.stack(rows).astype("<f4").tofile(args.logits)
    request.update(prompt_ids=list(ids), teacher_ids=teacher)
    write_json(args.request, request)
    write_json(args.report, {
        "engine": "Venturi", "prompt_ids": ids, "runs": runs,
        "execution": args.venturi_execution,
        "preparation_seconds": getattr(session, "preparation_seconds", 0.0),
        "vocab_size": model.config.vocab_size, "logit_rows": len(rows),
        "actual_context_capacity": session.capacity, "cache_dtype": "F16",
        "kv_cache_bytes": session.cache_bytes,
        "cuda_allocator_peak_bytes": torch.cuda.max_memory_allocated(model.device),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    })


def verify_gguf(checkpoint, gguf_path, llama_root):
    """Check every learned value after the documented QKV row permutation."""
    import re

    import numpy as np
    from safetensors import safe_open

    sys.path.insert(0, str(Path(llama_root) / "gguf-py"))
    import gguf

    config = json.loads((Path(checkpoint) / "config.json").read_text())
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GPTNEOX, config["num_hidden_layers"])
    reader = gguf.GGUFReader(gguf_path)
    converted = {tensor.name: tensor for tensor in reader.tensors}
    checked = set()
    dtypes = {}
    for file in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(file, framework="np") as source:
            for name in source.keys():
                if re.search(r"\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$", name):
                    continue
                target = names.get_name(name, try_suffixes=(".weight", ".bias"))
                if target not in converted or target in checked:
                    raise RuntimeError(f"Unmapped or duplicate learned tensor: {name}")
                actual = converted[target]
                if actual.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                    raise RuntimeError(f"Quantization unexpectedly present: {target}")
                expected = source.get_tensor(name)
                if ".attention.query_key_value." in name:
                    heads = config["num_attention_heads"]
                    dim = config["hidden_size"] // heads
                    shape = (heads, 3, dim) + expected.shape[1:]
                    grouped = expected.reshape(shape)
                    expected = np.concatenate([grouped[:, i].reshape((-1,) + expected.shape[1:]) for i in range(3)])
                if actual.data.shape != expected.shape or not np.array_equal(
                    actual.data.astype(np.float32), expected.astype(np.float32)
                ):
                    raise RuntimeError(f"Learned values changed during conversion: {name}")
                checked.add(target)
                dtype = str(actual.tensor_type.name)
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
    if checked != converted.keys():
        raise RuntimeError("GGUF contains unverified extra learned tensors")
    return {"passed": True, "learned_tensors_verified": len(checked), "gguf_dtype_counts": dtypes,
            "gguf_sha256": digest(gguf_path), "source_weights_sha256": {
                p.name: digest(p) for p in sorted(Path(checkpoint).glob("*.safetensors"))
            }}


def logit_diagnostics(actual_path, expected_path, vocab):
    import numpy as np

    actual = np.fromfile(actual_path, dtype="<f4").reshape(-1, vocab)
    expected = np.fromfile(expected_path, dtype="<f4").reshape(-1, vocab)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("Logit shape mismatch or nonfinite values")
    error = np.abs(actual - expected)
    return {
        "rows": len(actual), "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "argmax_agreement": float((actual.argmax(-1) == expected.argmax(-1)).mean()),
        "bit_exact": bool(np.array_equal(actual, expected)),
        "scope": "Diagnostic only, not a numerical equivalence pass. Equal learned values and F16 caches do not imply identical arithmetic/activation/logit precision.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--llama-root")
    parser.add_argument("--llama-runner")
    parser.add_argument("--llama-commit")
    parser.add_argument("--gguf")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--venturi-execution", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument(
        "--allow-gpu-process", action="append", default=[],
        help="Exact process path for a known background desktop service; recorded in report",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", help=argparse.SUPPRESS)
    parser.add_argument("--report", help=argparse.SUPPRESS)
    parser.add_argument("--logits", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        venturi_worker(args)
        return
    if not all((args.llama_root, args.llama_runner, args.llama_commit, args.gguf, args.output_dir)):
        parser.error("--llama-root, --llama-runner, --llama-commit, --gguf, --output-dir are required")
    if args.max_new_tokens < 1 or args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("Invalid benchmark sizes")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    request = out / "request.json"
    write_json(request, {
        "prompt": args.prompt, "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup, "repeats": args.repeats, "threads": args.threads,
    })
    conversion = verify_gguf(args.checkpoint, args.gguf, args.llama_root)
    gpu_before = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,power.draw,utilization.gpu",
        "--format=csv",
    ], text=True)
    other_processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader",
    ], text=True).strip()
    background_processes = check_gpu_processes(other_processes, args.allow_gpu_process)
    memory = {}
    memory["venturi"] = run_process([
        sys.executable, str(Path(__file__).resolve()), "--worker", "--checkpoint", args.checkpoint,
        "--request", str(request), "--report", str(out / "venturi.json"),
        "--logits", str(out / "venturi.f32"),
        "--venturi-execution", args.venturi_execution,
    ], out / "venturi.log")
    for label, graphs in (("llama", True), ("llama-no-graphs", False)):
        env = dict(os.environ)
        env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        if not graphs:
            env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        memory[label] = run_process([
            args.llama_runner, args.gguf, str(request),
            str(out / f"{label}.json"), str(out / f"{label}.f32"),
        ], out / f"{label}.log", env=env)
    reports = {key: json.loads((out / f"{key}.json").read_text()) for key in memory}
    comparison = {}
    for key in ("llama", "llama-no-graphs"):
        comparison[key] = {
            "greedy_ids_equal": reports[key]["runs"][0]["generated_ids"] == reports["venturi"]["runs"][0]["generated_ids"],
            "teacher_forced_logits": logit_diagnostics(
                out / f"{key}.f32", out / "venturi.f32", reports["venturi"]["vocab_size"]
            ),
        }
    report = {
        "schema_version": 1, "llama_commit": args.llama_commit,
        "venturi_execution": args.venturi_execution,
        "gpu_before": gpu_before,
        "allowed_background_processes": background_processes,
        "gpu_after": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,power.draw,utilization.gpu",
            "--format=csv",
        ], text=True),
        "llama_runner_binary_sha256": digest(args.llama_runner),
        "conversion": conversion, "request": json.loads(request.read_text()),
        "summaries": {key: summarize(value) for key, value in reports.items()},
        "correctness_diagnostics": comparison, "memory": memory,
        "runs": reports,
        "contract": {
            "precision": "Original learned values preserved exactly; GGUF F16 matrices/F32 vectors; F16 KV caches in both engines; activations/logits/reductions differ, not identical FP16 arithmetic",
            "timing": "Single unpadded sequence, all transformer layers/output offloaded, raw prompt/no inserted BOS, greedy selection with finite check, EOS ignored for fixed output lengths; first token excluded from decode count; context/KV allocated before warmup and reused with logical reset in BOTH engines. Not the fresh-KV-session Venturi CLI benchmark.",
            "exclusions": "Loading/tokenization/context allocation/process startup/teacher-forced validation excluded from timing; resident memory samples include loading and untimed validation. llama tokenization is independently checked against exact prompt IDs.",
            "order": ["venturi", "llama", "llama-no-graphs"],
            "venturi_preparation": "Graph capture, if selected, is excluded from warmed timings and explicitly reported by the worker; one graph per exact decode position, no padded KV approximation.",
            "limitations": "Tiny Pythia model only; no serving/network/batching, no Qwen/Q4/general engine superiority claim. Native llama-bench synthetic timings must not be mixed with these greedy timings.",
        },
    }
    write_json(out / "comparison.json", report)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
