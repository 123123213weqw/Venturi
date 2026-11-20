"""Persistent, opt-in Qwen execution through the external native NInfer backend.

No HTTP server, implicit downloads, weight conversion or silent fallback.
One owned subprocess loads one resident model; RPCs are sequential, while a
generate_many call can batch native requests on one GPU. Qwen math,
CUDA kernels and speculative verification belong to the attributed dependency.
"""

import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import tempfile
import threading
from time import monotonic, perf_counter

from ..ninfer import NInferArtifact


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


class QwenNativeSession:
    def __init__(
        self,
        checkpoint,
        *,
        runner,
        max_context=2048,
        mode="single",
        max_concurrency=None,
        kv_capacity=None,
        prefill_chunk=128,
        cuda_graph=True,
        cpu_wait="blocking",
        kv_dtype="bf16",
        speculative="none",
        draft_tokens=8,
        verify_tree=False,
        optimized_proposal=False,
        device=0,
        timeout=300.0,
    ):
        self.artifact = NInferArtifact(checkpoint)
        components = self.artifact.manifest["components"]
        config = components["text"]["config"]
        if config.get("architectures") != ["Qwen3_5ForCausalLM"]:
            raise ValueError(
                "This backend explicitly supports dense Qwen3_5ForCausalLM text execution"
            )
        model_context = _integer(
            config.get("max_position_embeddings"), "model context", 16, 2**32 - 1
        )
        _integer(config.get("vocab_size"), "model vocabulary", 1, 2**31 - 1)
        _integer(max_context, "max_context", 16, model_context)
        if mode not in ("single", "parallel"):
            raise ValueError("mode must be single or parallel")
        if max_concurrency is None:
            max_concurrency = 1 if mode == "single" else 4
        _integer(max_concurrency, "max_concurrency", 1, 8)
        if mode == "parallel" and max_concurrency == 1:
            raise ValueError("parallel mode requires max_concurrency >= 2")
        pages = (max_context + 63) // 64
        if kv_capacity is None:
            kv_capacity = max_concurrency * pages * 64
        _integer(kv_capacity, "kv_capacity", max_context, max_concurrency * pages * 64)
        if (kv_capacity + 63) // 64 < max(pages, max_concurrency):
            raise ValueError(
                "kv_capacity must reserve at least one 64-token page per lane"
            )
        _integer(prefill_chunk, "prefill_chunk", 1, max_context)
        if cpu_wait not in ("auto", "blocking"):
            raise ValueError("cpu_wait must be auto or blocking")
        _integer(device, "device", 0, 128)
        _integer(draft_tokens, "draft_tokens", 1, 15)
        for name, value in (
            ("cuda_graph", cuda_graph),
            ("verify_tree", verify_tree),
            ("optimized_proposal", optimized_proposal),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        if kv_dtype not in ("bf16", "k8v4") or speculative not in ("none", "dflash2"):
            raise ValueError("Unsupported KV or speculative mode")
        if (
            speculative == "dflash2"
            and components.get("dflash2", {}).get("target") != "text"
        ):
            raise ValueError("Checkpoint lacks a DFlash2 component paired with text")
        if speculative == "none" and (verify_tree or optimized_proposal):
            raise ValueError("Tree/proposal options require DFlash2")
        if verify_tree and max_concurrency != 1:
            raise ValueError("verify-tree requires max_concurrency=1")
        if verify_tree and (
            draft_tokens != 15
            or kv_dtype != "k8v4"
            or config.get("num_attention_heads") != 24
            or config.get("head_dim") != 256
        ):
            raise ValueError(
                "This native verify-tree route requires K=15, explicit k8v4 KV and 24x256 attention"
            )
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        executable = Path(runner).expanduser().resolve(strict=True)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("runner must be an executable native backend")
        self.config = dict(
            max_context=max_context,
            max_concurrency=max_concurrency,
            kv_capacity=kv_capacity,
            pending_timeout_ms=max(
                1, min(2**32 - 1, math.ceil(min(timeout, (2**32 - 1) / 1000) * 1000))
            ),
            prefill_chunk=prefill_chunk,
            cuda_graph=cuda_graph,
            cpu_wait=cpu_wait,
            kv_dtype=kv_dtype,
            speculative=speculative,
            draft_tokens=draft_tokens,
            verify_tree=verify_tree,
            optimized_proposal=optimized_proposal,
            device=device,
        )
        self._mode = mode
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._pending = bytearray()
        self._closed = False
        self._stderr = tempfile.TemporaryFile()
        self._selector = selectors.DefaultSelector()
        # Research flags can change arithmetic/weight representation. Do not
        # inherit them invisibly from the launching shell.
        environment = {
            k: v for k, v in os.environ.items() if not k.startswith("NINFER_")
        }
        try:
            self._process = subprocess.Popen(
                [str(executable), str(self.artifact.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=environment,
                bufsize=0,
            )
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self.ready = self._exchange(self.config)
            if self.ready.get("event") != "ready" or self.ready.get("protocol") != 3:
                raise RuntimeError(
                    "Incompatible native runner protocol; rebuild the protocol-3 runner"
                )
            if self.ready.get("cpu_wait") != cpu_wait:
                raise RuntimeError(
                    "Native runner did not apply the requested CPU wait policy"
                )
            limits = self.ready.get("limits", {})
            if (
                limits.get("max_concurrency") != max_concurrency
                or limits.get("kv_capacity") != ((kv_capacity + 63) // 64) * 64
            ):
                raise RuntimeError(
                    "Native runner did not reserve the requested concurrency/KV capacity"
                )
        except Exception:
            self.close()
            raise

    def _diagnostics(self):
        size = os.fstat(self._stderr.fileno()).st_size
        return os.pread(
            self._stderr.fileno(), min(size, 8192), max(0, size - 8192)
        ).decode("utf-8", errors="replace")

    def _exchange(self, request):
        if self._closed:
            raise RuntimeError("Qwen session is closed")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen sessions accept one sequential request at a time")
        try:
            encoded = (json.dumps(request, allow_nan=False) + "\n").encode()
            if len(encoded) > 16 * 2**20:
                raise ValueError("Native request exceeds 16 MiB")
            view = memoryview(encoded)
            while view:
                written = self._process.stdin.write(view)
                if not written:
                    raise RuntimeError("Native input pipe closed")
                view = view[written:]
            self._process.stdin.flush()
            deadline = monotonic() + self.timeout
            while b"\n" not in self._pending:
                remaining = deadline - monotonic()
                if remaining <= 0 or not self._selector.select(remaining):
                    raise TimeoutError("Qwen native response timed out")
                block = os.read(self._process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError(
                        f"Qwen native process exited: {self._diagnostics()}"
                    )
                self._pending.extend(block)
                if len(self._pending) > 32 * 2**20:
                    raise RuntimeError("Native response exceeds 32 MiB")
            line, _, remainder = self._pending.partition(b"\n")
            self._pending = bytearray(remainder)
            result = json.loads(line)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise RuntimeError("Malformed native response")
            if not result["ok"]:
                if not result.get("available", False):
                    self.close()
                raise ValueError(
                    f"Qwen native error: {result.get('error', 'unspecified')}"
                )
            return result
        except (OSError, RuntimeError, json.JSONDecodeError, UnicodeError) as error:
            diagnostics = self._diagnostics()
            if self._process.poll() is None:
                self._process.terminate()
            self.close()
            raise RuntimeError(
                f"Native backend failed: {error}\n{diagnostics}"
            ) from error
        finally:
            self._lock.release()

    def tokenize(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return self._exchange({"op": "tokenize", "prompt": text})["token_ids"]

    def _generation_request(
        self,
        prompt=None,
        *,
        token_ids=None,
        max_new_tokens=32,
        raw_prompt=False,
        ignore_eos=False,
        profile=False,
    ):
        _integer(max_new_tokens, "max_new_tokens", 1, self.config["max_context"])
        if (prompt is None) == (token_ids is None):
            raise ValueError("Provide exactly one of prompt or token_ids")
        for name, value in (
            ("raw_prompt", raw_prompt),
            ("ignore_eos", ignore_eos),
            ("profile", profile),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        request = dict(
            op="generate",
            max_new_tokens=max_new_tokens,
            raw_prompt=raw_prompt,
            ignore_eos=ignore_eos,
            profile=profile,
        )
        if token_ids is not None:
            if not isinstance(token_ids, (tuple, list)) or not token_ids:
                raise ValueError("token_ids must be a nonempty list/tuple")
            vocab = self.artifact.manifest["components"]["text"]["config"]["vocab_size"]
            for token in token_ids:
                _integer(token, "token ID", 0, vocab - 1)
            if len(token_ids) + max_new_tokens > self.config["max_context"]:
                raise ValueError("Prompt plus output exceeds max_context")
            request["token_ids"] = token_ids
        else:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            request["prompt"] = prompt
        return request

    def _validate_generation(self, result, max_new_tokens):
        if not isinstance(result, dict):
            self.close()
            raise RuntimeError("Malformed native generation result")
        ids = result.get("generated_ids")
        if (
            not isinstance(ids, list)
            or len(ids) > max_new_tokens
            or any(
                type(t) is not int
                or not 0
                <= t
                < self.artifact.manifest["components"]["text"]["config"]["vocab_size"]
                for t in ids
            )
        ):
            self.close()
            raise RuntimeError("Native response contains invalid generated IDs")
        if result.get("reused_prompt_tokens") != 0:
            self.close()
            raise RuntimeError("Unexpected prefix reuse in isolated-request execution")
        return result

    @property
    def mode(self):
        return self._mode

    def _check_mode(self, mode):
        if mode not in ("single", "parallel"):
            raise ValueError("mode must be single or parallel")
        if mode == "parallel" and self.config["max_concurrency"] == 1:
            raise ValueError(
                "parallel mode needs max_concurrency >= 2 reserved at startup"
            )
        return mode

    def set_mode(self, mode):
        """Change idle dispatch policy; reserved GPU memory/limits do not change."""
        mode = self._check_mode(mode)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot switch mode during an active request")
        try:
            if self._closed:
                raise RuntimeError("Qwen session is closed")
            self._mode = mode
        finally:
            self._lock.release()

    def generate(self, prompt=None, **generation):
        """Generate one isolated request, regardless of the current batch mode."""
        request = self._generation_request(prompt, **generation)
        started = perf_counter()
        result = self._exchange(request)
        result["client_request_seconds"] = perf_counter() - started
        return self._validate_generation(result, request["max_new_tokens"])

    def generate_many(self, prompts, *, mode=None, **generation):
        """Execute 1..16 strings in order, serially or with native decode batching.

        One RPC owns the complete batch. This is not a concurrent-client API.
        All requests are prepared/validated before any native submission.
        """
        mode = self._check_mode(self.mode if mode is None else mode)
        if not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 16:
            raise ValueError("prompts must be a list/tuple of 1..16 strings")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must contain only strings")
        requests = [
            self._generation_request(prompt, **generation) for prompt in prompts
        ]
        started = perf_counter()
        result = self._exchange(dict(op="generate_many", mode=mode, requests=requests))
        result["client_request_seconds"] = perf_counter() - started
        results = result.get("results")
        if (
            result.get("mode") != mode
            or not isinstance(results, list)
            or len(results) != len(requests)
            or any(
                not isinstance(r, dict)
                or type(r.get("index")) is not int
                or r["index"] != i
                for i, r in enumerate(results)
            )
        ):
            self.close()
            raise RuntimeError("Malformed native batch result/order")
        for item, request in zip(results, requests):
            self._validate_generation(item, request["max_new_tokens"])
        seconds = result.get("wall_seconds")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            self.close()
            raise RuntimeError("Malformed native batch wall time")
        result["output_tokens_per_second"] = (
            sum(len(r["generated_ids"]) for r in results) / seconds
        )
        return result

    def benchmark_many(self, prompts, *, mode=None, warmup=1, repeats=3, **generation):
        """Batch throughput uses elapsed wall time, never summed overlapping timings."""
        _integer(warmup, "warmup", 0, 10000)
        _integer(repeats, "repeats", 1, 10000)
        mode = self._check_mode(self.mode if mode is None else mode)
        for _ in range(warmup):
            self.generate_many(prompts, mode=mode, **generation)
        runs = [
            self.generate_many(prompts, mode=mode, **generation) for _ in range(repeats)
        ]
        tokens = sum(len(r["generated_ids"]) for run in runs for r in run["results"])
        return {
            "backend": "external-ninfer-cinference",
            "ready": self.ready,
            "mode": mode,
            "warmup": warmup,
            "repeats": repeats,
            "runs": runs,
            "output_tokens_per_second": tokens / sum(r["wall_seconds"] for r in runs),
            "client_output_tokens_per_second": tokens
            / sum(r["client_request_seconds"] for r in runs),
            "median_first_token_seconds": statistics.median(
                r["timings"]["first_token_seconds"]
                for run in runs
                for r in run["results"]
            ),
            "identical_generated_ids": all(
                [r["generated_ids"] for r in run["results"]]
                == [r["generated_ids"] for r in runs[0]["results"]]
                for run in runs
            ),
            "scope": "One resident model; bounded offline request list, prefix reuse disabled. All output tokens / elapsed batch wall time (including preparation, queueing and prefill); startup/warmup excluded. Per-request timings overlap in parallel mode; throughput is not per-stream speed.",
        }

    def benchmark(self, prompt, *, warmup=1, repeats=3, **generation):
        _integer(warmup, "warmup", 0, 10000)
        _integer(repeats, "repeats", 1, 10000)
        for _ in range(warmup):
            self.generate(prompt, **generation)
        runs = [self.generate(prompt, **generation) for _ in range(repeats)]
        tokens = sum(max(0, len(r["generated_ids"]) - 1) for r in runs)
        decode_seconds = sum(r["timings"]["decode_seconds"] for r in runs)
        wall_seconds = sum(r["timings"]["generation_wall_seconds"] for r in runs)
        return {
            "backend": "external-ninfer-cinference",
            "ready": self.ready,
            "warmup": warmup,
            "repeats": repeats,
            "runs": runs,
            "decode_tokens_per_second": tokens / decode_seconds
            if tokens and decode_seconds > 0
            else None,
            "generation_wall_tokens_per_second": tokens / wall_seconds
            if tokens and wall_seconds > 0
            else None,
            "median_first_token_seconds": statistics.median(
                r["timings"]["first_token_seconds"] for r in runs
            ),
            "identical_generated_ids": all(
                r["generated_ids"] == runs[0]["generated_ids"] for r in runs
            ),
            "scope": "Resident model, sequential requests, prefix reuse disabled. Startup excluded; first token excluded from decode numerator. Native execution timings and generation-wall timings are separate. Upstream backend kernels and speculative algorithm, not a new Venturi CUDA implementation.",
        }

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = getattr(self, "_process", None)
        try:
            if process is not None:
                if process.stdin:
                    process.stdin.close()
                if process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired as error:
                                raise RuntimeError(
                                    f"Owned native PID {process.pid} has not exited after SIGKILL"
                                ) from error
        finally:
            if process is not None and process.stdout:
                process.stdout.close()
            self._selector.close()
            self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
