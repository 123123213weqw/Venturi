"""Persistent, opt-in Qwen execution through the external native NInfer backend.

No HTTP server, implicit downloads, weight conversion or silent fallback.
One owned subprocess loads one resident model; calls are sequential. Qwen math,
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
        prefill_chunk=128,
        cuda_graph=True,
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
        _integer(prefill_chunk, "prefill_chunk", 1, max_context)
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
            prefill_chunk=prefill_chunk,
            cuda_graph=cuda_graph,
            kv_dtype=kv_dtype,
            speculative=speculative,
            draft_tokens=draft_tokens,
            verify_tree=verify_tree,
            optimized_proposal=optimized_proposal,
            device=device,
        )
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
            if self.ready.get("event") != "ready" or self.ready.get("protocol") != 1:
                raise RuntimeError("Incompatible native runner protocol")
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

    def generate(
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
        started = perf_counter()
        result = self._exchange(request)
        result["client_request_seconds"] = perf_counter() - started
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
