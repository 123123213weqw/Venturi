"""Opt-in exact-shape FP16 prefill/decode graphs, with unchanged eager math."""

from dataclasses import dataclass
from time import perf_counter

import torch

from .gpt_neox import GPTNeoXSession
from ..text.tokens import validate_token_ids


@dataclass
class _CapturedForward:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    logits: torch.Tensor
    checked_token: torch.Tensor


class GPTNeoXCUDAGraphSession(GPTNeoXSession):
    """One private graph per exact KV length, not a padded attention approximation.

    Preparation is expensive and must run without other CUDA work in this
    process. Reuse this session across requests whose decode positions fit the
    prepared range. Graphs have independent private pools, so EOS/reset cannot
    violate shared-pool replay ordering. At most 256 decode positions are
    prepared. Up to eight exact prompt lengths can also be prepared; other
    lengths, cached multi-token chunks and all-logits prefills use eager math.
    All execution/reset follows the eager session's stream ownership. Forward
    logits are borrowed until the same graph is replayed again.
    """

    MAX_DECODE_GRAPHS = 256
    MAX_PREFILL_GRAPHS = 8

    def __init__(self, model, capacity, *, decode_start, decode_steps, prefill_lengths=()):
        if model.device.type != "cuda" or model.dtype != torch.float16:
            raise ValueError("CUDA graph decode requires CUDA FP16 learned weights")
        if type(decode_start) is not int or decode_start < 1:
            raise ValueError("decode_start must be a positive prompt length")
        if type(decode_steps) is not int or not 0 <= decode_steps <= self.MAX_DECODE_GRAPHS:
            raise ValueError("decode_steps must be an integer in [0,256]")
        if type(capacity) is not int or decode_start + decode_steps > capacity:
            raise ValueError("Prepared decode range exceeds session capacity")
        if (
            not isinstance(prefill_lengths, (tuple, list))
            or len(prefill_lengths) > self.MAX_PREFILL_GRAPHS
            or any(type(n) is not int or not 1 <= n <= capacity for n in prefill_lengths)
            or len(set(prefill_lengths)) != len(prefill_lengths)
        ):
            raise ValueError("prefill_lengths must contain at most eight distinct lengths fitting capacity")
        started = perf_counter()
        super().__init__(model, capacity)
        self.decode_start, self.decode_steps = decode_start, decode_steps
        self._graphs = {}
        self._prefill_graphs = {}
        self._last_graph = None
        owner = torch.cuda.current_stream(model.device)
        preparation = torch.cuda.Stream(device=model.device)
        preparation.wait_stream(owner)
        try:
            with torch.cuda.stream(preparation), torch.inference_mode(), torch.autocast("cuda", enabled=False):
                # Capture setup must not read uninitialized prefix cache values.
                self._keys.zero_()
                self._values.zero_()
                for length in prefill_lengths:
                    ids = torch.zeros(length, dtype=torch.long, device=model.device)
                    self._prefill_graphs[length] = self._capture(ids, 0, 3, preparation)
                for position in range(decode_start, decode_start + decode_steps):
                    ids = torch.zeros(1, dtype=torch.long, device=model.device)
                    self._graphs[position] = self._capture(
                        ids, position, 3 if position == decode_start else 1, preparation
                    )
            owner.wait_stream(preparation)
            owner.synchronize()
        except Exception:
            # Finish preparation before captured tensor owners can be freed.
            preparation.synchronize()
            raise
        self._position = 0
        self.preparation_seconds = perf_counter() - started

    def _capture(self, ids, position, warmup, stream):
        for _ in range(warmup):
            self._position = position
            self._forward_tensor(ids, False)
        self._position = position
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = self._forward_tensor(ids, False)
            # One host read returns the greedy token or an error sentinel.
            checked = torch.where(torch.isfinite(logits).all(), logits.argmax(), -1)
        return _CapturedForward(graph, ids, logits, checked)

    def validate_request(self, prompt_length, max_new_tokens):
        if max_new_tokens <= 1:
            return
        if (
            prompt_length < self.decode_start
            or prompt_length + max_new_tokens - 1 > self.decode_start + self.decode_steps
        ):
            raise ValueError("Request does not fit this prepared CUDA graph decode range")

    def reset(self):
        super().reset()
        self._last_graph = None

    def forward(self, token_ids, *, all_logits=False):
        validate_token_ids(token_ids, self.model.config.vocab_size)
        if type(all_logits) is not bool:
            raise ValueError("all_logits must be boolean")
        self._check_stream()
        if self._failed:
            raise RuntimeError("Session failed during execution; reset it before reuse")
        if self.position + len(token_ids) > self.capacity:
            raise ValueError("Input would exceed the session KV capacity")
        self._last_graph = None
        prefill = self.position == 0 and not all_logits
        entry = self._prefill_graphs.get(len(token_ids)) if prefill else None
        if entry is None and (len(token_ids) != 1 or self.position < self.decode_start):
            return super().forward(token_ids, all_logits=all_logits)
        if entry is None:
            if self.position not in self._graphs:
                raise ValueError("Decode position is not prepared; no implicit graph capture")
            entry = self._graphs[self.position]
        try:
            with torch.inference_mode():
                if prefill:
                    # Blocking host copy avoids reusing pinned staging memory
                    # while a previous asynchronous transfer is still in flight.
                    host_ids = torch.tensor(token_ids, dtype=torch.long, device="cpu")
                    entry.input_ids.copy_(host_ids, non_blocking=False)
                else:
                    entry.input_ids.fill_(token_ids[0])
                entry.graph.replay()
            self._position += len(token_ids)
            self._last_graph = entry
            return entry.logits.unsqueeze(0) if all_logits else entry.logits
        except Exception:
            self._failed = True
            raise

    def select_token(self, logits):
        from ..generation import _greedy_token

        self._check_stream()
        if self._failed:
            raise RuntimeError("Session failed during execution; reset it before reuse")
        if self._last_graph is None or logits is not self._last_graph.logits:
            return _greedy_token(logits)
        token = int(self._last_graph.checked_token.item())
        if token < 0:
            self._failed = True
            raise RuntimeError("Model produced nonfinite logits; no token was selected")
        return token
