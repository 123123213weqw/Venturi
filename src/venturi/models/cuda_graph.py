"""Opt-in position-specialized FP16 decode graphs, with unchanged eager math."""

from dataclasses import dataclass
from time import perf_counter

import torch

from .gpt_neox import GPTNeoXSession, validate_token_ids


@dataclass
class _DecodeGraph:
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
    prepared. All execution/reset follows the eager session's stream ownership.
    Forward logits are borrowed until this position's graph is replayed again.
    """

    MAX_DECODE_GRAPHS = 256

    def __init__(self, model, capacity, *, decode_start, decode_steps):
        if model.device.type != "cuda" or model.dtype != torch.float16:
            raise ValueError("CUDA graph decode requires CUDA FP16 learned weights")
        if type(decode_start) is not int or decode_start < 1:
            raise ValueError("decode_start must be a positive prompt length")
        if type(decode_steps) is not int or not 0 <= decode_steps <= self.MAX_DECODE_GRAPHS:
            raise ValueError("decode_steps must be an integer in [0,256]")
        if type(capacity) is not int or decode_start + decode_steps > capacity:
            raise ValueError("Prepared decode range exceeds session capacity")
        started = perf_counter()
        super().__init__(model, capacity)
        self.decode_start, self.decode_steps = decode_start, decode_steps
        self._graphs = {}
        self._last_graph = None
        owner = torch.cuda.current_stream(model.device)
        preparation = torch.cuda.Stream(device=model.device)
        preparation.wait_stream(owner)
        try:
            with torch.cuda.stream(preparation), torch.inference_mode(), torch.autocast("cuda", enabled=False):
                # Capture setup must not read uninitialized prefix cache values.
                self._keys.zero_()
                self._values.zero_()
                for position in range(decode_start, decode_start + decode_steps):
                    ids = torch.zeros(1, dtype=torch.long, device=model.device)
                    for _ in range(3 if position == decode_start else 1):
                        self._position = position
                        self._forward_tensor(ids, False)
                    self._position = position
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=preparation):
                        logits = self._forward_tensor(ids, False)
                        # A single host read returns either the greedy token or
                        # an error sentinel; no .item()/CPU branch in capture.
                        checked = torch.where(
                            torch.isfinite(logits).all(), logits.argmax(), -1
                        )
                    self._graphs[position] = _DecodeGraph(graph, ids, logits, checked)
            owner.wait_stream(preparation)
            owner.synchronize()
        except Exception:
            # Finish preparation before captured tensor owners can be freed.
            preparation.synchronize()
            raise
        self._position = 0
        self.preparation_seconds = perf_counter() - started

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
        if len(token_ids) != 1 or self.position < self.decode_start:
            return super().forward(token_ids, all_logits=all_logits)
        if self.position not in self._graphs:
            raise ValueError("Decode position is not prepared; no implicit graph capture")
        entry = self._graphs[self.position]
        try:
            with torch.inference_mode():
                entry.input_ids.fill_(token_ids[0])
                entry.graph.replay()
            self._position += 1
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
