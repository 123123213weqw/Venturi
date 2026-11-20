"""Fresh-cohort Qwen adapter for the reusable measured-cost MTP policy.

Selection is once per generate call, never mid-transaction. Banks share immutable
weights, not KV/GDN/draft state. All token execution remains in existing native
entry points. No continuation, confidence head or automatic arithmetic change.
"""

import copy
import threading
from time import perf_counter, process_time

import torch

from ..adaptive_mtp import AdaptiveMTPPolicy, MTPWorkload
from ..ops.vocabulary import _ids
from .qwen_native import NativeQwenModel
from .qwen_mtp_native import NativeQwenMTPModel


class AdaptiveQwenMTPBatch:
    def __init__(self, model, capacities, *, policy=None):
        if (
            type(model) is not NativeQwenMTPModel
            or type(model.target) is not NativeQwenModel
            or model.dtype != torch.float16
            or model._mode != "resident"
        ):
            raise ValueError(
                "Adaptive MTP requires a resident FP16 single/TP target and MTP model, not PP"
            )
        if (
            not isinstance(capacities, (list, tuple))
            or not 1 <= len(capacities) <= min(16, model._max_tokens // 2)
            or any(
                type(c) is not int
                or not 0 < c <= model.config["max_position_embeddings"]
                for c in capacities
            )
        ):
            raise ValueError("Invalid adaptive batch capacities/slot count")
        if policy is not None and not isinstance(policy, AdaptiveMTPPolicy):
            raise TypeError("Explicit AdaptiveMTPPolicy required")
        self.model, self.capacities = model, tuple(capacities)
        self.policy = policy if policy is not None else AdaptiveMTPPolicy()
        self.supported = (0, 1, 2) if model._max_tokens >= 3 else (0, 1)
        self._banks, self._last = {}, None
        self._closed, self._failed = False, False
        self._lock = threading.RLock()

    def _check(self):
        if self._closed:
            raise RuntimeError("Adaptive MTP batch closed")
        if self._failed:
            raise RuntimeError("Adaptive MTP execution failed; explicit reset required")

    def prepare(self, depths=None):
        """Allocate selected banks before timing. Errors never switch strategy."""
        depths = self.supported if depths is None else depths
        if (
            not isinstance(depths, (list, tuple))
            or not depths
            or any(type(d) is not int or d not in self.supported for d in depths)
            or len(set(depths)) != len(depths)
        ):
            raise ValueError("Explicit supported unique depths required")
        with self._lock:
            self._check()
            created = []
            try:
                for depth in depths:
                    if depth in self._banks:
                        continue
                    if depth == 2:
                        bank = self.model.create_session(max(self.capacities))
                        try:
                            bank.prepare_prefix()
                        except BaseException:
                            bank.close()
                            raise
                    else:
                        bank = (
                            self.model.target if depth == 0 else self.model
                        ).create_batch(self.capacities)
                    self._banks[depth] = bank
                    created.append(depth)
            except BaseException:
                for depth in reversed(created):
                    self._banks.pop(depth).close()
                raise

    def _inputs(self, prompts, maximum, eos, slots):
        if not isinstance(prompts, (tuple, list)) or not prompts:
            raise ValueError("Nonempty tokenized prompts required")
        slots = list(range(len(prompts))) if slots is None else slots
        if (
            not isinstance(slots, (tuple, list))
            or len(slots) != len(prompts)
            or any(
                type(s) is not int or not 0 <= s < len(self.capacities) for s in slots
            )
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("One unique valid slot per prompt required")
        maximum = [maximum] * len(prompts) if type(maximum) is int else maximum
        if (
            not isinstance(maximum, (list, tuple))
            or len(maximum) != len(prompts)
            or any(type(n) is not int or not 0 < n < 2**31 for n in maximum)
        ):
            raise ValueError("Positive int32 budget per request required")
        if not isinstance(eos, (list, tuple)):
            raise ValueError("Explicit EOS lists required")
        eos = eos if eos and isinstance(eos[0], (list, tuple)) else [eos] * len(prompts)
        if len(eos) != len(prompts):
            raise ValueError("One EOS list per request required")
        for prompt, count, stop, slot in zip(prompts, maximum, eos, slots):
            _ids(prompt, self.model.config["vocab_size"])
            if len(prompt) + count > self.capacities[slot]:
                raise ValueError("Adaptive generation exceeds slot capacity")
            if not isinstance(stop, (list, tuple)):
                raise ValueError("Explicit EOS lists required")
            if stop:
                _ids(stop, self.model.config["vocab_size"])
        return list(slots), list(maximum), [list(e) for e in eos]

    def generate(
        self,
        prompts,
        *,
        max_new_tokens,
        eos_token_ids=(),
        slots=None,
        strategy=None,
        learn=True,
    ):
        """Auto or explicit depth 0/1/2, always fresh; learned state survives reset.

        Depth two is scalar prefix recovery only. No serialized MTP2 fallback is
        presented as batched MTP2. Learning excludes bank allocation but includes
        selected-bank reset, native generation and metadata collection.
        """
        if type(learn) is not bool:
            raise ValueError("learn must be boolean")
        slots, budgets, eos = self._inputs(
            prompts, max_new_tokens, eos_token_ids, slots
        )
        workload = MTPWorkload(tuple(map(len, prompts)), tuple(budgets), any(eos))
        with self._lock:
            self._check()
            self._last = None
            began = perf_counter()
            decision = self.policy.choose(
                workload, supported=self.supported, override=strategy
            )
            depth = decision.depth
            prepare = perf_counter()
            self.prepare([depth])
            preparation = perf_counter() - prepare
            bank = self._banks[depth]
            start, cpu = perf_counter(), process_time()
            try:
                if depth == 2:
                    bank.reset()
                    output = [
                        bank.generate(
                            prompts[0],
                            max_new_tokens=budgets[0],
                            eos_token_ids=eos[0],
                            draft_tokens=2,
                            recovery="prefix",
                        )
                    ]
                    request = dict(bank.info(), **bank.generation_timings())
                    rows = [dict(capacity=c, active=False) for c in self.capacities]
                    rows[slots[0]] = dict(request, active=True)
                    native = dict(requests=rows, scalar_mtp2=True)
                    stats = [request["mtp"]]
                else:
                    bank.reset(slots=slots)
                    output = bank.generate(
                        prompts, max_new_tokens=budgets, eos_token_ids=eos, slots=slots
                    )
                    native = bank.info()
                    stats = (
                        [native["requests"][s]["mtp"] for s in slots] if depth else []
                    )
                elapsed, used = perf_counter() - start, process_time() - cpu
                emitted = sum(map(len, output))
                proposed = sum(s["proposed_tokens"] for s in stats)
                accepted = sum(s["accepted_tokens"] for s in stats)
                if learn:
                    self.policy.observe(
                        workload,
                        depth,
                        wall_seconds=elapsed,
                        emitted_tokens=emitted,
                        proposed_tokens=proposed,
                        accepted_tokens=accepted,
                    )
            except BaseException:
                self._failed = True
                raise
            self._last = dict(
                decision=decision.as_dict(),
                selected_depth=depth,
                selected_slots=slots,
                native=native,
                prepared_depths=sorted(self._banks),
                preparation_wall_seconds=preparation,
                execution_wall_seconds=elapsed,
                execution_process_cpu_seconds=used,
                call_wall_seconds=perf_counter() - began,
                emitted_tokens=emitted,
                proposed_tokens=proposed,
                accepted_tokens=accepted,
                acceptance_rate=accepted / proposed if proposed else None,
                learned=learn,
                timing_scope="One whole fresh cohort; reset/native execution/metadata included; bank preparation separate. Not post-first decode TPS.",
                state_scope="Independent banks share weights only; selection is between fresh cohorts, not mid-generation",
            )
            return output

    def info(self):
        with self._lock:
            self._check()
            if self._last is None:
                raise RuntimeError("No completed adaptive cohort")
            return copy.deepcopy(self._last)

    def snapshot(self, slot):
        with self._lock:
            self._check()
            if (
                self._last is None
                or type(slot) is not int
                or slot not in self._last["selected_slots"]
            ):
                raise ValueError(
                    "Snapshot requires an active slot from the last completed cohort"
                )
            depth = self._last["selected_depth"]
            return (
                self._banks[depth].snapshot()
                if depth == 2
                else self._banks[depth].snapshot(slot)
            )

    def reset(self):
        """Explicit failure recovery; discard banks' states but retain policy history."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Adaptive MTP batch closed")
            for bank in self._banks.values():
                bank.reset()
            self._last, self._failed = None, False

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                for bank in reversed(list(self._banks.values())):
                    bank.close()
                self._banks.clear()
                self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
