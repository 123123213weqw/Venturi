"""Backend-independent, cohort-boundary MTP selection from measured costs.

This is bounded online exploration, not DSpark or a trained confidence head.
An observation is one complete successful cohort, never summed request times.
Policies are execution-instance local: do not reuse one across models/devices.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
import threading


def _positive_ints(values):
    return (
        isinstance(values, (tuple, list))
        and bool(values)
        and all(type(v) is int and 0 < v < 2**31 for v in values)
    )


def _bucket(n):
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class MTPWorkload:
    prompt_lengths: tuple[int, ...]
    output_budgets: tuple[int, ...]
    stopping: bool = False

    def __post_init__(self):
        if (
            not _positive_ints(self.prompt_lengths)
            or not _positive_ints(self.output_budgets)
            or len(self.prompt_lengths) != len(self.output_budgets)
            or type(self.stopping) is not bool
        ):
            raise ValueError(
                "Positive prompt lengths and one output budget per request required"
            )
        object.__setattr__(self, "prompt_lengths", tuple(self.prompt_lengths))
        object.__setattr__(self, "output_budgets", tuple(self.output_budgets))

    @property
    def key(self):
        return (
            len(self.prompt_lengths),
            _bucket(max(self.prompt_lengths)),
            _bucket(sum(self.prompt_lengths)),
            _bucket(max(self.output_budgets)),
            _bucket(sum(self.output_budgets)),
            self.stopping,
        )


@dataclass(frozen=True)
class MTPDecision:
    depth: int
    reason: str
    workload_key: tuple
    estimated_seconds_per_output: dict

    def as_dict(self):
        return asdict(self)


class AdaptiveMTPPolicy:
    """Learn wall seconds/emitted token; periodically recheck stale alternatives.

    Cold buckets first measure ordinary decoding, then supported MTP variants.
    Probes may regress latency and are explicitly labelled. Set probe_interval=0
    to disable automatic exploration (unmeasured workloads then use depth zero).
    """

    def __init__(
        self, *, alpha=0.3, minimum_gain=0.03, probe_interval=16, max_buckets=128
    ):
        if (
            type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or not 0 < alpha <= 1
            or type(minimum_gain) not in (int, float)
            or not math.isfinite(minimum_gain)
            or not 0 <= minimum_gain < 1
            or type(probe_interval) is not int
            or probe_interval < 0
            or type(max_buckets) is not int
            or not 1 <= max_buckets <= 4096
        ):
            raise ValueError("Invalid adaptive MTP policy parameters")
        self.alpha, self.minimum_gain = alpha, minimum_gain
        self.probe_interval, self.max_buckets = probe_interval, max_buckets
        self._buckets = OrderedDict()
        self._lock = threading.RLock()

    def _cell(self, workload):
        if not isinstance(workload, MTPWorkload):
            raise TypeError("Explicit MTPWorkload required")
        key = workload.key
        if key not in self._buckets:
            self._buckets[key] = dict(step=0, samples={})
            if len(self._buckets) > self.max_buckets:
                self._buckets.popitem(last=False)
        self._buckets.move_to_end(key)
        return self._buckets[key]

    def choose(self, workload, *, supported=(0, 1, 2), override=None):
        if (
            not isinstance(supported, (tuple, list))
            or not supported
            or any(type(v) is not int or v not in (0, 1, 2) for v in supported)
            or len(set(supported)) != len(supported)
            or 0 not in supported
        ):
            raise ValueError("Supported strategies must uniquely include depth zero")
        if not isinstance(workload, MTPWorkload):
            raise TypeError("Explicit MTPWorkload required")
        allowed = tuple(
            d
            for d in sorted(supported)
            if d == 0
            or (
                max(workload.output_budgets) > d
                and (d != 2 or len(workload.prompt_lengths) == 1)
            )
        )
        if override is not None and (
            type(override) is not int or override not in allowed
        ):
            raise ValueError("Forced MTP strategy is unsupported for this workload")
        with self._lock:
            cell = self._cell(workload)
            samples = cell["samples"]
            estimates = {
                d: samples[d]["seconds_per_output"] for d in allowed if d in samples
            }

            def decision(depth, reason):
                return MTPDecision(depth, reason, workload.key, estimates)

            if override is not None:
                return decision(override, "manual-override")
            if len(allowed) == 1:
                return decision(0, "no-useful-speculation")
            if self.probe_interval:
                for depth in allowed:
                    if depth not in samples:
                        return decision(depth, "cold-start-probe")
                oldest = min(allowed, key=lambda d: samples[d]["last_step"])
                if cell["step"] - samples[oldest]["last_step"] >= self.probe_interval:
                    return decision(oldest, "periodic-recheck")
            if 0 not in estimates:
                return decision(0, "unmeasured-baseline")
            best = min(estimates, key=lambda d: (estimates[d], d))
            if best and estimates[best] < estimates[0] * (1 - self.minimum_gain):
                return decision(best, "measured-cost")
            return decision(0, "baseline-or-insufficient-gain")

    def observe(
        self,
        workload,
        depth,
        *,
        wall_seconds,
        emitted_tokens,
        proposed_tokens=0,
        accepted_tokens=0,
    ):
        if not isinstance(workload, MTPWorkload):
            raise TypeError("Explicit MTPWorkload required")
        if (
            type(depth) is not int
            or depth not in (0, 1, 2)
            or (depth == 2 and len(workload.prompt_lengths) != 1)
            or type(wall_seconds) not in (int, float)
            or not math.isfinite(wall_seconds)
            or wall_seconds <= 0
            or type(emitted_tokens) is not int
            or emitted_tokens <= 0
            or emitted_tokens > sum(workload.output_budgets)
            or any(
                type(v) is not int or v < 0 for v in (proposed_tokens, accepted_tokens)
            )
            or accepted_tokens > proposed_tokens
            or (depth == 0 and (proposed_tokens or accepted_tokens))
        ):
            raise ValueError("Invalid successful-cohort observation")
        with self._lock:
            cell = self._cell(workload)
            cell["step"] += 1
            sample = cell["samples"].get(depth)
            cost = wall_seconds / emitted_tokens
            if sample is None:
                sample = dict(
                    observations=0,
                    seconds_per_output=cost,
                    proposed_tokens=0,
                    accepted_tokens=0,
                )
                cell["samples"][depth] = sample
            sample["seconds_per_output"] += self.alpha * (
                cost - sample["seconds_per_output"]
            )
            sample["observations"] += 1
            sample["last_step"] = cell["step"]
            sample["proposed_tokens"] += proposed_tokens
            sample["accepted_tokens"] += accepted_tokens

    def info(self):
        with self._lock:
            rows = []
            for key, cell in self._buckets.items():
                samples = {}
                for depth, sample in cell["samples"].items():
                    samples[depth] = dict(
                        sample,
                        acceptance_rate=(
                            sample["accepted_tokens"] / sample["proposed_tokens"]
                            if sample["proposed_tokens"]
                            else None
                        ),
                    )
                rows.append(
                    dict(
                        workload_key=key,
                        completed_cohorts=cell["step"],
                        samples=samples,
                    )
                )
            return dict(
                alpha=self.alpha,
                minimum_gain=self.minimum_gain,
                probe_interval=self.probe_interval,
                buckets=rows,
                scope="Execution-instance-local EWMA of complete cohort wall/emitted tokens; acceptance is diagnostic, not a trained confidence estimate",
            )
