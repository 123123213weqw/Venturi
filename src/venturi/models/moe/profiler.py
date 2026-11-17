"""Routing profile collection: per-layer expert histograms and coverage.

Answers 'which experts does a workload actually use' — the input to hot-set
pre-warming and (later) expert pruning. Recording is a side channel attached
to TopKRouter; it never touches routing math.
"""

from collections import Counter

import torch


class RouteProfiler:
    """Accumulates per-layer top-k expert selections and reports coverage."""

    def __init__(self, num_layers: int, num_experts: int):
        if type(num_layers) is not int or type(num_experts) is not int or num_layers <= 0 or num_experts <= 0:
            raise ValueError("Positive layer and expert counts required")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._counts = [Counter() for _ in range(num_layers)]
        self._layer = 0

    def enter_layer(self, layer: int):
        if type(layer) is not int or not 0 <= layer < self.num_layers:
            raise ValueError("Layer index out of range")
        self._layer = layer

    def record(self, indices: torch.Tensor):
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64:
            raise ValueError("Recording expects the router's int64 index tensor")
        self._counts[self._layer].update(indices.reshape(-1).tolist())

    def coverage_report(self, layer: int = None):
        """Descending expert frequencies with cumulative token coverage."""
        layers = range(self.num_layers) if layer is None else (layer,)
        report = {}
        for index in layers:
            total = sum(self._counts[index].values())
            if total == 0:
                report[index] = {"tokens": 0, "experts": []}
                continue
            ranked = sorted(self._counts[index].items(), key=lambda kv: (-kv[1], kv[0]))
            cumulative, entries = 0, []
            for expert, count in ranked:
                cumulative += count
                entries.append({"expert": expert, "tokens": count,
                                "coverage": cumulative / total})
            report[index] = {"tokens": total, "experts": entries}
        return report

    def hot_set(self, coverage: float = 0.95, layer: int = None):
        """Smallest expert set reaching the requested token coverage."""
        if not 0 < coverage <= 1:
            raise ValueError("coverage must lie in (0, 1]")
        result = {}
        for index, layer_report in self.coverage_report(layer).items():
            selected = []
            for entry in layer_report["experts"]:
                selected.append(entry["expert"])
                if entry["coverage"] >= coverage:
                    break
            result[index] = selected
        return result
