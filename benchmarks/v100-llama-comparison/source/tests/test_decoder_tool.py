"""Two-layer benchmark callback ordering and synthetic weight reproducibility."""

from pathlib import Path
import sys

import torch

from venturi.models import AttentionDimensions

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_decoder_layer import replay_sequence, synthetic_weights  # noqa: E402


def test_replay_callback_truncates_before_running_both_layers():
    trace = []

    class Cache:
        def __init__(self, name):
            self.name = name

        def truncate(self, length):
            trace.append((self.name, "truncate", length))

    class Graph:
        def __init__(self, name):
            self.name = name

        def run(self):
            trace.append((self.name, "run"))

    replay_sequence([Cache("a"), Cache("b")], [Graph("a"), Graph("b")], 128)
    assert trace == [
        ("a", "truncate", 128),
        ("b", "truncate", 128),
        ("a", "run"),
        ("b", "run"),
    ]


def test_synthetic_weights_are_reproducible_without_model_download():
    d = AttentionDimensions(2, 1, 64)
    for q4 in (False, True):
        a, b = (synthetic_weights(16, 32, d, q4, 24) for _ in range(2))
        for left, right in (
            (a.query, b.query),
            (a.output, b.output),
            (a.mlp.down, b.mlp.down),
        ):
            if q4:
                assert torch.equal(left.codes, right.codes) and torch.equal(
                    left.scales, right.scales
                )
            else:
                assert torch.equal(left.data, right.data)
