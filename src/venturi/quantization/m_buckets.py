"""Bounded SM89 row-tile calibration. Buckets select kernels, not tensor shapes."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log2


class MKind(Enum):
    ONE = "m1"
    TWO = "m2"
    FEW = "m3-4"
    SMALL = "m5-8"
    MASK16 = "m9-15 (16-row tile, masked)"
    FULL16 = "m16 (16-row tile, full)"
    MASK32 = "m17-31 (32-row tile, masked)"
    FULL32 = "m32 (32-row tile, full)"


def classify_m(m):
    if type(m) is not int or not 1 <= m <= 32:
        raise ValueError("M outside the calibrated range 1..32")
    for end, kind in zip((1, 2, 4, 8, 15, 16, 31, 32), MKind, strict=True):
        if m <= end:
            return kind
    raise AssertionError("unreachable shape")


@dataclass(frozen=True)
class Anchor:
    role: str
    m: int
    n: int
    k: int
    arms: tuple[int, int]

    @property
    def key(self):
        return self.m, self.n, self.k


@dataclass(frozen=True)
class Calibration:
    """Selection rules for the experimental SM89 FP16 row-split Q5 backend."""

    narrow: tuple[int, int]
    wide: tuple[int, int]
    head: tuple[int, int]
    tail: tuple[tuple[int, int], tuple[int, int]]
    anchors: tuple[Anchor, ...]

    @classmethod
    def from_history(cls, geometries, trunk_m, head_m):
        """Build ten anchors using shape counts, not previously timed winners.

        Geometries maps (N,K) to the number of uses in a trunk traversal
        (one for the head). Head histograms describe *sampled* rows.
        """
        narrow = min(geometries, key=lambda q: (q[0], q[1]))
        head = max(geometries, key=lambda q: (q[0], q[1]))

        def cost(q):
            return geometries[q] * q[0] * q[1]

        ordered = sorted(
            (q for q in geometries if q not in (narrow, head)),
            key=lambda q: (-cost(q), q),
        )
        if len(ordered) < 3:
            raise ValueError("Need distinct narrow, wide, head and two tail geometries")
        wide = ordered[0]
        middle = [q for q in ordered[1:] if q[0] * 2 <= wide[0]]
        if len(middle) < 2:
            raise ValueError("Need two distinct medium-width geometries for masked M")
        tail = middle[0], middle[1]
        tails = sorted(
            (m for m, count in trunk_m.items() if 17 <= m <= 31 and count > 0),
            key=lambda m: (-trunk_m[m], m),
        )
        t0 = tails[0] if tails else 19
        t1 = tails[1] if len(tails) > 1 else 23
        h = min(range(5, 9), key=lambda m: (-head_m.get(m, 0), m))

        def a(role, m, q, arms):
            return Anchor(role, m, *q, arms)

        return cls(
            narrow,
            wide,
            head,
            tail,
            (
                a("narrow_single", 1, narrow, (0, 1)),
                a("narrow_small", 4, narrow, (0, 1)),
                a("wide_single", 1, wide, (0, 1)),
                a("wide_two", 2, wide, (1, 2)),
                a("wide_small", 4, wide, (1, 2)),
                a("wide_full", 32, wide, (0, 2)),
                a("head_single", 1, head, (0, 1)),
                a("head_active", h, head, (1, 2)),
                a("masked0", t0, tail[0], (2, 0)),
                a("masked1", t1, tail[1], (2, 0)),
            ),
        )

    def rules(self, geometries, medians, *, margin=0.05):
        """Materialize native [M,N,K,route] keys from ten tested cells.

        Medians only contains validated candidates. Where evidence is
        missing, contradictory, or unsupported, use a conservative
        candidate. No route from a masked M>16 cell propagates to aligned
        M=16/32, and no grouped route propagates above M=8.
        """
        if not isfinite(margin) or not 0 <= margin < 1:
            raise ValueError("Invalid switching margin")
        picks = {}
        exact = {}
        for a in self.anchors:
            tested = medians.get(a.key, {})
            valid = {
                route: tested[route]
                for route in a.arms
                if route in tested and isfinite(tested[route]) and tested[route] > 0
            }
            if not valid:
                raise ValueError(f"No valid sample for {a.role}")
            winner = min(valid, key=valid.get)
            incumbent = a.arms[0]
            if incumbent in valid and valid[winner] >= valid[incumbent] * (1 - margin):
                winner = incumbent
            picks[a.role] = exact[a.key] = winner
        result = []
        for n, k in sorted(geometries):
            q = n, k
            for m in range(1, 33):
                kind = classify_m(m)
                if q == self.head:
                    # Scheduler admits at most eight sampled head rows in this pilot.
                    label = "head_single" if m <= 4 else "head_active"
                    route = picks[label] if m <= 8 else 0
                elif q == self.narrow:
                    route = (
                        picks["narrow_single"]
                        if m <= 2
                        else picks["narrow_small"]
                        if m <= 4
                        else 0
                    )
                elif kind == MKind.ONE:
                    route = picks["wide_single"]
                elif kind == MKind.TWO:
                    route = (
                        picks["wide_two"]
                        if n * 2 > self.wide[0]
                        else picks["wide_single"]
                    )
                elif kind in (MKind.FEW, MKind.SMALL):
                    route = picks["wide_small"]
                elif kind == MKind.MASK32 and n * 2 <= self.wide[0]:
                    nearest = min(
                        range(2), key=lambda i: abs(log2(k / self.tail[i][1]))
                    )
                    route = picks[f"masked{nearest}"]
                else:
                    route = picks["wide_full"]
                route = exact.get((m, n, k), route)
                result.append((m, n, k, 0 if route == 1 and m > 8 else route))
        return result
