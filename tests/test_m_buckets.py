"""Test workload-aware M classes without CUDA."""

import pytest

from venturi.quantization.m_buckets import Calibration, MKind, classify_m


GEOMETRIES = {
    (1024, 5120): 32,
    (5120, 6144): 64,
    (5120, 17408): 64,
    (6144, 5120): 48,
    (17408, 5120): 128,
    (248320, 5120): 1,
}


def calibration():
    return Calibration.from_history(
        GEOMETRIES, {29: 1, 19: 48, 23: 19, 32: 200}, {5: 44, 7: 25, 4: 104}
    )


def test_m_classes_separate_full_and_masked_tiles():
    assert classify_m(1) is MKind.ONE
    assert classify_m(2) is MKind.TWO
    assert classify_m(3) is classify_m(4) is MKind.FEW
    assert classify_m(8) is MKind.SMALL
    assert classify_m(15) is MKind.MASK16
    assert classify_m(16) is MKind.FULL16
    assert classify_m(19) is classify_m(23) is classify_m(31) is MKind.MASK32
    assert classify_m(32) is MKind.FULL32
    with pytest.raises(ValueError):
        classify_m(33)


def test_ten_anchors_spend_samples_on_real_head_and_masked_extents():
    c = calibration()
    assert len(c.anchors) == len({a.key for a in c.anchors}) == 10
    assert [a.m for a in c.anchors if a.role.startswith("masked")] == [19, 23]
    assert [a.m for a in c.anchors if a.role.startswith("head")] == [1, 5]
    assert c.tail == ((5120, 17408), (5120, 6144))


def test_masked_route_never_overrides_aligned_or_small_decode():
    c = calibration()
    winners = {
        "wide_full": 2,
        "wide_single": 1,
        "wide_two": 2,
        "head_single": 1,
        "head_active": 2,
        "masked0": 0,
        "masked1": 0,
    }
    med = {
        a.key: {r: 1.0 if r == winners.get(a.role, a.arms[0]) else 1.2 for r in a.arms}
        for a in c.anchors
    }
    rules = {(m, n, k): mode for m, n, k, mode in c.rules(GEOMETRIES, med)}
    assert rules[19, 5120, 17408] == rules[23, 5120, 6144] == 0
    assert rules[16, 5120, 17408] == rules[32, 5120, 6144] == 2
    assert rules[19, 17408, 5120] == 2
    assert rules[1, 17408, 5120] == 1 and rules[2, 17408, 5120] == 2
    assert rules[5, 248320, 5120] == 2 and rules[16, 248320, 5120] == 0
    assert all(m <= 8 or v != 1 for (m, _, _), v in rules.items())
