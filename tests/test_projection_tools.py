"""Projection experiment orchestration, without invoking a local compiler."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(path):
    spec = importlib.util.spec_from_file_location("projection_tool", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_rejects_changed_external_source_before_build(monkeypatch, tmp_path):
    tool = load_tool("tools/projection/build_candidates.py")
    kernel = tmp_path / "engine" / tool.RELATIVE
    kernel.parent.mkdir(parents=True)
    kernel.write_text("changed upstream")
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool",
            "--source",
            str(tmp_path / "engine"),
            "--build",
            str(tmp_path / "build"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="re-review"):
        tool.main()
    assert not output.exists()


def test_finite_experiment_and_source_shape_guards():
    tool = load_tool("tools/projection/build_candidates.py")
    assert len({c[0] for c in tool.CANDIDATES}) == 12
    with pytest.raises(ValueError, match="Unsupported"):
        tool.adapt("", 4, 3, "shared", "ca", 7)
    with pytest.raises(ValueError, match="Unexpected external"):
        tool.adapt("changed source", 4, 3, "shared", "ca")


def test_evaluation_rejects_tampered_object_before_compile(monkeypatch, tmp_path):
    tool = load_tool("tools/projection/evaluate_candidate.py")
    obj = tmp_path / "kernel.o"
    obj.write_bytes(b"changed")
    manifest = tmp_path / "builds.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": "test",
                        "link_command": ["c++", "bench.o", str(obj)],
                        "object_sha256": "wrong",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool",
            "--manifest",
            str(manifest),
            "--candidate",
            "test",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.main()
    assert not (tmp_path / "out").exists()


def test_rowsplit_changes_only_named_t1_schedules():
    tool = load_tool("tools/projection/test_rowsplit.py")
    original = "prefix\nusing GemvR1W8 = old;\nusing GemvR1W8K17408 = old;\nusing MmaR32C32 = unchanged;\nsuffix"
    assert tool.adapt(original, "baseline") == original
    changed = tool.adapt(original, "r2w4")
    assert "Q4RowSplitGemvSchedule<2, 4, 16, 1" in changed
    assert changed.endswith("using MmaR32C32 = unchanged;\nsuffix")
    assert "Modified by Venturi" in changed
    assert "Q4GemvR4W1DirectSchedule" in tool.adapt(original, "word_r4w1")
    with pytest.raises(ValueError, match="Unknown"):
        tool.adapt(original, "unreviewed")


def test_rowsplit_rejects_changed_source_before_compiling(monkeypatch, tmp_path):
    tool = load_tool("tools/projection/test_rowsplit.py")
    kernel = tmp_path / "source" / tool.RELATIVE
    kernel.parent.mkdir(parents=True)
    kernel.write_text("changed external kernel")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool",
            "--source",
            str(tmp_path / "source"),
            "--build",
            str(tmp_path / "build"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="re-review"):
        tool.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name,rows", [("word_r2w1", 2), ("word_r4w1", 4), ("word_r8w1", 8)]
)
def test_block_ablation_keeps_one_warp_per_row_and_word_decode(name, rows):
    tool = load_tool("tools/projection/test_rowsplit.py")
    original = "before\nusing GemvR1W8 = old;\nusing GemvR1W8K17408 = old;\nusing MmaR32C32 = unchanged;\nafter"
    changed = tool.adapt(original, name)
    assert tool.BLOCK_ROWS[name] == rows
    assert changed.endswith("using MmaR32C32 = unchanged;\nafter")
    if rows == 4:
        # Reuse the prior reference's exact schedule alias, not a second policy.
        assert changed.count("Q4GemvR4W1DirectSchedule") == 2
    else:
        assert changed.count(f"Q4RowSplitGemvSchedule<{rows}, 1, 16, 1") == 2
        for setting in ("PackedWord8", "Fp16Mantissa", "AsyncVector16", "SharedPair32"):
            assert changed.count(setting) == 2
        assert "PackedByte2" not in changed


@pytest.mark.parametrize(
    "name",
    [
        "prefill_w32",
        "prefill_c64_s2",
        "prefill_c64_s3",
        "prefill_r128_w32",
        "prefill_r128_w64",
    ],
)
def test_prefill_patch_preserves_decode_and_dispatch(name):
    tool = load_tool("tools/projection/test_rowsplit.py")
    before = "decode aliases and narrow-T schedules\n"
    after = "\nselector uses the same named alias; no threshold changes\n"
    original = before + "using MmaR64C128 = original_schedule;" + after
    changed = tool.adapt(original, name)
    assert before in changed
    assert changed.endswith(after)
    parameters = ", ".join(map(str, tool.PREFILL_SCHEDULES[name]))
    assert f"Q4RowSplitMmaGemmSchedule<{parameters}," in changed
    assert (
        "Q4FragmentPipeline::Serial, Cache::cg, Cache::cg, Q4ScaleLoad::Pair32"
        in changed
    )
    assert tool.adapt(original, "prefill_baseline") == original


def test_full_prefill_candidates_fit_static_budget_without_expanding_previous_suite():
    tool = load_tool("tools/projection/test_rowsplit.py")
    assert set(tool.FULL_PREFILL_CANDIDATES) - set(tool.PREFILL_CANDIDATES) == {
        "prefill_r128_w32",
        "prefill_r128_w64",
    }
    for name in tool.FULL_PREFILL_CANDIDATES:
        rows, cols, k, warp_rows, warp_cols, stages, _ = tool.PREFILL_SCHEDULES[name]
        shared = rows * k * 2 + stages * cols * k * 2 + stages * rows * 36
        assert shared <= 48 * 1024
        assert rows % warp_rows == cols % warp_cols == 0
        assert rows // warp_rows * (cols // warp_cols) * 32 in (128, 256)


def test_selective_prefill_keeps_old_alias_and_validates_before_route_selection():
    tool = load_tool("tools/projection/test_rowsplit.py")
    original = """decode aliases unchanged;
using MmaR64C128 = existing_wide_schedule;
Q4LinearAddLaunch select_q4_linear_add(int rows, int k, int tokens) {
    if (rows != 5120 || tokens <= 0) throw invalid_input;
    if (k == 17408) {
        if (tokens == 1) return existing_decode;
    }
    return existing_full_chunk;
}
"""
    changed = tool.adapt(original, "prefill_selective")
    assert "using MmaR64C128 = existing_wide_schedule;" in changed
    assert (
        "using MmaPrefillPartial = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 1,"
        in changed
    )
    assert changed.index("throw invalid_input") < changed.index("if (tokens > 192")
    assert "return existing_decode;" in changed
    assert changed.endswith("    return existing_full_chunk;\n}\n")
    with pytest.raises(ValueError, match="selector boundary"):
        tool.adapt("using MmaR64C128 = schedule;", "prefill_selective")
