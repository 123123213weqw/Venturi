"""Remove a failed performance prototype without touching preexisting work."""

from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for d in ["src", "tests", "tools"]
        for p in (ROOT / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def main():
    assert not (OUT / "production-restoration.json").exists(), (
        "Never repeat a completed restoration"
    )
    metrics = read(OUT / "metrics.json")
    assert not any(c["adopted"] for c in metrics["adoption_checks"].values())
    assert not any(metrics["path_promotions"].values())
    terminal = read(OUT / "terminal-audit.json")
    assert (
        not terminal["owned_processes"] and not terminal["remaining_periodic_samplers"]
    )
    experiment = read(OUT / "source-manifest-v4.json")
    assert manifest() == experiment, "Unexpected concurrent work; do not overwrite it"
    baseline = OUT.parent / "v100-source-validation"
    original = read(baseline / "source-manifest-v1.json")
    assert set(original).issubset(experiment), (
        "Prototype unexpectedly removed a preexisting file"
    )
    # Preserve every experimental version before removing any working-tree code.
    assert all(sha(OUT / "executed-source-v4" / p) == h for p, h in experiment.items())
    restored, removed, patch = {}, {}, []
    for rel, digest in sorted(experiment.items()):
        current = ROOT / rel
        previous = baseline / "source" / rel
        if rel not in original:
            removed[rel] = digest
            patch.extend(
                difflib.unified_diff(
                    [],
                    current.read_text().splitlines(True),
                    fromfile="pre-experiment/" + rel,
                    tofile="prototype/" + rel,
                )
            )
            continue
        assert sha(previous) == original[rel]
        if digest == original[rel]:
            continue
        restored[rel] = dict(prototype_sha256=digest, restored_sha256=original[rel])
        patch.extend(
            difflib.unified_diff(
                previous.read_text().splitlines(True),
                current.read_text().splitlines(True),
                fromfile="pre-experiment/" + rel,
                tofile="prototype/" + rel,
            )
        )
    (OUT / "implementation.diff").write_text("".join(patch))
    for rel in restored:
        shutil.copy2(baseline / "source" / rel, ROOT / rel)
    for rel in removed:
        (ROOT / rel).unlink()
    assert manifest() == original
    report = dict(
        restored=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        reason="All three paired screen gates failed; a large syscall-count reduction is not a serving speed win",
        restored_files=restored,
        removed_prototype_only_files=removed,
        restored_source_manifest=original,
        current_inference_sources_identical_to_pre_experiment=True,
        prototype_sources_and_all_results_preserved=True,
        no_unrelated_user_changes_overwritten=True,
        no_git_reset_commit_or_push=True,
    )
    (OUT / "production-restoration.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                restored_files=len(restored),
                removed_prototype_files=len(removed),
                production_files=len(original),
                prototype_promoted=False,
            )
        )
    )


if __name__ == "__main__":
    main()
