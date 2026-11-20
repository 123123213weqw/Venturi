"""Audit declared external control compatibility without running inference."""

import ast
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_environment(worker):
    updates = [
        node
        for node in ast.walk(ast.parse(worker.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
        and node.func.attr == "update"
    ]
    assert len(updates) == 1
    values = {k.arg: ast.dump(k.value) for k in updates[0].keywords}
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_P2P_DISABLE",
        "LD_LIBRARY_PATH",
    ]
    return {key: values[key] for key in keys}


def normalize_port(command):
    result = list(command)
    i = result.index("--port")
    result[i + 1] = "<loopback-port>"
    return result


def main():
    fresh = OUT / "v100/campaign-v6"
    campaign = read(fresh / "campaign-report.json")
    fresh_env = explicit_environment(fresh / "worker.py")
    checks = []
    for capsule, folder, old_label, fresh_label in [
        (
            "v100-q8-wmma",
            "campaign-v5",
            "repeat-llama-tensor-off",
            "repeat-llama-nonspec-fresh",
        ),
        (
            "v100-strided-output",
            "external-depth-v9",
            "repeat-llama-tensor-mtp2-c1",
            "repeat-llama-mtp2-fresh",
        ),
    ]:
        prior = OUT.parent / capsule / "v100" / folder
        previous = read(prior / "campaign-report.json")
        old = next(e for e in previous["experiments"] if e["label"] == old_label)
        new = next(e for e in campaign["experiments"] if e["label"] == fresh_label)
        old_case = read(prior / old_label / "timed/report.json")["case"]
        new_case = read(fresh / fresh_label / "timed/report.json")["case"]
        assert old["completed"] and new["completed"] and old_case == new_case
        assert normalize_port(old["command"]) == normalize_port(new["command"])
        assert explicit_environment(prior / "worker.py") == fresh_env
        frozen_tool = OUT.parent / capsule / "source/tools/benchmark_completion_api.py"
        assert sha(frozen_tool) == sha(ROOT / "tools/benchmark_completion_api.py")
        checks.append(
            dict(
                historical_capsule=capsule,
                historical_report_sha256=sha(prior / "campaign-report.json"),
                historical_label=old_label,
                fresh_label=fresh_label,
                exact_case_equal=True,
                full_server_command_equal_except_port=True,
                normalized_server_command=normalize_port(new["command"]),
                explicit_environment_expression_equal=True,
                benchmark_tool_sha256=sha(frozen_tool),
                inherited_environment_not_exhaustively_snapshotted=True,
            )
        )
    pins = read(OUT / "fresh-payload-and-external-bundle-pins.json")
    assert pins["all_eighty_release_files_match"]
    result = dict(
        matched=True,
        checked_without_repeating_inference=True,
        checks=checks,
        explicit_environment_expressions=fresh_env,
        release_pin_report_sha256=pins["prior_release_pin_report_sha256"],
        fresh_payload_and_bundle_pins_sha256=sha(
            OUT / "fresh-payload-and-external-bundle-pins.json"
        ),
        limitation="Declared flags, cases, benchmark code and explicit environment match; this is not a proof of identical inherited environment or an exhaustive competitor configuration search. Stronger historical rates remain in acceptance.",
    )
    (OUT / "external-control-compatibility.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("Two stronger historical external configurations match declared controls")


if __name__ == "__main__":
    main()
