"""Qualify one recorded CUDA object through public Op calls on a CUDA host.

The baseline links the original archive without an override. The candidate
links the recorded object before that archive. No external build is modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = [r for r in manifest["candidates"] if r["name"] == args.candidate]
    if len(rows) != 1:
        raise ValueError("Expected one matching candidate in the manifest")
    row = rows[0]
    obj = Path(row["link_command"][2])
    if sha256(obj) != row["object_sha256"]:
        raise ValueError("Candidate object hash mismatch")
    source = Path(manifest["external_source"])
    if (
        sha256(source / "src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu")
        != manifest["external_file_sha256"]
    ):
        raise ValueError("External source changed since candidate compilation")
    build = Path(manifest["external_build"])
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    commands = []

    def run(command, log):
        commands.append(command)
        with (out / log).open("w") as sink:
            subprocess.run(
                command, cwd=build, stdout=sink, stderr=subprocess.STDOUT, check=True
            )

    includes = [
        "-I" + str(source / path)
        for path in (
            "include",
            "src",
            "tests",
            "bench/ops",
            "third_party/spdlog/include",
        )
    ]
    includes.append("-I/usr/local/cuda/include")
    nvcc = row["compile_command"][0]
    run(
        [
            nvcc,
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-arch=sm_89",
            *includes,
            "-c",
            str(tools / "q4_swiglu_stream_bench.cu"),
            "-o",
            str(out / "stream.o"),
        ],
        "stream-compile.log",
    )
    for cpp, name in (
        (tools / "q4_swiglu_correctness.cpp", "test"),
        (source / "tests/ops/linear_swiglu/linear_swiglu_test_common.cpp", "oracle"),
    ):
        run(
            [
                row["link_command"][0],
                "-std=c++20",
                "-O3",
                "-DNDEBUG",
                "-fno-fast-math",
                "-ffp-contract=off",
                *includes,
                "-c",
                str(cpp),
                "-o",
                str(out / f"{name}.o"),
            ],
            f"{name}-compile.log",
        )
    libraries = row["link_command"][row["link_command"].index("-o") + 2 :]
    for label, objects in (("baseline", []), ("candidate", [str(obj)])):
        for kind, inputs in (
            ("stream", ["stream.o"]),
            ("test", ["test.o", "oracle.o"]),
        ):
            run(
                [
                    row["link_command"][0],
                    *[str(out / f) for f in inputs],
                    *objects,
                    "-o",
                    str(out / f"{label}-{kind}"),
                    *libraries,
                ],
                f"{label}-{kind}-link.log",
            )
        # A failed or skipped oracle aborts before any performance conclusion.
        run([str(out / f"{label}-test")], f"{label}-test.log")
    for index, label in enumerate(("baseline", "candidate", "candidate", "baseline")):
        run([str(out / f"{label}-stream")], f"{index}-{label}-stream.csv")
    report = dict(
        candidate=row,
        manifest=str(Path(args.manifest).resolve()),
        commands=commands,
        numerical_exit_codes={"baseline": 0, "candidate": 0},
        test_source_sha256=sha256(tools / "q4_swiglu_correctness.cpp"),
        oracle_source_sha256=sha256(
            source / "tests/ops/linear_swiglu/linear_swiglu_test_common.cpp"
        ),
        benchmark_source_sha256=sha256(tools / "q4_swiglu_stream_bench.cu"),
        linked_archive_sha256={
            p: sha256(Path(p) if Path(p).is_absolute() else build / p)
            for p in libraries
            if p.endswith(".a")
        },
        note="Synthetic screening, not real-model throughput or measured DRAM bandwidth",
    )
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
