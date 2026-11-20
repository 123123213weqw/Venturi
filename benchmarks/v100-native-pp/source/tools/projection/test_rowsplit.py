"""Small, isolated SM89 LinearAdd experiment; never modifies external archives.

Compile on the CUDA host. The public Op owns residual arithmetic; this changes
only the selected T=1 or wide-T schedules. Numerical checks precede screening.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


RELATIVE = Path("src/ops/linear_add/q4/q4_linear_add.cu")
EXPECTED_SHA256 = "c16af0f2f1fa7511cb46f05cdee301a9c56a126e4d393acea08753ffab4e835d"
CANDIDATES = ("baseline", "r2w4", "r4w2", "word_r4w1")
BLOCK_ROWS = {"word_r2w1": 2, "word_r4w1": 4, "word_r8w1": 8}
BLOCK_CANDIDATES = tuple(BLOCK_ROWS)
PREFILL_SCHEDULES = {
    "prefill_baseline": (64, 128, 64, 64, 32, 2, 1),
    "prefill_w32": (64, 128, 64, 32, 32, 2, 1),
    "prefill_c64_s2": (64, 64, 64, 32, 32, 2, 1),
    "prefill_c64_s3": (64, 64, 64, 32, 32, 3, 1),
    "prefill_selective": (64, 64, 64, 32, 32, 2, 1),
    "prefill_r128_w32": (128, 64, 64, 32, 32, 2, 1),
    "prefill_r128_w64": (128, 64, 64, 64, 32, 2, 1),
}
PREFILL_CANDIDATES = (
    "prefill_baseline",
    "prefill_w32",
    "prefill_c64_s2",
    "prefill_c64_s3",
)
FULL_PREFILL_CANDIDATES = ("prefill_baseline", "prefill_r128_w32", "prefill_r128_w64")


def adapt(original, name):
    if name not in (*CANDIDATES, *BLOCK_CANDIDATES, *PREFILL_SCHEDULES):
        raise ValueError("Unknown finite row-split candidate")
    if name in ("baseline", "prefill_baseline"):
        return original
    if name in PREFILL_SCHEDULES:
        begin = original.index("using MmaR64C128 =")
        end = original.index(";", begin) + 1
        parameters = ", ".join(map(str, PREFILL_SCHEDULES[name]))
        if name == "prefill_selective":
            alias = f"""\nusing MmaPrefillPartial = Q4RowSplitMmaGemmSchedule<{parameters},
        Q4FragmentPipeline::Serial, Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;"""
            patched = original[:end] + alias + original[end:]
            anchor = "    if (k == 17408) {\n        if (tokens == 1)"
            if patched.count(anchor) != 1:
                raise ValueError("Unexpected LinearAdd selector boundary")
            branch = """    // Keep the existing full-chunk route. The smaller tile wins for
    // short MMA problems and partial 64-column tiles below a full 512 chunk.
    if (tokens > 192 && tokens < 512 && (tokens <= 256 || tokens % 64 != 0))
        return launch_q4_mma<MmaPrefillPartial, ResidualEpilogue>;
"""
            return (
                "// Modified by Venturi: selective prefill tile experiment; see adjacent attribution.\n"
                + patched.replace(anchor, branch + anchor, 1)
            )
        schedule = f"""using MmaR64C128 = Q4RowSplitMmaGemmSchedule<{parameters},
        Q4FragmentPipeline::Serial, Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;"""
        return (
            "// Modified by Venturi: prefill tile/pipeline experiment; see adjacent attribution.\n"
            + original[:begin]
            + schedule
            + original[end:]
        )
    start = original.index("using GemvR1W8 =")
    end = original.index("using MmaR32C32", start)
    if name == "word_r4w1":
        schedule = "Q4GemvR4W1DirectSchedule"
    elif name in BLOCK_CANDIDATES:
        rows = BLOCK_ROWS[name]
        # Only CTA row count changes: one warp owns each full dot product,
        # with the same decode, staging, lane ownership and reduction order.
        schedule = f"""Q4RowSplitGemvSchedule<{rows}, 1, 16, 1,
        Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedWord8,
        Q4GemvDecodeMode::Fp16Mantissa, Q4GemvCodeTransfer::AsyncVector16,
        Q4GemvScaleAccess::SharedPair32, Cache::ca, 0, 1>"""
    else:
        rows, warps = (2, 4) if name == "r2w4" else (4, 2)
        # Dynamic scale-pair ownership handles both K values without assuming
        # that all groups assigned to one warp fit a single 16-group tile.
        schedule = f"""Q4RowSplitGemvSchedule<{rows}, {warps}, 16, 1,
        Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedByte2,
        Q4GemvDecodeMode::ScalarInteger, Q4GemvCodeTransfer::SyncVector16,
        Q4GemvScaleAccess::Scalar16Shuffle, Cache::ca, 0, 1>"""
    return (
        "// Modified by Venturi: T=1 multi-row ownership experiment; see adjacent attribution.\n"
        + original[:start]
        + f"using GemvR1W8 = {schedule};\n"
        + f"using GemvR1W8K17408 = {schedule};\n"
        + original[end:]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--suite",
        choices=(
            "ownership",
            "block-size",
            "prefill",
            "prefill-selective",
            "prefill-full",
        ),
        default="ownership",
    )
    args = parser.parse_args()
    candidates = {
        "ownership": CANDIDATES,
        "block-size": BLOCK_CANDIDATES,
        "prefill": PREFILL_CANDIDATES,
        "prefill-selective": ("prefill_selective",),
        "prefill-full": FULL_PREFILL_CANDIDATES,
    }[args.suite]
    source, build, out = (
        Path(x).resolve() for x in (args.source, args.build, args.output)
    )
    original = (source / RELATIVE).read_text()
    if hashlib.sha256(original.encode()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("External source changed; re-review the schedule patch")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("LICENSE", "NOTICE", "upstream-provenance.json"):
        shutil.copy2(source / name, out / name)
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "apps/ninfer"], text=True
    )
    links = [shlex.split(s) for s in commands.splitlines() if " -o apps/ninfer " in s]
    if len(links) != 1 or links[0][:2] != [":", "&&"] or links[0][-2:] != ["&&", ":"]:
        raise ValueError("Unexpected native link command")
    link = links[0][2:-2]
    libs = [
        str(build / p) if p.endswith(".a") and not Path(p).is_absolute() else p
        for p in link[link.index("-o") + 2 :]
    ]
    includes = [
        "-I" + str(source / p)
        for p in ("include", "src", "tests", "bench/ops", "third_party/spdlog/include")
    ]
    includes.append("-I/usr/local/cuda/include")
    cuda = [
        "/usr/local/cuda/bin/nvcc",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-lineinfo",
        "-arch=sm_89",
        *includes,
    ]
    tools = Path(__file__).resolve().parent
    recorded = []

    def run(command, log):
        recorded.append(command)
        with (out / log).open("w") as sink:
            subprocess.run(
                command, cwd=build, stdout=sink, stderr=subprocess.STDOUT, check=True
            )

    run(
        [
            *cuda,
            "-c",
            str(
                tools
                / (
                    "q4_prefill_stream_bench.cu"
                    if args.suite.startswith("prefill")
                    else "q4_linear_add_stream_bench.cu"
                )
            ),
            "-o",
            str(out / "bench.o"),
        ],
        "bench-compile.log",
    )
    for path, name in (
        (
            tools
            / (
                "q4_prefill_correctness.cpp"
                if args.suite.startswith("prefill")
                else "q4_linear_add_correctness.cpp"
            ),
            "test",
        ),
        (source / "tests/ops/linear_add/linear_add_test_common.cpp", "oracle"),
    ):
        run(
            [
                link[0],
                "-std=c++20",
                "-O3",
                "-DNDEBUG",
                "-fno-fast-math",
                "-ffp-contract=off",
                *includes,
                "-c",
                str(path),
                "-o",
                str(out / f"{name}.o"),
            ],
            f"{name}-compile.log",
        )
    if args.suite == "block-size":
        run(
            [
                link[0],
                "-std=c++20",
                "-O3",
                "-DNDEBUG",
                *includes,
                "-c",
                str(tools / "q4_linear_add_snapshot.cpp"),
                "-o",
                str(out / "snapshot.o"),
            ],
            "snapshot-compile.log",
        )
    rows = []
    for name in candidates:
        directory = out / name
        directory.mkdir()
        cpp = directory / RELATIVE.name
        cpp.write_text(adapt(original, name))
        obj = directory / "candidate.o"
        run(
            [*cuda, "--ptxas-options=-v", "-c", str(cpp), "-o", str(obj)],
            f"{name}/compile.log",
        )
        binaries = [
            ("bench", [out / "bench.o"]),
            ("test", [out / "test.o", out / "oracle.o"]),
        ]
        if args.suite == "block-size":
            binaries.append(("snapshot", [out / "snapshot.o"]))
        for kind, inputs in binaries:
            run(
                [
                    link[0],
                    *map(str, inputs),
                    str(obj),
                    "-o",
                    str(directory / kind),
                    *libs,
                ],
                f"{name}/{kind}-link.log",
            )
        oracle_args = ["512"] if args.suite == "prefill-full" else []
        run([str(directory / "test"), *oracle_args], f"{name}/correctness.log")
        if args.suite == "block-size":
            run(
                [str(directory / "snapshot"), str(directory / "outputs.bin")],
                f"{name}/snapshot.log",
            )
        rows.append(
            dict(
                name=name,
                object=str(obj),
                source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
                independent_oracle_exit_code=0,
                prefill_schedule=PREFILL_SCHEDULES.get(name),
                block_threads=(
                    BLOCK_ROWS[name] * 32 if name in BLOCK_CANDIDATES else None
                ),
            )
        )
        print(name, "built; numerical checks passed", flush=True)
    if args.suite == "block-size":
        reference = (out / "word_r4w1/outputs.bin").read_bytes()
        for name in candidates:
            if (out / name / "outputs.bin").read_bytes() != reference:
                raise ValueError(
                    f"{name}: block-size ablation changed represented outputs"
                )
        order = (
            "word_r4w1",
            "word_r2w1",
            "word_r8w1",
            "word_r4w1",
            "word_r8w1",
            "word_r2w1",
            "word_r4w1",
        )
    elif args.suite == "prefill":
        order = (
            "prefill_baseline",
            "prefill_w32",
            "prefill_c64_s2",
            "prefill_c64_s3",
            "prefill_baseline",
            "prefill_c64_s3",
            "prefill_c64_s2",
            "prefill_w32",
            "prefill_baseline",
        )
    elif args.suite == "prefill-selective":
        order = ("prefill_selective", "prefill_selective")
    elif args.suite == "prefill-full":
        order = (
            "prefill_baseline",
            "prefill_r128_w32",
            "prefill_r128_w64",
            "prefill_baseline",
            "prefill_r128_w64",
            "prefill_r128_w32",
            "prefill_baseline",
        )
    else:
        order = (
            "baseline",
            "r2w4",
            "r4w2",
            "word_r4w1",
            "baseline",
            "word_r4w1",
            "r4w2",
            "r2w4",
            "baseline",
        )
    # Interleave the reference with the forward/reverse candidate order.
    for index, name in enumerate(order):
        run([str(out / name / "bench")], f"{name}/stream-{index}.csv")
    (out / "results.json").write_text(
        json.dumps(
            dict(
                external_source=str(source),
                suite=args.suite,
                external_file_sha256=EXPECTED_SHA256,
                candidates=rows,
                commands=recorded,
                block_snapshot_bit_identical=(
                    True if args.suite == "block-size" else None
                ),
                note="Synthetic public LinearAdd screening, not model TPS or measured DRAM bandwidth",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
