"""Finite SM89 Q4 SwiGLU scheduling experiment against an explicit native build.

Only the selected translation unit is copied/compiled; no external file/archive
is modified. Replacement objects precede the unchanged ops archive at link time.
All benchmark calls go through the public LinearSwiGLU contract.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


RELATIVE = Path("src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu")
EXPECTED_SOURCE_SHA256 = (
    "7d73414426b0c1a4314b33da2761af371290af90ecf2ea8215a0de373ecc55ed"
)
CANDIDATES = [
    ("baseline", 4, 3, "shared", "ca", 16),
    ("w4s2", 4, 2, "shared", "ca", 16),
    ("w8s2", 8, 2, "shared", "ca", 16),
    ("w8s1", 8, 1, "shared", "ca", 16),
    ("w4s3cached", 4, 3, "cached", "cg", 16),
    ("w8s2cached", 8, 2, "cached", "cg", 16),
    ("g8w4s3", 4, 3, "shared", "ca", 8),
    ("g8w8s3", 8, 3, "shared", "ca", 8),
    ("g8w8s2", 8, 2, "shared", "ca", 8),
    ("g8w4s4", 4, 4, "shared", "ca", 8),
    ("w4s2cg", 4, 2, "shared", "cg", 16),
    ("w4s3cg", 4, 3, "shared", "cg", 16),
]


def adapt(original, warps, stages, activation, cache, groups=16):
    if (
        warps not in (4, 8)
        or stages not in (1, 2, 3, 4)
        or activation not in ("shared", "cached")
        or cache not in ("ca", "cg")
        or groups not in (8, 16)
    ):
        raise ValueError("Unsupported finite scheduling experiment")
    if original.count("constexpr int kWarpsPerBlock     = 4;") != 1:
        raise ValueError("Unexpected external warp configuration")
    source = original.replace(
        "constexpr int kWarpsPerBlock     = 4;",
        f"constexpr int kWarpsPerBlock     = {warps};",
        1,
    )
    source = source.replace(
        "constexpr int kStages   = 3;", f"constexpr int kStages   = {stages};", 1
    )
    if groups == 8:
        source = source.replace(
            "constexpr int kGroupsPerWarpTile = 16;",
            "constexpr int kGroupsPerWarpTile = 8;",
            1,
        )
        source = source.replace(
            "static_assert(kVecsPerWarpTile == 32);",
            "static_assert(kVecsPerWarpTile == 16);",
            1,
        )
        start = source.index("    pipe_copy<16>(&s_code[0][lane]")
        end = source.index("    if (lane < 2)", start)
        source = (
            source[:start]
            + "    if (lane < kVecsPerWarpTile) {\n"
            + source[start:end]
            + "    }\n"
            + source[end:]
        )
        source = source.replace("    if (lane < 2)", "    if (lane < 1)", 1)
        source = source.replace("s_scale)[2]", "s_scale)[1]", 1)
        source = source.replace(
            "scale_tile[kWarpsPerBlock][kStages][2][2]",
            "scale_tile[kWarpsPerBlock][kStages][2][1]",
            1,
        )
    if activation == "cached":
        begin = source.index("    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];")
        source = source[:begin] + source[begin:].replace(
            "    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];\n", "", 1
        )
        begin = source.index("    auto* x_sh_v    =")
        end = source.index("    const int lane", begin)
        source = source[:begin] + source[end:]
        source = source.replace(
            "reinterpret_cast<const __nv_bfloat162*>(x_sh)",
            "reinterpret_cast<const __nv_bfloat162*>(x)",
            1,
        )
    if cache == "cg":
        begin = source.index("__device__ __forceinline__ void q4_issue_pair_tile")
        end = source.index("__global__ void q4_linear_swiglu_gemv_pair_kernel", begin)
        source = (
            source[:begin]
            + source[begin:end].replace("pipe_copy<16>", "cp_async<16, Cache::cg>")
            + source[end:]
        )
    if (warps, stages, activation, cache, groups) != (4, 3, "shared", "ca", 16):
        source = (
            "// Modified by Venturi: Ada paired-row scheduling/cache experiment; original attribution retained below.\n"
            + source
        )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate", action="append", choices=[c[0] for c in CANDIDATES])
    args = p.parse_args()
    src = Path(args.source).resolve()
    build = Path(args.build).resolve()
    out = Path(args.output).resolve()
    original = (src / RELATIVE).read_text()
    if hashlib.sha256(original.encode()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("External kernel changed; re-review the patch before building")
    out.mkdir(parents=True, exist_ok=False)
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "apps/ninfer"], text=True
    )
    matches = [shlex.split(s) for s in commands.splitlines() if " -o apps/ninfer " in s]
    if len(matches) != 1:
        raise ValueError("Expected one native application link command")
    framing = matches[0]
    if framing[:2] != [":", "&&"] or framing[-2:] != ["&&", ":"]:
        raise ValueError("Unexpected native link command framing")
    link = framing[2:-2]
    libraries = link[link.index("-o") + 2 :]
    libraries = [
        str(build / x) if x.endswith(".a") and not Path(x).is_absolute() else x
        for x in libraries
    ]
    includes = [
        "-I" + str(src / "include"),
        "-I" + str(src / "src"),
        "-I" + str(src / "tests"),
        "-I" + str(src / "bench/ops"),
        "-I" + str(src / "third_party/spdlog/include"),
    ]
    compile_base = [
        "/usr/local/cuda/bin/nvcc",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-lineinfo",
        "-arch=sm_89",
        *includes,
    ]
    bench_obj = out / "benchmark.o"
    subprocess.run(
        [
            *compile_base,
            "-c",
            str(src / "bench/ops/q4_linear_swiglu_bench.cu"),
            "-o",
            str(bench_obj),
        ],
        check=True,
    )
    reports = []
    for name, warps, stages, activation, cache, groups in CANDIDATES:
        if args.candidate and name not in args.candidate:
            continue
        directory = out / name
        directory.mkdir()
        cpp = directory / RELATIVE.name
        cpp.write_text(adapt(original, warps, stages, activation, cache, groups))
        obj = directory / "q4_linear_swiglu_gemv.cu.o"
        command = [*compile_base, "--ptxas-options=-v", "-c", str(cpp), "-o", str(obj)]
        with (directory / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        exe = directory / "bench"
        link_command = [link[0], str(bench_obj), str(obj), "-o", str(exe), *libraries]
        subprocess.run(link_command, cwd=build, check=True)
        row = dict(
            name=name,
            groups=groups,
            warps=warps,
            stages=stages,
            activation=activation,
            cache=cache,
            source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
            object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
            compile_command=command,
            link_command=link_command,
        )
        reports.append(row)
        print("built", name, flush=True)
    for name in ("LICENSE", "NOTICE", "upstream-provenance.json"):
        shutil.copy2(src / name, out / name)
    (out / "builds.json").write_text(
        json.dumps(
            dict(
                external_source=str(src),
                external_build=str(build),
                external_file_sha256=hashlib.sha256(original.encode()).hexdigest(),
                candidates=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
