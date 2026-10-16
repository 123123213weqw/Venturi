"""Link Venturi's resident runner against an explicitly selected NInfer build.

Run on the CUDA server, never copies or modifies the external source/build.
No downloading, dependency installation or upstream rebuild is implicit.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", default="/usr/local/cuda")
    parser.add_argument(
        "--runner-source",
        help="Explicit alternative transport/probe C++ source; does not change the production runner",
    )
    args = parser.parse_args()
    source, build, output = map(
        lambda p: Path(p).resolve(), (args.source, args.build, args.output)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".building-{os.getpid()}")
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "apps/ninfer"], text=True
    )
    links = [
        shlex.split(line)
        for line in commands.splitlines()
        if " -o apps/ninfer " in line
    ]
    if len(links) != 1:
        raise RuntimeError(
            "Expected one apps/ninfer link command in the selected Ninja build"
        )
    link = links[0]
    if link[:2] != [":", "&&"] or link[-2:] != ["&&", ":"]:
        raise RuntimeError("Unrecognized Ninja link command framing")
    link = link[2:-2]
    index = link.index("-o")
    cpp = (
        Path(args.runner_source).resolve(strict=True)
        if args.runner_source
        else (
            Path(__file__).resolve().parents[1]
            / "src/venturi/backends/csrc/qwen_runner.cpp"
        )
    )
    command = [
        link[0],
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-DSPDLOG_COMPILED_LIB",
        "-I" + str(source / "include"),
        "-I" + str(source / "third_party"),
        "-I" + str(source / "third_party/spdlog/include"),
        "-I" + str(Path(args.cuda) / "include"),
        str(cpp),
        "-o",
        str(temporary),
    ] + link[index + 2 :]
    try:
        subprocess.run(command, cwd=build, check=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    attribution = output.parent / "external-ninfer-attribution"
    attribution.mkdir(exist_ok=True)
    hashes = {}
    for name in ("LICENSE", "NOTICE", "upstream-provenance.json"):
        data = (source / name).read_bytes()
        (attribution / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    report = {
        "backend": "external-ninfer-cinference",
        "source": str(source),
        "build": str(build),
        "runner_source_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "engine_header_sha256": hashlib.sha256(
            (source / "include/ninfer/engine.h").read_bytes()
        ).hexdigest(),
        "external_attribution_sha256": hashes,
        "linked_archive_sha256": {
            str(
                Path(p).resolve() if Path(p).is_absolute() else (build / p).resolve()
            ): hashlib.sha256(
                (Path(p) if Path(p).is_absolute() else build / p).read_bytes()
            ).hexdigest()
            for p in link[index + 2 :]
            if p.endswith(".a")
        },
        "command": command,
        "scope": "Own transport linked to existing external native implementation; not a new Qwen kernel implementation.",
    }
    output.with_suffix(".build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
