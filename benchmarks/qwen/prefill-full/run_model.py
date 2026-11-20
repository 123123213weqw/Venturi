"""Remote-only, fixed-workload ABBA comparison for the full-chunk tile experiment."""

import json
from pathlib import Path
import subprocess

ROOT = Path("/home/test/wangyue/venturi-work/qwen")
REPO = ROOT / "src"
OUT = ROOT / "prefill-full"
PYTHON = "/home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python"
BASELINE = ROOT / "native/venturi-qwen-runner"
CANDIDATE = OUT / "runner-r128-w32"


def main():
    commands = []

    def run(command, log):
        commands.append(command)
        with (OUT / log).open("w") as sink:
            subprocess.run(
                command, cwd=REPO, stdout=sink, stderr=subprocess.STDOUT, check=True
            )

    run(
        [
            PYTHON,
            "tools/build_qwen_runner.py",
            "--source",
            "/home/test/wangyue/engine",
            "--build",
            "/home/test/wangyue/engine/build",
            "--output",
            str(CANDIDATE),
            "--ops-object",
            str(OUT / "prefill_r128_w32/candidate.o"),
        ],
        "runner-build.log",
    )
    checkpoint = (ROOT / "staged-path.txt").read_text().strip()
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    for name, runner in [
        ("baseline-a", BASELINE),
        ("r128-a", CANDIDATE),
        ("r128-b", CANDIDATE),
        ("baseline-b", BASELINE),
    ]:
        run(
            [
                PYTHON,
                "tools/benchmark_qwen_prefill.py",
                checkpoint,
                "--runner",
                str(runner),
                "--output-directory",
                str(OUT / ("model-" + name)),
                "--lengths",
                "128",
                "509",
                "512",
                "1021",
                "2048",
                "8192",
                "--repeats",
                "3",
            ],
            "model-" + name + ".log",
        )
        print(name, "completed", flush=True)
    (OUT / "model-commands.json").write_text(json.dumps(commands, indent=2) + "\n")


if __name__ == "__main__":
    main()
