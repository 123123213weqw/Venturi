"""Explicit CPU-only original Qwen3.5 HF source to a separate Q8 text artifact."""

import argparse
import json
from pathlib import Path

from venturi.qwen_export import export_qwen_q8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--source-manifest")
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument("--allow-unverified-source", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if args.report:
        report_path = Path(args.report).expanduser().absolute()
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError("Refusing to replace an existing export report")
        if not report_path.parent.is_dir():
            raise ValueError("Report parent directory must already exist")
        if report_path == Path(args.output).expanduser().absolute():
            raise ValueError("Artifact and report must have different paths")
    report = export_qwen_q8(
        args.source,
        args.output,
        rows_per_chunk=args.rows_per_chunk,
        source_manifest=args.source_manifest,
        allow_unverified_source=args.allow_unverified_source,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        with report_path.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
