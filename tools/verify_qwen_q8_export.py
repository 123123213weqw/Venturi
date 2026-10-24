"""CPU-only independent storage verification after complete Qwen Q8 publication."""

import argparse
import json
from pathlib import Path

from venturi.qwen_export_verify import verify_qwen_q8_export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("artifact")
    parser.add_argument("--export-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--component",choices=["text","mtp"],default="text")
    parser.add_argument("--source-manifest")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Verification report already exists")
    result = verify_qwen_q8_export(args.source, args.artifact, args.export_report,component=args.component,source_manifest=args.source_manifest)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
