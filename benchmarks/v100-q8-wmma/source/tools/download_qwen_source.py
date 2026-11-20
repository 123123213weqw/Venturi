"""Explicit, resumable official BF16 checkpoint acquisition; no model execution.

User-authorized remote-only download, pinned to a Hub commit. Preserves partial
files for retries and records complete-file SHA256 provenance. Never upgrades
Q4 weights into a falsely labeled original/Q8 checkpoint or deletes old assets.
"""

import argparse
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import sleep, time


PATTERNS = (
    "*.safetensors",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "*.jinja",
    "LICENSE*",
    "README.md",
)


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_config(raw):
    if raw.get("model_type") != "qwen3_5":
        raise ValueError("Expected official Qwen3.5 multimodal checkpoint container")
    text = raw.get("text_config", {})
    expected = dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=248320,
    )
    if any(type(text.get(k)) is not int or text[k] != v for k, v in expected.items()):
        raise ValueError(
            "Source checkpoint geometry differs from the current Qwen-27B target"
        )
    if (
        text.get("dtype") != "bfloat16"
        or raw.get("quantization_config") is not None
        or text.get("quantization_config") is not None
    ):
        raise ValueError("Expected original BF16 source, not a quantized substitute")
    return text


def selected_files(siblings):
    files = []
    for f in siblings:
        if not any(fnmatch(f.rfilename, pattern) for pattern in PATTERNS):
            continue
        if (
            Path(f.rfilename).is_absolute()
            or ".." in Path(f.rfilename).parts
            or type(f.size) is not int
            or f.size < 0
        ):
            raise ValueError("Invalid source filename/size metadata")
        lfs = getattr(f, "lfs", None)
        digest = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        files.append(dict(name=f.rfilename, bytes=f.size, lfs_sha256=digest))
    if not any(f["name"].endswith(".safetensors") for f in files):
        raise ValueError("No original safetensors shards found")
    return files


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    # Import only after explicit invocation, with process-local public-download
    # settings. Never print or forward user authentication environment values.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "venturi-source.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["repo"] != args.repo:
        raise ValueError("Existing source directory belongs to a different repository")
    revision = previous["revision"] if previous else args.revision
    info = HfApi(endpoint=args.endpoint).model_info(
        args.repo, revision=revision, files_metadata=True, token=False, timeout=20
    )
    files = selected_files(info.siblings)
    if previous is None and any(out.iterdir()):
        raise ValueError(
            "Refusing to mix a new source into a nonempty unmanaged directory"
        )
    record = dict(
        repo=args.repo,
        source_endpoint=args.endpoint,
        revision=info.sha,
        status="downloading",
        started_unix=time(),
        files=files,
        total_source_bytes=sum(f["bytes"] for f in files),
        max_workers=args.max_workers,
    )
    atomic_json(manifest_path, record)
    # Reserve space for source + a separate ~28.6 GB Q8 artifact + normal scratch.
    remaining = sum(
        max(0, f["bytes"] - (out / f["name"]).stat().st_size)
        if (out / f["name"]).is_file()
        else f["bytes"]
        for f in files
    )
    required = remaining + 28_601_606_144 + int(args.reserve_gib * 2**30)
    if remaining and shutil.disk_usage(out).free < required:
        raise ValueError(
            f"Insufficient disk space: require {required} free bytes including Q8 output/reserve"
        )
    config_path = hf_hub_download(
        args.repo,
        "config.json",
        revision=info.sha,
        local_dir=out,
        token=False,
        endpoint=args.endpoint,
    )
    record["text_config"] = validate_config(json.loads(Path(config_path).read_text()))
    atomic_json(manifest_path, record)
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": info.sha,
                "source_bytes": record["total_source_bytes"],
                "output": str(out),
            }
        ),
        flush=True,
    )
    for attempt in range(1, args.retries + 1):
        record["attempt"] = attempt
        atomic_json(manifest_path, record)
        try:
            snapshot_download(
                args.repo,
                revision=info.sha,
                local_dir=out,
                allow_patterns=list(PATTERNS),
                max_workers=args.max_workers,
                token=False,
                endpoint=args.endpoint,
            )
            break
        except Exception as error:
            record["last_error"] = f"{type(error).__name__}: {error}"
            atomic_json(manifest_path, record)
            print(
                f"Download attempt {attempt} failed; keeping partial files for resume: {type(error).__name__}",
                flush=True,
            )
            if attempt == args.retries:
                raise
            sleep(5 * attempt)
    record["status"] = "verifying"
    atomic_json(manifest_path, record)
    for f in files:
        path = out / f["name"]
        if not path.is_file() or path.stat().st_size != f["bytes"]:
            raise ValueError(f"Incomplete checkpoint file: {f['name']}")
        digest = sha256(path)
        if f["lfs_sha256"] is not None and digest != f["lfs_sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {f['name']}")
        f["actual_sha256"] = digest
        atomic_json(manifest_path, record)
    record["status"], record["completed_unix"] = "complete", time()
    record.pop("last_error", None)
    atomic_json(manifest_path, record)
    print("Original BF16 checkpoint download and verification complete.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--endpoint",
        choices=("https://huggingface.co", "https://hf-mirror.com"),
        default="https://huggingface.co",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reserve-gib", type=float, default=10)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.max_workers <= 4
        or not 1 <= args.retries <= 5
        or not 0 <= args.reserve_gib <= 100
    ):
        raise ValueError("Invalid download concurrency/retries/reserve")
    try:
        run(args)
    except Exception as error:
        manifest = Path(args.output).expanduser().resolve() / "venturi-source.json"
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if record.get("repo") == args.repo and record.get("status") in {
                "downloading",
                "verifying",
            }:
                record["status"], record["error_type"] = "failed", type(error).__name__
                atomic_json(manifest, record)
        raise


if __name__ == "__main__":
    main()
