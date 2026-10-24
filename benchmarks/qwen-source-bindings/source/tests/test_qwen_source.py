"""Independent source schemas and bounded floating-HF to Q8 export fixtures."""

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from venturi.checkpoint import CheckpointError, inspect_checkpoint
from venturi.models import GDNFrontendWeights
from venturi.models.qwen_source import QwenSourcePlan
from venturi.ninfer import NInferArtifact
from venturi.ops import NInferVocabulary, TorchBackend
from venturi.qwen_export import export_qwen_q8
import venturi.qwen_export as exporter


def source_config():
    return dict(
        model_type="qwen3_5",
        tie_word_embeddings=False,
        text_config=dict(
            model_type="qwen3_5_text",
            dtype="bfloat16",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=64,
            linear_value_head_dim=13,
            linear_conv_kernel_dim=4,
            vocab_size=17,
            max_position_embeddings=128,
            hidden_act="silu",
            attention_bias=False,
            attn_output_gate=True,
            rms_norm_eps=1e-6,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters=dict(
                rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25
            ),
        ),
    )


def fixture_tensors():
    # Explicit HF reference shapes, independent of the production plan builder.
    shapes = {
        "lm_head.weight": (17, 32),
        "model.language_model.embed_tokens.weight": (17, 32),
        "model.language_model.norm.weight": (32,),
    }
    for i in range(2):
        p = f"model.language_model.layers.{i}."
        shapes.update(
            {
                p + "input_layernorm.weight": (32,),
                p + "post_attention_layernorm.weight": (32,),
                p + "mlp.gate_proj.weight": (64, 32),
                p + "mlp.up_proj.weight": (64, 32),
                p + "mlp.down_proj.weight": (32, 64),
            }
        )
    p = "model.language_model.layers.0.linear_attn."
    shapes.update(
        {
            p + "in_proj_qkv.weight": (308, 32),
            p + "in_proj_z.weight": (52, 32),
            p + "in_proj_a.weight": (4, 32),
            p + "in_proj_b.weight": (4, 32),
            p + "out_proj.weight": (32, 52),
            p + "conv1d.weight": (308, 1, 4),
            p + "norm.weight": (13,),
            p + "A_log": (4,),
            p + "dt_bias": (4,),
        }
    )
    p = "model.language_model.layers.1.self_attn."
    shapes.update(
        {
            p + "q_proj.weight": (64, 32),
            p + "k_proj.weight": (16, 32),
            p + "v_proj.weight": (16, 32),
            p + "o_proj.weight": (32, 32),
            p + "q_norm.weight": (16,),
            p + "k_norm.weight": (16,),
        }
    )
    result = {}
    for i, (name, shape) in enumerate(shapes.items()):
        count = torch.tensor(shape).prod().item()
        data = (((torch.arange(count) + i) % 251 - 125) / 16384).reshape(shape)
        result[name] = data.to(
            torch.float32 if name.endswith("A_log") else torch.bfloat16
        )
    # Deliberately asymmetric taps and per-head Q/gate rows detect reordering.
    result["model.visual.unused.weight"] = torch.ones(3, 5, dtype=torch.bfloat16)
    result["mtp.layers.0.unused.weight"] = torch.ones(2, 5, dtype=torch.bfloat16)
    return result


def write_source(root, *, raw=None, tensors=None, verified=True, shards=2):
    root.mkdir()
    raw = source_config() if raw is None else raw
    tensors = fixture_tensors() if tensors is None else tensors
    (root / "config.json").write_text(json.dumps(raw))
    weight_map = {}
    for i in range(shards):
        selected = {k: v for j, (k, v) in enumerate(tensors.items()) if j % shards == i}
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        # Independent safetensors framing, with no optional NumPy save helper.
        # All fixture payloads are small CPU tensors; production export uses
        # bounded zero-copy writes, not this diagnostic byte-list construction.
        header, chunks, offset = {}, [], 0
        for key, tensor in selected.items():
            data = bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
            header[key] = dict(
                dtype={torch.bfloat16: "BF16", torch.float32: "F32", torch.int8: "I8"}[
                    tensor.dtype
                ],
                shape=list(tensor.shape),
                data_offsets=[offset, offset + len(data)],
            )
            chunks.append(data)
            offset += len(data)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        (root / name).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
        )
        weight_map.update({k: name for k in selected})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=weight_map))
    )
    if verified:
        record = dict(
            status="complete",
            repo="synthetic/test-fixture",
            revision="fixture-not-trained-checkpoint",
            files=[],
        )
        for p in sorted(root.iterdir()):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            record["files"].append(
                dict(
                    name=p.name,
                    bytes=p.stat().st_size,
                    actual_sha256=digest,
                    lfs_sha256=None,
                )
            )
        (root / "venturi-source.json").write_text(json.dumps(record))
    return root


def test_actual_pinned_index_all_text_roles_and_exact_capacity():
    root = Path(__file__).parent / "fixtures/qwen3_5_27b"
    config, index = (
        json.loads((root / n).read_text())
        for n in ("config.json", "model.safetensors.index.json")
    )
    plan = QwenSourcePlan.from_config(config)
    assert len(plan.bindings) == 851
    assert len(plan.ignored_names(index["weight_map"])) == 348
    assert sum(s.nbytes for s in plan.bindings) == 28_601_606_144
    assert {
        f: sum(s.format == f for s in plan.bindings)
        for f in ("q8_g32_fp16", "bf16", "fp32")
    } == {"q8_g32_fp16": 402, "bf16": 353, "fp32": 96}


@pytest.mark.parametrize(
    "bad",
    [
        "quantized",
        "nested-quantized",
        "bias",
        "gates",
        "dtype",
        "layers",
        "heads",
        "tied-conflict",
    ],
)
def test_source_config_rejects_unsupported_semantics(bad):
    raw = source_config()
    c = raw["text_config"]
    if bad == "quantized":
        raw["quantization_config"] = {"bits": 4}
    elif bad == "nested-quantized":
        c["quantization_config"] = {"bits": 8}
    elif bad == "bias":
        c["attention_bias"] = True
    elif bad == "gates":
        c["attn_output_gate"] = False
    elif bad == "dtype":
        c["dtype"] = "float8"
    elif bad == "layers":
        c["layer_types"] = ["full_attention"]
    elif bad == "heads":
        c["linear_num_value_heads"] = 3
    else:
        c["tie_word_embeddings"] = True
    with pytest.raises(ValueError):
        QwenSourcePlan.from_config(raw)


def test_metadata_rejects_missing_unexpected_shape_and_quantized_storage(tmp_path):
    tensors = fixture_tensors()
    for mode in ("missing", "unexpected", "shape", "dtype"):
        values = dict(tensors)
        name = "model.language_model.layers.0.linear_attn.in_proj_z.weight"
        if mode == "missing":
            values.pop(name)
        elif mode == "unexpected":
            values["model.language_model.extra.weight"] = torch.ones(1)
        elif mode == "shape":
            values[name] = values[name][:1]
        else:
            values[name] = values[name].to(torch.int8)
        source = write_source(tmp_path / mode, tensors=values)
        with pytest.raises(CheckpointError):
            QwenSourcePlan.from_config(source_config()).validate_checkpoint(
                inspect_checkpoint(source)
            )


@pytest.mark.parametrize("rows", [1, 7, 256])
def test_export_full_storage_roundtrip_independent_q8_rows_and_constants(
    tmp_path, rows
):
    tensors = fixture_tensors()
    source = write_source(tmp_path / "source", tensors=tensors)
    result = export_qwen_q8(source, tmp_path / "new-q8.ninfer", rows_per_chunk=rows)
    artifact = NInferArtifact(result["output"])
    assert (
        result["sha256"]
        == hashlib.sha256(Path(result["output"]).read_bytes()).hexdigest()
    )
    assert result["tensor_count"] == 28 and result["ignored_source_tensors"] == 2
    assert result["source"]["kind"] == "verified-acquisition"
    for spec in QwenSourcePlan.from_config(source_config()).bindings:
        obj = artifact.objects[artifact.manifest["bindings"][spec.binding]["object"]]
        value = tensors[spec.source]
        if spec.format == "q8_g32_fp16":
            selected_rows = {0, value.shape[0] - 1}
            if spec.binding.endswith("attention/q"):
                selected_rows.update((15, 16, 31, 32, 47, 48))
            for row in sorted(selected_rows):
                # Python round is nearest-even; expected scale uses BF16 source
                # values and the stored FP16 divisor, not an unrounded scale.
                raw = value[row].double().tolist()
                expected = []
                for start in range(0, len(raw), 32):
                    group = raw[start : start + 32]
                    scale = torch.tensor(
                        max(map(abs, group)) / 127, dtype=torch.float16
                    ).item()
                    expected.extend(
                        min(127, max(-127, round(x / scale))) * scale if scale else 0
                        for x in group
                    )
                assert artifact.decode_row(obj["id"], row) == expected
        else:
            raw = artifact.read_object(obj["id"])
            dtype = torch.float32 if spec.format == "fp32" else torch.bfloat16
            actual = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(obj["shape"])
            expected = value.squeeze(1).T if spec.transform == "conv-kc" else value
            assert torch.equal(actual, expected.to(dtype))
    # Existing native-operator binders consume the new source artifact unchanged.
    frontend = GDNFrontendWeights.from_ninfer(artifact, layer=0)
    assert torch.equal(
        frontend.conv,
        tensors["model.language_model.layers.0.linear_attn.conv1d.weight"].squeeze(1).T,
    )
    vocabulary = NInferVocabulary.upload_from_ninfer(
        artifact, TorchBackend(), device="cpu"
    )
    out = torch.empty(2, 32, dtype=torch.bfloat16)
    vocabulary.embedding.prepare_embedding([0, 16]).run(out=out)
    obj = artifact.manifest["bindings"]["text/token_embedding"]["object"]
    assert torch.equal(
        out, torch.tensor([artifact.decode_row(obj, i) for i in (0, 16)]).bfloat16()
    )
    # Raw zero-centered decoder gamma and Q/gate head ordering were not transformed.
    assert artifact.manifest["provenance"]["transforms"] == {
        "text/layers/0/gdn/convolution": "conv-kc"
    }
    assert not list(tmp_path.glob(".venturi-q8-*.partial"))


def test_streaming_reads_bounded_rows_and_no_gpu_work(tmp_path, monkeypatch):
    source = write_source(tmp_path / "source")
    original = exporter.safe_open
    rows_seen, full_reads = [], []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original(*args, **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_tensor(self, name):
            full_reads.append(name)
            return self.reader.get_tensor(name)

        def get_slice(self, name):
            sliced = self.reader.get_slice(name)

            class Slice:
                def get_shape(self):
                    return sliced.get_shape()

                def get_dtype(self):
                    return sliced.get_dtype()

                def __getitem__(self, span):
                    rows_seen.append(span.stop - span.start)
                    return sliced[span]

            return Slice()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU exporter used CUDA")

    monkeypatch.setattr(exporter, "safe_open", Reader)
    for name in ("init", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    export_qwen_q8(source, tmp_path / "new.ninfer", rows_per_chunk=7)
    assert rows_seen and max(rows_seen) <= 7
    quantized = {
        s.source
        for s in QwenSourcePlan.from_config(source_config()).bindings
        if s.format == "q8_g32_fp16"
    }
    assert not quantized.intersection(full_reads)


def test_partial_and_modified_verified_sources_never_start_conversion(
    tmp_path, monkeypatch
):
    source = write_source(tmp_path / "source")
    manifest = source / "venturi-source.json"
    original = json.loads(manifest.read_text())
    partial = copy.deepcopy(original)
    partial["status"] = "downloading"
    manifest.write_text(json.dumps(partial))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid source began conversion")

    monkeypatch.setattr(exporter, "safe_open", forbidden)
    with pytest.raises(CheckpointError, match="not complete"):
        export_qwen_q8(source, tmp_path / "new.ninfer")
    manifest.write_text(json.dumps(original))
    shard = next(source.glob("*.safetensors"))
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(CheckpointError, match="hash mismatch"):
        export_qwen_q8(source, tmp_path / "new.ninfer")
    assert not (tmp_path / "new.ninfer").exists()


def test_explicit_unverified_opt_in_is_labeled_and_existing_outputs_preserved(tmp_path):
    source = write_source(tmp_path / "source", verified=False)
    target = tmp_path / "new.ninfer"
    with pytest.raises(CheckpointError, match="Verified"):
        export_qwen_q8(source, target)
    export_qwen_q8(source, target, allow_unverified_source=True)
    artifact = NInferArtifact(target)
    assert (
        artifact.manifest["provenance"]["source"]["kind"]
        == "explicitly-unverified-local-floating-source"
    )
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        export_qwen_q8(source, target, allow_unverified_source=True)
    assert target.read_bytes() == before


def test_failed_conversion_and_publish_race_leave_no_partial_output(
    tmp_path, monkeypatch
):
    tensors = fixture_tensors()
    tensors["lm_head.weight"][0, 0] = float("inf")
    source = write_source(tmp_path / "bad", tensors=tensors)
    with pytest.raises(ValueError, match="finite"):
        export_qwen_q8(source, tmp_path / "new.ninfer")
    assert not (tmp_path / "new.ninfer").exists() and not list(
        tmp_path.glob(".venturi-q8-*.partial")
    )
    source = write_source(tmp_path / "good")
    target = tmp_path / "raced.ninfer"
    original_link = exporter.os.link

    def raced_link(src, dst):
        Path(dst).write_bytes(b"keep existing winner")
        return original_link(src, dst)

    monkeypatch.setattr(exporter.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        export_qwen_q8(source, target)
    assert target.read_bytes() == b"keep existing winner" and not list(
        tmp_path.glob(".venturi-q8-*.partial")
    )


def test_source_mutation_during_conversion_aborts_publication(tmp_path, monkeypatch):
    source = write_source(tmp_path / "source")
    original = exporter._write_tensor
    mutated = False

    def write_and_mutate(stream, tensor):
        nonlocal mutated
        original(stream, tensor)
        if not mutated:
            path = source / "config.json"
            path.write_text(path.read_text() + " ")
            mutated = True

    monkeypatch.setattr(exporter, "_write_tensor", write_and_mutate)
    with pytest.raises(CheckpointError, match="changed during export"):
        export_qwen_q8(source, tmp_path / "new.ninfer")
    assert not (tmp_path / "new.ninfer").exists() and not list(
        tmp_path.glob(".venturi-q8-*.partial")
    )


@pytest.mark.parametrize("tied", [False, True])
def test_standalone_text_prefix_and_explicit_tied_storage(tmp_path, tied):
    raw = source_config()["text_config"]
    raw["tie_word_embeddings"] = tied
    tensors = {
        k.replace("model.language_model.", "model."): v
        for k, v in fixture_tensors().items()
        if not k.startswith(("model.visual.", "mtp."))
    }
    if tied:
        tensors.pop("lm_head.weight")
    source = write_source(tmp_path / "source", raw=raw, tensors=tensors)
    result = export_qwen_q8(source, tmp_path / "new.ninfer")
    artifact = NInferArtifact(result["output"])
    bindings = artifact.manifest["bindings"]
    assert (bindings["text/token_embedding"] == bindings["text/output_head"]) == tied
    vocabulary = NInferVocabulary.upload_from_ninfer(
        artifact, TorchBackend(), device="cpu"
    )
    assert (vocabulary.embedding is vocabulary.head) == tied
    if tied:
        # Two stored candidates are not silently discarded on a tie flag.
        tensors["lm_head.weight"] = torch.zeros(17, 32, dtype=torch.bfloat16)
        bad = write_source(tmp_path / "duplicate", raw=raw, tensors=tensors)
        with pytest.raises(CheckpointError, match="Unexpected"):
            export_qwen_q8(bad, tmp_path / "bad.ninfer")


def test_low_disk_space_and_malformed_verification_fail_without_temporary_files(
    tmp_path, monkeypatch
):
    source = write_source(tmp_path / "source")
    monkeypatch.setattr(
        exporter.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})()
    )
    with pytest.raises(OSError, match="disk space"):
        export_qwen_q8(source, tmp_path / "new.ninfer")
    assert not list(tmp_path.glob(".venturi-q8-*.partial"))
    manifest = source / "venturi-source.json"
    record = json.loads(manifest.read_text())
    record["files"].append(dict(record["files"][0]))
    manifest.write_text(json.dumps(record))
    with pytest.raises(CheckpointError, match="unique direct"):
        export_qwen_q8(source, tmp_path / "new.ninfer")


def test_complete_runtime_bindings_preserve_encoded_and_small_float_weights(tmp_path):
    from venturi.models import QwenQ8Weights
    from venturi.ops import DenseWeight, Q8RowSplitWeight

    source = write_source(tmp_path / "source")
    result = export_qwen_q8(source, tmp_path / "new.ninfer")
    artifact = NInferArtifact(result["output"])
    weights = QwenQ8Weights.from_ninfer(artifact)
    assert len(weights.layer_bindings(0)) == 14
    assert len(weights.layer_bindings(1)) == 11
    qkv = weights.read_linear("text/layers/0/gdn/qkv")
    assert isinstance(qkv, Q8RowSplitWeight)
    assert qkv.codes.shape == (308, 128) and qkv.in_features == 32
    head = weights.read_linear("text/output_head")
    assert head.codes.dtype == torch.int8
    a = weights.read_linear("text/layers/0/gdn/a")
    assert isinstance(a, DenseWeight) and a.data.dtype == torch.bfloat16
    norm = weights.read_float("text/layers/1/attention/q_norm")
    assert torch.equal(
        norm, fixture_tensors()["model.language_model.layers.1.self_attn.q_norm.weight"]
    )
    with pytest.raises(ValueError, match="dequantizes"):
        weights.read_float("text/layers/0/gdn/qkv")
    with pytest.raises(ValueError, match="matrix"):
        weights.read_linear("text/layers/0/input_norm")
    for bad in (True, -1, 2):
        with pytest.raises(ValueError):
            weights.layer_bindings(bad)
    # A loaded weight descriptor is owned and remains usable independently.
    artifact.read_object = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("late payload read")
    )
    assert torch.isfinite(qkv.dequantize_reference()).all()


@pytest.mark.parametrize(
    "bad", ["missing", "wrong-format", "wrong-shape", "tying", "multipart"]
)
def test_all_runtime_roles_preflight_before_payload_reads(tmp_path, monkeypatch, bad):
    from venturi.models import QwenQ8Weights

    source = write_source(tmp_path / "source")
    result = export_qwen_q8(source, tmp_path / "new.ninfer")
    artifact = NInferArtifact(result["output"])
    name = "text/layers/1/attention/k_norm"
    binding = artifact.manifest["bindings"][name]
    if bad == "missing":
        artifact.manifest["bindings"].pop(name)
    elif bad == "wrong-format":
        artifact.objects[binding["object"]]["format"] = "q4_g64_fp16"
    elif bad == "wrong-shape":
        artifact.objects[binding["object"]]["shape"] = [32]
    elif bad == "tying":
        artifact.manifest["bindings"][name] = dict(
            artifact.manifest["bindings"]["text/layers/1/attention/q_norm"]
        )
    else:
        artifact.manifest["bindings"][name] = {
            "parts": [{"object": binding["object"], "range": [0, 16]}]
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid role metadata caused a payload read")

    monkeypatch.setattr(artifact, "read_object", forbidden)
    with pytest.raises(ValueError):
        QwenQ8Weights.from_ninfer(artifact)


@pytest.mark.cuda
def test_new_artifact_roles_upload_directly_to_native_q8_and_dense_operators(tmp_path):
    import os
    from venturi.models import QwenQ8Weights
    from venturi.ops import CudaBackend

    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native blocks library required")
    source = write_source(tmp_path / "source")
    result = export_qwen_q8(source, tmp_path / "new.ninfer")
    artifact = NInferArtifact(result["output"])
    weights, backend = QwenQ8Weights.from_ninfer(artifact), CudaBackend(library)
    x = (torch.arange(3 * 32).reshape(3, 32).double() / 64 - 0.5).bfloat16().cuda()
    for name in (
        "text/layers/0/gdn/qkv",
        "text/layers/0/gdn/a",
        "text/layers/1/attention/q",
    ):
        descriptor = weights.read_linear(name)
        resident = backend.upload_linear(
            descriptor, dtype=torch.bfloat16, device="cuda:0"
        )
        plan = resident.prepare(tokens=3)
        workspace = (
            torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
            if plan.workspace_bytes
            else None
        )
        out = torch.empty(3, resident.out_features, dtype=torch.bfloat16, device="cuda")
        plan.run(x, out=out, workspace=workspace)
        obj = artifact.objects[artifact.manifest["bindings"][name]["object"]]
        dense = (
            torch.tensor(
                [artifact.decode_row(obj["id"], row) for row in range(obj["shape"][0])],
                dtype=torch.float64,
            )
            if obj["format"] == "q8_g32_fp16"
            else fixture_tensors()[obj["id"]].double()
        )
        ideal = x.cpu().double() @ dense.T
        error = (out.cpu().double() - ideal.bfloat16().double()).abs()
        assert bool(
            (error <= 2 * torch.finfo(torch.bfloat16).eps * ideal.abs() + 2e-4).all()
        )
    norm = backend.upload_zero_centered_rmsnorm(
        weights.read_float("text/layers/0/input_norm"),
        dtype=torch.bfloat16,
        device="cuda:0",
    ).prepare(tokens=3)
    target = torch.empty_like(x)
    norm.run(x, out=target)
    raw = weights.read_float("text/layers/0/input_norm").double()
    value = x.cpu().double()
    ideal = (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * (1 + raw)
    )
    assert bool(
        (
            (target.cpu().double() - ideal.bfloat16().double()).abs()
            <= 2 * torch.finfo(torch.bfloat16).eps * ideal.abs() + 2e-5
        ).all()
    )


def test_export_tool_report_preflight_happens_before_conversion(tmp_path, monkeypatch):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "qwen_export_tool", Path(__file__).parents[1] / "tools/export_qwen_q8.py"
    )
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)

    def forbidden(*args, **kwargs):
        raise AssertionError("bad report path started conversion")

    monkeypatch.setattr(tool, "export_qwen_q8", forbidden)
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        tool.main(["source", str(tmp_path / "new.ninfer"), "--report", str(existing)])
    with pytest.raises(ValueError, match="different paths"):
        tool.main(
            [
                "source",
                str(tmp_path / "new.ninfer"),
                "--report",
                str(tmp_path / "new.ninfer"),
            ]
        )
    with pytest.raises(ValueError, match="parent"):
        tool.main(
            [
                "source",
                str(tmp_path / "new.ninfer"),
                "--report",
                str(tmp_path / "missing/report.json"),
            ]
        )
    assert existing.read_text() == "keep"
