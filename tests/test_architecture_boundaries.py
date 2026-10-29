"""Dependency contracts use fresh processes; pytest's Torch fixture cannot hide leaks."""

import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_architecture.py"


def isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


BLOCK_FRAMEWORKS = """
import importlib.abc, importlib, json, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'siphon'}:
            raise AssertionError('Unexpected framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
"""


@pytest.mark.parametrize(
    "module",
    [
        "venturi.cli",
        "venturi.commands.registry",
        "venturi.commands.checkpoint",
        "venturi.models",
        "venturi.ops",
        "venturi.text",
        "venturi.text.contracts",
        "venturi.checkpoint",
        "venturi.dtypes",
        "venturi.flash_next",
        "venturi.ninfer",
        "venturi.model_assets",
        "venturi.model_config",
        "venturi.tokenizer_assets",
        "venturi.gguf_inventory",
        "venturi.gguf_metadata",
    ],
)
def test_namespace_and_metadata_import_without_tensor_framework(module):
    report = isolated(
        BLOCK_FRAMEWORKS
        + f"""
importlib.import_module({module!r})
print(json.dumps({{'torch': 'torch' in sys.modules}}))
"""
    )
    assert report == {"torch": False}


@pytest.mark.parametrize(
    "arguments,exit_code",
    [
        (["--help"], 0),
        (["inspect", "--help"], 0),
        (["load", "--help"], 0),
        (["verify", "--help"], 0),
        (["generate", "--help"], 0),
        (["benchmark", "--help"], 0),
        (["qwen", "--help"], 2),
        (["flash-next", "--help"], 0),
        (["unknown-command"], 2),
        ([], 2),
    ],
)
def test_help_and_dispatch_errors_do_not_import_a_runtime(arguments, exit_code):
    report = isolated(
        BLOCK_FRAMEWORKS
        + f"""
import contextlib, io
from venturi.cli import main
output, error = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
    try:
        code = main({arguments!r})
    except SystemExit as e:
        code = e.code
print(json.dumps(dict(code=code, output=output.getvalue(), error=error.getvalue())))
"""
    )
    assert report["code"] == exit_code
    assert "usage:" in report["output"] + report["error"]


@pytest.mark.parametrize(
    "package,symbol,unrelated",
    [
        (
            "venturi.models",
            "GPTNeoXConfig",
            ("venturi.models.qwen_native", "venturi.models.llama", "venturi.ops"),
        ),
        (
            "venturi.ops",
            "DenseWeight",
            ("venturi.ops.blocks", "venturi.ops.ada", "venturi.models"),
        ),
        ("venturi.generation", "LocalTokenizer", ("venturi.models", "venturi.ops")),
    ],
)
def test_selected_symbol_does_not_import_unrelated_implementations(
    package, symbol, unrelated
):
    report = isolated(f"""
import importlib, json, sys
module = importlib.import_module({package!r})
getattr(module, {symbol!r})
print(json.dumps({{name: name in sys.modules for name in {unrelated!r}}}))
""")
    assert not any(report.values()), report


def test_public_exports_keep_identities_and_unknown_attributes_fail():
    for package in ("venturi.models", "venturi.ops"):
        namespace = importlib.import_module(package)
        assert list(namespace._EXPORTS) == namespace.__all__
        for name, module in namespace._EXPORTS.items():
            assert name in dir(namespace)
            value = getattr(namespace, name)
            assert value is getattr(importlib.import_module(module, package), name)
            assert getattr(namespace, name) is value
        with pytest.raises(AttributeError, match="has no attribute"):
            namespace.this_public_symbol_does_not_exist


def test_shared_text_keeps_legacy_aliases():
    from venturi.generation import LocalTokenizer as old_tokenizer
    from venturi.models.gpt_neox import validate_token_ids as old_validator
    from venturi.text import LocalTokenizer, validate_token_ids

    assert old_tokenizer is LocalTokenizer
    assert old_validator is validate_token_ids


def test_registry_is_closed_immutable_and_has_no_duplicate_names():
    from venturi.commands.registry import COMMANDS, COMMAND_BY_NAME

    assert len(COMMAND_BY_NAME) == len(COMMANDS) == 8
    with pytest.raises(TypeError):
        COMMAND_BY_NAME["other"] = COMMANDS[0]
    with pytest.raises(AttributeError):
        COMMANDS[0].module = "arbitrary_plugin"


def test_command_dispatch_forwards_only_the_selected_route(monkeypatch):
    from types import SimpleNamespace
    from venturi import cli

    calls = []

    def load(module):
        calls.append(("module", module))
        return SimpleNamespace(main=lambda *a, **kw: calls.append((a, kw)) or 17)

    monkeypatch.setattr(cli, "import_module", load)
    arguments = ["flash-next", "inspect", "somewhere", "--json"]
    assert cli.main(arguments) == 17
    assert arguments[0] == "flash-next"
    assert calls == [("module", "venturi.flash_next_cli"), ((arguments[1:],), {})]


def test_dense_dispatch_retains_legacy_injected_hook(monkeypatch):
    from types import SimpleNamespace
    from venturi import cli

    observed = {}

    def replacement(*args, **kwargs):
        return None

    def handler(argv, **kwargs):
        observed.update(kwargs)
        return 19

    monkeypatch.setattr(cli, "load_checkpoint", replacement)
    monkeypatch.setattr(
        cli, "import_module", lambda module: SimpleNamespace(main=handler)
    )
    assert cli.main(["load", "somewhere"]) == 19
    assert observed["load"] is replacement


def test_real_source_dependency_guard():
    report = runpy.run_path(str(TOOL))["inspect_source"](ROOT / "src")
    assert not report["violations"], report["violations"]
    assert all(row["path"].startswith("venturi/") for row in report["modules"])


@pytest.mark.parametrize(
    "module,source,rule",
    [
        (
            "ops.bad",
            "from ..models import Something",
            "operators-do-not-import-models-or-commands",
        ),
        (
            "ops.bad",
            "def run():\n    from ..models import Something",
            "operators-do-not-import-models-or-commands",
        ),
        ("models.bad", "from ..cli import main", "implementations-do-not-import-cli"),
        ("text.bad", "import torch", "shared-text-is-model-and-framework-independent"),
        ("checkpoint", "import torch", "metadata-does-not-eagerly-import-torch"),
        (
            "model_assets",
            "from .loader import load_checkpoint",
            "metadata-does-not-depend-on-execution",
        ),
        (
            "cli",
            "from .models import Anything",
            "dispatch-imports-command-metadata-only",
        ),
        (
            "ops.__init__",
            "from .blocks import CudaBackend",
            "public-namespaces-resolve-exports-lazily",
        ),
    ],
)
def test_guard_rejects_dependency_regressions(tmp_path, module, source, rule):
    p = tmp_path / "venturi" / (module.replace(".", "/") + ".py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source)
    report = runpy.run_path(str(TOOL))["inspect_source"](tmp_path)
    assert rule in {v["rule"] for v in report["violations"]}


def test_guard_distinguishes_deferred_and_type_only_tensor_imports(tmp_path):
    p = tmp_path / "venturi/dtypes.py"
    p.parent.mkdir(parents=True)
    p.write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\ndef resolve():\n    import torch\n"
    )
    report = runpy.run_path(str(TOOL))["inspect_source"](tmp_path)
    assert not report["violations"]
    edges = [e for e in report["modules"][0]["imports"] if e["target"] == "torch"]
    assert len(edges) == 1 and edges[0]["eager"] is False


# Frozen pre-refactor public names: new exports may be added, old ones stay.
LEGACY_PUBLIC_NAMES = {
    "venturi.models": (
        "AdaptiveQwenMTPBatch",
        "NativeQwenBatchSession",
        "NativeQwenMTPBatchSession",
        "QwenMTPPlan",
        "QwenMTPWeights",
        "NativeQwenMTPModel",
        "NativeQwenMTPSession",
        "NativeQwenModel",
        "NativeQwenPipelineModel",
        "NativeQwenPipelineSession",
        "NativeQwenSession",
        "StreamingQwenModel",
        "StreamingQwenSession",
        "qwen_request_bytes",
        "ResidentQwenModel",
        "QwenSession",
        "qwen_resident_bytes",
        "ResidentQwenAttentionLayer",
        "PreparedQwenAttentionLayer",
        "QwenAttentionWorkspace",
        "QwenAttentionStep",
        "QwenTextRoPE",
        "GDNLayerState",
        "PreparedQwenGDNLayer",
        "QwenGDNWorkspace",
        "ResidentQwenGDNLayer",
        "QwenSourceBinding",
        "QwenSourcePlan",
        "QwenQ8Weights",
        "GDNFrontendWeights",
        "GDNMixerWorkspace",
        "PreparedGDNMixerCore",
        "ResidentGDNFrontend",
        "LlamaConfig",
        "LlamaWeights",
        "LlamaModel",
        "LlamaSession",
        "GPTNeoXConfig",
        "GPTNeoXModel",
        "GPTNeoXSession",
        "GatedMLPWeights",
        "GatedMLPWorkspace",
        "PreparedGatedMLP",
        "ResidentGatedMLP",
        "AttentionDimensions",
        "DecoderLayerWeights",
        "ResidentDecoderLayer",
        "PreparedDecoderLayer",
        "DecoderStep",
        "CapturedDecoderStep",
        "DecoderWorkspace",
        "LayerKVCache",
    ),
    "venturi.ops": (
        "Backend",
        "BackendCapabilities",
        "LinearPlan",
        "LinearWeight",
        "OperatorBackend",
        "OperatorSupport",
        "UnsupportedOperationError",
        "create_backend",
        "PreparedQ8DP4A",
        "prepare_q8_dp4a",
        "PreparedQ8WMMA",
        "prepare_q8_wmma",
        "PreparedGreedy",
        "prepare_greedy",
        "ResidentQwenQKNorm",
        "PreparedQwenQKNormRoPE",
        "PreparedQwenOutputGate",
        "ResidentCausalConv",
        "PreparedCausalConv",
        "ResidentGDNGates",
        "PreparedGDNGates",
        "ResidentGatedRMSNorm",
        "PreparedGatedRMSNorm",
        "NInferVocabulary",
        "Q8VocabularyWeights",
        "ResidentQ8Vocabulary",
        "PreparedEmbedding",
        "PreparedGDN",
        "AdaBackend",
        "PreparedLinear",
        "Q4RowSplitWeight",
        "Q8RowSplitWeight",
        "DenseWeight",
        "CudaBackend",
        "TorchBackend",
        "ResidentLinear",
        "ResidentRMSNorm",
        "ResidentQ4Weight",
        "PreparedRoPE",
        "PreparedCausalAttention",
    ),
}


@pytest.mark.parametrize("package", LEGACY_PUBLIC_NAMES)
def test_legacy_public_names_are_not_dropped(package):
    module = importlib.import_module(package)
    assert set(LEGACY_PUBLIC_NAMES[package]) <= set(module.__all__)
