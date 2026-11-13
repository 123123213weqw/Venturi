"""Two concrete libraries: dependency direction, cache boundaries and error ownership."""

import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from import_budget import framework_import_timeout
from venturi.ops.build import _source_hashes

ROOT = Path(__file__).parents[1] / "src/venturi"


def test_operator_sources_do_not_include_model_runtime():
    files = _source_hashes(ROOT / "ops/csrc/blocks.cu", ROOT)
    assert files
    assert all(name.startswith("ops/") for name in files)
    assert not any("qwen_runtime" in name or "weight_source" in name for name in files)
    assert "ops/csrc/ops_api.h" in files


def test_qwen_includes_only_operator_contract_and_stateless_helpers():
    files = _source_hashes(ROOT / "models/qwen/csrc/qwen.cu", ROOT)
    assert {name for name in files if name.startswith("ops/")} == {
        "ops/csrc/ops_api.h",
        "ops/csrc/cuda_common.cuh",
    }
    assert "models/qwen/csrc/qwen_runtime.cuh" in files
    text = "\n".join(
        (ROOT / name).read_text() for name in files if name.startswith("models/")
    )
    for private in (
        "static_cast<Linear*>",
        "vdq::",
        "vproducer::",
        "vprojection::",
        "vrq::",
    ):
        assert private not in text


def test_qwen_change_does_not_invalidate_operator_build_inputs(tmp_path):
    root = tmp_path / "venturi"
    shutil.copytree(ROOT / "ops/csrc", root / "ops/csrc")
    shutil.copytree(ROOT / "models/qwen/csrc", root / "models/qwen/csrc")
    ops_entry = root / "ops/csrc/blocks.cu"
    model_entry = root / "models/qwen/csrc/qwen.cu"
    before = _source_hashes(ops_entry, root)
    model_before = _source_hashes(model_entry, root)
    with (root / "models/qwen/csrc/qwen_runtime.cuh").open("a") as f:
        f.write("\n// model-only change\n")
    assert _source_hashes(ops_entry, root) == before
    assert _source_hashes(model_entry, root) != model_before


def libraries():
    operators = os.environ.get("VENTURI_OPS_LIBRARY")
    qwen = os.environ.get("VENTURI_QWEN_LIBRARY")
    if not operators or not qwen:
        pytest.skip("Explicit split ops/Qwen libraries required")
    return Path(operators), Path(qwen)


def test_exported_symbols_and_concrete_link_dependency():
    operators, qwen = libraries()

    def symbols(path):
        return subprocess.check_output(
            ["nm", "-D", "--defined-only", str(path)], text=True
        )

    op_symbols = symbols(operators)
    qwen_symbols = symbols(qwen)
    assert " venturi_dense_run\n" in op_symbols
    assert " venturi_qwen_create\n" not in op_symbols
    assert " venturi_qwen_create\n" in qwen_symbols
    assert " venturi_dense_run\n" not in qwen_symbols
    deps = subprocess.check_output(["readelf", "-d", str(qwen)], text=True)
    soname = json.loads(operators.with_name("build.json").read_text())["soname"]
    assert soname in deps and "$ORIGIN" in deps


def test_operator_load_does_not_load_qwen_or_import_models():
    operators, _ = libraries()
    code = """
import sys
from venturi.ops import CudaBackend
b=CudaBackend(sys.argv[1])
assert not any(name.startswith('venturi.models') for name in sys.modules)
assert 'libventuri_qwen' not in open('/proc/self/maps').read()
assert b._native.venturi_blocks_abi()==17
import torch
from venturi.ops import DenseWeight
weight=b.upload_linear(DenseWeight(torch.tensor([[3.,4.],[5.,6.]],dtype=torch.float16)),dtype=torch.float16,device='cuda:0')
p=weight.prepare(tokens=1)
x=torch.tensor([[1.,2.]],dtype=torch.float16,device='cuda:0')
y=torch.empty((1,2),dtype=torch.float16,device='cuda:0')
workspace=torch.empty(p.workspace_bytes,dtype=torch.uint8,device='cuda:0')
p.run(x,out=y,workspace=workspace)
assert y.cpu().tolist()==[[11.,17.]]
assert 'libventuri_qwen' not in open('/proc/self/maps').read()
p.close()
"""
    subprocess.run(
        [sys.executable, "-c", code, str(operators)],
        check=True,
        timeout=framework_import_timeout(),
    )


def test_runtime_errors_are_not_operator_errors_and_wrong_pair_rejected(tmp_path):
    operators, qwen = libraries()
    from venturi.ops import CudaBackend
    from venturi.models.qwen.native_runtime import QwenRuntime
    from venturi.models.qwen_load.abi import _bind

    backend = CudaBackend(operators)
    runtime = QwenRuntime(qwen, backend)
    native = _bind(runtime)
    assert backend._native.venturi_rms_run(None, None, None, 0, 1, 0, 1e-6, None) < 0
    op_error = backend._native.venturi_blocks_error()
    assert not native.venturi_qwen_create(None, None, 0, None, None, None, None, None)
    model_error = runtime._native.venturi_qwen_error()
    assert b"descriptor" in model_error
    assert backend._native.venturi_blocks_error() == op_error
    assert backend._native.venturi_rms_run(None, None, None, 1, 1, 0, 1e-6, None) < 0
    assert runtime._native.venturi_qwen_error() == model_error
    copy = tmp_path / "other-ops.so"
    shutil.copyfile(operators, copy)
    wrong = CudaBackend(copy)
    with pytest.raises(RuntimeError, match="different operator library instance"):
        QwenRuntime(qwen, wrong)


def test_relocated_pair_loads_without_original_build_directories(tmp_path):
    operators, qwen = libraries()
    op_record = json.loads(operators.with_name("build.json").read_text())
    moved_ops = tmp_path / op_record["soname"]
    moved_qwen = tmp_path / qwen.name
    shutil.copyfile(operators, moved_ops)
    shutil.copyfile(qwen, moved_qwen)
    code = """
import sys
from venturi.ops import CudaBackend
from venturi.models.qwen.native_runtime import QwenRuntime
b=CudaBackend(sys.argv[1]);r=QwenRuntime(sys.argv[2],b)
assert r._native.venturi_qwen_abi()==1
"""
    subprocess.run(
        [sys.executable, "-c", code, str(moved_ops), str(moved_qwen)],
        check=True,
        env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
        timeout=framework_import_timeout(),
    )
