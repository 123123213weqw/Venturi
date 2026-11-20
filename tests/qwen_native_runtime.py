"""Explicit runtime pairing for GPU tests; uninitialized objects only for preflight unit tests."""

import os
from types import SimpleNamespace
from venturi.models.qwen.native_runtime import QwenRuntime


def native_runtime(operators):
    path = os.environ.get("VENTURI_QWEN_LIBRARY")
    if path and hasattr(operators, "architecture"):
        return QwenRuntime(path, operators)
    # CPU tests exercise rejection ordering without opening either native library.
    runtime = object.__new__(QwenRuntime)
    runtime.operators = operators
    runtime._native = getattr(operators, "_native", SimpleNamespace())
    return runtime
