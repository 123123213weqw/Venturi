"""One explicit Qwen DSO and its concrete operator dependency; no plugin lookup."""

import ctypes as C
from pathlib import Path
from venturi.ops import CudaBackend


class QwenRuntime:
    def __init__(self, library, operators):
        if not isinstance(operators, CudaBackend):
            raise TypeError(
                "QwenRuntime requires an explicit CudaBackend operator library"
            )
        self.operators = operators
        self.library = Path(library).expanduser().resolve(strict=True)
        self._native = C.CDLL(str(self.library))
        for name in (
            "venturi_qwen_abi",
            "venturi_qwen_arch",
            "venturi_qwen_required_ops_abi",
        ):
            function = getattr(self._native, name)
            function.argtypes, function.restype = [], C.c_int
        error = self._native.venturi_qwen_error
        error.argtypes, error.restype = [], C.c_char_p
        identity = self._native.venturi_qwen_ops_identity
        identity.argtypes, identity.restype = [], C.c_void_p
        if self._native.venturi_qwen_abi() != 1:
            raise RuntimeError("Qwen runtime ABI mismatch; rebuild explicitly")
        if self._native.venturi_qwen_arch() != operators.architecture:
            raise RuntimeError("Qwen and operator architectures differ")
        if (
            self._native.venturi_qwen_required_ops_abi()
            != operators._native.venturi_blocks_abi()
        ):
            raise RuntimeError("Qwen requires a different operator ABI")
        if identity() != C.cast(operators._native.venturi_blocks_abi, C.c_void_p).value:
            raise RuntimeError(
                "Qwen is linked to a different operator library instance"
            )

    def _check(self, result):
        if result < 0:
            raise RuntimeError(
                self._native.venturi_qwen_error().decode("utf-8", "replace")
            )
