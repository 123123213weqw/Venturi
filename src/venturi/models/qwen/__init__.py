"""Qwen family APIs; selecting a schema does not import the native runtime."""

from importlib import import_module

_EXPORTS = {
    "NativeQwenModel": ".qwen_native",
    "QwenWeights": ".qwen_weights",
    "ResidentQwenModel": ".qwen_model",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
