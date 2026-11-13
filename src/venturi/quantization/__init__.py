"""Quantization APIs; importing geometry never initializes execution or storage."""

from importlib import import_module

_EXPORTS = {
    "CODECS": ".codecs",
    "EXECUTION_QUALIFIED": ".codecs",
    "Codec": ".codecs",
    "codec_for": ".codecs",
    "verify_checkpoint": ".storage_verify",
    "write_evidence": ".storage_verify",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
