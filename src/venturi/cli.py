"""Lightweight explicit command dispatch; no model/runtime import until selected."""

import argparse
from importlib import import_module
import sys

from .commands.registry import COMMANDS, COMMAND_BY_NAME

# Transitional hook identities used by existing callers/tests. These remain
# lazy, and a patched hook is passed only to the dense checkpoint handler.
_COMPAT = {
    "inspect_checkpoint": ".checkpoint",
    "load_checkpoint": ".loader",
    "verify_loaded_checkpoint": ".verification",
}


def __getattr__(name):
    module = _COMPAT.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __package__), name)
    globals()[name] = value
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    command = COMMAND_BY_NAME.get(arguments[0]) if arguments else None
    if command is not None:
        handler = import_module(command.module)
        if command.family:
            return handler.main(arguments[1:])
        return handler.main(
            arguments,
            inspect=globals().get("inspect_checkpoint"),
            load=globals().get("load_checkpoint"),
            verify=globals().get("verify_loaded_checkpoint"),
        )
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        commands.add_parser(command.name, help=command.help)
    parser.parse_args(arguments)
    return 0
