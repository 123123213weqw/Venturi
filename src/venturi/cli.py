"""Lightweight explicit command dispatch; no model/runtime import until selected."""

import argparse
from importlib import import_module
import sys

from .commands.registry import COMMANDS, COMMAND_BY_NAME


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    command = COMMAND_BY_NAME.get(arguments[0]) if arguments else None
    if command is not None:
        handler = import_module(command.module)
        if command.family:
            return handler.main(arguments[1:])
        return handler.main(arguments)
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        commands.add_parser(command.name, help=command.help)
    parser.parse_args(arguments)
    return 0
