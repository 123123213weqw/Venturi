"""Immutable built-in command metadata, not runtime or model auto-discovery."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    help: str
    family: bool = False


COMMANDS = (
    Command(
        "flash-next",
        "venturi.flash_next_cli",
        "Experimental Qwen4-Exp inspection or floating reference generation",
        True,
    ),
    Command(
        "qwen-native",
        "venturi.qwen_native_cli",
        "Owned Q8 C++/CUDA runtime: inspect or explicit greedy generation",
        True,
    ),
    Command(
        "llama",
        "venturi.llama_cli",
        "Independent classic Llama checkpoint bindings and generation",
        True,
    ),
    Command(
        "inspect",
        "venturi.commands.checkpoint",
        "Inspect safetensors storage without loading weights",
    ),
    Command(
        "load",
        "venturi.commands.checkpoint",
        "Load tensors using an explicitly selected loader",
    ),
    Command(
        "verify",
        "venturi.commands.checkpoint",
        "Verify loaded tensors against stored bytes",
    ),
    Command(
        "generate",
        "venturi.commands.checkpoint",
        "Raw continuation with the dense GPT-NeoX adapter",
    ),
    Command(
        "benchmark",
        "venturi.commands.checkpoint",
        "Benchmark the explicit dense GPT-NeoX path",
    ),
)
COMMAND_BY_NAME = MappingProxyType({command.name: command for command in COMMANDS})
