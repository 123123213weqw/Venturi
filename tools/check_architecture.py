"""Check source dependency boundaries without importing Venturi or a framework.

This is a static guard for the incremental architecture, not a proof that all
legacy dependency cycles are gone. Optional/local imports are recorded separately
from eager imports. Scans src/ only, never frozen benchmark/operation snapshots.
"""

import argparse
import ast
from importlib.util import resolve_name
import json
from pathlib import Path


METADATA = frozenset(
    {
        "venturi.checkpoint",
        "venturi.dtypes",
        "venturi.model_assets",
        "venturi.model_config",
        "venturi.tokenizer_assets",
        "venturi.ninfer",
        "venturi.gguf_inventory",
        "venturi.gguf_metadata",
        "venturi.flash_next",
        "venturi.adaptive_mtp",
        "venturi.staging",
    }
)

# These are model engines/adapters, not CUDA/cuBLAS, format readers, Siphon or
# attributed individual operators. Offline comparator tools are outside src/.
EXTERNAL_ENGINES = (
    "venturi.backends",
    "venturi.qwen_cli",
    "llama_cpp",
    "vllm",
    "sglang",
    "ninfer",
    "cinference",
    "strata",
)


def within(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class Imports(ast.NodeVisitor):
    def __init__(self, module, is_package, known_modules):
        self.package = module if is_package else module.rpartition(".")[0]
        self.known_modules = known_modules
        self.deferred = 0
        self.edges = []

    def add(self, target, node):
        self.edges.append(
            {"target": target, "line": node.lineno, "eager": not self.deferred}
        )

    def visit_Import(self, node):
        for name in node.names:
            self.add(name.name, node)

    def visit_ImportFrom(self, node):
        target = (
            resolve_name("." * node.level + (node.module or ""), self.package)
            if node.level
            else node.module
        )
        self.add(target, node)
        for name in node.names:
            child = target + "." + name.name
            if child in self.known_modules:
                self.add(child, node)

    def visit_FunctionDef(self, node):
        # Decorators/defaults execute eagerly; only the function body is deferred.
        for value in node.decorator_list + node.args.defaults + node.args.kw_defaults:
            if value is not None:
                self.visit(value)
        self.deferred += 1
        for value in node.body:
            self.visit(value)
        self.deferred -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        if (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
            isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
        ):
            for value in node.orelse:
                self.visit(value)
        else:
            self.generic_visit(node)


def check_edge(module, edge):
    target = edge["target"]
    rules = []
    if within(target, EXTERNAL_ENGINES):
        rules.append("native-runtime-does-not-import-external-engines")
    higher = ("venturi.models", "venturi.commands", "venturi.backends", "venturi.cli")
    if within(module, ("venturi.ops",)) and within(
        target, higher + ("venturi.generation",)
    ):
        rules.append("operators-do-not-import-models-or-commands")
    if within(module, ("venturi.models", "venturi.backends")) and within(
        target, ("venturi.commands", "venturi.cli")
    ):
        rules.append("implementations-do-not-import-cli")
    if within(module, ("venturi.text",)) and within(
        target,
        higher
        + (
            "venturi.ops",
            "venturi.loader",
            "venturi.generation",
            "torch",
            "safetensors",
        ),
    ):
        rules.append("shared-text-is-model-and-framework-independent")
    if module in METADATA and within(
        target, higher + ("venturi.ops", "venturi.loader", "venturi.generation")
    ):
        rules.append("metadata-does-not-depend-on-execution")
    if module in METADATA and edge["eager"] and within(target, ("torch",)):
        rules.append("metadata-does-not-eagerly-import-torch")
    if (
        module in ("venturi.cli", "venturi.commands.registry")
        and edge["eager"]
        and within(
            target,
            higher
            + (
                "venturi.ops",
                "venturi.loader",
                "venturi.generation",
                "torch",
                "safetensors",
            ),
        )
    ):
        # Command metadata is the sole allowed application dependency of dispatch.
        if not within(target, ("venturi.commands.registry",)):
            rules.append("dispatch-imports-command-metadata-only")
    if (
        module in ("venturi.models", "venturi.ops")
        and edge["eager"]
        and within(target, ("venturi", "torch"))
    ):
        rules.append("public-namespaces-resolve-exports-lazily")
    return [{"rule": rule, "module": module, **edge} for rule in rules]


def inspect_source(source_root):
    source_root = Path(source_root)
    files = sorted((source_root / "venturi").rglob("*.py"))
    if not files:
        raise ValueError("Expected a source root containing venturi/*.py")
    modules = {}
    for path in files:
        parts = path.relative_to(source_root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[module] = path
    records = []
    violations = []
    for module, path in modules.items():
        source = path.read_text()
        visitor = Imports(module, path.name == "__init__.py", set(modules))
        visitor.visit(ast.parse(source, filename=str(path)))
        for edge in visitor.edges:
            violations.extend(check_edge(module, edge))
        records.append(
            {
                "module": module,
                "path": str(path.relative_to(source_root)),
                "lines": len(source.splitlines()),
                "imports": visitor.edges,
            }
        )
    return {
        "scope": "Static source dependency guard; no model/backend imports or device probes",
        "module_count": len(records),
        "violations": violations,
        "modules": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_source(args.source_root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"{report['module_count']} modules, {len(report['violations'])} boundary violations"
        )
        for violation in report["violations"]:
            print(
                f"{violation['module']}:{violation['line']}: {violation['rule']}: {violation['target']}"
            )
    return int(bool(report["violations"]))


if __name__ == "__main__":
    raise SystemExit(main())
