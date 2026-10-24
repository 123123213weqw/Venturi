# Preserved initial parser-test collection issue

The first focused local pytest command failed at collection with
`ModuleNotFoundError: No module named 'summarize_native_source_validation'`.
The new test initially imported the tool as a top-level module, but this
repository does not globally prepend `tools` to pytest's module path.

Changed only the test import to the repository's existing explicit
`spec_from_file_location` convention. The repaired four parser tests pass locally
and remotely; the full local suite passes822 tests with2105 GPU-dependent skips.
No GPU inference run or production source guard failed, and no retry relaxed a
numerical, source-integrity or performance threshold. This file is a retrospective
issue note, not a claimed raw pytest log.
