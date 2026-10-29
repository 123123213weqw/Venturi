# Pinned offline numerical references

Three unchanged upstream Apache-2.0 files are retained under
`benchmarks/qwen38-flash-next-generation-v1/measured-source/` for the CPU oracle.
They are **test data**, not an imported external model engine or runtime fallback.
The original copyright/license headers are preserved. The Apache-2.0 license is
in `tests/licenses/transformers-APACHE-2.0.txt`; it does not license Venturi itself.

The modeling reference is Qwen/Qwen3.8-Flash-Next revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The cache and SDPA helper snapshots
are pinned by content, not a claimed Transformers release. Qualification tools
extract selected original AST definitions and document narrow ABI shims; they do
not execute the entire remote modeling module. No network fetch occurs in tests.

| File | SHA256 |
| --- | --- |
| `reference.py` | `e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df` |
| `reference_cache.py` | `36c4cd52d66f39fdb51fa2fc92cfc43e2e8b152e07f7c2a17186eac1e81237e7` |
| `reference_sdpa.py` | `ba7f7c4e1ac7a0ec457e5c3035e54d0e0183e5bd20f938784814774c7724a616` |
