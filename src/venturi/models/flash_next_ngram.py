"""Bounded CPU PLE ngram row lookup with transactional, request-private history.

Hash semantics follow the pinned Qwen4-Exp reference: order-2..N XOR hashes,
prime head vocabularies, EOS-delimited history, contiguous table shards. This
is NOT ngram speculative decoding and does not implement a Flash-Next forward.
"""

from concurrent.futures import Executor
from contextlib import ExitStack
from dataclasses import dataclass
import threading

from safetensors import safe_open

from ..flash_next import FlashNextConfig


@dataclass(frozen=True)
class NGramProposal:
    owner: object
    version: int
    table_key: tuple
    token_ids: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]


class NGramRequest:
    """Preview does not commit history; rejected speculative suffixes disappear."""

    def __init__(self, config: FlashNextConfig, multipliers, *, ple_index=0):
        self.config = config
        self.sizes, self.offsets, self.padded_rows = config.ngram_geometry(ple_index)
        self.multipliers = tuple(multipliers)
        bound = (2**63 - 1) // max(1, config.vocab_size - 1)
        if len(self.multipliers) != config.ngram_size or any(
            type(x) is not int or not 0 < x <= bound for x in self.multipliers
        ):
            raise ValueError("Use valid serialized I64 layer_multipliers from the checkpoint")
        self.table_key = (config, ple_index, self.multipliers)
        self._owner, self._version, self._history = object(), 0, ()
        self._lock = threading.RLock()

    @property
    def history(self):
        with self._lock:
            return self._history

    def _advance(self, token_ids, history):
        c, rows = self.config, []
        history = list(history)
        for token in token_ids:
            mixed, out = token * self.multipliers[0], []
            for order in range(2, c.ngram_size + 1):
                position = order - 1
                previous = history[-position] if len(history) >= position else c.eos_token_id
                mixed ^= previous * self.multipliers[position]
                start = (order - 2) * c.heads_per_ngram
                out.extend(mixed % self.sizes[h] + self.offsets[h]
                           for h in range(start, start + c.heads_per_ngram))
            rows.append(tuple(out))
            if token == c.eos_token_id:
                history.clear()
            else:
                history.append(token)
                history = history[-(c.ngram_size - 1):]
        return tuple(rows), tuple(history)

    def preview(self, token_ids):
        ids = tuple(token_ids)
        if any(type(x) is not int or not 0 <= x < self.config.vocab_size for x in ids):
            raise ValueError("Ngram token ids must lie inside the vocabulary")
        with self._lock:
            rows, _ = self._advance(ids, self._history)
            return NGramProposal(self._owner, self._version, self.table_key, ids, rows)

    def commit(self, proposal, accepted_tokens=None):
        with self._lock:
            if not isinstance(proposal, NGramProposal) or proposal.owner is not self._owner or proposal.version != self._version:
                raise ValueError("Stale/foreign ngram proposal")
            count = len(proposal.token_ids) if accepted_tokens is None else accepted_tokens
            if type(count) is not int or not 0 <= count <= len(proposal.token_ids):
                raise ValueError("Invalid accepted prefix length")
            _, self._history = self._advance(proposal.token_ids[:count], self._history)
            self._version += 1

    def reset(self):
        with self._lock:
            self._history = ()
            self._version += 1


class ShardedNGramTable:
    """Read-only safetensors mmap slices; never upload/materialize the full table.

    checkpoint is the existing architecture-independent inspected checkpoint.
    Each lookup opens only addressed files, deduplicates rows within a call,
    and owns its result. The OS page cache is not a guaranteed resident RAM cache.
    """

    _storage_dtypes = ("BF16", "F16", "F32")

    def __init__(self, checkpoint, config, *, ple_index=0, max_tokens=4096):
        if type(max_tokens) is not int or not 0 < max_tokens <= 65536:
            raise ValueError("Explicit bounded lookup size required")
        self.config, self.max_tokens = config, max_tokens
        _, _, self.padded_rows = config.ngram_geometry(ple_index)
        self.rows_per_part = self.padded_rows // config.split_ngram_parts
        self.width = config.ple_embed_dim // config.ngram_heads
        prefix = f"model.language_model.layers.{config.ple_layers[ple_index] - 1}.ple.ple_embedding"
        self.parts = []
        for part in range(config.split_ngram_parts):
            name = f"{prefix}.ngram_embedding.shard_{part}.weight"
            spec = checkpoint.tensors.get(name)
            if spec is None or spec.shape != (self.rows_per_part, self.width) or spec.dtype not in self._storage_dtypes:
                raise ValueError("Ngram shard schema/dtype mismatch; quantized tables need a separate reader")
            self.parts.append(spec)
        if len({t.dtype for t in self.parts}) != 1:
            raise ValueError("Ngram shards must share a dtype")
        spec = checkpoint.tensors.get(prefix + ".layer_multipliers")
        if spec is None or spec.dtype != "I64" or spec.shape != (config.ngram_size,):
            raise ValueError("Serialized ngram multipliers missing")
        with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
            self.multipliers = tuple(reader.get_tensor(spec.name).tolist())
        # Validate before accepting requests, without allocating a full table.
        NGramRequest(config, self.multipliers, ple_index=ple_index)
        self.table_key = (config, ple_index, self.multipliers)

    def lookup(self, proposal):
        import torch
        from ..dtypes import torch_dtype

        if not isinstance(proposal, NGramProposal) or len(proposal.rows) > self.max_tokens:
            raise ValueError("Expected a bounded ngram proposal")
        if proposal.table_key != self.table_key:
            raise ValueError("Ngram proposal belongs to a different table/configuration")
        if len(proposal.rows) != len(proposal.token_ids) or any(
            len(row) != self.config.ngram_heads for row in proposal.rows
        ):
            raise ValueError("Invalid ngram lookup geometry")
        output = torch.empty((len(proposal.rows), self.config.ngram_heads, self.width),
                             dtype=torch_dtype(self.parts[0].dtype))
        grouped = {}
        for token, row in enumerate(proposal.rows):
            for head, global_row in enumerate(row):
                if type(global_row) is not int or not 0 <= global_row < self.padded_rows:
                    raise ValueError("Ngram row outside the table")
                part, local_row = divmod(global_row, self.rows_per_part)
                grouped.setdefault(part, {}).setdefault(local_row, []).append((token, head))
        for part, rows in grouped.items():
            spec = self.parts[part]
            with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
                view = reader.get_slice(spec.name)
                if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                    raise ValueError("Ngram backing metadata changed")
                for local_row, destinations in rows.items():
                    value = view[local_row:local_row + 1][0]
                    for token, head in destinations:
                        output[token, head].copy_(value)
        return output.flatten(-2)

    def prefetch(self, proposal, executor: Executor):
        """Caller-owned bounded executor; no GPU stream/automatic history commit."""
        return executor.submit(self.lookup, proposal)

    def lookup_many(self, proposals):
        """Coalesce addressed rows across private requests without merging history.

        Returned views own a new common result allocation, never a reused scratch
        buffer. Each request still commits its own accepted token prefix.
        """
        import torch

        proposals = tuple(proposals)
        if any(not isinstance(p, NGramProposal) or p.table_key != self.table_key for p in proposals):
            raise ValueError("Ngram batch contains a foreign table/configuration")
        lengths = [len(p.token_ids) for p in proposals]
        if sum(lengths) > self.max_tokens:
            raise ValueError("Ngram batch exceeds bounded total lookup size")
        merged = NGramProposal(
            None, 0, self.table_key,
            tuple(token for p in proposals for token in p.token_ids),
            tuple(row for p in proposals for row in p.rows),
        )
        output = self.lookup(merged)
        return tuple(torch.split(output, lengths)) if lengths else ()

    def prefetch_many(self, proposals, executor: Executor):
        return executor.submit(self.lookup_many, tuple(proposals))


class FP8ShardedNGramTable(ShardedNGramTable):
    """Explicit official FP8 PLE encoding: E4M3FN rows + one BF16 scale.

    This is not INT8, block-scaled FP8 GEMM, or BF16-quality parity. Decode
    addressed rows on the CPU into BF16; no FP8 Tensor Core is required. Match
    the reference order: FP8 -> BF16, then multiply by the BF16 weight_scale.
    """

    _storage_dtypes = ("F8_E4M3",)

    def __init__(self, checkpoint, config, *, ple_index=0, max_tokens=4096):
        import torch

        super().__init__(checkpoint, config, ple_index=ple_index, max_tokens=max_tokens)
        prefix = f"model.language_model.layers.{config.ple_layers[ple_index] - 1}.ple.ple_embedding"
        spec = checkpoint.tensors.get(prefix + ".ngram_embedding.weight_scale")
        if spec is None or spec.dtype != "BF16" or spec.shape != (1,):
            raise ValueError("FP8 PLE requires the explicit BF16 per-tensor weight_scale")
        with safe_open(str(spec.file), framework="pt", device="cpu") as reader:
            scale = reader.get_tensor(spec.name)
        if not torch.isfinite(scale.float()).all() or scale.item() <= 0:
            raise ValueError("FP8 PLE scale must be positive and finite")
        self.weight_scale = scale.clone()

    def lookup(self, proposal):
        import torch

        return super().lookup(proposal).to(torch.bfloat16) * self.weight_scale


class PreparedNGramTable:
    """Opt-in bounded persistent mmap/slice handles for a validated PLE reader.

    Preparation opens each distinct file once and validates only slice metadata;
    it does NOT get_tensor/materialize/pin the large embedding tables. Keep the
    checkpoint files immutable for this object's lifetime. Preparation cost is
    explicit and must be reported separately from steady-state lookup latency.

    A lock serializes lookup/close, including caller-executor prefetches. Close
    waits for active reads; queued reads after close fail rather than accessing
    stale mappings. Returned tensors own their storage and survive close. This
    is not an async H2D pipeline or a model scheduler.
    """

    def __init__(self, store, *, max_open_files=128):
        if type(store) not in (ShardedNGramTable, FP8ShardedNGramTable):
            raise TypeError("Prepare an explicit original or official FP8 PLE reader")
        if type(max_open_files) is not int or not 1 <= max_open_files <= 1024:
            raise ValueError("Bounded open-file budget required")
        self.store = store
        self.table_key = store.table_key
        self.max_tokens = store.max_tokens
        self.max_open_files = max_open_files
        self._lock = threading.RLock()
        self._state, self._stack, self._views = "new", None, ()

    def prepare(self):
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("Prepared ngram reader is closed")
            if self._state == "ready":
                return self
            paths = {str(spec.file) for spec in self.store.parts}
            if len(paths) > self.max_open_files:
                raise ValueError("Ngram files exceed the explicit open-file budget")
            stack = ExitStack()
            views = []
            try:
                readers = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                           for path in sorted(paths)}
                for spec in self.store.parts:
                    view = readers[str(spec.file)].get_slice(spec.name)
                    if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                        raise ValueError("Ngram backing metadata changed")
                    views.append(view)
            except BaseException:
                views.clear()
                stack.close()
                raise
            self._stack, self._views, self._state = stack, tuple(views), "ready"
            return self

    def close(self):
        with self._lock:
            self._views = ()
            if self._stack is not None:
                self._stack.close()
                self._stack = None
            self._state = "closed"

    def __enter__(self):
        return self.prepare()

    def __exit__(self, *exc):
        self.close()

    def lookup(self, proposal):
        import torch
        from ..dtypes import torch_dtype

        with self._lock:
            if self._state != "ready":
                raise RuntimeError("Prepared ngram reader must be prepared and not closed")
            s = self.store
            if not isinstance(proposal, NGramProposal) or len(proposal.rows) > self.max_tokens:
                raise ValueError("Expected a bounded ngram proposal")
            if proposal.table_key != self.table_key:
                raise ValueError("Ngram proposal belongs to a different table/configuration")
            if len(proposal.rows) != len(proposal.token_ids) or any(
                len(row) != s.config.ngram_heads for row in proposal.rows
            ):
                raise ValueError("Invalid ngram lookup geometry")
            output = torch.empty((len(proposal.rows), s.config.ngram_heads, s.width),
                                 dtype=torch_dtype(s.parts[0].dtype))
            grouped = {}
            for token, row in enumerate(proposal.rows):
                for head, global_row in enumerate(row):
                    if type(global_row) is not int or not 0 <= global_row < s.padded_rows:
                        raise ValueError("Ngram row outside the table")
                    part, local_row = divmod(global_row, s.rows_per_part)
                    grouped.setdefault(part, {}).setdefault(local_row, []).append((token, head))
            for part, rows in grouped.items():
                view = self._views[part]
                for local_row, destinations in rows.items():
                    value = view[local_row:local_row + 1][0]
                    for token, head in destinations:
                        output[token, head].copy_(value)
            output = output.flatten(-2)
            if isinstance(s, FP8ShardedNGramTable):
                output = output.to(torch.bfloat16) * s.weight_scale
            return output

    # Reuse the batch/async contracts, not the original per-call reader path.
    lookup_many = ShardedNGramTable.lookup_many
    prefetch = ShardedNGramTable.prefetch
    prefetch_many = ShardedNGramTable.prefetch_many
