"""Single-request transactional floating reference, not a serving scheduler.

Coordinates one official-style PLE layer's private hash history with all decoder
caches. Partial acceptance replays the accepted prefix from the base snapshot;
no recurrent-matrix crop. EOS resets hash segments, not model histories. Caller
chooses request termination. Only explicit reset clears the entire request.
"""

from dataclasses import dataclass
import threading

import torch

from .flash_next_ngram import NGramRequest, PreparedNGramTable, ShardedNGramTable
from .flash_next_text import FlashNextTextCore


@dataclass(frozen=True, eq=False)
class TextProposal:
    logits: torch.Tensor
    token_ids: tuple


@dataclass
class _Pending:
    public: TextProposal
    ngram: object
    rows: torch.Tensor
    mask: torch.Tensor | None
    state: object
    logits: torch.Tensor


class FlashNextRequest:
    """One outstanding proposal, private immutable-ID/owned-cache transaction.

    A successful new preview supersedes its old proposal; failed preview/replay
    leaves the committed state AND previous outstanding proposal unchanged.
    Returned logits/state snapshots cannot mutate the internal pending state.
    Model weights and reader files must remain immutable while requests exist.
    This correctness reference intentionally serializes each request via RLock.
    """

    def __init__(self, core, table):
        if not isinstance(core, FlashNextTextCore) or len(core.ple_layers) != 1 or not isinstance(
                table, (ShardedNGramTable, PreparedNGramTable)):
            raise ValueError("Qualified floating text core with exactly one PLE layer and explicit reader required")
        c, ple_index, multipliers = table.table_key
        layer = core.ple_layers[0]
        if ple_index != 0 or c.ple_layers != (layer+1,) or c.hidden_size != core.embed_tokens.embedding_dim or (
                c.vocab_size != core.embed_tokens.num_embeddings or c.layers != len(core.layers) or
                c.hc_count != core.hyper_connection_mixer.streams or c.ple_embed_dim != core.layers[layer].ple.geometry.embedding_dim):
            raise ValueError("Request PLE table/core configuration mismatch")
        self.core, self.table, self._ple_layer = core, table, layer
        self._ngram = NGramRequest(c, multipliers)
        self._state, self._pending, self._last_logits, self._lock = None, None, None, threading.RLock()

    @property
    def history(self):
        with self._lock:
            return self._ngram.history

    @property
    def state(self):
        with self._lock:
            return None if self._state is None else self._state.snapshot()

    @property
    def last_logits(self):
        """Owned last ACCEPTED token logits, including prefix-replay rounding."""
        with self._lock:
            return None if self._last_logits is None else self._last_logits.clone()

    def preview(self, token_ids, *, mask=None):
        ids = tuple(token_ids)
        c = self._ngram.config
        if not ids or any(type(t) is not int or not 0 <= t < c.vocab_size for t in ids):
            raise ValueError("Nonempty valid integer token sequence required")
        bits = None if mask is None else tuple(mask)
        if bits is not None and (len(bits) != len(ids) or any(type(v) is not bool for v in bits)):
            raise ValueError("Current request mask must contain one bool per token")
        with self._lock, torch.inference_mode():
            device, dtype = self.core.embed_tokens.weight.device, self.core.embed_tokens.weight.dtype
            seen = 0 if self._state is None else self._state.tokens_seen
            if seen+len(ids) > self.core.max_sequence or len(ids)*(seen+len(ids)) > self.core.max_mask_elements or (
                    len(ids)*c.vocab_size > self.core.max_logits_elements):
                raise ValueError("Request text sequence/mask/logits budget exceeded")
            # The source replaces masked PLE IDs with EOS BEFORE token hashing.
            ple_ids = ids if bits is None else tuple(t if v else c.eos_token_id for t, v in zip(ids, bits))
            plan = self._ngram.preview(ple_ids)
            rows = self.table.lookup(plan).unsqueeze(0).to(device=device, dtype=dtype)
            conv_mask = None if bits is None else torch.tensor([bits], device=device, dtype=torch.bool)
            tokens = torch.tensor([ids], device=device, dtype=torch.long)
            logits, state = self.core(tokens, ple_rows={self._ple_layer: rows}, conv_mask=conv_mask, past=self._state)
            with torch.inference_mode(False):
                public = TextProposal(logits.clone(), ids)
            self._pending = _Pending(public, plan, rows, conv_mask, state, logits)
            return public

    def commit(self, proposal, accepted_tokens=None):
        with self._lock, torch.inference_mode():
            pending = self._pending
            if pending is None or proposal is not pending.public:
                raise ValueError("Stale/foreign text proposal")
            n = len(pending.public.token_ids)
            count = n if accepted_tokens is None else accepted_tokens
            if type(count) is not int or not 0 <= count <= n:
                raise ValueError("Invalid accepted prefix length")
            state, accepted_logits = pending.state, pending.logits
            if 0 < count < n:
                ids = torch.tensor([pending.public.token_ids[:count]], device=self.core.embed_tokens.weight.device,
                                   dtype=torch.long)
                accepted_logits, state = self.core(ids, ple_rows={self._ple_layer: pending.rows[:, :count]},
                    conv_mask=None if pending.mask is None else pending.mask[:, :count], past=self._state)
            # Allocate the accepted prediction BEFORE publishing either cache.
            # A caller may have mutated public proposal logits; those never
            # supply committed prediction state. A partial replay may round
            # differently from the original multi-token proposal.
            last_logits = None if not count else accepted_logits[:, count-1].clone()
            # Private hash request is never exposed; after successful model
            # replay this commit cannot race a reset/foreign mutation. All
            # fallible validation/tensor work precedes the two owned publishes.
            self._ngram.commit(pending.ngram, count)
            if count:
                self._state, self._last_logits = state, last_logits
            self._pending = None

    def reset(self):
        with self._lock:
            self._ngram.reset()
            self._state, self._pending, self._last_logits = None, None, None
