"""Native MTP1 target-verification cohorts, private target/draft caches.

One native call owns generation. Drafting and shifted prefill remain per-request
native operations; only target verification is batched. No continuation/paging.
"""

import ctypes as C
import threading
from types import SimpleNamespace
import weakref
import torch
from .qwen_batch import NativeQwenBatchSession
from .qwen_mtp_native import NativeQwenMTPModel, _base_mtp_stats
from .qwen_pipeline import NativeQwenPipelineModel
from ..ops.vocabulary import _ids


def _bind(backend):
    p, i, u = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "create": ([p, p, i], p),
        "destroy": ([p], None),
        "generate": ([p, p, i, p, p, p, p, p, p, i, p], i),
        "reset": ([p, p, i], i),
        "info": ([p, p, p, p], i),
        "snapshot": ([p, i, i, i, p, u], i),
        "debug_failure": ([p, i, i, i], i),
        "debug_proposals": ([p, i, p, i], i),
        "draft_snapshot": ([p, i, i, p, u], i),
        "stats": ([p, i, p, p], i),
        "extra_info": ([p, p, p], i),
        "cohort_timings": ([p, p], i),
    }
    values = {}
    try:
        for name, (args, result) in signatures.items():
            fn = getattr(backend._native, "venturi_qwen_mtp_batch_" + name)
            fn.argtypes, fn.restype = args, result
            values[name] = fn
    except AttributeError as e:
        raise RuntimeError("Native MTP batch exports absent; rebuild explicitly") from e
    return SimpleNamespace(**values)


class NativeQwenMTPBatchSession(NativeQwenBatchSession):
    @classmethod
    def create(cls, model, capacities):
        if (
            type(model) is not NativeQwenMTPModel
            or isinstance(model.target, NativeQwenPipelineModel)
            or model.dtype != torch.float16
        ):
            raise ValueError(
                "MTP batch requires a complete resident FP16 single/TP MTP model, not PP"
            )
        if not isinstance(capacities, (list, tuple)) or not 1 <= len(capacities) <= min(
            16, model._max_tokens // 2
        ):
            raise ValueError("MTP batch slots must be 1..min(16,max_tokens/2)")
        if any(
            type(c) is not int or not 0 < c <= model._cfg["max_position_embeddings"]
            for c in capacities
        ):
            raise ValueError("MTP batch capacities exceed context")
        native = _bind(model._backend)
        with model._lock:
            if not model._finalizer.alive:
                raise RuntimeError("MTP model closed")
            model._metadata()
            cap = (C.c_int * len(capacities))(*capacities)
            handle = native.create(model._handle, cap, len(capacities))
            if not handle:
                model._backend._check(-1)
        self = cls()
        self.model, self.capacities, self._handle, self._native = (
            model,
            tuple(capacities),
            handle,
            native,
        )
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, native.destroy, handle)
        return self

    def forward(self, *a, **kw):
        raise ValueError(
            "MTP batch supports fresh/reset generate only, not forward/continuation"
        )

    def generate(self, prompts, *, max_new_tokens, eos_token_ids=(), slots=None):
        # Check fresh state before the inherited bounded allocation/preflight.
        selected = (
            self._slots(list(range(len(prompts))) if slots is None else slots)
            if isinstance(prompts, (list, tuple)) and prompts
            else None
        )
        with self._lock:
            self._check()
            if selected is not None:
                info = super().info()
                if any(info["requests"][i]["position"] for i in selected):
                    raise RuntimeError(
                        "MTP batch generation requires fresh/reset requests"
                    )
            return super().generate(
                prompts,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                slots=slots,
            )

    def stats(self, slot):
        self._slots([slot])
        with self._lock:
            self._check()
            values, times = (C.c_uint64 * 20)(), (C.c_double * 5)()
            self.model._backend._check(
                self._native.stats(self._handle, slot, values, times)
            )
            result = _base_mtp_stats(values, times)
            result.update(
                target_recovery="request-prefix1-trace",
                prefix_trace_extra_primary_bytes=0,
                prefix_trace_extra_secondary_bytes=0,
                verification_timing_scope="Shared cohort verification wall repeated per participating request; do not sum across slots.",
            )
            return result

    def info(self):
        with self._lock:
            result = super().info()
            extra, seconds = (C.c_uint64 * 4)(), (C.c_double * 1)()
            cohort = (C.c_double * 2)()
            self.model._backend._check(
                self._native.extra_info(self._handle, extra, seconds)
            )
            result.update(
                zip(
                    (
                        "mtp_model_device_bytes",
                        "all_mtp_request_device_bytes",
                        "all_target_backup_device_bytes",
                        "shared_mtp_selection_device_bytes",
                    ),
                    map(int, extra),
                )
            )
            result["shared_verification_wall_seconds"] = seconds[0]
            self.model._backend._check(
                self._native.cohort_timings(self._handle, cohort)
            )
            result["shared_prefill_complete_seconds"] = cohort[0]
            result["shared_native_generation_seconds"] = cohort[1]
            for i, r in enumerate(result["requests"]):
                r["mtp"] = self.stats(i)
            result.update(
                prefill_strategy="serial native shifted warmup",
                draft_strategy="serial native one-proposal/catch-up",
                target_verification="native packed cohort, request-local prefix1 recovery",
                scope="Target model/workspace and MTP model/selection bytes count once; target request, additional draft/request and backup bytes per slot are disjoint. Fixed fresh/reset cohorts, no PP/continuous batching/paging. Per-slot verification wall overlaps; do not sum it.",
            )
            return result

    def draft_snapshot(self, slot):
        self._slots([slot])
        with self._lock:
            self._check()
            position = self.stats(slot)["draft_position"]
            cfg = self.model._cfg
            result = dict(position=position)
            for kind, name in [(2, "keys"), (3, "values")]:
                out = torch.empty(
                    (position, cfg["num_key_value_heads"], cfg["head_dim"]),
                    dtype=self.model.dtype,
                    device="cpu",
                )
                self.model._backend._check(
                    self._native.draft_snapshot(
                        self._handle,
                        slot,
                        kind,
                        out.data_ptr(),
                        out.numel() * out.element_size(),
                    )
                )
                result[name] = out
            return result

    def _force_proposals(self, slot, ids):
        self._slots([slot])
        _ids(ids, self.model._cfg["vocab_size"])
        with self._lock:
            self._check()
            values = (C.c_int64 * len(ids))(*ids)
            self.model._backend._check(
                self._native.debug_proposals(self._handle, slot, values, len(ids))
            )
