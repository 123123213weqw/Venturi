"""Qwen4-Exp/Flash-Next intake and explicit capacity planning, not inference.

The dense Qwen3.5/3.8 adapter must not bind this architecture. In particular,
MoE, QSA, four-stream gated residuals and PLE are architectural requirements,
not optional performance switches. Header-only manifests never claim verified
checkpoint payloads. No remote model code is imported.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from .checkpoint import _read_json
from .dtypes import DTYPE_INFO
from .ninfer import row_geometry


def _positive(value, name):
    if type(value) is not int or not 0 < value < 2**31:
        raise ValueError(f"{name} must be a positive int32")
    return value


@lru_cache(maxsize=32)
def _head_vocabularies(base, count):
    _positive(base, "ngram_vocab_size_base")
    if base > 10**9 or not 0 < count <= 128:
        raise ValueError("Ngram geometry exceeds intake bounds")
    sizes, candidate = [], base
    while len(sizes) < count:
        if candidate >= 2 and (candidate == 2 or candidate % 2):
            if all(candidate % d for d in range(3, math.isqrt(candidate) + 1, 2)):
                sizes.append(candidate)
        candidate += 1
    return tuple(sizes)


@dataclass(frozen=True)
class FlashNextConfig:
    hidden_size: int
    layers: int
    layer_types: tuple[str, ...]
    vocab_size: int
    eos_token_id: int
    hc_count: int
    hc_lowrank: int
    num_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ple_embed_dim: int
    ngram_padding: int
    split_ngram_parts: int
    ple_layers: tuple[int, ...]
    kv_heads: int
    head_dim: int
    max_context: int

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("model_type") != "qwen4_exp":
            raise ValueError("Expected Qwen3.8-Flash-Next qwen4_exp, not dense Qwen3.8")
        c = raw.get("text_config")
        if not isinstance(c, dict) or c.get("model_type") != "qwen4_exp_text":
            raise ValueError("Explicit qwen4_exp_text configuration required")
        keys = (
            "hidden_size", "num_hidden_layers", "vocab_size", "hc_count", "hc_lowrank",
            "num_experts", "num_experts_per_tok", "moe_intermediate_size", "ngram_size",
            "heads_per_ngram", "ngram_vocab_size_base", "ple_embed_dim",
            "make_ngram_vocab_size_divisible_by", "split_ngram_parts",
            "num_key_value_heads", "head_dim", "max_position_embeddings",
        )
        v = {k: _positive(c.get(k), k) for k in keys}
        kinds = c.get("layer_types")
        if not isinstance(kinds, list) or len(kinds) != v["num_hidden_layers"] or any(
            x not in ("linear_attention", "full_attention") for x in kinds
        ):
            raise ValueError("Every Flash-Next layer needs an explicit attention kind")
        ple = c.get("ple_layer_ids")
        if not isinstance(ple, list) or not ple or any(
            type(x) is not int or not 1 <= x <= v["num_hidden_layers"] for x in ple
        ) or len(set(ple)) != len(ple):
            raise ValueError("PLE layer ids must be unique, valid one-based indices")
        eos = c.get("eos_token_id")
        if isinstance(eos, list) and len(eos) == 1:
            eos = eos[0]
        if type(eos) is not int or not 0 <= eos < v["vocab_size"]:
            raise ValueError("One explicit valid EOS id required for ngram boundaries")
        heads = (v["ngram_size"] - 1) * v["heads_per_ngram"]
        if not 2 <= v["ngram_size"] <= 8 or not 0 < heads <= 128:
            raise ValueError("Unsupported ngram order/head count")
        if v["ple_embed_dim"] % heads or v["num_experts_per_tok"] > v["num_experts"]:
            raise ValueError("Inconsistent PLE/MoE dimensions")
        if raw.get("tie_word_embeddings", False) or c.get("tie_word_embeddings", False):
            raise ValueError("Tied Flash-Next embeddings are not qualified")
        return cls(
            v["hidden_size"], v["num_hidden_layers"], tuple(kinds), v["vocab_size"], eos,
            v["hc_count"], v["hc_lowrank"], v["num_experts"], v["num_experts_per_tok"],
            v["moe_intermediate_size"], v["ngram_size"], v["heads_per_ngram"],
            v["ngram_vocab_size_base"], v["ple_embed_dim"],
            v["make_ngram_vocab_size_divisible_by"], v["split_ngram_parts"], tuple(ple),
            v["num_key_value_heads"], v["head_dim"], v["max_position_embeddings"],
        )

    @property
    def ngram_heads(self):
        return (self.ngram_size - 1) * self.heads_per_ngram

    def ngram_geometry(self, ple_index=0):
        if type(ple_index) is not int or not 0 <= ple_index < len(self.ple_layers):
            raise ValueError("Invalid PLE index")
        primes = _head_vocabularies(
            self.ngram_vocab_size_base, (ple_index + 1) * self.ngram_heads
        )[ple_index * self.ngram_heads:]
        offsets, total = [], 0
        for size in primes:
            offsets.append(total)
            total += size
        padded = (total + self.ngram_padding - 1) // self.ngram_padding * self.ngram_padding
        if padded % self.split_ngram_parts:
            raise ValueError("Ngram padded vocabulary does not split evenly")
        return primes, tuple(offsets), padded


@dataclass(frozen=True)
class FlashNextTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    filename: str

    @property
    def nbytes(self):
        return math.prod(self.shape) * DTYPE_INFO[self.dtype][1]

    @property
    def component(self):
        if ".ple.ple_embedding.ngram_embedding.shard_" in self.name:
            return "ngram_tables"
        if self.name.startswith("mtp."):
            return "mtp"
        if self.name.startswith("model.visual."):
            return "vision"
        return "text_core"

    @property
    def is_expert(self):
        return self.name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))


@dataclass(frozen=True)
class FlashNextManifest:
    config: FlashNextConfig
    tensors: tuple[FlashNextTensor, ...]
    source_scope: str
    repo: str | None = None
    revision: str | None = None

    @classmethod
    def from_headers(cls, directory, headers):
        directory, headers = Path(directory), Path(headers)
        cfg = FlashNextConfig.from_dict(_read_json(directory / "config.json"))
        index = _read_json(directory / "model.safetensors.index.json")
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(n, str) or not n or not isinstance(f, str)
            or Path(f).name != f or "\\" in f or not f.endswith(".safetensors")
            for n, f in mapping.items()
        ):
            raise ValueError("Invalid shard index")
        result, provenance = {}, set()
        for filename in sorted(set(mapping.values())):
            record = _read_json(headers / (filename + ".json"))
            repo, revision = record.get("repo"), record.get("revision")
            if not isinstance(repo, str) or not isinstance(revision, str) or not repo or not revision:
                raise ValueError("Header provenance is required")
            provenance.add((repo, revision))
            size, length = record.get("file_bytes"), record.get("header_bytes")
            if record.get("filename") != filename or type(size) is not int or type(length) is not int:
                raise ValueError("Header filename/length differs")
            if not 2 <= length <= 16 * 2**20 or not 8 + length <= size:
                raise ValueError("Invalid header/file size")
            raw = record.get("header")
            if not isinstance(raw, dict):
                raise ValueError("Header must be an object")
            ranges = []
            for name, tensor in raw.items():
                if name == "__metadata__":
                    continue
                if mapping.get(name) != filename or name in result or not isinstance(tensor, dict):
                    raise ValueError("Header/index tensor disagreement")
                shape, dtype = tensor.get("shape"), tensor.get("dtype")
                if not isinstance(shape, list) or len(shape) > 8 or any(
                    type(d) is not int or d < 0 or d >= 2**63 for d in shape
                ) or dtype not in DTYPE_INFO:
                    raise ValueError("Invalid tensor shape/dtype")
                t = FlashNextTensor(name, tuple(shape), dtype, filename)
                offsets = tensor.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2 or any(
                    type(x) is not int for x in offsets
                ) or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length:
                    raise ValueError("Invalid tensor byte offsets")
                if offsets[1] - offsets[0] != t.nbytes:
                    raise ValueError("Tensor shape/dtype/byte span differs")
                ranges.append(tuple(offsets))
                result[name] = t
            end = 0
            for lo, hi in sorted(ranges):
                if lo != end:
                    raise ValueError("Overlapping/noncontiguous safetensors payloads")
                end = hi
            if end != size - 8 - length:
                raise ValueError("Shard payload size differs")
        if len(provenance) != 1 or set(result) != set(mapping):
            raise ValueError("Mixed revisions or incomplete header set")
        declared = index.get("metadata", {}).get("total_size")
        if declared is not None and sum(t.nbytes for t in result.values()) != declared:
            raise ValueError("Index total_size differs from tensor payloads")
        repo, revision = provenance.pop()
        instance = cls(cfg, tuple(result[k] for k in sorted(result)),
                       "header-only; tensor payloads absent/unverified", repo, revision)
        instance.validate_schema()
        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint):
        instance = cls(
            FlashNextConfig.from_dict(dict(checkpoint.config)),
            tuple(FlashNextTensor(t.name, t.shape, t.dtype, t.file.name)
                  for t in checkpoint.tensors.values()),
            "local checkpoint metadata; payload hashes not verified by intake",
        )
        instance.validate_schema()
        return instance

    def validate_schema(self):
        c, tensors = self.config, {t.name: t for t in self.tensors}

        def require(name, shape, dtypes=None):
            t = tensors.get(name)
            if t is None or t.shape != shape or (dtypes and t.dtype not in dtypes):
                raise ValueError(f"Missing/incompatible Flash-Next tensor: {name}")

        root = "model.language_model"
        require(root + ".embed_tokens.weight", (c.vocab_size, c.hidden_size))
        require("lm_head.weight", (c.vocab_size, c.hidden_size))
        for layer in range(c.layers):
            prefix = f"{root}.layers.{layer}"
            require(prefix + ".mlp.experts.gate_up_proj",
                    (c.num_experts, 2 * c.expert_intermediate_size, c.hidden_size))
            require(prefix + ".mlp.experts.down_proj",
                    (c.num_experts, c.hidden_size, c.expert_intermediate_size))
            for part in ("attn_hyper_connection", "mlp_hyper_connection"):
                require(f"{prefix}.{part}.hc_norm.weight", (c.hc_count * c.hidden_size,))
                require(f"{prefix}.{part}.input_mix_weight_down.weight",
                        (c.hc_lowrank, c.hc_count * c.hidden_size))
                require(f"{prefix}.{part}.input_mix_weight_up.weight",
                        (c.hc_count * c.hidden_size, c.hc_lowrank))
                require(f"{prefix}.{part}.block_inject_weight.weight",
                        (c.hc_count, c.hc_count * c.hidden_size))
        for ple_index, layer in enumerate(c.ple_layers):
            prefix = f"{root}.layers.{layer - 1}.ple.ple_embedding"
            require(prefix + ".layer_multipliers", (c.ngram_size,), ("I64",))
            _, _, padded = c.ngram_geometry(ple_index)
            for part in range(c.split_ngram_parts):
                require(f"{prefix}.ngram_embedding.shard_{part}.weight",
                        (padded // c.split_ngram_parts, c.ple_embed_dim // c.ngram_heads))

    def plan(self, *, profile="source", gpu_budgets=(), ram_budget=0,
             include_mtp=False, include_vision=False, batch_size=1, context=256):
        if profile not in ("source", "expert-q8-g32", "expert-q4-g64"):
            raise ValueError("Unknown explicit storage-estimate profile")
        if type(include_mtp) is not bool or type(include_vision) is not bool:
            raise ValueError("Component inclusion must be explicit boolean")
        if type(ram_budget) is not int or ram_budget < 0 or any(
            type(x) is not int or x < 0 for x in gpu_budgets
        ):
            raise ValueError("Budgets must be nonnegative integer bytes")
        _positive(batch_size, "batch_size")
        _positive(context, "context")
        if context > self.config.max_context:
            raise ValueError("Context exceeds model configuration")
        source, proposed = Counter(), Counter()
        for t in self.tensors:
            source[t.component] += t.nbytes
            if t.component == "mtp" and not include_mtp or t.component == "vision" and not include_vision:
                continue
            nbytes = t.nbytes
            if profile != "source" and t.is_expert:
                if t.dtype != "BF16":
                    raise ValueError("Expert quantization estimate requires original BF16 tensors")
                nbytes = row_geometry(
                    (math.prod(t.shape[:-1]), t.shape[-1]),
                    "q8_g32_fp16" if profile == "expert-q8-g32" else "q4_g64_fp16",
                )[-1]
            proposed[t.component] += nbytes
        c = self.config
        # Dense K/V backing lower bound only: QSA selection does not by itself
        # prove that old K/V may be discarded. Indexer and all GDN state excluded.
        kv = batch_size * context * c.layer_types.count("full_attention") * c.kv_heads * c.head_dim * 4
        gpu_weights = sum(v for k, v in proposed.items() if k != "ngram_tables")
        host_tables = proposed["ngram_tables"]
        gpu_capacity = sum(gpu_budgets)
        overflow = max(0, gpu_weights + kv - gpu_capacity)
        return dict(
            architecture="qwen4_exp", native_generation_supported=False,
            source_scope=self.source_scope, repo=self.repo, revision=self.revision,
            tensor_count=len(self.tensors), source_component_bytes=dict(source),
            proposed_component_bytes=dict(proposed), storage_profile=profile,
            profile_scope="capacity estimate only; no exporter/quantized MoE executor or quality qualification",
            placement=dict(ngram_tables="CPU RAM/mmap; gather only addressed rows",
                           text_core="GPU preferred; explicit expert cache/offload required if it does not fit",
                           mtp="GPU only if explicitly included", vision="excluded unless explicitly included"),
            gpu_budgets_bytes=list(gpu_budgets), ram_budget_bytes=ram_budget,
            batch_size=batch_size, context=context, dense_fp16_kv_lower_bound_bytes=kv,
            gpu_weight_bytes=gpu_weights, cpu_ngram_table_bytes=host_tables,
            ideal_gpu_overflow_bytes=overflow,
            minimum_host_bytes_with_overflow=host_tables + overflow,
            capacity_lower_bound_fits=(bool(gpu_budgets) and host_tables + overflow <= ram_budget),
            all_weights_gpu_lower_bound_fits=(bool(gpu_budgets) and overflow == 0 and host_tables <= ram_budget),
            exclusions=["GDN/QSA-indexer/PLE state", "activations/workspaces", "CUDA graphs/context",
                        "allocator and per-rank padding", "OS reserve", "TP/EP topology and execution support"],
            missing_native_components=["MoE routing/expert execution", "QSA indexer/attention",
                                       "gated-residual integration", "PLE integration", "new-model MTP verification"],
        )
