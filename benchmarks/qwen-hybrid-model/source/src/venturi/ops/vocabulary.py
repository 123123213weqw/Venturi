"""Independent row-split Q8 vocabulary bindings, lookup and output projection.

The Qwen artifact's embedding/head are Q8, not the Q4 projection format. This
module preserves signed codes/group-32 FP16 scales; it never expands an entire
vocabulary to dense weights in native execution. The simple head kernel is a
correctness baseline, not an optimized prefill GEMM.
"""

from dataclasses import dataclass

import torch

from .blocks import _no_overlap, _shape, _tensor


def _ids(ids, vocab):
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("Vocabulary IDs must be a nonempty list/tuple")
    if len(ids) >= 2**31 or any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise ValueError("Vocabulary ID/count is out of range")


def _read_plane(artifact, name, offset, shape, dtype):
    # Bound each physical read. Never duplicate a multi-GB bytes object just to
    # turn it into a tensor; the destination has exactly one owned CPU allocation.
    out = torch.empty(shape, dtype=dtype)
    raw = out.view(torch.uint8).view(-1)
    chunk = 16 * 2**20
    for start in range(0, raw.numel(), chunk):
        size = min(chunk, raw.numel() - start)
        data = bytearray(
            artifact.read_object(
                name, offset=offset + start, size=size, max_bytes=chunk
            )
        )
        raw[start : start + size].copy_(torch.frombuffer(data, dtype=torch.uint8))
    return out


@dataclass(frozen=True)
class Q8VocabularyWeights:
    codes: torch.Tensor
    scales: torch.Tensor
    hidden_size: int

    def validate(self):
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("Q8 hidden size must be positive")
        if not isinstance(self.codes, torch.Tensor) or self.codes.ndim != 2:
            raise ValueError("Q8 codes require [vocab,padded_hidden]")
        n, padded = self.codes.shape
        _shape(n, self.hidden_size)
        if padded != (self.hidden_size + 127) // 128 * 128:
            raise ValueError("Q8 hidden padding must be a multiple of 128")
        _tensor(self.codes, (n, padded), torch.int8, torch.device("cpu"), "codes")
        _tensor(
            self.scales, (n, padded // 32), torch.float16, torch.device("cpu"), "scales"
        )
        if bool((self.codes == -128).any()):
            raise ValueError("Q8 code -128 is outside the artifact contract")
        if not bool(torch.isfinite(self.scales).all()) or bool((self.scales < 0).any()):
            raise ValueError("Q8 scales must be finite and nonnegative")

    @classmethod
    def from_ninfer(cls, artifact, binding):
        """Bind a direct named matrix using already validated NInfer metadata.

        No multipart flattening, format guessing, model execution or repacking.
        Embedding/head bindings in the current Qwen artifact are direct objects.
        """
        from ..ninfer import NInferArtifact, row_geometry

        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        spec = artifact.manifest["bindings"][binding]
        if spec.keys() != {"object"}:
            raise ValueError("Q8 vocabulary requires a direct matrix binding")
        name = spec["object"]
        obj = artifact.objects[name]
        if (
            obj.get("format") != "q8_g32_fp16"
            or obj.get("layout") != "row_split_k128_v1"
        ):
            raise ValueError("Vocabulary requires q8_g32_fp16/row_split_k128_v1")
        n, k = obj["shape"]
        _, _, groups, _, scale_offset, _ = row_geometry(obj["shape"], obj["format"])
        codes = _read_plane(artifact, name, 0, (n, groups * 32), torch.int8)
        scales = _read_plane(artifact, name, scale_offset, (n, groups), torch.float16)
        weights = cls(codes, scales, k)
        weights.validate()
        return weights

    def upload(self, backend, *, device="cuda:0"):
        self.validate()
        target = backend._device(torch.bfloat16, device)
        codes, scales = (
            x.detach().to(target, copy=True) for x in (self.codes, self.scales)
        )
        if target.type == "cuda":
            torch.cuda.current_stream(target).synchronize()
        return ResidentQ8Vocabulary(backend, codes, scales, self.hidden_size)


@dataclass(frozen=True)
class ResidentQ8Vocabulary:
    backend: object
    _codes: torch.Tensor
    _scales: torch.Tensor
    hidden_size: int
    dtype = torch.bfloat16

    @property
    def device(self):
        return self._codes.device

    @property
    def vocab_size(self):
        return self._codes.shape[0]

    @property
    def storage_tensors(self):
        return (self._codes, self._scales)

    @property
    def storage_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage_tensors)

    def prepare_embedding(self, ids):
        _ids(ids, self.vocab_size)
        self.backend._device(self.dtype, self.device)
        indices = torch.tensor(ids, dtype=torch.int64, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return Q8Embedding(self, indices)

    def prepare(self, *, tokens):
        _shape(tokens, self.vocab_size)
        if tokens * self.vocab_size >= 2**31:
            raise ValueError("Q8 head row count must fit int32")
        self.backend._device(self.dtype, self.device)
        return Q8OutputHead(self, tokens)


@dataclass(frozen=True)
class Q8Embedding:
    resident: ResidentQ8Vocabulary
    _indices: torch.Tensor

    def run(self, *, out):
        r = self.resident
        _tensor(out, (self._indices.numel(), r.hidden_size), r.dtype, r.device, "out")
        _no_overlap(out, r.storage_tensors + (self._indices,))
        if r.backend.name == "torch-reference":
            codes, scales = r._codes[self._indices], r._scales[self._indices]
            out.copy_(
                (codes.float() * scales.float().repeat_interleave(32, -1))[
                    :, : r.hidden_size
                ].to(r.dtype)
            )
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_q8_embedding_run(
                        r._codes.data_ptr(),
                        r._scales.data_ptr(),
                        self._indices.data_ptr(),
                        out.data_ptr(),
                        self._indices.numel(),
                        r.vocab_size,
                        r.hidden_size,
                        r._codes.shape[1],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out


@dataclass(frozen=True)
class Q8OutputHead:
    resident: ResidentQ8Vocabulary
    tokens: int
    workspace_bytes = 0

    def run(self, x, *, out):
        r = self.resident
        _tensor(x, (self.tokens, r.hidden_size), r.dtype, r.device, "x")
        _tensor(out, (self.tokens, r.vocab_size), r.dtype, r.device, "out")
        _no_overlap(out, (x,) + r.storage_tensors)
        if r.backend.name == "torch-reference":
            # Explicit allocating, tiled FP32 reference. No full-vocabulary dense
            # reconstruction; the block bound keeps peak scratch under control.
            for start in range(0, r.vocab_size, 512):
                end = min(start + 512, r.vocab_size)
                weight = r._codes[start:end].float() * r._scales[
                    start:end
                ].float().repeat_interleave(32, -1)
                out[:, start:end].copy_(
                    (x.float() @ weight[:, : r.hidden_size].T).to(r.dtype)
                )
        else:
            with torch.cuda.device(r.device):
                r.backend._check(
                    r.backend._native.venturi_q8_head_run(
                        x.data_ptr(),
                        r._codes.data_ptr(),
                        r._scales.data_ptr(),
                        out.data_ptr(),
                        self.tokens,
                        r.vocab_size,
                        r.hidden_size,
                        r._codes.shape[1],
                        torch.cuda.current_stream(r.device).cuda_stream,
                    )
                )
        return out


@dataclass(frozen=True)
class NInferVocabulary:
    """Validated text embedding/head bindings, deduplicated by physical object ID.

    Q8 row-split format only for now. This binds vocabulary storage, not decoder
    layers or a whole model. Different Qwen/other NInfer adapters can reuse it.
    """

    embedding: ResidentQ8Vocabulary
    head: ResidentQ8Vocabulary
    object_ids: tuple[str, str]

    @classmethod
    def upload_from_ninfer(cls, artifact, backend, *, device="cuda:0"):
        from ..ninfer import NInferArtifact

        if not isinstance(artifact, NInferArtifact):
            raise TypeError("Expected validated NInferArtifact")
        config = artifact.manifest["components"]["text"]["config"]
        vocab, hidden = config.get("vocab_size"), config.get("hidden_size")
        _shape(vocab, hidden)
        names = ("text/token_embedding", "text/output_head")
        objects = []
        # Validate both bindings before any large reads or GPU allocation.
        for name in names:
            spec = artifact.manifest["bindings"].get(name)
            if not isinstance(spec, dict) or spec.keys() != {"object"}:
                raise ValueError(f"{name} requires a direct vocabulary binding")
            obj = artifact.objects[spec["object"]]
            if (
                obj["shape"] != [vocab, hidden]
                or obj.get("format") != "q8_g32_fp16"
                or obj.get("layout") != "row_split_k128_v1"
            ):
                raise ValueError(
                    f"{name} requires Q8 [vocab_size,hidden_size] row-split storage"
                )
            objects.append(spec["object"])
        tied = config.get("tie_word_embeddings", False)
        if type(tied) is not bool or (tied and objects[0] != objects[1]):
            raise ValueError("Tied vocabulary must reference the same physical object")
        residents = {}
        for name, obj in zip(names, objects):
            if obj not in residents:
                weights = Q8VocabularyWeights.from_ninfer(artifact, name)
                residents[obj] = weights.upload(backend, device=device)
                del weights
        return cls(residents[objects[0]], residents[objects[1]], tuple(objects))

    @property
    def storage_nbytes(self):
        unique = {
            t.data_ptr(): t
            for r in (self.embedding, self.head)
            for t in r.storage_tensors
        }
        return sum(t.numel() * t.element_size() for t in unique.values())
