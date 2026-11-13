"""Model-bundle identity and storage admission, separate from operator dispatch.

This module never imports torch, downloads files, allocates GPU memory, evicts
pages or starts a runtime. An admission plan is NOT model support or a speed
prediction. Sparse lookup files are disk assets, not resident matrix weights.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import re
from urllib.parse import urlsplit, unquote


_ROLES = frozenset(("backbone", "sparse_lookup", "draft", "metadata"))


def _bytes(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be nonnegative integer bytes")
    return value


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    bytes: int
    sha256: str
    url: str
    repository: str
    revision: str
    source_filename: str
    role: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,191}", self.filename) or ".." in self.filename:
            raise ValueError("Asset filename must be a direct safe filename")
        _bytes("asset", self.bytes)
        if not self.bytes or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Asset requires positive size and SHA256")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Immutable 40-character repository revision required")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ValueError("Explicit Hugging Face owner/repository required")
        source = PurePosixPath(self.source_filename)
        if source.is_absolute() or ".." in source.parts or str(source) != self.source_filename or source.name != self.filename:
            raise ValueError("Canonical relative source path must retain asset filename")
        parsed = urlsplit(self.url)
        expected = f"/{self.repository}/resolve/{self.revision}/{self.source_filename}"
        if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
                or parsed.query or parsed.fragment or unquote(parsed.path) != expected):
            raise ValueError("Source URL must match immutable repository/revision/path")
        if self.role not in _ROLES:
            raise ValueError("Unknown asset role")

    def verify(self, root):
        """Explicit full-file hash check; no trust based only on equal length."""
        path = Path(root) / self.filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != self.bytes:
            raise ValueError(f"Missing, symlinked or wrong-size asset: {self.filename}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 2**20), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ValueError(f"Asset SHA256 mismatch: {self.filename}")
        return path


@dataclass(frozen=True)
class ModelBundle:
    model: str
    official_revision: str
    variant: str
    files: tuple[ModelAsset, ...]

    def __post_init__(self):
        object.__setattr__(self, "files", tuple(self.files))
        if not self.model or not self.variant or not re.fullmatch(r"[0-9a-f]{40}", self.official_revision):
            raise ValueError("Model, variant and official immutable revision required")
        if not self.files or any(not isinstance(a, ModelAsset) for a in self.files):
            raise ValueError("Typed assets required")
        if len({a.filename for a in self.files}) != len(self.files):
            raise ValueError("Duplicate asset filename")
        # Different quant repositories are allowed (e.g. backbone and drafter),
        # but a source repo cannot silently mix revisions within one bundle.
        for repo in {a.repository for a in self.files}:
            if len({a.revision for a in self.files if a.repository == repo}) != 1:
                raise ValueError("Mixed revisions from the same source repository")

    @classmethod
    def from_dict(cls, raw):
        return cls(raw["model"], raw["official_revision"], raw["variant"],
                   tuple(ModelAsset(**a) for a in raw["files"]))

    def role_bytes(self, role):
        if role not in _ROLES:
            raise ValueError("Unknown asset role")
        return sum(a.bytes for a in self.files if a.role == role)

    @property
    def disk_bytes(self):
        return sum(a.bytes for a in self.files)


@dataclass(frozen=True)
class StorageBudget:
    """FREE VRAM, AVAILABLE RAM and FREE disk, not installed totals.

    RAM reservations include runtime heaps/activations/OS needs. Expert RAM is
    one budget for either file cache OR anonymous copies, never both. Runtime
    workspaces, KV and draft weights must be counted explicitly by the caller.
    """
    vram_free: int
    ram_available: int
    disk_free: int
    vram_reserve: int
    ram_reserve: int
    disk_reserve: int

    def __post_init__(self):
        for name, value in vars(self).items():
            _bytes(name, value)
        if self.vram_reserve > self.vram_free or self.ram_reserve > self.ram_available or self.disk_reserve > self.disk_free:
            raise ValueError("Reserve exceeds free/available capacity")


@dataclass(frozen=True)
class MoEStoragePlan:
    dense_and_draft_vram: int
    hot_experts_vram: int
    cold_experts_ram: int
    cold_experts_disk: int
    sparse_lookup_disk: int
    sparse_lookup_ram_cache: int
    additional_download_bytes: int

    @property
    def experts_fully_memory_resident(self):
        return self.cold_experts_disk == 0


def plan_moe_storage(bundle, budget, *, dense_and_draft_bytes, routed_expert_bytes,
                     expert_alignment, sparse_cache_bytes=0, verified_existing_bytes=0):
    """Conservative byte-level admission; no implicit precision/top-k changes.

    Splittable expert storage is rounded to a caller-specified allocation unit.
    Its size MUST come from an actual tensor/pack inventory, not average bpw or
    the GGUF shard size. This is a capacity plan, not an adaptive cache policy.
    """
    for name, value in dict(dense_and_draft_bytes=dense_and_draft_bytes,
                            routed_expert_bytes=routed_expert_bytes,
                            sparse_cache_bytes=sparse_cache_bytes,
                            verified_existing_bytes=verified_existing_bytes,
                            expert_alignment=expert_alignment).items():
        _bytes(name, value)
    if not expert_alignment or routed_expert_bytes % expert_alignment:
        raise ValueError("Routed experts must respect a positive allocation unit")
    if verified_existing_bytes > bundle.disk_bytes:
        raise ValueError("Verified existing bytes exceed the bundle")
    remaining = bundle.disk_bytes - verified_existing_bytes
    if remaining + budget.disk_reserve > budget.disk_free:
        raise ValueError("Insufficient disk for remaining assets plus reserve")
    gpu = budget.vram_free - budget.vram_reserve - dense_and_draft_bytes
    if gpu < 0:
        raise ValueError("Dense/draft weights do not fit VRAM; no automatic CPU fallback")
    lookup = bundle.role_bytes("sparse_lookup")
    if sparse_cache_bytes > lookup:
        raise ValueError("Sparse RAM cache exceeds lookup asset size")
    ram = budget.ram_available - budget.ram_reserve - sparse_cache_bytes
    if ram < 0:
        raise ValueError("Sparse cache exceeds RAM budget")
    hot = min(routed_expert_bytes, gpu // expert_alignment * expert_alignment)
    cold = routed_expert_bytes - hot
    resident = min(cold, ram // expert_alignment * expert_alignment)
    return MoEStoragePlan(dense_and_draft_bytes, hot, resident, cold - resident,
                          lookup, sparse_cache_bytes, remaining)
