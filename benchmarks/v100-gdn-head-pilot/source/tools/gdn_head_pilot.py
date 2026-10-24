"""Preparation/diagnostics for an isolated native GDN head-partition pilot.

No model/session dispatch is installed. Python owns allocations and untimed
canonical joining; the timed block loop and transport are entirely C++.
"""

import ctypes as C
from dataclasses import dataclass
import threading

import torch

from venturi.models.gdn_frontend import GDNFrontendWeights
from venturi.ops import CudaBackend, Q8RowSplitWeight
from venturi.ops.q8_column_tp import slice_q8_columns


def semantic_rows(qk, heads, dk, dv, rank):
    if (
        any(type(n) is not int or n <= 0 for n in (qk, heads, dk, dv))
        or qk % 2
        or heads % qk
        or rank not in (0, 1)
    ):
        raise ValueError(
            "Semantic head split requires even Q/K heads and integral value groups"
        )
    q, v = qk * dk, heads * dv
    return tuple(
        list(range(rank * q // 2, (rank + 1) * q // 2))
        + list(range(q + rank * q // 2, q + (rank + 1) * q // 2))
        + list(range(2 * q + rank * v // 2, 2 * q + (rank + 1) * v // 2))
    )


def take_rows(weight, indices):
    weight.validate()
    if weight.codes.device.type != "cpu":
        raise ValueError("Encoded row preparation requires CPU source planes")
    ids = torch.tensor(indices, dtype=torch.int64)
    return Q8RowSplitWeight(
        weight.codes[ids].contiguous(),
        weight.scales[ids].contiguous(),
        weight.in_features,
    )


def join_channels(values, qk, heads, dk, dv):
    """Untimed diagnostic scatter, not two concatenated local-QKV histories."""
    shape = (*values[0].shape[:-1], 2 * qk * dk + heads * dv)
    out = torch.empty(shape, dtype=values[0].dtype)
    for rank in (0, 1):
        out[..., list(semantic_rows(qk, heads, dk, dv, rank))] = values[rank]
    return out


class Config(C.Structure):
    _fields_ = [
        (name, C.c_int)
        for name in (
            "mode",
            "hidden",
            "qk",
            "heads",
            "dk",
            "dv",
            "kernel",
            "sequences",
            "steps",
            "prefixes",
            "gamma_fp32",
        )
    ] + [
        ("epsilon", C.c_float),
        ("scale", C.c_float),
        ("input", C.c_void_p),
        ("output", C.c_void_p),
        ("norm", C.c_void_p),
        ("projected", C.c_void_p),
    ]


class Rank(C.Structure):
    _fields_ = (
        [("codes", C.c_void_p * 3), ("scales", C.c_void_p * 3)]
        + [
            (n, C.c_void_p)
            for n in (
                "a",
                "b",
                "conv",
                "rate",
                "bias",
                "gamma",
                "normalized",
                "qkv",
                "z",
                "av",
                "bv",
                "convolved",
                "decay",
                "beta",
                "recurrent",
                "mixed",
                "local_output",
                "partial",
                "remote_partial",
                "quant",
            )
        ]
        + [("quant_bytes", C.c_uint64)]
        + [
            (n, C.c_void_p * count)
            for n, count in (
                ("history", 8),
                ("state", 8),
                ("prefix_history", 16),
                ("prefix_state", 16),
            )
        ]
    )


@dataclass(frozen=True)
class Parts:
    qkv: object
    z: object
    output: object
    a: torch.Tensor
    b: torch.Tensor
    frontend: GDNFrontendWeights
    epsilon: float

    @classmethod
    def from_weights(cls, weights, layer=0):
        prefix = f"text/layers/{layer}/gdn/"
        return cls(
            *(weights.read_linear(prefix + name) for name in ("qkv", "z", "output")),
            *(weights.read_float(prefix + name).float() for name in ("a", "b")),
            GDNFrontendWeights.from_ninfer(weights.artifact, layer=layer),
            weights.config["rms_norm_eps"],
        )


class Pilot:
    def __init__(
        self,
        parts,
        blocks,
        library,
        *,
        mode,
        sequences,
        steps,
        inputs,
        histories,
        states,
    ):
        if (
            mode not in ("row", "head")
            or sequences not in (1, 8)
            or steps not in (1, 2, 3)
            or (steps != 1 and sequences != 1)
        ):
            raise ValueError(
                "Pilot supports scalar1/3 or eight independent one-token sequences"
            )
        f = parts.frontend
        h, qk, heads, dk, dv, kernel = (
            parts.qkv.in_features,
            f.qk_heads,
            f.value_heads,
            f.key_dim,
            f.value_dim,
            f.conv.shape[0],
        )
        semantic_rows(qk, heads, dk, dv, 0)
        if h % 32 or heads * dv // 2 % 32:
            raise ValueError(
                "Pilot requires aligned hidden and local output-projection columns"
            )
        if (
            inputs.dtype != torch.float16
            or tuple(inputs.shape) != (sequences * steps, h)
            or inputs.device.type != "cpu"
        ):
            raise ValueError("Explicit contiguous CPU FP16 inputs required")
        if (
            histories.dtype != torch.float16
            or tuple(histories.shape)
            != (sequences, kernel - 1, 2 * qk * dk + heads * dv)
            or states.dtype != torch.float32
            or tuple(states.shape) != (sequences, heads, dv, dk)
        ):
            raise ValueError(
                "Explicit private canonical histories/packed FP32 states required"
            )
        if torch.cuda.device_count() != 2 or any(
            torch.cuda.get_device_capability(i) != (7, 0) for i in (0, 1)
        ):
            raise ValueError("Exactly two explicitly selected V100s required")
        self._lock, self._handle = threading.RLock(), None
        self.parts, self.mode, self.sequences, self.steps = (
            parts,
            mode,
            sequences,
            steps,
        )
        self.rows, self.h, self.qk, self.heads, self.dk, self.dv = (
            sequences * steps,
            h,
            qk,
            heads,
            dk,
            dv,
        )
        self.initial_history, self.initial_state = histories.clone(), states.clone()
        (
            self.backend,
            self.keep,
            self.tensors,
            self.state_tensors,
            self.prefix_tensors,
        ) = CudaBackend(blocks), [], [], [], []
        self.native = C.CDLL(str(library))
        n = self.native
        n.venturi_gdn_head_error.restype = C.c_char_p
        n.venturi_gdn_head_sizes.argtypes = [C.POINTER(C.c_uint64)]
        sizes = (C.c_uint64 * 2)()
        n.venturi_gdn_head_sizes(sizes)
        if list(sizes) != [C.sizeof(Config), C.sizeof(Rank)]:
            raise RuntimeError("Pilot descriptor ABI mismatch")
        n.venturi_gdn_head_create.argtypes, n.venturi_gdn_head_create.restype = (
            [C.c_char_p, C.POINTER(Config), C.POINTER(Rank)],
            C.c_void_p,
        )
        n.venturi_gdn_head_destroy.argtypes = [C.c_void_p]
        n.venturi_gdn_head_run.argtypes, n.venturi_gdn_head_run.restype = (
            [C.c_void_p, C.c_int, C.POINTER(C.c_double)],
            C.c_int,
        )
        n.venturi_gdn_head_info.argtypes, n.venturi_gdn_head_info.restype = (
            [C.c_void_p, C.POINTER(C.c_uint64)],
            C.c_int,
        )
        # Bounded conservative admission includes both model-independent scratch
        # and private prefix/history/state allocations; no implicit offload.
        for rank in (0, 1):
            need = (
                sum(w.nbytes for w in (parts.qkv, parts.z, parts.output))
                + sequences * heads * dk * dv * 4 * 3
                + self.rows * (h + 4 * (2 * qk * dk + heads * dv) + 8 * heads * dv) * 8
                + 64 * 2**20
            )
            if need + 512 * 2**20 > torch.cuda.mem_get_info(rank)[0]:
                raise MemoryError(
                    "Pilot memory admission failed before GPU allocations"
                )

        def owned(value, rank):
            t = value.contiguous().to(f"cuda:{rank}", copy=True)
            self.keep.append(t)
            return t

        norm = self.backend.upload_zero_centered_rmsnorm(
            f.input_norm, dtype=torch.float16, device="cuda:0", epsilon=parts.epsilon
        )
        self.keep.append(norm)
        self.input = owned(inputs, 0)
        self.output = torch.empty_like(self.input)
        self.keep.append(self.output)
        self.projected = torch.empty_like(self.input)
        self.keep.append(self.projected)
        descriptors = (Rank * 2)()
        for rank in (0, 1):
            desc, t = descriptors[rank], {}
            local = mode == "head"
            ch, vh = (
                (2 * qk * dk + heads * dv) // (2 if local else 1),
                heads // (2 if local else 1),
            )
            val = vh * dv
            indices = (
                semantic_rows(qk, heads, dk, dv, rank)
                if local
                else tuple(
                    range(
                        rank * parts.qkv.out_features // 2,
                        (rank + 1) * parts.qkv.out_features // 2,
                    )
                )
            )
            weights = (
                take_rows(parts.qkv, indices),
                take_rows(
                    parts.z,
                    tuple(
                        range(
                            rank * parts.z.out_features // 2,
                            (rank + 1) * parts.z.out_features // 2,
                        )
                    ),
                ),
                slice_q8_columns(
                    parts.output, rank * heads * dv // 2, (rank + 1) * heads * dv // 2
                )
                if local
                else take_rows(
                    parts.output, tuple(range(rank * h // 2, (rank + 1) * h // 2))
                ),
            )
            for j, w in enumerate(weights):
                code, scale = owned(w.codes, rank), owned(w.scales, rank)
                desc.codes[j], desc.scales[j] = code.data_ptr(), scale.data_ptr()

            def empty(name, shape, dtype=torch.float16):
                value = torch.empty(shape, dtype=dtype, device=f"cuda:{rank}")
                self.keep.append(value)
                t[name] = value
                setattr(desc, name, value.data_ptr())
                return value

            empty("normalized", (self.rows, max(h, heads * dv) if rank == 1 else h))
            if local or rank == 1:
                qn, zn = (ch, val) if local else (ch // 2, val // 2)
                packed = torch.empty(
                    self.rows * (qn + zn), device=f"cuda:{rank}", dtype=torch.float16
                )
                self.keep.append(packed)
                t["qkv"], t["z"] = (
                    packed[: self.rows * qn].view(self.rows, qn),
                    packed[self.rows * qn :].view(self.rows, zn),
                )
                desc.qkv, desc.z = t["qkv"].data_ptr(), t["z"].data_ptr()
            else:
                empty("qkv", (self.rows, ch))
                empty("z", (self.rows, val))
            empty("local_output", (self.rows, h // 2))
            qb = self.rows * ((max(h, heads * dv) + 31) // 32) * 72
            empty("quant", (qb,), torch.uint8)
            desc.quant_bytes = qb
            if local:
                empty("partial", (self.rows, h), torch.float32)
                if rank == 0:
                    empty("remote_partial", (self.rows, h), torch.float32)
            hslots, sslots, ph, ps = [], [], [[], []], [[], []]
            if local or rank == 0:
                head_indices = (
                    list(range(rank * heads // 2, (rank + 1) * heads // 2))
                    if local
                    else list(range(heads))
                )
                for name, value in (
                    ("a", parts.a[head_indices]),
                    ("b", parts.b[head_indices]),
                    (
                        "conv",
                        f.conv.float()[:, list(indices)] if local else f.conv.float(),
                    ),
                ):
                    value = owned(value, rank)
                    setattr(desc, name, value.data_ptr())
                gates = self.backend.upload_gdn_gates(
                    f.a_log[head_indices],
                    f.dt_bias[head_indices],
                    dtype=torch.float16,
                    device=f"cuda:{rank}",
                )
                self.keep.append(gates)
                desc.rate, desc.bias = (v.data_ptr() for v in gates.storage_tensors)
                gamma = owned(
                    f.gated_norm
                    if f.gated_norm.dtype == torch.float32
                    else f.gated_norm.half(),
                    rank,
                )
                desc.gamma = gamma.data_ptr()
                for name, width, dtype in (
                    ("av", vh, torch.float16),
                    ("bv", vh, torch.float16),
                    ("convolved", ch, torch.float16),
                    ("decay", vh, torch.float32),
                    ("beta", vh, torch.float32),
                    ("recurrent", val, torch.float16),
                    ("mixed", val, torch.float16),
                ):
                    empty(name, (self.rows, width), dtype)
                for s in range(sequences):
                    hi = histories[s, :, list(indices)] if local else histories[s]
                    st = states[s, head_indices]
                    hv, sv = owned(hi, rank), owned(st, rank)
                    hslots.append(hv)
                    sslots.append(sv)
                    desc.history[s], desc.state[s] = hv.data_ptr(), sv.data_ptr()
                    if steps == 3:
                        for p in (0, 1):
                            hp, sp = (
                                owned(torch.zeros_like(hi), rank),
                                owned(torch.zeros_like(st), rank),
                            )
                            ph[p].append(hp)
                            ps[p].append(sp)
                            (
                                desc.prefix_history[p * 8 + s],
                                desc.prefix_state[p * 8 + s],
                            ) = hp.data_ptr(), sp.data_ptr()
            self.tensors.append(t)
            self.state_tensors.append((hslots, sslots))
            self.prefix_tensors.append((ph, ps))
        self.config = Config(
            int(mode == "head"),
            h,
            qk,
            heads,
            dk,
            dv,
            kernel,
            sequences,
            steps,
            2 if steps == 3 else 0,
            int(f.gated_norm.dtype == torch.float32),
            parts.epsilon,
            dk**-0.5,
            self.input.data_ptr(),
            self.output.data_ptr(),
            norm.storage_tensors[0].data_ptr(),
            self.projected.data_ptr(),
        )
        self.descriptors = descriptors
        for rank in (0, 1):
            torch.cuda.synchronize(rank)
        self._handle = n.venturi_gdn_head_create(
            str(blocks).encode(), C.byref(self.config), descriptors
        )
        if not self._handle:
            raise RuntimeError(n.venturi_gdn_head_error().decode())

    def close(self):
        with self._lock:
            if self._handle:
                self.native.venturi_gdn_head_destroy(self._handle)
                self._handle = None

    def run(self, iterations=1):
        with self._lock:
            if not self._handle:
                raise RuntimeError("GDN pilot closed")
            elapsed = C.c_double()
            rc = self.native.venturi_gdn_head_run(
                self._handle, iterations, C.byref(elapsed)
            )
            if rc < 0:
                raise RuntimeError(self.native.venturi_gdn_head_error().decode())
            counters = (C.c_uint64 * 7)()
            if self.native.venturi_gdn_head_info(self._handle, counters) < 0:
                raise RuntimeError(self.native.venturi_gdn_head_error().decode())
            return dict(
                native_wall_seconds=elapsed.value,
                **dict(
                    zip(
                        (
                            "h2d_bytes",
                            "d2h_bytes",
                            "h2d_calls",
                            "d2h_calls",
                            "d2d_calls",
                            "iterations",
                            "pinned_bytes",
                        ),
                        counters,
                    )
                ),
            )

    def reset(self, slots=None):
        with self._lock:
            if not self._handle:
                raise RuntimeError("GDN pilot closed")
            ids = list(range(self.sequences)) if slots is None else list(slots)
            if (
                not ids
                or any(type(s) is not int or not 0 <= s < self.sequences for s in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("Invalid private pilot slots")
            for rank in (0, 1) if self.mode == "head" else (0,):
                channels = (
                    semantic_rows(self.qk, self.heads, self.dk, self.dv, rank)
                    if self.mode == "head"
                    else tuple(range(2 * self.qk * self.dk + self.heads * self.dv))
                )
                heads = (
                    slice(rank * self.heads // 2, (rank + 1) * self.heads // 2)
                    if self.mode == "head"
                    else slice(None)
                )
                for s in ids:
                    self.state_tensors[rank][0][s].copy_(
                        self.initial_history[s, :, list(channels)]
                    )
                    self.state_tensors[rank][1][s].copy_(self.initial_state[s, heads])
                torch.cuda.synchronize(rank)

    def snapshot(self):
        with self._lock:
            if not self._handle:
                raise RuntimeError("GDN pilot closed")

            def joined(name):
                values = [
                    t[name].cpu()
                    for t in self.tensors[: 2 if self.mode == "head" else 1]
                ]
                if self.mode == "row":
                    return values[0]
                return (
                    join_channels(values, self.qk, self.heads, self.dk, self.dv)
                    if name in ("qkv", "convolved")
                    else torch.cat(values, dim=-1)
                )

            result = {
                name: joined(name)
                for name in (
                    "qkv",
                    "z",
                    "av",
                    "bv",
                    "convolved",
                    "decay",
                    "beta",
                    "recurrent",
                    "mixed",
                )
            }
            result["output"] = self.output.cpu()
            result["projected"] = self.projected.cpu()
            ranks = (0, 1) if self.mode == "head" else (0,)
            result["history"] = torch.stack(
                [
                    join_channels(
                        [self.state_tensors[r][0][s].cpu() for r in ranks],
                        self.qk,
                        self.heads,
                        self.dk,
                        self.dv,
                    )
                    if self.mode == "head"
                    else self.state_tensors[0][0][s].cpu()
                    for s in range(self.sequences)
                ]
            )
            result["state"] = torch.stack(
                [
                    torch.cat([self.state_tensors[r][1][s].cpu() for r in ranks], dim=0)
                    for s in range(self.sequences)
                ]
            )
            if self.steps == 3:
                for p in (0, 1):
                    hs = [self.prefix_tensors[r][0][p][0].cpu() for r in ranks]
                    result[f"prefix{p + 1}_history"] = (
                        join_channels(hs, self.qk, self.heads, self.dk, self.dv)
                        if self.mode == "head"
                        else hs[0]
                    )
                    result[f"prefix{p + 1}_state"] = torch.cat(
                        [self.prefix_tensors[r][1][p][0].cpu() for r in ranks], dim=0
                    )
            return result

    def restore_prefix(self, prefix):
        """Untimed operator transaction check, not model/session integration."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("GDN pilot closed")
            if self.steps != 3 or type(prefix) is not int or prefix not in (1, 2):
                raise ValueError("A three-token verification prefix1/2 is required")
            for rank in (0, 1) if self.mode == "head" else (0,):
                self.state_tensors[rank][0][0].copy_(
                    self.prefix_tensors[rank][0][prefix - 1][0]
                )
                self.state_tensors[rank][1][0].copy_(
                    self.prefix_tensors[rank][1][prefix - 1][0]
                )
                torch.cuda.synchronize(rank)
