"""Floating gated-delta core and sigmoid-gated RMS for Flash-Next qualification.

Explicit FP32 state layout [B, value_heads, key_dim, value_dim]. Callers must
expand K heads to V heads first (official16->48), and keep convolution state
separately. Not a full GDN frontend/native kernel/automatic backend selector.
Chunk and recurrent evaluation have different rounding and are not bit-equal.
"""

import math

# Core math lives in the shared mechanism layer; re-export for compatibility.
from venturi.models.linear_attention import SigmoidGatedRMSNorm, gated_delta_rule  # noqa: F401


import torch
from torch import nn
from torch.nn import functional as F


