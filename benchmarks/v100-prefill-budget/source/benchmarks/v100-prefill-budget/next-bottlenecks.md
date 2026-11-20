# Remaining work after projection-budget tuning

Only C4 currently clears the pinned external HTTP control's >=5% margin. C1,
C2 and C8 still miss; MTP1 remains materially below the strongest tested external
post-first rate. Budget tuning is not a substitute for the remaining native
algorithm/verification work.

## Native MTP2 first

Implement a bounded two-proposal native chain, not an external forward. Verify
normalized-draft-hidden recurrence against primary implementation/reference
math. The first pending-token draft cache entry is based on a verified prior
hidden and can be retained; speculative later entries must be rewound and
rebuilt from verified target hidden states. Draft is attention-only, but cursor
rewind is not sufficient for the target's GDN/history: restore its checkpoint
and replay exactly the committed input prefix on partial rejection/EOS/budget
boundaries. Preserve last-emitted-but-unconsumed token semantics. Qualify forced
accept/reject at every depth, natural acceptance, failures/reset, exact ordinary-
target outputs/committed state and per-request ownership before speed. MTP1's
shortcut is invalid at depth two. Do not compare only a weaker competitor depth.

## Larger prefill token tiles / repeated GDN work

A 256-row budget can feed T184 in one call, but the projection shader still
reuses each weight over at most eight token rows. A token16 K-tiled prototype
could trade more shared memory/registers for weight reuse. It needs independent
byte/FMA/tail/graph gates, exact resource queries and real C1/C8 screening; no
higher-occupancy or 2x-serving claim follows from a tile size. Preserve the
negative WMMA/shared-T1 and copy-only studies rather than repeat them unchanged.

Alternatively precompute original-order FP32 GDN Q/K normalization and decay
once per head, without an extra FP16 cast. Additional launch/scratch/traffic
must be accounted for and beat current compact GDN at T1/T2 as well as prefill.
Changed reduction association requires explicit quality qualification.

## Workspace memory

Rows256 adds 1722 MiB primary NVML usage for only a 3.43% C8 HTTP gain here.
Inspect duplicated batch-private forward workspaces separately from shared work
and per-request cache/RoPE/state. Do not remove buffers required for isolation,
failure/reset or snapshots, and keep admission estimates equal to allocations.
A memory-only improvement must not be relabelled as a throughput win.
