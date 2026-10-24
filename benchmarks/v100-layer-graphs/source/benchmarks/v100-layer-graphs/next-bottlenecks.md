# Next: reduce the work, not only the launch count

The native/external MTP2 story case both accept17/26 proposals. Do not attribute
the decode gap to an unmeasured acceptance deficit or silently weaken precision.
Graphs removed most eager launches but retained the projection and host-bridge
work, and failed the screen. Do not begin another unbounded block/tile sweep.

Current TP row-shards every projection, broadcasts each full input and gathers
all projection outputs to the primary device. A canonical column/row TP pair
would retain half the attention/GDN heads and MLP intermediates on each rank,
then reduce only O/down full-hidden outputs. Start with a **research-only FP32
partial projection and sum**: bit-preserving Q8 column slicing at stored group
boundaries, the same two-pass activation quantizer, FP32 partials across transport,
one final FP16 cast. Never round each partial to FP16 before summing.

For the actual hidden5120/layers64/63 target-input-token example, two reductions
per layer would request247,726,080 bytes in each host-bridge direction: two times
(4-byte remote partial +2-byte full-hidden broadcast) per element. This excludes
head/draft/reset and is a logical payload estimate, NOT measured wire/DRAM traffic
or a promised speedup. Existing whole-call copies are535,484,944/976 bytes; these
scopes cannot simply be subtracted to predict TPS. Full canonical TP also needs
head-aware QKV/z slicing, private per-rank GDN/convolution/KV/prefix images,
replicated norms/residuals and correct MTP ownership. Operator tests alone cannot
claim model support.

Changing FP32 association across two partial sums is explicit new arithmetic,
not byte parity with the previous profile. Require independent FP64 represented
references, no lower-precision transport, unchanged model-quality bounds and
fresh full model/state/trajectory qualification before enabling an engine path.
