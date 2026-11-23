# Budgeted tuning along the M dimension

`M` is the number of activation rows in a particular projection. One
request keeps its own KV/GDN even as its packed trunk and vocabulary head
change M independently. The policy selects a kernel, never a padded
logical request or a different scheduler batch.

For the aligned SM89 Q5-G64 experiment, classify M as 1, 2, 3–4, 5–8,
9–15, 16, 17–31, or 32. The distinctions preserve the grouped GEMV
limit, the full 16-/32-row WMMA tile boundaries, and masked variants.
Do not extrapolate a speed winner from a full tile to a tail by
rounding M to its nearest aligned point. Other strides and N/K
alignment have their own kernel admission checks; M alone is not the
dispatch key.

From an earlier scheduler trace, choose ten representative M/N/K
anchors without consulting prior timing winners. Keep a dedicated M=1
choice and a separate wide M=2 crossover. Reserve two places for
frequent masked 32-row trunk calls and one for sampled head rows
above four, but never spend the head budget on M greater than the
prepared sample-row capacity of this pilot. Each anchor compares two
eligible routes and passes the same operator precision checks. Only a
stable margin justifies trading away the anchor incumbent.

For deployment in the experiment, the policy expands those decisions
to the native immutable per-batch table. Keys preserve the actual M,
N and K, without measuring each numerical M. The narrow, head and
medium/large trunk classes have separate propagation rules. A grouped
choice cannot escape its M≤8 support, masked rules cannot replace
M=16/32, and missing exact native keys keep cuBLAS. The calibration
pilot does not claim tuned support for M>32, other weight formats or
new architectures.

Compare the resulting table with the old ten-anchor and previously
measured atlas policies, preferably on a new prompt-length mix.
Separate pure decode and fixed masked-tile replays from rolling C4/C8,
and separate scored runs from CUDA capture. Record P95 as well as TPS;
neither sampled matrix timing nor agreement on one fixed token tape
qualifies a new numerical mode for default serving.

See [the single-card result and regression procedure](sm89-m-bucket-results.md).
