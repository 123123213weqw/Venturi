# Finite recovery — final controls remain unfinished

Initial dispatch failed during TCP connect to the configured V100 endpoint
100.70.188.115:9022. No remote PID/artifact/result has been assumed. Three bounded
same-host observations with5/15s backoff failed. Each used15s connect timeout,
BatchMode, no agent forwarding, strict host-key verification and no blind replay.
The exact serialized dispatch is preserved, but must not be replayed until a
healthy observation confirms the unique artifact directory is absent.

The pre-existing same-host control master exists locally but a bounded40s read
was unresponsive; it was not terminated/reconfigured. Tailscale control status
reports the node online through HKG relay, but three5s dataplane pings receive
no reply. Therefore control-plane presence is not treated as reachable SSH.
The configured rescue port9023 also times out. The existing rescue jump route
through the4080 fails during banner exchange. A configured campus LAN route
through School_4090D is separately observed with strict verification and no agent
forwarding; see its raw log. These are routes to the same selected V100, not a
substitute benchmark host. No router, driver, firewall, SSH/Tailscale policy,
clock, host key or authentication configuration was changed.

Independent work continues locally:43 frozen capsules pinned, new comparison
protocol prepared, finite aligned probability checker and fail-closed parser
checks prepared. No Rust/CUDA compilation, new GPU result, fresh external
comparison or goal completion is claimed. This is unfinished network-dependent
work, not concrete proof of a permanent block. Next: inspect the unique remote
artifact/process via a healthy authorized route; resume existing work or replay
exact serialized dispatch only after absence is verified. Then run the separate
aligned-probability diagnostic and completion audit.

## Second goal turn: recovery, real GPU progress, then another transport outage

A fresh direct read recovered and authoritatively found the initial controls-v1
artifact absent. Only then was the exact preserved dispatch launched, PID3291428.
The live /proc command was checked. It completed ordinary row256 qualification,
four alternating C1/C8 screens and two three-repeat1/2/4/8 matrices. Candidate
C8 is83.85089 TPS. The first external server then exited127 before loading;
aligned-v2 PID3294734 stopped at its prerequisite gate and did not run inference.
Both terminal markers were read through the live monitor; neither was restarted.

An actual read through an existing authenticated control master identifies the
loader error: `libllama-server-impl.so` is not on the dynamic loader route.
Recovery-v4 restores the same per-competitor-process LD_LIBRARY_PATH used by the
previous pinned campaigns, verifies ldd and records actual process settings and
loaded-library hashes. It deliberately reuses the completed native qualification
and matrices rather than repeating them. Direct dispatch hit TCP timeout; a
subsequent successful master read found v4 absent before one exact replay through
that session. That replay then exceeded80s without a PID response. The v5 aligned
submission similarly exceeded60s. **Both outcomes are unknown until inspected**;
no duplicate dispatch is permitted merely because an observation expired.

Fresh small-query attempts encountered connection closure, banner timeout and
TCP timeout with bounded backoff. Configured rescue and campus jump routes also
timed out. No host selection, authentication/host-key policy or global network
setting changed. The small successful master read establishes only that one
request worked, not continuing health. Preserve its response and the later
unknown-outcome probe. Observe v4/v5 directories, actual /proc handles and terminal
markers before deciding the next safe action. Do not treat either as stopped.
