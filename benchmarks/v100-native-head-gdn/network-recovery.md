# Network recovery record

Temporary SSH closures/timeouts did not trigger worker restarts or a host switch.

- Fixture-v2 synchronization completed, but its first dispatch lost the SSH
  connection before acknowledgement. A bounded inspection first timed out,
  then connected and proved both the artifact directory and matching process
  handles absent. Only then was the exact serialized dispatch replayed; it
  started PID3147036 and completed its tests.
- Two initial source-v3 copy/initialization requests lost the connection. An
  authenticated diagnostic confirmed the first had not created that snapshot.
  Initialization then used an idempotent dedicated directory and bounded rsync,
  rather than assuming a remote job had failed or resetting any checkout.
- The first source-v3 rsync dropped; the second completed and full source hashes
  matched before PID3164680 began qualification. No partially synchronized source
  was executed. Terminal/qualified jobs are reused by hash, not rerun after an
  observation failure.
- Raw fetches use a finite two-attempt transfer budget with longer keepalive
  tolerance for short transport outages. Strict host-key checks, the same
  WZU_Server destination, disabled agent forwarding and no proxy override route
  changes are preserved throughout.

The successful SSH diagnostic and per-attempt transfer logs are retained here.
These were transport failures, not a hard prerequisite requiring user action.
