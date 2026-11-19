# Early-warning replay candidates

These are native mixed ORD scenarios. The 32-sample Anchor model, anomaly
cutoff, two-second issue cadence and fictional demo zone were kept fixed.

| Scenario | Drone ID | Record from | Show | First qualifying forecast | Next forecast | First reference entry | Lead from first warning |
|---|---|---:|---:|---:|---:|---:|---:|
| 000073 | `~drone01` | 92 s | 98–128 s | 100 s, 9/32 | 102 s, 14/32 | 108.67 s | 8.67 s |
| 000052 | `~drone02` | 78 s | 84–114 s | 86 s, 20/32 | 88 s, 30/32 | 99.67 s | 13.67 s |

All times are seconds from scenario start. The 000073 visible clip is
2026-10-12 17:01:38–17:02:08 UTC. The 000052 visible clip is
2026-10-10 17:16:24–17:16:54 UTC. Capturing six seconds before playback
supplies its opening history and a common timestamp for synchronization.

The forecasts above were checked again with the fixed set of tracks that
cover the entire clip plus history, as used by the D12 replay. In 000073 the
target has 0/32 at 98 s and then 9/32, 14/32, 9/32, 30/32 and 32/32 up to
entry. Its observed first entry is 109 s, and two aircraft are continuous.
In 000052 the target rises from 3/32 at 84 s to 20/32 and 30/32, and stays
above threshold through the last pre-entry update checked. Its reference
and noisy track first enter at the same timestamp. Three aircraft and only
the target drone satisfy the full display-track criterion in this window.

The screening pass found 33 drone tracks with entry in both position
streams and 19 with an eligible pair of pre-entry observations. Anchor
inference checked 13 cases including the earlier 000073 window. Ten
passed the two-update rule in screening. Constant velocity prioritized
which candidates to check and was not used as warning evidence.
This is selection for a demo, with camera coverage still to be confirmed.
New clips have not been rendered. Existing clip files and replay request
are preserved. Local evidence is in `outputs/unreal_replay_shortlist/early_warning/`.
