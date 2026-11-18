Hi Joshua,

I screened the new mixed ORD dataset and picked two clips for the initial side-by-side demo. Could you replay these original scenarios in Unreal and share the synchronized camera clips? If available, please include the YOLO detections and tracker IDs in the camera views.

| Scenario in `ord_mixed_100_3hz` | What happens | Capture from* | Side-by-side clip* | Side-by-side time (UTC) |
|---|---|---:|---:|---|
| `scenario_000073` | `~drone01` circles, with four aircraft continuously present in the reference tracks. | 102 s | 108–138 s | Oct 12, 2026, 17:01:48–17:02:18 |
| `scenario_000094` | `~drone01` transitions from hover to departure at 150 s, with four aircraft continuously present in the reference tracks. | 132 s | 138–168 s | Oct 13, 2026, 17:47:18–17:47:48 |

\* Times are relative to each scenario's start. For each camera export, please include the camera ID and first-frame scenario timestamp, plus the frame rate or per-frame timestamps. A full-scenario recording works too. I'll align the clips to the BEV timeline on my side.

If we have time for a third clip, `scenario_000097` has two drones active together (`~drone01`, `~drone02`). Capture from 84 s and use 90–120 s in the demo (Oct 14, 2026, 17:01:30–17:02:00 UTC).

Thanks!
