# sygard-motion

BC aircraft trajectory Demo on the 800-scenario SYGARD ORD dataset. Current scope follows `../SYGARD_MOTION_PLAN.md` D1, §4.1 items 6–10, M6 Demo, M7 Demo GIFs and M8/M8b anomaly scoring MVP. No CAT-K or NN retraining is part of this scope.

Run from this repository in the existing catk environment:

```bash
scripts/catk.sh -m scripts.run_demo
```

This command merges the existing date indexes if needed, loads `outputs/air_pre_bc/checkpoints/epoch_016.ckpt` on CPU, generates 32 joint rollouts with topk_prob K=5/temp=1, evaluates 3D constant velocity and exact GT replay, and writes:

- `outputs/eval/summary.csv` and `summary.md`: 10/20/30s horizontal/vertical minADE, ADE, FDE and 30 s horizontal miss rate, split by date and current ground status; explicit BC-vs-CV comparisons.
- `outputs/demo/*.png`: six windows selected by seeded GT phase strata (climbing/descending/level/ground), never by prediction error.
- `outputs/demo_rollouts/{bc,constant_velocity,gt_replay}/*.npz`: metre-valued positions `[32,N,90,3]`, raw GT `[N,109,3]`, validity/target masks, date and agent metadata.
- `outputs/eval/acceptance.json`: runtime, checkpoint identity and unchanged protected-artifact hashes.

The active split is train=09-28..10-02 (618 scenarios), val=10-03+10-04 (182 scenarios, 912 existing windows). There is no test split in this set. Original date indexes are retained under ignored `data/splits/legacy_date_splits/`; merging never rebuilds windows. Train indexes, type vocabulary, motion dictionary and BC checkpoint remain byte-for-byte unchanged. 10-03 selected epoch 16 and dictionary configuration (K2048/chord heading); 10-04 did not participate in either choice, so results are reported separately and combined.

To render GIFs from existing BC rollout files, without loading a model:

```bash
scripts/catk.sh -m src.eval.demo_gif --output-dir outputs/demo_gif_v2
```

This writes ten 1200×900 top-down animations, `contact_sheet.png` and a selection/validation `manifest.json` under ignored `outputs/demo_gif_v2/`. Omit `--output-dir` to use `outputs/demo_gif/`. Windows are the six existing examples plus four seed-0 random windows with at least three valid aircraft. Each scene uses one joint rollout minimizing the equal-aircraft mean 3D ADE over targets' full valid future horizons; this visualization selection does not change M6 metrics. OSM tiles use a descriptive User-Agent and are reused from ignored `data/basemap_cache/`. Rendering needs network access only for uncached tiles.

Draft v2 follows Yi-Chung's feedback on v1's grey GIF colours: a dark navy map, saturated aircraft colours, thicker trajectories and larger icons/labels. `--theme light` provides a light map alternative. A shared 256-entry palette reserves exact track colours plus antialias blends before allocating the remaining entries to the map. Explicit transparent frame deltas preserve both colour and compression. The contact sheet uses decoded GIF frames so it reflects the encoded animations. Original v1 outputs are retained in `outputs/demo_gif/`. Each revision stops after one visual self-review for Yi-Chung's feedback.

M8 anomaly scoring runs frozen-checkpoint teacher-forced inference:

```bash
scripts/catk.sh -m src.eval.anomaly_score
```

CPU is the default. After checking GPU availability, `--device cuda:0` uses
physical GPU1 under the wrapper's `CUDA_VISIBLE_DEVICES=1,3`. Each sample is an
isolated agent, scored at node3→4 after6s history; the target is the default
autoregressive GT tokenizer label. Scenarios are scanned at stride6, with invalid
tail padding, independently of the old stride90 Demo windows. Full history and
the next2s token must be valid. The native drone pose file is read directly and
interpolated to3Hz, without extrapolation. Residuals separately match the scored
segment from its raw anchor using the existing dictionary metric, avoiding
accumulated chain drift.

Outputs under ignored `outputs/anomaly/` are `token_scores.csv`, `summary.csv`,
`summary.md`, `nll_hist.png`, `drone_timeline.png` and `acceptance.json`. Summary
groups separate10-04 airborne/ground aircraft and10-03 reference aircraft. AUROC
uses drone-positive ranks with average ties, implemented in NumPy. All drone
samples are from one simulated route, so token-level AUROC is descriptive.
No state-plausibility/fallback scores are added. M8 stops after reporting;
optional M9 requires a new request.

Yi-Chung's M8 visualization follow-up replays GT with the existing per-object
NLL and a strict red-outline threshold of7:

```bash
scripts/catk.sh -m src.eval.anomaly_gif
```

This reads `outputs/anomaly/token_scores.csv` without model execution. Outputs
under `outputs/anomaly/gifs/` include `drone01_full.gif` (the entire300s route),
the ten original aircraft examples with predictions removed, a contact sheet,
low/high drone snapshots and a manifest. Native3Hz positions play at about9fps.
Each object label displays the **latest completed2s token** and its scenario-time
interval; its score appears only at the interval end and expires after2s without
a new valid score. N/A is never interpreted as normal. NLL>7 adds a red box
around the object and a thick red halo under the corresponding completed GT
segment. Historical red segments persist when the current score becomes low.
These are offline, single-agent, teacher-forced diagnostics shown over a GT scene;
no new score, inference, training or anomaly metric is introduced.
The drone uses an original vector quadcopter icon (four rotor rings and a
heading-oriented body); aircraft retain the aircraft silhouette.

M8b adds the plan's fixed non-neural location-conditioned state density:

```bash
scripts/catk.sh -m src.eval.anomaly_state --gifs
```

The density fits only eligible3Hz train states (09-28..10-02) with250m cells,
five height bins,14speed bins and12course bins,3×3 spatial pooling and alpha0.5.
Speed and course use only the past2s displacement. It scores the exact M8 token
rows; 10-03 aircraft, both airborne and ground, provide each score's median/IQR.
Combined score is `max(z_transition, z_state)`. No drone-dependent tuning,
neural inference, training or new packages are involved.

Current smoothed results in `outputs/anomaly_state_smoothed/` contain per-token and group CSVs, `summary.md`, the
saved histogram/probability arrays, reference parameters, `traffic_density.png`,
`drone_timeline_state.png` and an acceptance receipt. `--gifs` additionally writes
eleven GT-only combined-score replays and a contact sheet under `gifs/`. Their
red threshold is the10-04-airborne combined p95; labels show combined z-units,
not NLL nats. Combined appears only when the transition's2s token completes.
10-04 p95 exceedance is descriptive; one drone/route does not validate a detector.
M8b stops after reporting and a local commit; optional M9 remains out of scope.

Yi-Chung's speed/course smoothing correction applies the fixed `[1,2,1]/4`
kernel after spatial pooling and before alpha. Course wraps circularly; speed
reflects at the first/last bin. It preserves histogram mass and leaves raw,
joint and height counts unchanged. This is adjacency by bin index, not a
uniform m/s bandwidth. Hard lookup remains, so boundary steps and false alerts
can remain. The advisor's provisional rule is recorded in `QUESTIONS.md` for
inclusion in the plan.

The current default writes to `outputs/anomaly_state_smoothed/`, preserving the
original `outputs/anomaly_state/`. `--legacy-bins` reproduces the original
spatial/alpha-only method. Each method uses its own10-03 state median/IQR and
10-04-airborne p95. `smoothing_comparison.csv` also evaluates the new scores
with frozen original calibration/cutoffs; recalibrated airborne exceedance is
approximately5% by construction, so that figure alone cannot show improvement.
`aa0300_comparison.csv` preserves the reported false-positive example over time.

The BC checkpoint comes from the historical 128-epoch M5 run. Future `air_pre_bc` config uses 20 epochs and keeps best val/loss; the Demo never starts training. Historical GPU acceptance scripts are not Demo entry points. The old M6 training wait/verification scripts are disabled.

Timing: 3 Hz, 6 s history, 2 s tokens, 30 s future. The committed dictionary is K=2048/shift=6/causal chord heading. Rollout starts at tokenized current pose and never reads future GT. ADE/FDE average the 32 rollouts; minADE picks the best full trajectory per agent independently for horizontal/vertical errors. Miss rate is the fraction of eligible agents whose 32 endpoints all exceed 500 m horizontally; the mean per-rollout miss fraction is also shown. Eligibility requires every GT frame through the horizon. Counts refer to agent-window instances.

```bash
scripts/test.sh -q
```

Geometry uses float64; neural features use float32. Unit conversions are centralized in `src/common/units.py`. Heights are altitude minus 166 m; altitude-fallback flags are preserved. Flights may continue across midnight at split boundaries. Raw data stays outside this repo; all generated data and outputs are ignored. No Git remote is configured.

Copied SMART/CAT-K code retains original headers; see LICENSE and LICENSE.SMART. Dataset provenance: ADSB.lol historical traces (ODbL 1.0), as documented in the source dataset README. Historical dictionary reconstruction results (day 10-03) are in `docs/motion_dict.md`; they are not forecast errors.

Presentation exports from existing anomaly scores:

```bash
scripts/catk.sh -m scripts.export_anomaly_demo
```

Writes `outputs/anomaly_demo/scenes/` (11 individual1200×900 PNGs, last frame
of each replay), `gifs/` (matching animations and contact sheet), and
`manifest.json`; `scene_images.zip` packages all individual PNGs. Uses the restored five-height smoothed baseline. Demo labels
say “Anomaly score”, with scene-local aircraft/drone numbers and no token
intervals, internal scenario IDs or formula. Source IDs, exact cutoff, timing
and input hashes stay in the manifest; scores and alert logic are unchanged.
This export reads saved score tables and recorded GT only, without fitting
histograms, model execution or training. Scientific output directories remain
separate and unchanged.

Quarterly report figures from the six existing seeded Demo cases:

```bash
scripts/catk.sh -m scripts.export_rollout_report
```

`outputs/quarterly_rollouts/` contains six individual horizontal/vertical
figures, a six-case overview (300dpi PNG + vector PDF), `CAPTION.md` and a ZIP
package. All32 saved rollout samples are drawn alongside6s history and30s
recorded future. No model execution, training or error-ranked selection.

For presentation-style completed simulations matching the dark-map v2 GIFs:

```bash
scripts/catk.sh -m scripts.export_rollout_snapshots
```

`outputs/rollout_demo_stills/` contains10 individual2400×1800 PNGs at+30s and
`rollout_demo_stills.zip`. Solid/filled aircraft show recorded movement;
dashed/outline aircraft show simulated movement. The same joint rollout as
the original GIF is used (illustrative best scene-average3D error of32,
selected with recorded future); provenance is disclosed in CAPTION.md and
manifest.json. No model execution or training.

Camera-conditioned trajectory Demo (frozen checkpoint, no training):

```bash
scripts/catk.sh -m scripts.run_camera_demo --gifs
# Re-render saved predictions without running a model:
scripts/catk.sh -m scripts.run_camera_demo --render-only --gifs
```

`outputs/camera_demo/` contains10three-way cases:8aircraft and2drone. Cyan solid
is camera tracker, yellow dashed is sample0 of one model rollout, green dotted
is GT. Each takes6s camera history and predicts30s. Camera/GT lines stop at
unsupported observations and endpoint labels show their last available time.
`scenes/`: individual2400×1800 PNGs; `gifs/`: matching animations; `rollouts/`:
saved arrays; `camera_demo_images.zip`: PNG bundle. Track audit includes all43
native tracker IDs, including excluded/unmapped cases.

Inputs use the last camera posterior plus its own velocity×age≤1s, never a
future camera interpolation. Types are UNK and ground status is a fixed
camera-height proxy (enter<10m, exit>20m), not GT. Camera includes the existing
tracker's Kalman predictions and uses its RGB+simulator-depth reconstruction.
Each tracker ID is inferred alone; offline segmentation identities only pair
GT overlays, with no GT type/heading/ground features or ID stitching. Origins
are first eligible camera histories, not selected by prediction error; sample0
of1 uses seed817/topk5/temp1. This qualitative transfer demo does not establish
camera anomaly-detection performance. See docs/camera_demo_acceptance.json.

Integrated MVP v1 (approved D0–D5, 2026-10-29):

- Plan: `INTEGRATED_DEMO_PLAN.md`; immutable parameters: `configs/integrated_demo/v1.json` and its SHA256 lock.
- Artifacts: `outputs/integrated_demo/D5/index.html`, two offline interactive
  viewers, MP4/GIF and four3200×1800 PNGs per scene. Download/extract
  `outputs/integrated_demo/integrated_demo_v1.zip` and open index.html locally.
- Frozen model:6s simulated noisy-tracker history →32joint30s futures, updated
  every2s. Main display is the GT-free joint3D medoid. Numeric-only CV is saved
  alongside forecasts but never displayed or substituted.
- Threat requires an observed current anomaly in the preceding10s. Other
  objects show Normal traffic with no threat score; no anomaly+5 modifier.
- Day/night,noise and zone are explicitly scripted. Anomaly density remains
  frozen location/height,speed,direction only, without weather/time-of-day bins.
- First fixed run: daytime drone anomalous/Low; nighttime approach model-based
  High at56s before reference entry80.67s (24.67s lead). Future anomaly is not
  earlier than current anomaly here; CV does not establish a model advantage.

D4 first-pass evaluation is deliberately protected against an automatic repeat
or outcome-driven tuning. To re-render accepted arrays only:

```bash
scripts/catk.sh -m scripts.integrated_d5
```

This rendering path is offline, uses only locally cached maps and does not run
the model. Complete observations, per-sample forecasts, scores, CV arrays,
rules and timeline remain in the two scene directories under
`outputs/integrated_demo/`; acceptance receipts are in `docs/integrated_demo_D*.json`.
