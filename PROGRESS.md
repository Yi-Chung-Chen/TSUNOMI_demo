# Progress

Implementation follows M0–M7 in order. No milestone is complete until its acceptance checks pass.

## M0 — Bootstrap (2026-10-16)

Implemented repository layout, copied run/utils and Hydra skeleton with original headers, preserved LICENSE/SMART license, configured SYGARD paths, offline WandB and physical GPU visibility 1,3. No remote.

Validation:
```text
scripts/catk.sh -m src.run --help: exit 0 (Hydra configuration displayed)
scripts/test.sh -q: 1 passed in 0.04s (catk Python 3.11)
```

Environment: approved pip install pytest failed because DNS is unavailable and site-packages is read-only. Tests use pure-Python pytest 8.3.4 / pluggy 1.5.0 / iniconfig / py shim copied from base into ignored `.cache/test-tools`; catk runtime and its other packages are unchanged. Reproduce with base `python scripts/prepare_test_tools.py` then `scripts/test.sh -q`. Advisor question pending; this passes the runtime test check without an install. GPU execution remains unverified; `nvidia-smi` exits 9 inside sandbox.

Source tracking checked before and after commit via scripts/check_tracked.py.

## M1 — Ingest (2026-10-16)

Implemented strict dense JSONL ingest, contiguous-pair velocity statistics, fallback/type retention, common unit conversion, and row-by-row source verification. Missing frames remain invalid without interpolation. NPZ files are atomic and allow_pickle=False compatible. Bounds retained as an additional JSON scalar for evaluation.

Acceptance:
```text
scripts/catk.sh -m src.preprocess.ingest --workers 4: exit 0
NPZ scenario count: 800
Sum aircraft tracks: 4967
Sum valid: 1231609 == sum index.reference_samples
scripts/test.sh -q: 3 passed in 8.03s
```

Full-data extrema exceed preliminary plan statistics; see docs/data_stats.md. No raw data changed. Advisor call on filtering/tolerance rejected by automatic review; logged in QUESTIONS.md.

Environment: standard pytest 8.4.2 installed into catk under §4.1 approval. GPU escalation works after user selected Approve for me; GPUs 1/3 occupied at 92%/98%, no training launched. Initial advisor call returned ERROR (DNS), also logged.

Tracking after commit: 18/18 source Python files.

## M2 — UTC split and window index (2026-10-16)

Implemented configurable date splits, train-only top-31 ICAO vocabulary plus UNK, and CSV window indexing without materialized chunks. D10 checks every frame in history and initial 10 s future.

Acceptance:
```text
scenarios: train=618 val=91 test=91; no overlap
vocabulary_size: 32 (train only)
train: 24570 windows, 49598 target instances
val: 488 windows, 920 target instances
test: 424 windows, 620 target instances
scripts/test.sh -q: 6 passed in 8.52s
```

Tests independently inspect every generated window for bounds and target counts, train-only vocabulary, split overlap rejection and missing-frame eligibility. Tracking after commit: 20/20 source Python files.

Advisor call now succeeded after user explicitly authorized sending all implementation questions. M3 provisional decisions: tol=0.5 internal units, retry ×0.8 until exact K; token zero exactly stationary; configurable horizontal segment cap 200 m/s; shared seeded pool max 500k segments per shift; retain raw ingest. Full Q/A in ../SYGARD_QA_LOG.md; these are provisional pending Yi-Chung review.

## M3 — Motion dictionary (2026-10-16)

Built full 18-case grid: shifts 3/6/9 × data/chord × K 512/1024/2048. Train-only shared seeded pools: 341493 / 168217 / 110506 segments. All dictionaries verified for exact K, finite arrays, shift+1 subframes and exact zero stationary token. No pool reached the 500k cap.

Selected fixed shift=6, K=2048, causal chord heading; asset copied into assets/ and paths config updated. Complete results and arrow-fan PNG in docs/motion_dict.md and docs/motion_dict_fan.png. 30 s horizontal FDE=2.5599 m (<10 m target met). Shift=3 gives lower quantization error (~1.6 m at K2048) but timing remains 6 as required; no switch made.

Acceptance:
```text
scripts/test.sh -q: 10 passed in 8.55s
18 dictionaries verified: exact K, full subframes, finite values, exact stationary token
Full val sweep: 18 configurations × 2 modes × 2 horizons × 3 groups
{'mode': 'autoregressive', 'horizon_s': 10, 'group': 'all', 'n_agents': 920, 'horizontal_ade_m': 2.3250322341918945, 'horizontal_fde_m': 2.242997407913208, 'vertical_ade_m': 1.222062349319458, 'vertical_fde_m': 0.8880302309989929}
{'mode': 'autoregressive', 'horizon_s': 10, 'group': 'ground', 'n_agents': 318, 'horizontal_ade_m': 1.8693069219589233, 'horizontal_fde_m': 2.3138809204101562, 'vertical_ade_m': 0.3876897096633911, 'vertical_fde_m': 0.3480924069881439}
{'mode': 'autoregressive', 'horizon_s': 10, 'group': 'airborne', 'n_agents': 602, 'horizontal_ade_m': 2.5657641887664795, 'horizontal_fde_m': 2.2055535316467285, 'vertical_ade_m': 1.6628108024597168, 'vertical_fde_m': 1.1732466220855713}
{'mode': 'autoregressive', 'horizon_s': 30, 'group': 'all', 'n_agents': 668, 'horizontal_ade_m': 2.6353418827056885, 'horizontal_fde_m': 2.5598530769348145, 'vertical_ade_m': 1.3438886404037476, 'vertical_fde_m': 1.3111904859542847}
{'mode': 'autoregressive', 'horizon_s': 30, 'group': 'ground', 'n_agents': 262, 'horizontal_ade_m': 2.4646224975585938, 'horizontal_fde_m': 2.793461799621582, 'vertical_ade_m': 0.6982719302177429, 'vertical_fde_m': 0.6871119737625122}
{'mode': 'autoregressive', 'horizon_s': 30, 'group': 'airborne', 'n_agents': 406, 'horizontal_ade_m': 2.7455108165740967, 'horizontal_fde_m': 2.4091010093688965, 'vertical_ade_m': 1.7605184316635132, 'vertical_fde_m': 1.7139209508895874}
```

Counts: 920 at 10 s, 668 at 30 s, with every frame valid through horizon. Evaluated GT-conditioned reconstruction rather than forecast performance. Partial horizon token matched at exact subframe (needed by shift=9 at 10s). Full statistics include vertical artifacts; filtering applied only to train dictionary segments. Provisional advisor decisions recorded in M2 and docs; implementation by Codex. Builder metadata git_hash is parent M2 commit; builder sources are committed in this milestone. Tracking after commit: 23/23 source Python files.

Explicit `motion_dict_eval --dict-path assets/motion_dict_K2048_s6.pkl --autoregressive` acceptance CLI completed with exit 0; result saved to data/motion_dict/acceptance/single.json.

## M4 — Dataset, DataModule and 3D tokens (2026-10-16)

Implemented lazy NPZ window dataset with per-worker cache, train-only type IDs, current-time aircraft selection, PyG variable-agent collation, target builder, and SMART-style autoregressive GT/sampled token matching with joint xy/dz distance. BOS plus 18 endpoints yields 19 nodes; every frame within a token is checked. Future-only agents are excluded.

Acceptance:
```text
scripts/test.sh -q: 14 passed, 1 upstream deprecation warning in 27.21s
Additional sampled-path test: 1 passed in 3.08s
Full CPU-worker train epoch: 6143 batches, 24570 windows, 60368 agents, 49598 targets, workers=4
Torch/NumPy AR and one-shot reconstruction agree within 1e-3 m for all ground/airborne/all 10s/30s ADE/FDE and exact counts.
Independent detokenisation from IDs agrees with processor output within 1e-5 internal units.
Mid-token gaps masked; modifying future positions/headings leaves all history tokens unchanged.
```

Provisional advisor correction to M4 gate: AR roundtrip compares with M3 AR, one-shot helper compares separately with M3 one-shot. One-shot never supplies training targets. Reconstruction acceptance uses identical raw-current initialization (start_frame=18) as M3; default training tokenizer chains from frame0 through history and future. No raw-current reset is inserted.

Precision fix: nearest-token geometry now uses float64 to avoid float32 near-tie drift between NumPy and Torch; trainable model features remain float32. Full 18-case eval rerun; selected dictionary unchanged, corrected 30s horizontal FDE=2.5598717 m. Existing M3 report regenerated. No dictionary rebuild or timing change.

CPU worker IPC is blocked inside sandbox; initial hung acceptance process and only its workers were stopped, and the final full epoch ran outside sandbox. Tracking after commit: 28/28 source Python files.

M4 advisor follow-up resolved a contradictory answer: exact default-chain detokenisation replaces the incorrect <= one-shot bound (provisional). Additional default-chain test passed (1 passed in 2.78s). Default-chain raw-GT reconstruction is reported without threshold in docs/motion_dict.md. Current 3D drift mean=2.5562 m, p95=4.1357 m across 920 val target instances. Rollout starts at tokenized current pose; metric GT remains raw.

## M5 — Implementation ready; GPU acceptance pending (2026-10-16)

Ported SMART temporal and fully connected agent attention (4 layers, 4 heads, hidden64); copied shared layers with attribution; no map branch. Added contour+dz token embeddings, current z/ground/type features, delta-z relative encoding, causal masking, and masked CE that stops at the first future gap. Lightning module includes LR decay and teacher-forced validation accuracy/open-loop minADE (six token-rank modes, 3D metres).

Completed checks (these do NOT satisfy GPU gates):
```text
Model causality, batch isolation, loss-gap mask and delta-z tests: 3 passed
Single-batch overfit, 89 tokens: step117 unsmoothed CE=0.0288959, accuracy=1.0 (CPU)
Hydra/Lightning CPU fast_dev_run=2: fit+validate completed, exit0
```

GPU1 and GPU3 still have existing compute PIDs 2136030 and 2136032. No training is launched while they are occupied. scripts/run_m5_when_idle.sh waits for one authorized GPU to have no compute PID, utilization<10%, memory<1GiB on two consecutive 30s checks; then CUDA smoke (1epoch,16 train/4val batches) and full128epoch BC with batch32/all windows. It rechecks after smoke, verifies actual CUDA log evidence, both checkpoints, full-run completion and accuracy/minADE/horizon counts, then creates the single M5 milestone commit. M5 remains incomplete until this passes. M6/M7 have not started.

Full-run outputs: outputs/air_pre_bc; execution/wait logs: outputs/m5_execution. Source tracking will be 36/36 after staging and after M5 commit.

M5 combined test suite: 21 passed (19 upstream deprecation warnings) in29.30s. Advisor review found no model/loss/metric blocker; GPU mapping was hardened with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA-runtime PCI-to-NVIDIA-UUID verification before both runs. Torch2.4 has no properties.uuid, so the existing CUDA12 cudaDeviceGetPCIBusId API is used. verify_m5 already writes actual GPU evidence into PROGRESS.md before staging/commit.

M5 queued in local tmux session `sygard_motion_m5` (2026-10-16). Only GPU waiting is currently active; existing compute PIDs remain untouched. The runner writes errors to QUESTIONS.md/PROGRESS.md and creates no M5 commit on failure. All implementation Python files are staged in Git, with 36/36 source files tracked; there is still no remote.

M5 runner stopped with ERROR (exit 1); see QUESTIONS.md and outputs/m5_execution/.

## M5 — GPU acceptance passed

Actual CUDA smoke and complete 128-epoch / 98304-step BC verified. The retained checkpoint is selected by best validation loss; it is distinct from the final epoch metrics.

```json
{
  "smoke_physical_gpu": 1,
  "bc_physical_gpu": 1,
  "cuda_visible_devices": "1,3",
  "overfit": {
    "step": 117,
    "ce": 0.02889588661491871,
    "accuracy": 1.0,
    "tokens": 89
  },
  "smoke_global_step": 16,
  "completed_epochs": 128,
  "completed_global_steps": 98304,
  "retained_checkpoint": {
    "path": "outputs/air_pre_bc/checkpoints/epoch_016.ckpt",
    "epoch": 16,
    "global_step": 13056,
    "metrics": {
      "val/loss": 4.346920490264893,
      "val/next_token_accuracy": 0.18332262337207794,
      "val/open_loop_minADE_30s_m": 6.056827897207742,
      "val/open_loop_agents_30s": 668.0
    }
  },
  "final_epoch": {
    "epoch": 127,
    "weights_saved": false,
    "metrics": {
      "val/loss": 4.863179683685303,
      "val/next_token_accuracy": 0.16090938448905945,
      "val/open_loop_minADE_30s_m": 6.430518894284198,
      "val/open_loop_agents_30s": 668.0
    }
  }
}
```

The original save_last=link left last.ckpt pointing at the best epoch16 checkpoint. Final epoch127 weights were not saved. Future runs use save_last=true. Best val loss at epoch16, followed by worse final validation loss, suggests overfitting. Provisional advisor decision: use best epoch16 for M6; verify full training using complete CSV epoch history and completion log. No rerun required by the plan.

M5 acceptance correction regression tests: 2 passed in1.32s; both complete epoch history and missing-epoch rejection verified. Previous combined suite21 passed. GPU smoke/full training evidence and retained/final metrics were checked directly; acceptance.json generated successfully. Source files36/36 tracked; git diff --cached --check passed before milestone commit.

## M6 — Implementation ready; GPU acceptance pending

Implemented joint scene rollout replication, GT-free inference, CAT-K top32 probability candidates selected by xy-contour+dz distance at temp1e-5 during training, and CE targets recomputed from actual rolled poses. Temporal/spatial attention and next-token head are trainable; embeddings are frozen as upstream. Inference holds current ground phase and keeps only current aircraft, with separate scene IDs per rollout. Per-window atomic NPZ exports preserve 3Hz 3D positions in metres, raw GT, IDs, validity, targets, fallback flags and bounds.

Acceptance completed so far:
```text
Full test suite: 27 passed, 19 upstream deprecation warnings in28.31s
M6 tests: future GT changes leave inference unchanged; 2-rollout shape and metre exports; joint-world separation; dz-aware CAT-K; invalid-GT fallback; only intended modules receive gradients; first recomputed target matches default GT chain.
CPU Hydra integration: experiment=air_clsft, BC epoch16 loaded, fast_dev_run=1; training and validation completed; 4 validation files each with 32 rollouts x90frames x3 coordinates.
```

These are not GPU acceptance. Initial CPU integration failed on upstream torch.load without map_location; fixed src/run.py to load checkpoint weights on CPU before Trainer moves them to the selected GPU. This also prevents accidental allocation on logical GPU0 when authorized logical GPU1 is selected.

Read-only advisor review found no blocking correctness issue; suggested gradient/target chaining checks were added and passed. Full training plan:8epochs, lr5e-5, batch8, full val with32 joint rollouts. scripts/run_m6_when_idle.sh waits for an unoccupied authorized GPU, verifies PCI/UUID mapping, runs CUDA fine-tuning, checks checkpoint/CSV/488 rollout files and668 full-horizon targets, then creates the M6 milestone commit. M7 remains unstarted until this gate passes.

GPU1/3 currently occupied (100%, about45GB each). M6 queued in tmux sygard_motion_m6. Logs: outputs/m6_execution/runner.log and finetune.log. Implementation files staged,39/39 src Python files tracked; no remote.

## M6 scope replacement — BC-only Demo (2026-10-16)

User superseded the former CAT-K stage and old M7 with the updated plan D1, §4.1 items6/7 and M6 Demo. Stopped the waiting tmux sygard_motion_m6 before it launched training, disabled its runner, and removed the uncommitted CAT-K training path. No fine-tuning or BC retraining was launched. The intentional stop may appear as an old runner ERROR; it is a scope cancellation, not a Demo failure.

Merged existing val488 + test424 windows byte-for-byte into active val912,182 scenarios, without invoking preprocessing/rebuilding. Original files retained in data/splits/legacy_date_splits/. Train manifest/windows, vocabulary, dictionary and epoch16 checkpoint SHA256 identities recorded in data/splits/merge_receipt.json. Future BC config changed128→20epochs only; nothing trained.

One-command Demo: scripts/catk.sh -m scripts.run_demo (CPU).32 joint rollouts, topk_prob5/temp1, raw-current-history immutable, tokenized current initialization, full3Hz subtrajectories. Existing BC weights loaded strictly from epoch_016.ckpt. Output schema position[32,N,90,3], rawGT[N,109,3], date, validity/target masks and IDs. CV/GT baselines same format; GT replay must be exactly zero.

Provisional advisor choices: ADE/FDE average rollouts; per-agent whole-trajectory minADE, independent horizontal/vertical minima; all32 horizontal endpoints>500m defines30s miss, with mean-rollout miss also shown; plot RNG0 selects phase-stratified6 unique windows using target dz10s±20m or current ground flag, never prediction errors. Selection method matches the updated scope.

Tests:31 passed (19 upstream deprecation warnings) in42.36s. Six-window one-command smoke passed in10.51s, including saved32-rollout files, baselines, metrics, plots, protected hashes and exact GT replay0. Full912-window CPU Demo is running; final acceptance/results will be appended below.

Advisor Demo review found no blocking issue. Added disclosure:10-03 selected both epoch16 and dictionary K2048/chord heading;10-04 did not participate in either choice. Dictionary centres remain train-only. No artifacts rebuilt.

## M6 — Demo acceptance passed (CPU, no training)

One command `scripts/catk.sh -m scripts.run_demo` completed all912 merged val windows:488 from10-03 and424 from10-04.32 BC rollouts per window with K5/temp1, plus32 identical CV/GT baseline rollouts each. Full CPU runtime511.92s (8.53min). Outputs:3×912 NPZ files,162 summary rows (3 models×3 populations/dates×3ground strata×3horizons×2axes),6 seeded phase-stratified PNGs. All GT replay ADE/FDE/minADE/miss values are exactly0.

Acceptance:
```json
{
  "windows": 912,
  "dates": {
    "2026-10-03": 488,
    "2026-10-04": 424
  },
  "rollouts": 32,
  "num_k": 5,
  "temp": 1.0,
  "device": "cpu",
  "checkpoint_epoch": 16,
  "plot_count": 6,
  "elapsed_s": 511.9214086429565,
  "gt_replay_exact_zero": true,
  "no_training": true,
  "full_demo": true
}
```

All five protected artifacts (train manifest/windows, vocabulary, dictionary and BC epoch16 checkpoint) retain identical SHA256 before/after merge and full Demo. Original val/test indexes archived; merge is byte concatenation only. Active split has182val scenarios/912windows, no test. BC future config20epochs confirmed; no training launched.

Tests:full suite31 passed in42.36s; additional whole-trajectory minADE regression1 passed in2.96s. Source tracking44/44 after commit; every non-cache Python file tracked.

Combined all-agent comparison (horizontal, metres):
```json
[
  {
    "model": "bc",
    "horizon_s": "10",
    "n_agents": "1540",
    "minADE_m": "9.887872452670464",
    "ADE_m": "19.57813871131043",
    "FDE_m": "43.70318055774069",
    "miss_rate": ""
  },
  {
    "model": "bc",
    "horizon_s": "20",
    "n_agents": "1321",
    "minADE_m": "25.4497055549549",
    "ADE_m": "53.061076234648674",
    "FDE_m": "137.31564258187365",
    "miss_rate": ""
  },
  {
    "model": "bc",
    "horizon_s": "30",
    "n_agents": "1090",
    "minADE_m": "50.25036106815061",
    "ADE_m": "104.97686252724833",
    "FDE_m": "294.410639775111",
    "miss_rate": "0.060550458715596334"
  },
  {
    "model": "constant_velocity",
    "horizon_s": "10",
    "n_agents": "1540",
    "minADE_m": "28.257469357016102",
    "ADE_m": "28.257469357016102",
    "FDE_m": "68.09865784536085",
    "miss_rate": ""
  },
  {
    "model": "constant_velocity",
    "horizon_s": "20",
    "n_agents": "1321",
    "minADE_m": "83.19731739916995",
    "ADE_m": "83.19731739916995",
    "FDE_m": "220.45646867436722",
    "miss_rate": ""
  },
  {
    "model": "constant_velocity",
    "horizon_s": "30",
    "n_agents": "1090",
    "minADE_m": "161.4828010083975",
    "ADE_m": "161.48280100839753",
    "FDE_m": "450.0290125946904",
    "miss_rate": "0.3275229357798165"
  }
]
```

BC beats CV on combined/all horizontal and vertical minADE/ADE/FDE at each10/20/30s horizon. This is not a statement about every ground/date subgroup; full per-subgroup yes/no comparisons are in outputs/eval/summary.md and docs/demo_results.md. Counts at10/20/30s:1540/1321/1090 agent-window instances. Report explicitly discloses10-03 dictionary-configuration and checkpoint selection;10-04 unused before Demo.

Done-when outputs verified:outputs/eval/summary.csv, summary.md, outputs/demo/6PNG files; selection.json lists seed0 and phase-selected targets, never ranked by error. PNGs visually inspected for history/GT/32predictions and altitude panel. The old M7 is merged into this M6; optional M8 was not started.

## M7 — Demo GIFs draft v1 (2026-10-16)

Implemented plan §4.1 item8 / new M7 in `src/eval/demo_gif.py`.
Command: `scripts/catk.sh -m src.eval.demo_gif`. Reads existing BC NPZ only;
no model execution, training, GPU use or package installation. OSM standard tiles
are fetched with a descriptive User-Agent and reused under ignored
`data/basemap_cache/`. Network sandbox escalation was used for those downloads.

Selection: the six existing `outputs/demo/selection.json` windows, then four
seed-0 random windows among remaining windows with at least three aircraft having
valid GT. No window error ranking. One coherent joint rollout per scene minimizes
equal-aircraft mean 3D ADE over each target's full valid future horizon. This is
visualization-only selection and leaves the M6 evaluation unchanged.

Rendering: fixed top-down equal-aspect view; all GT and chosen-rollout xy bounds
with approximately 10% padding; local tangent plane about (41.980103,-87.903868)
to Web Mercator; desaturated OSM blended with 60% white. Solid GT/history, faint
full-GT ghost, lighter dashed predictions, heading-rotated aircraft, subtle IDs,
minimal key and OSM attribution. Prediction trails and markers stop at the first
invalid GT after current time; GT gaps break lines. Stable ID-derived qualitative
colours are used before GIF quantization. Full animation is -6 through +30 s,
109 frames at native 3 Hz, playback 110 ms/frame and final 1500 ms hold, looping.
One shared 128-colour palette and transparent GIF deltas meet the size limit
without frame skipping.

Acceptance output (`docs/demo_gif_acceptance.json` contains per-window details):

```json
{"gifs":10,"frames_each":109,"size":[1200,900],"max_bytes":745132,
 "total_bytes":6338442,"playback_frame_ms":110,"last_hold_ms":1500,
 "rollout_argmin_verified":10,"input_hashes_unchanged":true,
 "protected_artifacts_unchanged":true}
```

`scripts/test.sh -q tests/test_demo_gif.py`: **3 passed**, 18 existing matplotlib /
pyparsing deprecation warnings. Tests independently check origin and ENU/pixel
round-trip, bounds inclusion, joint ADE argmin with non-target/invalid outliers,
and actual synthetic 109-frame GIF encoding including loop and final hold.
Acceptance additionally decoded every frame of all ten real GIFs, checked timing,
resolution, size, deterministic selection, argmin, input SHA256, and protected
train indexes/vocabulary/dictionary/checkpoint hashes against the M6 receipt.

One visual self-review completed: inspected `outputs/demo_gif/contact_sheet.png`
and decoded frames 0/18/36/60/84/108 of both
`scenario_000667_aircraft_f0720.gif` and `scenario_000497_aircraft_f0540.gif`.
Paths grow and the view stays fixed. No major label clutter in these examples;
the map is readable but competes with the thin trajectories at reduced size.
**Known visual limitation:** quantization noticeably washes out aircraft colours
in the GIFs compared with the RGB contact sheet, making some aircraft colours
and overlapping GT/prediction lines harder to distinguish. Small IDs and the
tiny dashed key are also hard to read in thumbnails. These are recorded for
Yi-Chung's visual feedback; no taste iteration was performed after self-review.

Ten GIFs, contact sheet and manifest remain under ignored `outputs/demo_gif/`.
Code and acceptance record committed locally; no remote added. Draft v1 stops
here pending Yi-Chung's review. Optional M8 remains unstarted.

## M7 — Demo GIFs draft v2, colour revision (2026-10-16)

Yi-Chung reviewed v1: 「大致上不差，但黑白的好醜，你跟你的顧問請教一下怎麼美化」.
Asked the upgraded advisor script once with the milestone, implementation,
observations and consolidated palette/style questions. The read-only advisor
returned `ANSWER:` (not a provisional decision or DEFER); the complete exchange
is in parent `SYGARD_QA_LOG.md`. Advisor inspected the RGB contact sheet, not the
GIFs, and recommended a dark flight-radar theme, exact reserved track colours,
larger marks, and contact sheets from decoded GIF frames. Codex implemented it.

Confirmed the original failure quantitatively: two sampled v1 final frames
contained zero pixels with RGB channel spread >80. Adaptive quantization had
allocated nearly every palette entry to the map and lost the saturated lines.

Implemented dark navy `#0b1622` map/frame with desaturated inverted OSM mapped
to navy/blue-grey. Saturated coral/yellow/cyan/green/purple aircraft colours are
assigned by sorted ID within each scene (no hash collisions for these windows).
Solid GT/history uses 3.5 pt strokes with a navy halo; lighter dashed predictions
use 2.6 pt strokes; filled/hollow aircraft markers are 26/22 pt and boxed labels
11 pt. Full-path ghosts use an explicit muted colour, avoiding alpha blending.
`--theme light` retains a light OSM alternative. No additional visual variant
was rendered in this revision.

Encoding reserves 124 entries for background/text, exact trajectory colours,
prediction/ghost variants and antialias blends; 131 entries serve the map-only
quantization and one entry is reserved for transparency. All frames share this
palette. Explicit transparent deltas plus `optimize=False, disposal=1` compress
the static map while preserving the reserved indices. The contact sheet is now
assembled from the decoded GIF final frames, not pre-encoding RGB frames.

Command (existing cached tiles, CPU rendering only):

```bash
scripts/catk.sh -m src.eval.demo_gif --output-dir outputs/demo_gif_v2
scripts/test.sh -q tests/test_demo_gif.py
```

Original v1 files remain under `outputs/demo_gif/`; the new ten GIFs, contact
sheet and manifest are under ignored `outputs/demo_gif_v2/`. No model execution,
training, packages, remote, input data or checkpoint changes. Selection, joint
rollout indices and NPZ input hashes match v1 exactly.

Acceptance (`docs/demo_gif_v2_acceptance.json` includes every window):

```json
{"draft":"v2","theme":"dark","gifs":10,"frames_each":109,
 "size":[1200,900],"max_bytes":982727,"total_bytes":7948730,
 "playback_frame_ms":110,"last_hold_ms":1500,
 "inputs_and_rollout_indices_match_v1":true,"input_hashes_unchanged":true,
 "protected_artifacts_unchanged":true,"exact_gt_colours_survive_all_ten":true}
```

**4 tests passed**, 18 existing dependency deprecation warnings. Added a
regression with tiny saturated marks on a neutral gradient background: decoded
GIF GT and prediction centre pixels must equal the reserved RGB exactly; the
complete composited frame must equal the original quantized frame, including
erasure of old icons. Existing synthetic animation also compares its decoded
final frame against the renderer. Real-output acceptance decoded all109 frames
of each GIF, checked dimensions/timing/loop/size, exact GT colour presence in
all10 final frames, original input and rollout identity, and protected hashes.

One visual self-review: viewed the decoded-GIF contact sheet and frames
0/18/36/60/84/108 of scenarios000667_f0720 and000497_f0540. Saturated aircraft
colours survive encoding and stand out clearly from the dark map. No obvious
animation remnants or major ID-label clutter in these examples. GT/prediction
remain visually close where paths almost coincide; attribution/key and map
labels are intentionally subdued and small at contact-sheet scale. The map
could feel too dark depending on display; leave that judgement to Yi-Chung.
No taste iterations after this self-review. Commit locally and stop for feedback.

## M8 — Anomaly scoring MVP (2026-10-18)

Re-read plan §4.1 item9 and the new M8. Before implementation, invoked
the advisor script (`--refork`) once to synchronize the read-only advisor with the
updated main planning session. Full Q/A is in parent `SYGARD_QA_LOG.md`.
Advisor returned **DECISION (provisional, Yi-Chung to confirm)**: scan all
10-04/10-03 scenario tracks at stride6 instead of reusing M6 windows; batch
isolated agents with distinct scene IDs; report the drone's track-mean NLL rank.
These three clarifications are followed provisionally pending Yi-Chung's review.
Other concrete advice included invalid tail padding, full-history eligibility,
heading-convention validation, raw-anchor residuals and source-domain caveats.
No DEFER or ERROR was returned.

Implemented `src/eval/anomaly_score.py` with a CLI. Uses the frozen
`epoch_016.ckpt` and existing K2048/shift6 dictionary. State dict loads strictly;
checkpoint tokenizer buffers are checked against the requested dictionary file.
No model/source-model changes, training, installs or state-plausibility scores.

Data and masks:

- Drone: read `simulation_poses_native.jsonl` directly. Its902 reads have901
  unique timestamps: reads195/196 repeat the exact same pose/heading at65.49s.
  Collapse identical duplicates in memory; reject conflicting duplicates.
  Piecewise-linear xyz interpolation and unwrapped heading interpolation to
  frames0..900 at3Hz. Do not extrapolate: t=0 precedes the first read at0.075s
  and is invalid; frames1..900 are valid. Native-knot interpolation error is
  exactly0m. Height = ENU z +2250−166m; `on_ground=False`, type=UNK0,
  same radius and position scale as aircraft.
  Native heading vs velocity median absolute difference0.00041578rad
  (0.02382°), consistent with CCW from east; no heading replacement needed.
- Aircraft: all91 scenarios per10-03 and10-04 from the existing val manifest,
  read from `data/processed/`. No source dataset or existing indexes rebuilt.
- Slide109-frame windows at starts0,6,12,… over each scenario/track. Pad the
  unneeded tail past scenario end invalid. Require frames0..24 all valid:
  6s history/current plus the first2s future. Score every eligible agent once
  per start; no additional10s future requirement. Drone produces146 tokens
  at scored-token starts8,10,…,298s. No duplicate score keys.

Scoring:

- One agent per graph, distinct graph batch IDs: no spatial context for either
  class. Unsmoothed `cross_entropy(logits[:,3], gt_idx[:,4])` is the one-step
  NLL in nats, using the unchanged default autoregressive GT tokenizer chain.
  No history nodes or later future tokens are scored. Model is in eval mode.
- Residual is a separate `reconstruct(..., start_frame=18, horizon_frames=6,
  one_shot=True)` call with the existing raw-anchor/chord-heading matcher.
  Report mean pointwise horizontal distance and mean absolute vertical error
  over frames19..24, in metres. Accumulated chain error is excluded.
- Split aircraft at `on_ground` at the scored-token start. NumPy AUROC uses
  average ranks for ties; drone is positive and larger NLL/residual is anomalous.

Execution: `nvidia-smi` outside the sandbox confirmed GPUs1/3 both idle (0MiB,
0%); the wrapper fixed `CUDA_VISIBLE_DEVICES=1,3`. Ran
`scripts/catk.sh -m src.eval.anomaly_score --device cuda:0`, using physical GPU1
only. Full inference: **20.67s**, **28,413 scored tokens** from182 aircraft
scenarios plus the drone. No GPU0/2 use.

Results (primary aircraft comparison is10-04;10-03 reference only):

| group | tokens | NLL mean / median / p95 | horizontal residual mean / p95 (m) | vertical residual mean / p95 (m) | NLL AUROC | horizontal / vertical residual AUROC |
| --- | ---: | --- | --- | --- | ---: | --- |
| Drone | 146 | 3.392 / 2.658 / 7.858 | 1.702 / 2.960 | 0.292 / 1.178 | — | — |
|10-04 airborne|9,200|4.068 / 3.898 / 7.083|2.214 / 4.691|1.432 / 3.509|0.336|0.400 / 0.111|
|10-04 ground|2,638|2.078 / 1.458 / 5.773|1.198 / 2.498|0.110 / 0.691|0.760|0.705 / 0.870|
|10-03 airborne (reference)|11,097|4.453 / 4.313 / 7.543|2.178 / 4.781|1.447 / 3.700|0.283|0.417 / 0.130|
|10-03 ground (reference)|5,332|2.123 / 1.527 / 5.766|1.276 / 2.692|0.175 / 0.799|0.747|0.685 / 0.836|

Plain conclusion: **The drone is typically more likely under the transition
model than airborne aircraft, not less likely. High one-step NLL does not
reliably flag this route. It reconstructs at least as well as airborne aircraft
on average (both horizontal and vertical residuals are lower).** Some drone
time segments still show high NLL, as the timeline makes clear. Drone track-mean
NLL is at the16.7th percentile among 354 10-04 airborne aircraft track-segment
means. This descriptive rank is not a new anomaly score. All positives come
from one simulated route and are correlated; native smooth near3Hz drone reads
vs interpolated near1Hz ADS-B aircraft can shift both diagnostics. No claim of
general drone detection or detection threshold is made.

Required outputs exist under ignored `outputs/anomaly/`:
`summary.md`, `summary.csv`, `nll_hist.png`, `drone_timeline.png`; also
`token_scores.csv` and `acceptance.json` for reproducibility. Both plots were
visually inspected: distinct overlaid distributions, labelled counts/units, and
drone timeline with10-04 airborne median/p95 reference lines. Committed copies
of tables/conclusion/receipt are `docs/anomaly_results.md`,
`docs/anomaly_summary.csv`, `docs/anomaly_acceptance.json`.

Validation:

```text
scripts/test.sh -q
41 passed, 19 existing dependency deprecation warnings in48.83s
```

Five new tests verify model unsmoothed CE equality on an actual val window;
causality under perturbations after frame24 and isolated batching; native-knot
interpolation, duplicate handling, grid bounds, height and heading; AUROC1/.5/0
and tied pairwise reference; stride, eligibility and tail padding. Acceptance
also independently recomputed every reported AUROC by pairwise comparison,
checked all28,413 score keys unique, and reproduced one GPU score per group on
CPU: maximum NLL difference4.77e-7 nats; residual differences≤1.78e-15m.
Raw drone/checkpoint/dictionary hashes and protected train indexes/vocabulary
match their original receipts. All source Python tracked; no Git remote added.

M8 Done-when met. Commit locally and **stop here for Yi-Chung's review**.
No state-plausibility fallback, further experiment or optional M9 was started.

## M8 follow-up — GT replay with object NLL (2026-10-18)

Yi-Chung explicitly requested the existing rollout GIF style with GT trajectories
only, per-object NLL anomaly scores, red object boxes for **NLL >7**, and a thick
red outline around the matching GT route segments. This authorizes this M8
visualization follow-up; optional M9 and additional anomaly scores remain out of
scope.

Asked the continuing read-only advisor once about score timing and gaps. Its
`ANSWER:` confirmed the latest-completed-token display rule: token[f,f+6] is
available at frame f+6, shown for the next6frames until a new token completes.
Labels show the actual scored interval in absolute scenario time. Expired or
missing tokens show N/A, never a carried-forward or zero/normal score. High GT
segments turn red only once their token completes. Full exchange is recorded
in parent `SYGARD_QA_LOG.md`; no DEFER/ERROR or new provisional score decision.

Implemented `src/eval/anomaly_gif.py`, reusing the M7 v2 dark OSM map, geometry,
palette and GIF writer. It reads existing `outputs/anomaly/token_scores.csv`
without model/checkpoint loading or inference; drone GT is deterministically
interpolated with the existing M8 loader and aircraft GT comes from the same ten
saved Demo windows. No predicted paths or future GT ghosts are drawn. Aircraft
and drone use the saved single-agent NLL even when displayed in a multi-object
GT scene. The footer explicitly says offline, single-agent, teacher-forced NLL.

Objects have a numeric NLL and scored-time interval. A strict >7 comparison adds
a red rectangle around the icon and a red label border. For the path, a red
8pt stroke sits under its ordinary coloured3pt GT core on the exact six edges of
each completed high token. Historical red segments remain after the current
score becomes low, while the object's red box clears. Invalid GT breaks paths
and hides icons. Score grids must align with the displayed absolute frame grid;
misaligned objects get a recorded warning and no rounded/interpolated score.
All actual input grids align. Reserved red `#ff3038` plus antialias blends are
added to the exact GIF palette; normal track colours avoid red.

Command:

```bash
scripts/catk.sh -m src.eval.anomaly_gif
scripts/test.sh -q tests/test_anomaly_gif.py tests/test_demo_gif.py
```

Rendering used CPU and existing packages, with network escalation only for
uncached OSM tiles. Outputs under ignored `outputs/anomaly/gifs/`:

- `drone01_full.gif`: the full0..300s drone GT route,901 native3Hz frames.
- Ten original aircraft examples,109frames each, GT-only, per-object scores.
- `contact_sheet.png`, built from decoded GIF final frames.
- `drone_low.png`, `drone_high.png`, `drone_low_after_high.png`, extracted from
  actual decoded GIF frames (t10,162,164s), plus reproducibility `manifest.json`.

Acceptance is committed as `docs/anomaly_gif_acceptance.json`:

```json
{"gif_count":11,"total_frames":1991,"drone_frames":901,
 "aircraft_frames_each":109,"size":[1200,900],"max_bytes":3436926,
 "threshold":7.0,"strict_greater_than":true,
 "source_scores_unchanged":true,"protected_artifacts_unchanged":true,
 "prediction_trajectories_drawn":false,"model_execution":false,
 "training":false,"all_frames_decoded":true,"all_score_grids_aligned":true}
```

All GIFs retain native frames (no skip), play110ms/frame, hold the final frame
1500ms and loop. All1991 encoded frames were decoded and checked for counts,
dimensions, durations, loop, file sizes and source hashes. Original M8 scores,
checkpoint/dictionary, raw drone source, aircraft rollout inputs and protected
train artifacts remain unchanged. No package installs or new anomaly metric.

**7 tests passed**,19 existing dependency warnings: score availability and
expiry at exact boundaries; unscored gaps; strict threshold7 exclusion; absolute
frame alignment and invalid GT edges; actual GIF pixels switching from no red
to high-score box/path and then back to low-score object while retaining the
old high trail; existing M7 palette/encoding/geometry regressions also pass.

One visual self-review completed: inspected the decoded contact sheet, drone
t10/162/164s snapshots, and aircraft scenario000667_f0720 decoded frames
0/18/36/60/84/108. Numeric labels and scored intervals are readable at full size.
At t162s the drone shows NLL7.19 for160–162s with a red icon box and red path
halo; at t164s NLL2.54 for162–164s clears the object box, preserving the prior
short red segment. Exact red-pixel counts in these snapshots are0/1528/43.
Drone has14 high tokens among146; first160–162s, maximum10.49 at262–264s.
The full-route camera makes individual2s red segments short, but they remain
visible. No taste iteration after this review. Commit locally and stop for
Yi-Chung to inspect the animations.

## M8 visualization follow-up — drone icon (2026-10-18)

Yi-Chung requested a drone-specific icon. Replaced the drone's aircraft marker
with an original vector quadcopter: four open rotor rings, X arms and a small
east-facing body/nose, rotated by the same heading convention. Aircraft retain
their original marker. No external icon assets or packages are needed.

Re-rendered only `outputs/anomaly/gifs/drone01_full.gif` and its low/high/return-low
snapshots, then updated the drone cell of the contact sheet. Added a close-up
`drone_icon_preview.png` from the decoded high-score frame. All ten aircraft GIF
hashes and the saved NLL CSV hash are unchanged. The drone GIF remains901frames,
1200×900,110ms/frame with1500ms final hold, now3,872,853bytes. No model execution.

`scripts/test.sh -q tests/test_anomaly_gif.py`: **3 passed**,19 existing dependency
warnings. Decoded all901 updated frames and checked dimensions/timing; visually
inspected the actual high-score snapshot to confirm four rotor rings remain
recognizable inside the red box and the NLL interval label remains readable.
Updated `docs/anomaly_gif_acceptance.json`. Commit locally and stop for feedback.

## M8b — Location-conditioned state density (2026-10-18)

Read the updated plan §4.1 item10, M8b and §4.2. Continued Yi-Chung's already
initialized **plan-based** advisor session with one consolidated question; no
--refork or expensive --fork-main was used. Advisor returned `ANSWER:` confirming
the proposed units, fixed grid/binning, pooling/normalisation, reference split
and time alignment. Optional combined GIFs use the **10-04 airborne combined
p95**, matching the plan's timeline and exceedance reference. Nothing was
missing from the plan; no unanswered decision/DEFER/ERROR to log. Q/A remains
in parent `SYGARD_QA_LOG.md`.

Implemented `src/eval/anomaly_state.py`. Fixed non-neural density fit uses only
the618 train scenarios from09-28..10-02 and **1,007,315 eligible3Hz states**.
Eligibility requires valid endpoints t−6 and t; speed/course come only from that
past2s displacement. Aircraft processed z and the existing drone loader already
contain height above166m, so no second height conversion is applied.

Metadata bounds south41.94/north42.015/west−87.95/east−87.845 are converted by
the M7 fixed-origin tangent plane. Grid is35×34 cells of250m, plus one outside
bucket (1,191 cells total). Last cells are partial and exact upper bounds are
included in those cells. Height boundaries15/100/300/600m; speed boundaries
5/10/20/…/120m/s (14bins);12course bins begin at−pi. Stationary atan2(0,0)=0.

Raw counts are retained for audit. Joint, speed and course histograms pool over
3×3 in-grid neighbours at fixed height with zero padding. Outside pools itself.
With alpha=.5, joint p(cell,height) normalises over all5,955 categories;
conditional speed/course denominators are pooled_joint + alpha×14/12. All
conditionals normalise independently. State NLL is the sum of location/height,
conditional speed, and conditional course NLL, all in nats. No drone tuning.

Scored the **exact28,413 M8 token rows**, including146drone tokens, with no
drops or replacements. State is measured at token_start_frame; existing M8
transition NLL is joined unchanged. Standardisation uses only all16,42910-03
aircraft tokens (airborne+ground):

| score |10-03 median|10-03 IQR|
|---|---:|---:|
|transition NLL|3.7179174423|3.0186932087|
|state NLL|9.2971473064|1.9010103151|

Combined = max(z_transition,z_state). Zero/invalid IQR raises an error; there is
no fitted fallback.10-04 remains the clean evaluation set.

Primary results, drone vs9,20010-04 airborne tokens:

| score | AUROC | drone fraction > that score's10-04 airborne p95 |
|---|---:|---:|
|transition NLL|0.336244|9.59%|
|location/height NLL|0.965998|75.34%|
|speed NLL|0.987441|91.10%|
|course NLL|0.975782|91.78%|
|state NLL|0.999902|100%|
|combined|0.999902|100%|

State median drone23.8186 vs airborne8.3655nats; airborne state p95=11.5871.
Combined median drone7.6388 vs airborne0.2106; airborne combined p95=1.466696.
The largest mean component excesses are course+5.335 and speed+5.151nats;
location/height adds+3.646nats. Full tables include10-04 ground and10-03 reference
groups, with count/mean/median/p95/AUROC and the fixed exceedance statistic.

Plain conclusion: **location-conditioned state density flags this drone route
where the transition score largely did not. All146drone tokens exceed the
10-04 airborne state p95; unusual course and speed at the given place/height
drive most of the excess.** Combined has the same AUROC as state alone here.
This is one drone and one correlated route, a demonstration rather than a
validated detector.10-04 p95 is descriptive, not a validated deployment threshold.

Run `scripts/catk.sh -m src.eval.anomaly_state --gifs`. The required density,
scoring and two plots took4.18s on CPU before optional GIF rendering. No neural
training, model execution, new packages, GPU allocation or reference-repo edits.
Outputs under ignored `outputs/anomaly_state/`:

- `summary.md`, `summary.csv`, `token_scores.csv` with all component/state/z
  scores and position/speed/course bins, `standardization.json`, `acceptance.json`.
- `state_density.npz` contains raw/pooled counts and normalised probabilities.
- `traffic_density.png`: five height panels of log(1+raw eligible train counts)
  over faded cached OSM, with drone path only in matching height panels.
- `drone_timeline_state.png`: separate state-NLL/transition-NLL/combined panels
  and each10-04-airborne p95 line. Separate panels preserve score units.
- Optional `gifs/`:11GT-only combined replays, contact sheet and drone preview;
 901drone frames and109frames per aircraft example. Largest4,303,801bytes.
  All use the quadcopter/aircraft icons and display **combined**, not NLL. Red
  threshold1.466696 comes only from10-04 airborne combined p95, never the drone.
  State is known at token start; combined is displayed only when the following
  transition token completes, with its actual interval. Original M8 GIFs remain.

Committed copies: `docs/anomaly_state_results.md`, `docs/anomaly_state_summary.csv`,
`docs/anomaly_state_acceptance.json`. Both required plots and the decoded
combined contact sheet/drone frames0/30/300/486/792/900 were visually inspected.
The new combined display flags steady drone flight from the first completed
score at10s, unlike the old transition-only display. No further taste iteration.

Validation:

```text
scripts/test.sh -q
51 passed, 19 existing dependency deprecation warnings in45.46s
```

Six density tests cover eligible count equality/train-file isolation,
pooled-count consistency and all distribution sums, empty-vs-busy synthetic
states, causality/invalid endpoints,10-03-only reference and zero-IQR failure,
fixed bin boundaries and outside-cell routing. The actual GIF regression now
tests both NLL and combined labels. Runtime acceptance independently verified
all28,413 original token keys/transition scores, count totals, every probability
sum, every z-score/component sum/max, every reported AUROC by pairwise
comparison and every exceedance fraction. All1,991 combined GIF frames decoded
with resolution/timing/loop/size checks. Protected train indexes, vocabulary,
dictionary, checkpoint and original M8 score CSV hashes are unchanged.

M8b Done-when met. Commit locally, keep no remote, and **stop for Yi-Chung's
review**. Optional M9 and NN retraining remain unstarted.

## M8b follow-up — circular course and reflecting speed smoothing (2026-10-18)

Yi-Chung authorized smoothing after the normal-aircraft aa0300 false alert.
Continued the cheap plan-based advisor once with the kernel, nonuniform speed
bins, open tail, calibration and comparison questions batched. Advisor returned
**DECISION (provisional, Yi-Chung to confirm)**: fixed[1,2,1]/4 by adjacent-bin
index, circular course, reflecting speed, after spatial3×3 pooling and before
alpha. Do not introduce physical bandwidths, artificial tail centres or smooth
height/joint. The missing plan specification is recorded in QUESTIONS.md for
Yi-Chung to add to §4.1; implementation proceeds under user authorization.

Each interior bin keeps1/2 mass and sends1/4 to each neighbour. Course bins11/0
are neighbours. Speed end bins keep3/4 and send1/4 to their only neighbour.
This preserves mass exactly, including the >=120m/s tail; it is not equal-m/s
smoothing because speed bins have unequal widths. The numeric speed/course of
aircraft and their trajectories are unchanged. No temporal smoothing or tuning.

`StateDensity` now retains raw/pooled counts and additionally saves
`smoothed_speed`/`smoothed_course`. Joint probabilities, spatial/height handling
and alpha are unchanged. Conditional probabilities use the same denominators,
with smoothed counts in their numerators. Hard bin lookup remains: this reduces
adjacent-bin differences but does not make a continuous density or eliminate
all steps/false alerts.

Current command: `scripts/catk.sh -m src.eval.anomaly_state --gifs`.
Default output is **outputs/anomaly_state_smoothed/**; the22 original files under
outputs/anomaly_state are preserved byte-for-byte. `--legacy-bins` reproduces
the original method. Smoothed10-03 state median/IQR and10-04 reference p95 are
recomputed; transition reference and all28,413 original token rows are unchanged.
Comparison additionally freezes the *old* medians/IQRs and cutoffs, preventing
calibration-scale mixing when comparing combined false-alert counts.

The reported aa0300 example is **still flagged**, without special-case changes:

| quantity, token28–30s (display t30s) | before | smoothed |
|---|---:|---:|
|speed NLL|2.357830|2.049774|
|course NLL|2.727231|2.733274|
|state NLL|12.447265|12.145251|
|combined, own10-03 reference|1.657075|1.384783|
|10-04 airborne combined p95|1.466696|1.384230|
|flagged|true|true|

Across the−179.82°/+179.20° seam in tokens26/28s, the course-NLL difference
decreases **0.610144→0.223548**. The target-bin course NLL itself does not drop;
the speed component drops. Frozen-reference combined at28s is1.498205, also
above the old cutoff. No cutoff was raised to conceal this example. Because
the new value is only0.000553 above cutoff, GIFs show5decimal threshold/nearby
score values instead of rounding both to1.38.

Combined flagged-token counts (normal aircraft treated as clean):

| group | before | after, frozen old calibration/cutoff | after, recalibrated |
|---|---:|---:|---:|
|drone|146/146|146/146|146/146|
|10-04 airborne|460/9200|436/9200|459/9200|
|10-04 ground|75/2638|76/2638|81/2638|
|10-03 airborne reference|1249/11097|1064/11097|1106/11097|
|10-03 ground reference|469/5332|374/5332|379/5332|

Recalibrated10-04 airborne exceedance stays approximately5% by construction
(strict ties can make it slightly lower). **This does not demonstrate a general
false-positive reduction:** ground flags slightly increase. Course/velocity
boundary noise is reduced, but the reported plane and other normal rare states
can remain red. Do not claim the false-positive problem is solved.

Drone-vs10-04-airborne AUROC: speed0.995804, course0.980103, state0.999764,
combined0.999751. State/combined still flag146/146drone tokens above the new
airborne p95. These figures are not used to select a kernel; there was one
fixed advisor choice, no sweep or drone-dependent tuning. One-route caveat
and descriptive-threshold limitations remain.

Outputs in the new directory include refreshed tables, plots,11combined GIFs,
`smoothing_comparison.csv/.json` and `aa0300_comparison.csv`. Per-component and
frozen/recalibrated comparisons are explicit. Updated GIF example:
gifs/scenario_000536_aircraft_f0090.gif and decoded gifs/aa0300_t30.png. Required
plots and that decoded frame were visually inspected; actual display reads
combined1.38478 and red cutoff1.38423. No visual change hides the residual flag.
Committed result/summary/acceptance and comparison tables have smoothed/smoothing
names under docs/, preserving earlier committed results.

Validation:

```text
scripts/test.sh -q
54 passed,19 existing dependency warnings in50.09s
scripts/test.sh -q tests/test_anomaly_gif.py
4 passed,19 existing dependency warnings in5.15s
```

Three new regressions verify every source-bin mass sums to1, exact circular and
reflecting edges, raw/pooled/joint invariance and conditional sums, and actual
near-westward course scoring borrowing across the seam. Existing causality,
train-only fit,10-03-only standardisation and rendering tests pass. Acceptance
rechecks all raw arrays/joint probabilities against the original; exact kernel
outputs and totals; per-token conditional NLL;10-03-only parameters; every
AUROC independently by pairwise comparison. All1,991 updated GIF frames decode
with correct counts/timing/size/loop. All22 original output hashes and protected
checkpoint/dictionary/train artifacts are unchanged. CPU only; no NN/model
execution, training, installs or M9 work. Commit locally and stop for review.

## M8b height100 comparison — 2026-10-18

Yi-Chung authorized merging <15/15–100m after reviewing normal taxi alerts.
Implemented by Codex; the advisor provided one consolidated read-only provisional
review (logged in parent SYGARD_QA_LOG.md and QUESTIONS.md). No neural training,
model execution, packages or GPU. Density still fits1,007,315 train states from
618scenarios; score joins are the same28,413 rows. New raw height boundaries
100/300/600m,1191×4 joint cells; fixed spatial pooling, course/speed kernel,
alpha and max combination retained. Class defaults retain legacy compatibility;
CLI defaults to separate outputs/anomaly_state_height100. Prior smoothed CLI
is available via --keep-height-15, original via --legacy-bins.

Results versus speed/course-smoothed five-height baseline:
- Selected10-03 aabc11/scenario000667 token starts238–274s:8/19 flags →0/19,
 16s continuous alert →0s; aa0300/scenario000536 starts24–60s:5/19 →0/19.
 The advisor mistakenly called these10-04 cases; actual row/GIF dates are10-03.
- Frozen baseline calibration/cutoff:10-04 all-aircraft540/11838 →467/11838;
 airborne459/9200 →388/9200, ground81/2638 →79/2638. Recalibrated:
 airborne460/9200 (5% by construction), ground112/2638=4.2456%, all572/11838.
 Ground maximum run18s →60s; tracks with≥6s ground alert6/149 →7/149.
- Drone retention146/146 →138/146, both frozen and recalibrated. All8misses
 below100m:32tokens, median state NLL18.3992 →13.5477,24flagged.100–300m:
 114tokens, median23.04183 →23.04176,114flagged. Combined AUROC.999751 →.992693.
- Therefore the selected normal examples improve, but recalibrated ground
 false alerts and low-drone misses worsen. This is a comparison variant,
 not an overall validated improvement. No additional tuning/score applied.
 Both10-03 examples and10-04 aggregates informed development; neither an
 inspected example nor inspected10-04 evaluation is a pristine holdout.

Acceptance output:
- `scripts/catk.sh -m src.eval.anomaly_state --gifs`: successful CPU run,
 11GIFs decoded at1200×900; all under8MB; four-height density and drone plot.
- Full `scripts/test.sh -q`:59passed,19dependency warnings,45.83s.
- `scripts/test.sh -q tests/test_anomaly_state.py`:14passed,19dependency
 warnings,3.62s. Tests cover train-only fit,4/5-bin normalization and busy/empty
 cells, exact raw merge before alpha, causality and gap/track-aware alert runs.
- Actual fitted raw/pooled/kernel counts exactly equal baseline first-two sums;
 all higher layers unchanged; joint shape(1191,4). All47 original and smoothed
 artifact hashes unchanged. Protected source scores/checkpoint/dictionary and
 train inputs unchanged (acceptance receipt).
- Visual self-review: decoded aabc11 at260s has no red box/track; other aircraft
 can still alert. Four-panel height density plot has correct labels/drone path.

Artifacts: outputs/anomaly_state_height100/{summary.md,token_scores.csv,
state_density.npz,height_comparison.csv,alert_runs.csv,aircraft_cases.csv,
drone_by_height.csv,traffic_density.png,drone_timeline_state.png,gifs/}.
Reports and numerical acceptance archived under docs/anomaly_state_height100*.
Stopped after this requested comparison; no additional detector changes.

## Restore five-height smoothed default — 2026-10-18

Yi-Chung requested reverting after the height100 tradeoff review. Restored the
CLI default to original15/100/300/600m edges with the fixed course/speed
smoothing and outputs/anomaly_state_smoothed. --keep-height-15 remains accepted;
--height100 explicitly selects the retained experiment. Conflicting profile
flags fail before any output writes. Existing tables/GIFs remain unchanged;
no scoring, rendering, training, model execution or packages were needed.

Validation:14 state-density tests passed (19 dependency warnings),3.63s;
intercepted fit calls confirm default five bins and opt-in four bins, both
smoothed; all47 preserved baseline artifact SHA256 checks pass; CLI help and
git diff --check pass. Height100 report remains archived as comparison evidence.

## Presentation demo export — 2026-10-19

Yi-Chung requested individual full-scene images and cleaner demo visualization:
use “Anomaly score”, omit token intervals and unexplained implementation text.
Implemented an opt-in presentation style and saved-score-only exporter:
`scripts/catk.sh -m scripts.export_anomaly_demo`. It uses the restored five-height
smoothed baseline, existing trajectories and cached map tiles, with no density
fit, neural/model execution, training, score changes or added packages.

Outputs are separate under outputs/anomaly_demo/: scenes/ holds11 individual
1200×900 PNGs matching each replay's final/contact-sheet frame; gifs/ holds11
matching animations and contact_sheet.png. scene_images.zip packages all11
PNGs. Header uses SYGARD / AIRSPACE MONITOR with friendly scene names/date;
labels use scene-local Aircraft/Drone numbers, Anomaly score and alert/tracking
status. Removed raw scenario/object IDs, combined formula, NLL, interval and
N/A from the presentation. Retained relative-score explanation, recorded-path
replay context and OpenStreetMap attribution. Source IDs and timing remain in
the manifest/README for reproducibility. Existing scientific visuals remain
unchanged; presentation=False is the renderer's backwards-compatible default.

Validation:20 tests passed,19 dependency warnings,6.51s (state-density and
renderer tests, including presentation mode's red-pixel timing/expiry checks).
Decoded all11GIFs, checked dimensions/frame counts and exact PNG equality to
the last decoded GIF frame. Threshold remains1.3842295463291823; all47 prior
baseline artifacts verified byte-identical. ZIP integrity verified. Visual
self-review of drone snapshot, scenario000536 aircraft snapshot and contact
sheet completed; headers/labels readable with no interval/formula. Numerical
receipt archived at docs/anomaly_demo_acceptance.json. git diff --check passed.

## Quarterly report rollout figures — 2026-10-19

Yi-Chung requested simulation rollout figures for a quarterly report. Exported
six existing seeded Demo cases as presentation/report figures from saved
32-rollout BC arrays (epoch016), without rerunning a model or training. New
command: scripts/catk.sh -m scripts.export_rollout_report. Output directory:
outputs/quarterly_rollouts. Each case has horizontal trajectories and vertical
profiles; dark blue6s history, orange recorded30s future, light blue all32
sampled futures. A six-case horizontal overview, suggested English caption,
provenance manifest and ZIP package are included.300dpi PNGs (individual3900×1740,
overview4500×3000) and vector PDFs. Equal-distance map geometry retained;
straight tracks get expanded axis limits so axes stay readable without cropping
rollouts. No error-ranked scene selection or best-rollout selection introduced.
Flight-phase labels retain the original Demo's GT-based classification.

Validation: all six input SHA256 checks unchanged; all32 plotted samples in
both horizontal and vertical panels match source arrays. Six PNG dimensions
and seven PDF signatures checked; overview dimensions checked;16-file ZIP
integrity passed. Visual review of overview, climb and descent panels passed
after fixing collapsed straight-track axes and header spacing. No packages,
GPU, density fitting or model execution. Receipt: docs/quarterly_rollout_acceptance.json.
Caption documents validation reuse and sampled trajectories rather than
calibrated confidence intervals. Original rollout/data artifacts unchanged.

## Quarterly rollout stills — corrected visual direction, 2026-10-19

Yi-Chung clarified that the requested quarterly Demo is the existing dark-map
simulation GIF style at the completed rollout frame, rather than scientific
axes/overview figures. Exported all10 existing v2 scenes as individual2400×1800
PNGs at t=+30s (frame108), including scenario000197/f0180. Colourful recorded
solid paths/filled aircraft, simulated dashed paths/outline aircraft, clean
SYGARD / MOTION SIMULATION title, friendly aircraft labels and an explanatory
legend. No plot-grid overview. OSM attribution retained. Source v2 GIFs and
rollout arrays unchanged. Export: scripts/catk.sh -m scripts.export_rollout_snapshots;
outputs/rollout_demo_stills/{individual PNGs,rollout_demo_stills.zip,CAPTION.md,manifest.json}.

Renderer has opt-in direct final-frame/presentation/high-resolution parameters;
default animation behavior is retained. No model execution/training or new
packages. Uses exactly the existing GIF's joint rollout index (best scene-wide
3D error of32); this selection uses recorded future and is disclosed in the
caption/manifest, rather than presenting it as an unbiased sample.

Validation:4 renderer tests passed,18 dependency warnings,3.72s, including exact
comparison of direct last-frame output against the original animation's final
composition after identical palette quantization. All10 scenes have matching
original indices and SHA256 input hashes, dimensions2400×1800 and final time30s.
12-entry ZIP integrity verified. Visual self-review of the user's requested
scenario000197 final image passed. Receipt: docs/rollout_demo_stills_acceptance.json.

## Camera-derived input rollout Demo — 2026-10-27

Yi-Chung authorized a camera-input version with camera/prediction/GT views,
arbitrary prediction sample (no minADE). Implemented by Codex; one consolidated
read-only advisor consultation provided provisional causal adapter rules
recorded in QUESTIONS.md. No architecture/checkpoint/dictionary changes,
training or package installs. Ran on CPU; nvidia-smi driver unavailable in this
environment. CUDA_VISIBLE_DEVICES remains1,3 via scripts/catk.sh.

Added src/data/camera_trajectory.py: native multi-camera rows grouped by
tracker_id/time, posterior at largest event_index retained, any actual measured
event at that time tracked separately. Each3Hz sample takes the most recent
posterior and advances with its own Kalman velocity×age (≤1s). No interpolation
from future camera rows. Skip uninitialized zero-velocity states before second
measurement. Require all19history frames,≥3 distinct measured timestamps in6s,
last measurement≤1s at current frame. First eligible origin per independent
tracker; raw heading from camera velocity with retained fallback, existing
causal-chord tokenizer unchanged. Camera-only ground proxy: height z+2250−166,
enter<10m, exit>20m; all types UNK. Exactly19historical camera samples are passed
to the frozen checkpoint, one single-agent scene per track. Sample0 of1,
topk5/temp1/seed817,30s forecast. No forecast-error selection.

GT references are loaded separately for timestamp-aligned overlays only;
interpolation requires valid brackets≤1s, with no extrapolation. Segmentation
mapping is used only to select offline-paired examples/identity labels; no GT
features or GT-based ID stitching. Camera data are existing RGB detection plus
simulator-depth/camera-pose tracker estimates, including Kalman predicted
states. This is camera-input trajectory prediction, not camera anomaly scoring.

Results:43native tracker IDs audited,10eligible paired cases (8aircraft,
2drone),33excluded as unmapped or lacking sufficient fresh camera history.
Camera future coverage7.0–24.33s, GT coverage5.0–30.0s; predictions remain full30s.
One ground-flag transition across the10input histories, reported in manifest.
No claim of general prediction accuracy or anomaly-detection performance.

Visuals: src/eval/camera_rollout.py draws cyan solid camera, yellow dashed
prediction, green dotted GT on dark OSM maps with plane/quadcopter icons.
Endpoint labels preserve last-observed time after camera/GT stops. Sample0
is explicit. All10individual completed-rollout PNGs2400×1800, matching109-frame
GIFs1200×900 (<8MB each), saved NPZs, auditCSV, manifest/acceptance and PNG ZIP
are under outputs/camera_demo. No overview substituted for the requested demo.
Visual review: aircraft track05 and drone track25, source colors and shorter
camera coverage distinguishable. OSM attribution retained.

Validation:
- Full scripts/test.sh -q:70passed,19dependency warnings,47.17s.
- Camera-focused tests:9passed,19warnings,3.90s. Covers sequential updates,
 causal CV resampling, measurement freshness, initialization, gap masks,
 future-input independence, UNK type, ground hysteresis, GT interpolation and
 all three renderer source colors without changing input arrays.
- All10saved NPZ shapes, PNG/GIF dimensions/frame counts/durations and ZIP
 integrity verified. All6raw input SHA256 hashes and5protected artifacts
 unchanged. Exact first forecast reproduced after deleting every post-origin
 camera row. GT is not an argument to the inference adapter/call.
- Initial smoke exposed missing map tiles under sandbox DNS; reviewed network
 access downloaded public OSM tiles. A saved single-agent output-axis indexing
 error in the new renderer was fixed before the complete run; all10final cases
 and focused renderer checks pass. No unresolved execution blocker remains.

Commands:
 scripts/catk.sh -m scripts.run_camera_demo --gifs
 scripts/catk.sh -m scripts.run_camera_demo --render-only --gifs

Acceptance, complete manifest and43-track audit archived under docs/camera_demo*.
Stop after this authorized input-transfer demo; no retraining or detector
modifications added.

## Integrated MVP planning draft — 2026-10-29

Yi-Chung requested a reviewable Markdown plan after sharing David's feedback,
and asked to settle outstanding commits first. Read-only audit found clean
working trees in both sygard-motion (HEAD2822d3a) and the Overleaf bi-weekly
report (its report edits were already committed separately). No uncommitted
implementation needed reconciliation; generated outputs remain ignored.

Added INTEGRATED_DEMO_PLAN.md as a review draft only: noisy reference inputs,
frozen forecasting, current/future state anomaly, forecast-based intent,
protected-zone/day-night threat heuristics, explanations, and two scripted
stories. Includes proposed parameter defaults, data provenance, causal checks,
no GT-based sample selection, milestone Done-when gates and review checklist.
No new inference, training, synthetic data generation or runtime code changes.
Validation: documentation diff/whitespace check; existing module/artifact paths
confirmed. User review precedes implementation of this new phase.

## Integrated v1 D0 — approved scope and fixed config, 2026-10-29

Read parent §4.1.11 and approved plan; clean baseline HEAD522afb9. No advisor
call this run. Fixed configs/integrated_demo/v1.json, SHA256 lock for11config/
input/model/density files, scenario/zone sources and rules before model runs.
A: daytime circle [2200,-2000,80],radius100m,speed8m/s; B: nighttime circle
[1800,300,110],radius60m, then westward acceleration at20–32s to38m/s. Zone
fictional x[-600,-300],y[150,550],height[0,180]. Both105s, background aa4e47
reference frame500 onward, unmodified and explicitly time-shifted composition.

Four optional decisions:
1. NO time-conditioned histograms: retain frozen detector/calibration and avoid
new fit/calibration scope. Day/night is scripted threat context only; no
weather/time-of-day-inside-detector claims.
2. YES joint3D medoid over32rollouts (mean Euclidean distance over agents and90
future frames, no GT), more representative while keeping joint sample coherence.
3. YES numeric-only CV forecast from the same noisy past2s displacement; useful
for attribution/comparison, never displayed/substituted for model prediction.
4. YES scripted night sigma3m/axis versus daytime1m, labelled as scripted
Gaussian noise rather than measured nighttime tracking quality.

Required correction: anomalous latch is any current score strictly above
frozen cutoff during last10s, per object; future scores cannot activate it.
Unflagged objects including background: Normal traffic, null threat score;
no anomaly+5. Unknown input/forecast is explicit, not an implicit low threat.
D4 first-pass gate will stop the run if B has no pre-entry high warning.
D0 acceptance: config parses,11checksums saved, both scenarios described with
sources; docs/integrated_demo_D0.json passed. Documentation/config only,
no inference/training. Updated plan/AGENTS; parent §4.1.11 already records scope.

## Integrated v1 D1 — noisy observations and frozen forecast

Implemented keyed per-source/object seeded Gaussian observation streams,
causal noisy-position velocity/heading/ground features, UNK types,19-state
history graphs,32joint sampled forecasts, joint3D medoid and numeric-only CV.
Background source segment verified fully observed; synthetic route generation
is configured but story B has not been inferred or adjusted at this stage.
Features can carry prior estimator state (like a tracker); future rows are
never read. No clean heading/ground/type enters the graph.

Acceptance:4new data/forecast tests passed (19dependency warnings,25.90s),
covering sigma0/prefix overlap, future perturbation invariance, actual-sample
medoid selection and exact linear CV. CPU smoke on existing aircraft and native
drone produced[32,1,90,3]. For BOTH inputs, sigma0 forecasts repeat exactly;
removing all future observations leaves noisy forecasts bitwise identical.
All11locked artifacts/config hashes unchanged. D1 passed; saved arrays under
outputs/integrated_demo/D1, receipt docs/integrated_demo_D1.json. Corrected D0
prose checksum count from12 to actual11; lock/acceptance already recorded11.

## Integrated v1 D2 — frozen current/future anomaly and intent evidence

Read the existing five-height probability arrays without calling fit/finish.
Raw state cutoff=11.387990034013754 nats; equivalent z cutoff=.9641637639319811
using the frozen10-03 state median/IQR. Never use the combined cutoff.
Current trace scores use2s observed displacements; forecasts score each sample
at+2..+30s from its own positions and the observed current anchor. Future alerts
require>=8/32strict exceedances per horizon and cannot set the current latch.
Added configured polygon+height volume geometry, distance and3Hz sampled entry
counts/time summaries as forecast-derived intent, no GT. Endpoint outputs are
saved without selecting against a reference.

Acceptance:4scoring/geometry tests passed,19dependency warnings,6.26s. Reproduced
all28,413original token rows with exactly0error in all4state/components. Tests
cover own-sample speed/course, strict cutoff/sample count, missing endpoints,
height/boundary inclusion, volume distance and entry time. Aircraft/drone D1
forecast score/intent outputs saved under outputs/integrated_demo/D2;
docs/integrated_demo_D2.json passed. All11locked hashes unchanged; no refit,
training, calibration updates or story B inference.
