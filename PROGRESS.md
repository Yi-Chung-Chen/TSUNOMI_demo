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
