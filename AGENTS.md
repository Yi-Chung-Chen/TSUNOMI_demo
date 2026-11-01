# Repository instructions


The implementer is Codex or Yi-Chung. The plan author wrote this plan only and does not implement it.

1. Create the new repo at **`/local/scratch/a/chen5262/PythonProject/sygard-motion`**. This is the same directory as `~/disk_a/PythonProject/sygard-motion`.
2. **Never modify** any of these reference repos. They are read-only references:
   - `SYGARD-Dataset`. This is someone else's repo (`robin6205/SYGARD-Dataset`, author Josh Chang).
   - `catk_starcraft`
   - `catk_starcraft_release`
   - `starcraft-motion-private`
   - `starcraft-motion-processing`

   Copy code into the new repo. Do not import from these repos and do not symlink into them.
3. Do not copy raw data into the repo. Read it via a config path. Processed outputs go to `data/` (git-ignored).
4. Work milestone by milestone (M0 through the current M8b location-conditioned state density). Each milestone has a **Done when** check. Do not start the next milestone until that check passes. Commit once per milestone.
5. `.gitignore` must use **anchored** patterns (`/data/`, `/logs/`, `/outputs/`).
   - Lesson learned: `catk_starcraft_release` had `datasets/`, which silently dropped `src/starcraft/datasets/sc_dataset.py` from git.
   - After each commit, run `git ls-files src | wc -l` and confirm that every `src/**/*.py` is tracked.
6. Use the existing conda env `catk` (`/local/scratch/a/chen5262/anaconda3/envs/catk`). Verified versions: Python 3.11.9, torch 2.4.1+cu121, lightning 2.4.0, torch_geometric 2.7.0, hydra 1.3.2. Do not install packages into it without asking. **Exception (approved 2026-10-16): `pip install pytest` into `catk` is allowed.** If something is missing, list it in `install/requirements.txt` and tell the user.
7. Copied CAT-K/SMART code carries NVIDIA / SMART license headers. Keep them and copy the `LICENSE` files too. See `catk_starcraft_release/LICENSE` and `src/smart/LICENSE.SMART`.

Path shorthands used below:

| Short | Absolute path |
|---|---|
| `SYG` | `/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_aircraft_training_800_3hz` |
| `CS` | `/local/scratch/a/chen5262/PythonProject/catk_starcraft` (branch `main`, the code actually used for experiments) |
| `REL` | `/local/scratch/a/chen5262/catk_starcraft_release` (cleaned, Waymo-free version of `CS`) |
| `PRIV` | `/local/scratch/a/chen5262/PythonProject/starcraft-motion-private` (preprocessing + motion dictionary) |


## Layout

```
sygard-motion/
  AGENTS.md                 # repo rules for Codex (copy section 0 + the layout + commands)
  README.md
  LICENSE, LICENSE.SMART    # copied from REL
  .gitignore                # anchored: /data/ /logs/ /outputs/ /.cache/ __pycache__/
  install/requirements.txt
  configs/                  # Hydra; skeleton copied from REL/configs
    run.yaml  paths/default.yaml  trainer/  logger/  callbacks/  hydra/
    data/sygard.yaml  model/air_smart.yaml
    experiment/air_pre_bc.yaml  experiment/air_clsft.yaml
  src/
    run.py                  # copied from REL/src/run.py
    utils/                  # copied from REL/src/utils
    common/                 # SMART layers + geometry: attention_layer, fourier_embedding, mlp_layer, weight_init, rollout geometry
    preprocess/
      ingest.py             # JSONL -> dense per-scenario npz              (M1)
      split.py              # date split -> manifests                        (M2)
      windows.py            # window index (scenario, start_frame)           (M2)
      motion_dict.py        # K-disk build                                   (M3)
      motion_dict_eval.py   # reconstruction ADE/FDE (one-shot + autoregressive) (M3)
    data/  dataset.py  datamodule.py  target_builder.py                     (M4)
    tokens/  token_processor.py                                              (M4)
    model/   air_smart.py  decoder.py  agent_decoder.py                      (M5)
    metrics/ cross_entropy.py                                                (M5)
    rollout/ sampling.py  rollout_io.py                                      (M6)
    eval/    load_rollout.py  demo_plots.py  metrics/*  aggregate.py  (M6 Demo)
  scripts/                  # thin CLI wrappers + shell scripts for each stage
  tests/
  data/                     # git-ignored: processed/, splits/, motion_dict/
  assets/                   # final motion dictionary .pkl (committed, small)
```


## Commands
Use `scripts/catk.sh -m src.run --help` and `scripts/test.sh -q`. Follow the parent SYGARD_MOTION_PLAN.md sections 4.1 and 4.2. Record validation in PROGRESS.md; unresolved advisor decisions in QUESTIONS.md. GPU visibility must be 1,3 and usage checked before training.

## Current scope override (2026-10-16)
D1 / plan §4.1 items6/7 override earlier text: val=10-03+10-04, merge existing indexes only; no test split. M6 is BC-only Demo from epoch_016.ckpt, 32 rollouts, topk5/temp1, CPU if GPU busy. No CAT-K, no retraining, no old M7. Future BC config20epochs only; do not launch training. Preserve train manifests/windows, vocabulary, dictionary and checkpoint bytes.

Plan §4.1 item8 adds the new M7 Demo GIFs: read existing rollout files only, no model execution, training or package installs. Follow the rendering/selection/tests in M7, commit draft v1 after one visual self-review, then stop for Yi-Chung's feedback. Later revisions require Yi-Chung's feedback.

Plan §4.1 item9 (2026-10-18) adds M8 Anomaly scoring MVP; the old optional M8 is
now M9. Use frozen epoch_016.ckpt and the existing dictionary to score isolated
agents' first future token, with the default teacher-forced GT tokenizer chain.
Compare the native-pose drone interpolated to3Hz against10-04 aircraft, and10-03
for reference. Report NLL, raw-anchor token residuals and NumPy AUROC. No training,
model changes, package installs or state-plausibility fallback score. Commit
after required outputs/tests and stop for Yi-Chung. Optional M9 is not authorized.

Plan §4.1 item10 authorizes M8b: fit fixed250m location/height, speed and course
histograms only on09-28..10-02 train states. Standardise state/transition NLL by
all10-03 aircraft medians/IQRs; combine with max. Evaluate exact M8 token rows on
10-04, with10-03 reference. No NN training, model changes, new packages or drone
tuning. Required outputs/tests are specified in M8b. Optional combined GIFs may
reuse the existing GT renderer; use10-04-airborne combined p95 as the visual red
threshold. Commit and stop for Yi-Chung's review. M9 remains unauthorized.

Advisor usage follows current plan §4.2: default continues a plan-based session;
--refork starts a cheap new plan-based session after a plan change; --fork-main
is reserved for hard questions the plan cannot resolve. Batch questions, never
call in parallel, and log missing plan information in QUESTIONS.md.

Yi-Chung subsequently authorized speed/course smoothing. The provisional
advisor rule in QUESTIONS.md is fixed[1,2,1]/4, circular course and reflecting
speed, after spatial pooling and before alpha. Do not smooth joint/height or
tune to examples. Default outputs are outputs/anomaly_state_smoothed; preserve
the original outputs/anomaly_state for comparison and report residual false
alerts honestly with frozen and recalibrated cutoff comparisons.

Yi-Chung also authorized a separate height100 variant: merge raw <15/15–100m
counts into <100m before pooling/kernel/alpha, retaining100/300/600m edges.
Yi-Chung reverted the default to the five-height smoothed baseline after
reviewing the tradeoff: outputs/anomaly_state_smoothed. --keep-height-15
remains a compatible alias for that default; --height100 explicitly opts into
the preserved outputs/anomaly_state_height100 experiment. --legacy-bins reproduces the
original five-bin unsmoothed baseline. Preserve both previous directories.
Compare frozen and recalibrated scores, consecutive alert durations and drone
retention; disclose10-04 as inspected development data. Do not add further
score changes or train a neural model under this revision.

Yi-Chung authorized a camera-conditioned rollout Demo on2026-10-27: frozen
checkpoint, camera-only6s history,30s forecast, single-agent tracker scenes,
first eligible origin, sample0 of1 (no minADE). GT is overlay/association only;
no GT subtype/heading/ground input or track stitching. Adapter provisional
rules and execution issues are recorded in QUESTIONS.md. Save separate
outputs/camera_demo; no training, new packages or anomaly-model changes.

Active2026-10-29 scope is INTEGRATED_DEMO_PLAN.md D0–D5 per parent plan §4.1.11.
Do not call the advisor script this run. Each stage requires its Done-when and local
commit. Threat is only computed for tracks with any current state score above
cutoff during the preceding10s; otherwise Normal traffic and null threat score.
No anomaly+5 modifier. D4 B must have high warning before reference zone entry;
if it fails, save results, commit stage outcome and stop without tuning to pass.

User-approved presentation revision2026-11-01 (INTEGRATED_DEMO_PLAN.md D6)
overrides the10s latch and numeric threat rules for the active demo. Current-only
object anomaly gets orange boxes/reasons. Scenario statuses aggregate same-object
anomaly and zone evidence (>=8/32 entry); current intrusion is explicit. No numeric
threat, intent card or future-anomaly warning in media. Reinterpret D4 saved
outputs without inference; export D6/v2, preserve D4/D5 and frozen artifacts.

User-approved D7 preview adds up to4 foreground representative futures weighted
by group membership among all32samples. Full remaining3D zone entry stratifies
red/amber groups; mean3D trajectory medoids select actual sampled paths. Not
likelihood softmax or calibrated probabilities. Existing scenario rules remain
unchanged. Read existing D4 only, export D7/v3; preserve D6/v2. No advisor calls.

D8 user review removes A–D/entry-outside legend; percentages go beside endpoints.
Group up to4 futures independently for every currently anomalous object, including
aircraft. Keep original model paths, current anomaly/scene rules and per-object
normalization. Export D8/v4 without inference; preserve earlier outputs. Audit
raw saved curvature, never bend forecasts for appearance. No advisor calls.

D9 user review requests persistent scenario High threat after confirmed observed
entry by a currently anomalous object. Store first intrusion time/trackID; keep
High after exit, clearing anomaly or missing observations, reset per scenario.
Normal traffic and forecast-only entry do not latch. Orange boxes/rollout groups
remain current-only. Export D9/v5 from saved D4; preserve earlier versions.
