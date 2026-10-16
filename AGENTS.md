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
4. Work milestone by milestone (M0 to M7 below). Each milestone has a **Done when** check. Do not start the next milestone until that check passes. Commit once per milestone.
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
    eval/    load_rollout.py  kinematics.py  log_kde.py  metrics/*  aggregate.py  (M7)
  scripts/                  # thin CLI wrappers + shell scripts for each stage
  tests/
  data/                     # git-ignored: processed/, splits/, motion_dict/
  assets/                   # final motion dictionary .pkl (committed, small)
```


## Commands
Use `scripts/catk.sh -m src.run --help` and `scripts/test.sh -q`. Follow the parent SYGARD_MOTION_PLAN.md sections 4.1 and 4.2. Record validation in PROGRESS.md; unresolved advisor decisions in QUESTIONS.md. GPU visibility must be 1,3 and usage checked before training.
