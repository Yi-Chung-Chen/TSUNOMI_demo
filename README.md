# sygard-motion

BC aircraft trajectory Demo on the 800-scenario SYGARD ORD dataset. Current scope follows `../SYGARD_MOTION_PLAN.md` D1, §4.1 items 6–8, M6 Demo and M7 Demo GIFs. No CAT-K or retraining is part of the Demo.

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

To render draft-v1 GIFs from existing BC rollout files, without loading a model:

```bash
scripts/catk.sh -m src.eval.demo_gif
```

This writes ten 1200×900 top-down animations, `contact_sheet.png` and a selection/validation `manifest.json` under ignored `outputs/demo_gif/`. Windows are the six existing examples plus four seed-0 random windows with at least three valid aircraft. Each scene uses one joint rollout minimizing the equal-aircraft mean 3D ADE over targets' full valid future horizons; this visualization selection does not change M6 metrics. OSM tiles use a descriptive User-Agent and are reused from ignored `data/basemap_cache/`. Rendering needs network access only for uncached tiles. Draft v1 stops after one visual self-review, pending Yi-Chung's feedback.

The BC checkpoint comes from the historical 128-epoch M5 run. Future `air_pre_bc` config uses 20 epochs and keeps best val/loss; the Demo never starts training. Historical GPU acceptance scripts are not Demo entry points. The old M6 training wait/verification scripts are disabled.

Timing: 3 Hz, 6 s history, 2 s tokens, 30 s future. The committed dictionary is K=2048/shift=6/causal chord heading. Rollout starts at tokenized current pose and never reads future GT. ADE/FDE average the 32 rollouts; minADE picks the best full trajectory per agent independently for horizontal/vertical errors. Miss rate is the fraction of eligible agents whose 32 endpoints all exceed 500 m horizontally; the mean per-rollout miss fraction is also shown. Eligibility requires every GT frame through the horizon. Counts refer to agent-window instances.

```bash
scripts/test.sh -q
```

Geometry uses float64; neural features use float32. Unit conversions are centralized in `src/common/units.py`. Heights are altitude minus 166 m; altitude-fallback flags are preserved. Flights may continue across midnight at split boundaries. Raw data stays outside this repo; all generated data and outputs are ignored. No Git remote is configured.

Copied SMART/CAT-K code retains original headers; see LICENSE and LICENSE.SMART. Dataset provenance: ADSB.lol historical traces (ODbL 1.0), as documented in the source dataset README. Historical dictionary reconstruction results (day 10-03) are in `docs/motion_dict.md`; they are not forecast errors.
