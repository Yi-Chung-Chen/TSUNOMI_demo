# sygard-motion

Aircraft trajectory prediction on the 800-scenario SYGARD ORD dataset. Work follows `../SYGARD_MOTION_PLAN.md`. Raw data stays outside this repository; all processed outputs are ignored.

Use the existing `catk` environment through `scripts/catk.sh`. Training uses only physical GPUs 1 and 3 after verifying availability. No remote is configured.

Timing: 3 Hz, 6 s history, 2 s tokens, 30 s future. Split by UTC date; flights may continue across midnight at split boundaries. Heights use altitude minus 166 m and retain altitude fallback flags.

Code attribution: copied SMART/CAT-K files retain their original headers; see LICENSE and LICENSE.SMART. Raw data originates from ADSB.lol historical traces (ODbL 1.0); see the source dataset README for provenance.

Implemented preprocessing commands (run from this repository):

```bash
scripts/catk.sh -m src.preprocess.ingest --workers 4
scripts/catk.sh -m src.preprocess.split
scripts/catk.sh -m src.preprocess.windows
scripts/catk.sh -m src.preprocess.motion_dict
scripts/catk.sh -m src.preprocess.motion_dict_eval
scripts/test.sh -q
```

The committed dictionary is K2048, shift 6, causal chord heading. See [motion dictionary validation](docs/motion_dict.md) for the full train-only vocabulary sweep and val reconstruction results. Dictionary selection never uses test. Forecasting will use only aircraft present at current time; future entrants are excluded.

Geometry uses float64 during nearest-token matching and detokenization for consistent NumPy/Torch reconstruction. Trainable neural embeddings use float32. Position scaling and altitude reference conversions are centralized in `src/common/units.py`.
