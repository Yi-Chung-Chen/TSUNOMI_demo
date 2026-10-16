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
