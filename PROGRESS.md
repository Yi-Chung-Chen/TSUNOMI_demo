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
