# M10 runway-anchor experiment

`runway_anchors` defaults to false. This leaves the legacy parameter/buffer keys
and numerical path intact, including strict loading of `epoch_016.ckpt`. True
adds 16 learned runway-end identities, fixed metre-to-internal-unit coordinate
buffers, a geometry embedding and four bipartite attention layers. No identity
dropout or taxiway/polyline features are included. Geometry is recomputed from
the current sampled pose at every rollout call, with one edge per anchor per
valid agent-time node. All scenes share the same 16 ORD sources.

`assets/ord_runway_ends.csv` contains the plan's fixed OurAirports public-domain
coordinates (8 active runways, two endpoints each). XY uses WGS84 geodetic/ECEF
ENU at the dataset origin; Z uses the train-only median processed height near
the finite runway segment, with all-runway fallback. Heading is radians from
ENU east toward north, pointing from each endpoint toward its opposite end.
The unused 4L/22R runway remains among the anchors.

The supplied `../m10_reference` directory is read-only. First run its reference
alignment script unmodified using catk Python `-I` and compare stdout byte for
byte. `scripts/catk.sh -m scripts.runway_alignment` requires that gate first,
then applies the exact corrected masks with the precise-ENU coordinates and
checks ±5% per runway. Neither gate can be tuned to pass.

Training commands (outside sandbox, after tests and idle GPU checks):

```bash
scripts/run_runway_job.sh base20 1
scripts/run_runway_job.sh anchor 3
```

Run these as independent parallel processes. Each exposes only its named GPU,
checks idle/PIDs, verifies CUDA PCI->UUID, and uses seed817, batch32/16, workers4,
full train batches and20epochs from scratch. Experiments use the original lr,
weight decay, scheduler, label smoothing and dropout. A separate CSV selects the
488 existing10-03 validation windows; no original split is rebuilt or modified.
Best checkpoints are selected only by10-03 val loss, never10-04 metrics.

After both jobs complete, evaluate each model on CPU:

```bash
scripts/catk.sh -m scripts.evaluate_runway_model --model epoch_016
scripts/catk.sh -m scripts.evaluate_runway_model --model base20
scripts/catk.sh -m scripts.evaluate_runway_model --model anchor
scripts/catk.sh -m scripts.runway_attention_figure
scripts/catk.sh -m scripts.summarize_runway
scripts/catk.sh -m scripts.verify_runway_experiment
```

Independent CPU evaluations can run concurrently. The evaluator reuses existing
M6 rollout/metric and M8 isolated-token scoring functions unchanged. M8 token
identities/residuals must reproduce the original28413 rows. Raw arrays/metrics,
training completion records and final comparison tables are under
`outputs/runway_anchor`. The attention figure uses6s observed history only and
pre-existing cached map tiles. Its head/layer-mean weights are diagnostic, not
causal explanations. No demo uses the new weights without a separate decision.

Flow labels provisionally follow the supplied reference exactly: nominal2s
compressed-valid ground displacements over30m/s, minimum3 roll frames per track,
median normalised dx with magnitude>.9, then scenario majority vote. This
reproduces10-03east68/west0 and10-04east15/west39, skipping37none/tie on10-04. It
approximates E-W runway usage by direction, without a centreline membership mask.
The interpretation and advisor response are recorded in QUESTIONS.md.
