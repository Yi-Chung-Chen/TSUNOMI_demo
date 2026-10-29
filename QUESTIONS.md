# Questions

## Environment — pending

GPU validation is unavailable inside the sandbox (`nvidia-smi` driver communication failure). No GPU stage can be marked complete here. Only GPUs 1,3 are authorized.

M0: approved pytest installation failed (DNS unavailable; catk site-packages read-only). Advisor asked about an isolated temporary test harness; awaiting response.

### GPU execution escalation — rejected by tool policy

Exact command attempted: `CUDA_VISIBLE_DEVICES=1,3 nvidia-smi` with `sandbox_permissions=require_escalated`.
The tool rejected the request before execution: `sandbox_approval: false`; escalated permissions cannot be requested in this session. This is not GPU occupancy evidence. No GPU process has been launched. User/session execution permission changes are required before M5 GPU acceptance can pass. CPU implementation and preprocessing remain unblocked.

### Environment update — GPU execution access resolved

User changed permissions to Approve for me. Escalated `CUDA_VISIBLE_DEVICES=1,3 nvidia-smi` succeeded at 2026-10-16 02:48 local: physical GPU 1 at 92%, GPU 3 at 98%, both occupied. Do not launch training until availability is confirmed. The earlier escalation restriction is resolved.

### M0 advisor response — ERROR

The advisor script failed: `ERROR (exit 1)` / API unreachable (`EAI_AGAIN`). Question: isolated temporary pytest harness vs external install. Temporary harness passed 1 test; standard approved pytest installation is being retried with newly available escalation. No design decision depends on the advisor response.

## M1/M3 advisor call — rejected by automatic review

Question prepared: full-dataset extrema differ from plan preliminary statistics; recommend preserving raw ingest. M3 K-disk tolerance unspecified: propose configurable 0.5 internal units, reduced if clustering exhausts before exact K; reserve token zero exactly stationary. Horizontal outlier cap needs advice.

Automatic review rejected sending these dataset statistics and implementation details through the advisor script to the advisor, stating external payload export needs explicit authorization. User was asked for approval; no workaround attempted. M1 acceptance counts and M2 are unaffected.

### Advisor access — resolved

User explicitly authorized all implementation questions to the advisor. Retry succeeded; M1 statistics retained and M3 tolerance/filtering/sampling decisions recorded in PROGRESS.md. The advisor is read-only; Codex implements.

## M5 resource wait — unresolved

GPU access is working, but authorized physical GPUs1/3 both contain active compute processes. Per user instruction, do not share or terminate them. Recommendation: wait until one is unoccupied, then run M5 acceptance and full BC automatically. Only Yi-Chung can authorize changing resources. M6/M7 depend on M5 gate and remain unstarted.

## M5 runner ERROR — 2026-10-16T05:55:35-04:00

Exit code: 1. See outputs/m5_execution/runner.log, smoke.log and bc.log. M5 is not complete; M6/M7 must wait.

## M5 checkpoint gate — resolved provisionally

Full GPU run finished 128 epochs / 98304 steps; previous ERROR was an incorrect verifier assertion that last.ckpt must contain epoch127. Lightning2.4 save_last=link retained best epoch16. The advisor approved verifying complete CSV/log evidence and using best epoch16 checkpoint for M6; final epoch127 weights are unavailable. Future config now uses save_last=true. No resource change or rerun requested. Full Q/A in ../SYGARD_QA_LOG.md.

## M6 resource wait

Implementation and27 tests pass, including a32-rollout CPU integration. GPU1/3 both occupied at100%, about45GB each. Wait for one to be entirely unoccupied; no sharing/termination authorized. M7 waits for M6 GPU gate. Any runner ERROR is appended here automatically.

### M6 CPU checkpoint load ERROR — resolved

Copied run.py deserialized GPU checkpoint storages without map_location, failing under CPU sandbox. Fixed to map_location=cpu before Trainer placement; CPU train+32-rollout validation now passes. This prevents loading onto a different logical GPU than the selected one.

## Prior M6 resource wait — superseded

User replaced CAT-K/GPU M6 with BC-only CPU Demo, merged val/test indexes without rebuilding, and prohibited retraining. Stopped waiting tmux before training; no GPU wait remains. Old M6 resource questions/runner cancellation are closed by the new scope. Metrics and plot ambiguities resolved provisionally via the advisor; full Q/A in ../SYGARD_QA_LOG.md.

## M6 Demo complete

No unresolved blocker. Full CPU Demo passed; protected inputs unchanged; no training. Provisional metric/plot details and validation-selection disclosure are documented in PROGRESS.md and docs/demo_results.md. Old GPU/CAT-K wait is closed.

## M8b speed/course smoothing — provisional rule to add to plan

Yi-Chung explicitly authorized the smoothing correction after the aa0300 false
positive. The current plan specifies only spatial3×3 pooling and alpha; please
record the new categorical smoothing rule in §4.1/M8b.

Options considered: physical-distance smoothing with a new speed bandwidth and
an artificial centre for the open speed tail; or a fixed adjacent-bin kernel.
Recommendation and advisor DECISION (provisional, Yi-Chung to confirm): use
`[1,2,1]/4` on the existing bins, circular for course and reflecting for speed.
Apply after spatial pooling, before alpha; preserve counts and normalisation.
Speed edges send3/4 mass to themselves and1/4 to their only neighbour. This is
by bin index rather than uniform m/s; there is no new bandwidth/tail centre.
Do not smooth height/joint or tune to the drone/aa0300. Recompute10-03 reference
parameters and10-04 p95; compare with frozen old calibration/cutoffs as well.

Implementation proceeds under the user's authorization and binding provisional
advisor rule. No work is blocked; plan documentation remains pending Yi-Chung.
Full consolidated Q/A is in parent SYGARD_QA_LOG.md.

## M8b height100 revision — user override and provisional run definition

Yi-Chung requested merging <15 and15–100m into a single <100m layer to
investigate normal landing/taxi false positives, then explicitly authorized
implementation (“處理一下吧”). Please update plan §4.1/M8b: raw-count height
edges are now100/300/600m for this separate variant. The speed/course kernel
recorded above remains unchanged. Preserve both prior result directories.

Advisor DECISION (provisional): merge raw train counts before spatial pooling,
fixed speed/course smoothing and alpha; joint denominator now has four height
categories. Fit on train only, recalibrate on10-03, compare frozen baseline
calibration/cutoffs as well as recalibrated outputs. Report drone count/median
state NLL in each physical height group. Stationary course remains zero.

New reporting convention: group alert runs by(scenario,agent), consecutive
flagged starts must differ by6 frames; missing/unflagged tokens break runs.
Duration=2seconds×token count. Report max/median duration and fraction of
scenario-agent tracks with any run≥6seconds, separately by aircraft phase.
The advisor referred to aabc11/aa0300 as10-04 examples, but the actual
scenario000667/000536 records and GIF labels are10-03.10-04 aggregate
results have also informed revision and are inspected development evidence,
not a pristine holdout. No blocker; full Q/A in ../SYGARD_QA_LOG.md.

## Camera-derived trajectory rollout Demo — 2026-10-27

Yi-Chung explicitly authorized connecting camera trajectories to the frozen
trajectory model and making camera/prediction/GT rollouts, with an arbitrary
sample rather than minADE. This overrides the prior camera-out-of-scope text
for this demo only; no training or detector changes. Plan documentation of the
adapter is pending. Full advisor Q/A is in ../SYGARD_QA_LOG.md.

Advisor DECISION (provisional, binding pending review): same timestamp takes
largest event_index; resample causally from last known camera posterior plus
that posterior's velocity×age, age≤1s. No future camera interpolation. Require
19 valid6s history states,≥3 distinct measured timestamps, latest measurement
≤1s at forecast origin, and skip uninitialized zero-velocity rows before the
second measurement. Raw heading uses camera Kalman velocity, preserving last
usable value below1m/s; existing causal chord tokenizer unchanged. Ground is
camera-only height z+2250−166: enter<10m, exit>20m, initialize<10m; report flips.
All types UNK. First eligible origin per offline-mapped independent tracker;
single-agent scenes avoid duplicate tracker IDs interacting with themselves.
GT associations only pair overlays, never infer inputs/stitch IDs/select best
prediction. Sample0 of1, topk5/temp1/seed817; full30s forecast despite shorter
camera visibility. Audit unmapped/insufficient tracks; disclose upstream
Kalman predictions and segmentation-based offline identities. New output:
outputs/camera_demo. Thresholds are unvalidated demo proxies, not tuned to GT.

Execution notes: initial advisor call was rejected by automatic approval
review for external sharing of potentially private paths/details. Yi-Chung
explicitly replied “允許傳給顧問”; retry succeeded. nvidia-smi reports driver
unavailable in this environment; inference proceeds on CPU. Initial smoke
inference completed but map rendering hit sandbox DNS restriction for missing
OSM tile; rerun requests normal reviewed network access for public map tiles.

## Integrated v1 notes — 2026-10-29 (no advisor call)

No blocking design question required a user decision. Optional choices were
made in D0 and recorded in PROGRESS.md. Time-conditioned density was deferred;
caption explicitly says time-of-day/weather are not anomaly-detector inputs.

D5 network approval review rejected a command that might fetch OSM tiles due
to possible location disclosure. Resolved with a strictly offline mosaic
loader using existing cached zoom14tiles; cache misses raise without HTTP.
No retry to transmit scene coordinates, no new network authorization needed.

Review points after v1 (not implemented beyond approved rules):
- Future anomaly starts later than current anomaly in both stories; predictive
benefit demonstrated here is protected-zone entry warning, not earlier initial
anomaly detection. CV entry at52s precedes model>=8/32high at56s; do not claim
model superiority based on this illustration.
- After B leaves the demo volume and no longer approaches/predicts entry, the
approved instantaneous zone rules can return threat toLow despite earlier
entry. A persistence/incident-state rule would require a later scope decision;
no unrequested threat latch was added.
- No graphical browser executable is available for automation. Offline viewer
JS syntax and DOM/canvas-stub interaction tests pass; visual review uses the
matching rendered GIFs/PNGs. Browser-specific layout remains a user-review item.
