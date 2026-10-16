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
