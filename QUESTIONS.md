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
