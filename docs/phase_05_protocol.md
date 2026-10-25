# Phase05 protocol and runner handoff

The frozen final matrix completed on October 26, 2026 (Asia/Manila). All 24 stages and 70,000 planned Adam steps passed independent saved-record readback. Phase05 is completed locally and awaits root closeout review; analytical accuracy and conservation remain Phase06 work. The preparation and recovery details below describe the unchanged protocol used for this execution.

## Frozen study

`configs/final_protocol.json` records protocol `core-v1-pilot-a03`, version 1, with content digest `e73e9a1bd6973cebaeb62435cb3e96da57f3bf6308afb0694a2c36bc916216db`. Its file SHA256 is `1382e09f2acb3d6da92e8089f36dbfe0cd12719e79f91d813791cb7f877c9904`.

The 24 core stages are two analytical references, eight unlimited Lax-Wendroff grids, seven training jobs and seven physical predictions. E1 fixed/adaptive pairs use seeds 11/22/33; E2 baseline uses seed 11 and exactly zero prescribed downstream depth. Training retains CPU float32/four threads, four 64-unit tanh layers, 10,000 interior/1,000 initial/500 points per boundary and 10,000 Adam steps at 0.001. LBFGS remains disabled. The exact clipping/EMA/common-parameter gradient rule and fixed separate-validation checkpoint selection remain unchanged.

Final evaluation has 400 centres and 50 positive times 0.01 through 0.5 s. Validation retains its separate 73-centre/five-time coordinates. No final analytical errors select a model or setting.

Conservation axes are materialized in the protocol: midpoint 2,000/4,000 positions at t0 and all 50 reported times; trapezoidal 1,001/2,001 time nodes at x=-5,5, including 0,T and every reported time. There are 36 planned auxiliary IDs for the nine reference/prediction sources. Runtime bindings require actual configuration and checkpoint hashes; no nonexistent checkpoint hash or successful auxiliary output is prefilled. Numerical mass uses native cell sums and saved stepwise interface flux.

The loader enforces the protocol digest, complete matrix, scientific input bytes, 11 core source modules, locked-environment files and installed Python/PyTorch/NumPy/SciPy identity. CLI/entrypoint hashes and Git revision remain historical provenance. Later additive evaluator/reproduction commands and separate modules can coexist with these unchanged scientific sources. Changed core code, scientific inputs, lock or runtime are rejected.

## Execution and recovery

`run-matrix --protocol configs/final_protocol.json` creates the real incremental manifest only when invoked for execution. An OS advisory lock excludes another writer. The manifest records explicit unexecuted/running/completed/failed/blocked/interrupted states, real attempt IDs, hashes, checkpoint selection, stopping reasons and step progress at 1/100/.../10,000. Workflow and scientific status are separate. A finite budget stop establishes no convergence or accuracy.

Completed records are verified before reuse. A completed run left behind by a missed manifest update is reconciled; unowned existing directories are refused. Only a saved `NumericalRunFailed` with its diagnostic category is classified as a numerical research outcome. Reference failures block their dependent stages. Artifact readback errors retain both the original error and observed record status instead of masking the failure.

A compatible failed/interrupted training attempt can resume explicitly with `--resume-stage STAGE_ID --resume-from SOURCE_RUN_ID --resume-run-id NEW_RUN_ID`. The source must be a recorded attempt of that stage, use frozen science/software/partitions and have fewer than 10,000 completed steps. The new attempt preserves ancestry and the old files. A source still marked running, or a noncompleted source already at the frozen cap, requires explicit technical recovery; the runner changes neither its status nor the budget.

## Separate paper execution API

`execution_plan(protocol, namespace=..., output_root=..., manifest_path=...)` maps all 60 core/auxiliary IDs while retaining the parent protocol digest. `run_matrix` accepts the corresponding optional `execution_namespace`, `output_root` and `manifest_path` arguments. Destinations stay inside the project. Overrides require a namespace; a completed namespace cannot be presented as a fresh execution.

The ordinary CLI retains its main-study destinations. A future paper caller can choose, for example, namespace `paper-v1`, output root `reproduction/paper-v1/runs` and manifest `reproduction/paper-v1/manifest.json`. This derives new actual IDs and destinations from the same frozen stage identities and scientific settings. Resumed actual checkpoint bindings remain execution-specific. The namespace fixture proves original/fresh scientific checkpoint agreement and preservation of the original protocol and main outputs. No actual paper reproduction was launched.

## Verification and boundary

Ten protocol checks passed in 21.57 s; eight final runner checks passed in 52.05 s. The earlier seven-test runner attempt remains recorded before the known paper namespace requirement was added. Bounded runner fixtures invoke the actual reference, solver, training and prediction APIs; only the strict production-protocol boundary is replaced with an explicitly labeled tiny software fixture. Those fixtures establish no final study result.

CLI help and incomplete-resume argument checks passed. The actual freeze command exited 0. Independent readback verified 24 core stages, three pairs, 70,000 planned Adam steps, 36 auxiliary specifications, exact quadrature axes, all scientific hashes and absence of final execution. All 364 original inventory files and 366 files from prior runs remain unchanged. Scoped Ruff/format and Git diff checks passed.

Preparation evidence is under `docs/evidence/phase_05/`: test records, CLI/freeze command outputs, `freeze_readback.json`, preservation inventory and executed checklist copy. These prelaunch records retain their historical state. Root owns technical acceptance and commits.

## Actual matrix execution and independent readback

Root accepted the launch in `docs/coordination/phase_05_launch_review.json` at source commit `aca15d3df406fc906ae5b2a51804e83ed52a8bef`. The unchanged command `rtk proxy uv run --frozen python -m hydro_pinn run-matrix --protocol configs/final_protocol.json` exited 0 at `2026-10-25T16:58:25.567746+00:00`. Its measured command wall time was 6,955.021 s (1 h 55 min 55 s). The append-only combined output, launch/session records and command result remain under `docs/evidence/phase_05/`.

`docs/final_run_manifest.json` reports all 24 expected stages completed, with no failed, blocked or interrupted stage and no retry or resume. Both references, all eight numerical grids, seven full training jobs and seven predictions have saved records. All four E1 classical comparators are valid for later evaluation.

The completion helper ran after the writer exited and returned 0. Its report, `docs/evidence/phase_05/matrix_readback.json`, independently verifies 24 stage identities, seven 10,000-row Adam histories totaling 70,000 steps, 17 required output records, all three matched pairs, source/input/runtime bindings and exact final output axes. Each pair retains identical sampling/validation partitions and the same budget, architecture, backend and selection rule. Adaptive weight changes occur only at the 100 scheduled boundaries; baseline weights remain fixed.

| Case | Variant | Seed | Completed steps | Selected step | Separate-validation objective | Training and saving (s) |
|---|---|---:|---:|---:|---:|---:|
| E1 | baseline | 11 | 10,000 | 6,512 | 0.0046532921 | 2,226.985 |
| E1 | adaptive | 11 | 10,000 | 8,626 | 0.0047156028 | 1,492.418 |
| E1 | baseline | 22 | 10,000 | 1,387 | 0.0124579146 | 469.741 |
| E1 | adaptive | 22 | 10,000 | 8,385 | 0.0052585980 | 483.522 |
| E1 | baseline | 33 | 10,000 | 4,126 | 0.0062952083 | 796.237 |
| E1 | adaptive | 33 | 10,000 | 4,516 | 0.0081221818 | 698.995 |
| E2 | baseline | 11 | 10,000 | 9,405 | 0.0001432969 | 762.413 |

These objectives are the predeclared separate-validation physics/IC/BC objective, not analytical solution errors. Reloading each saved selected parameter state reproduced its logged selected objective exactly in this readback, within the predeclared float32 tolerance. Physical predictions match a fresh forward pass from those selected states exactly. Every job stopped because its finite Adam budget was exhausted; convergence is not established. The last optimizer state and selected prediction state remain distinct and preserved.

Training and saving totaled 6,930.311 s; manifest training-attempt wall time including runner readback totaled 6,932.614 s (1.926 h). This is below the pilot's 7,360.195–22,390.759 s observed-rate envelope. Host conditions varied and the pilot envelope was neither a bound nor a confidence interval. Using 6,932.614 s for another 70,000-step paper run is only an arithmetic projection under comparable conditions; fresh setup, nontraining stages and later host/thermal changes are excluded. No full paper retraining was performed here.

Readback verified all 364 original inventory files and 366 prior-run files against their saved hashes. It tracked 167 main-run artifacts and confirmed that neither these artifacts nor the terminal manifest changed during verification. The protocol digest and frozen scientific modules remain unchanged. CLI provenance records the additive Phase06/07 work that was permitted during training.

The full conservation specification and 36 auxiliary destinations remain frozen. Actual auxiliary predictions, analytical errors, conservation metrics, figures and comparative timing remain for Phase06 production execution after root review. This closeout establishes execution and saved-record integrity; it does not establish model quality, scientific superiority, reproduction, collaborator rerun or course acceptance.
