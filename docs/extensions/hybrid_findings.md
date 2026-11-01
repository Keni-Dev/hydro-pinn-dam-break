# Hybrid E1 feasibility pilot (historical record)

This record describes the initial 20-step-per-variant feasibility pilot, before the two 10,000-step fits. It supports a short-run software and resource finding and supplies no final accuracy result. The completed study is reported in [the full findings](hybrid_full_findings_v01.md).

The experiment uses one predesignated E1 seed-11 pair on the available M5 Air. Both variants retain the core four-layer, 64-unit tanh network, physical scaling, 10,000 PDE points, 1,000 IC points, 500 points at each boundary, fixed 1/10/10 physics weights, Adam settings, CPU float32 and four threads. The fresh control has no interior-label payload. The hybrid adds only a weight-1 mean of squared scaled h/q differences, summed over the two fields and averaged over 256 labels. Neither variant uses adaptive weights, viscosity or an architecture change.

The labels are generated idealized analytical E1 values, with CPU float64 arithmetic and the reference's bracketed wet-star root tolerance. They are not field observations. The independent PCG64 label seed is 110211. Training labels use `0 < t < 0.35 s`; the 128 separate validation labels use `0.35 ≤ t < 0.40 s`. Final axes contain 400 equal cell centres and 11 times from 0.40 through 0.50 s, including both endpoints. Coordinates, memberships, units, values, source versions, costs and hashes are saved in `runs/p11_hybrid_labels_v02/`. Its final artifact contains axes only. Physics collocation covers the full PDE domain, so the final window is held out from solution labels only.

Both variants select checkpoints by the same fixed physics/IC/BC validation score, including step zero and every Adam update. Ties retain the earlier checkpoint. Data-validation values and final solution values do not enter fitting, selection or tuning. The pilot evaluated no final analytical model errors.

Nine focused tests pass. The full-workload zero-data check gives exactly the same objective, all ten parameter-gradient tensors, model parameters after one Adam update and Adam state as the unchanged baseline path. The maximum gradient difference is zero. The saved pilot readback confirms identical initial parameters, physics training and validation partitions, separate checkpoint selection, 20 history rows per variant, no control labels and an exact copy of the declared training labels in the hybrid run. The three CLI commands are registered; the full evaluation branch is implemented and statically reviewed but has not executed. Pilot accuracy evaluation is refused before generating a reference.

| Observed pilot quantity | Control | Hybrid |
|---|---:|---:|
| Completed full-coverage steps | 20 | 20 |
| Training wall time | 1.011 s | 1.480 s |
| Objective/gradient median per step | 0.0310 s | 0.0332 s |
| Objective/gradient mean per step | 0.0331 s | 0.0488 s |
| Complete timed-step median | 0.0374 s | 0.0399 s |
| Complete timed-step mean | 0.0393 s | 0.0583 s |

Scaling the two medians to 10,000 steps each gives 12.9 minutes for the pair; the means give 16.3 minutes, and inclusive training-wall scaling gives 20.8 minutes. These are short-pilot projections, not a confidence interval. Hybrid step times varied from 0.0292 to 0.1610 s in a sequential run, so this pilot cannot isolate label overhead from host variation. Sustained thermal behavior and final evaluation cost remain unmeasured. Inclusive wall scaling contains memory-query overhead and short-run setup; the individual step timers exclude memory queries outside those timers. Analytical label evaluation took 0.00617 s; coordinate checks and array saving brought label work through saved arrays to 0.01149 s. The whole label CLI process took 1.889 s, including startup and metadata collection.

The process-lifetime peak RSS was 409,485,312 bytes, about 391 MiB. That cumulative peak includes earlier work in the same process and cannot assign separate peaks to the variants. Current RSS and the process inventory were unavailable because the sandbox denied `ps`. No free-memory, memory-pressure or within-step allocator-peak measurement is claimed.

Label run v01 remains intact. Its final x coordinates used mathematically equivalent arithmetic that differed from the core evaluator at 114 positions by at most 1.78e-15 m. Exact axis validation would reject it. The generator and extension evaluator now use the core's arithmetic; v02 is the declared input. No fitting used v01. Its generator source and all earlier test logs are retained. The first test attempt also retained an overly strict floating-point round-trip assertion; the corrected test uses a 2e-16 relative tolerance. Neither change used final model errors.

All 20 protected non-CLI core modules and the core archive SHA256 remain unchanged. The CLI diff contains 73 inserted lines and no removed lines; the original function statements remain unchanged and in order. Core defaults, original runs, manifests, artifacts and the paper archive were preserved.

After technical release, execute the frozen pair with the proposed command in `hybrid_manifest.json`, then run its separate evaluator. The evaluator saves final h/q and reference-wet velocity errors, shock-band and outside-band errors, fixed shock-position diagnostics, initial-volume bias, boundary-corrected mass, doubled-quadrature sensitivity, raw arrays, costs and integrity records. Full fits and this final evaluation must complete before any accuracy finding or extension completion claim. One E1 seed cannot establish training variability, new-case generalization or field validity.
