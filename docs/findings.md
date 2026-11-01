# Frozen core study findings

Recorded October 26, 2026 (Asia/Manila). These are local results from the completed frozen core study. This dated report distinguishes original execution from saved-output replay and fresh training reproduction; current instructions are in the [reproducibility guide](../REPRODUCIBILITY.md).

Within each case, the N=400 Lax–Wendroff comparator has lower pooled depth and discharge errors than its tested PINN runs. Adaptive weighting improves pooled errors for seed 33 and worsens them for seeds 11 and 22. The trained networks produce the requested fields faster than fresh integration in this timing window, while cached classical reconstruction and exact-reference generation are cheaper. Host load, precision, training cost, and approximation quality limit that timing comparison.

## Evidence and measurement scope

The protocol is `core-v1-pilot-a03`, digest `e73e9a1bd6973cebaeb62435cb3e96da57f3bf6308afb0694a2c36bc916216db`. The terminal main manifest SHA256 is `8f8b755024877061e119e8b9befcf0e2fd8d63bcb5f3031d21440f8850c77f2d`. All 24 stages completed, including seven 10,000-step training jobs. No checkpoint, training budget, mask, or scientific setting was changed after final results were examined.

The comparison uses 400 common centres and 50 times from 0.01 to 0.5 s. Reference and PINN outputs are point values; numerical cell averages use the prescribed right-cell piecewise-constant reconstruction. All three E1 seed pairs are retained. E2 has one baseline seed and supports a diagnostic result only.

Saved scientific records: [metrics JSON](../results/metrics.json), [metrics CSV](../results/metrics.csv), [timings with raw samples](../results/timings.json), and [figure manifest](figure_manifest.json). The historical pooled-summary and independent-readback records are not distributed separately; metric definitions, source bindings and raw records are retained in the listed scientific files and the `core-data` bundle in [ARTIFACTS.json](../ARTIFACTS.json).

## RQ-01: Depth and discharge accuracy

Pooled errors below weight all 20,000 space-time points equally. RMSE units are m for depth and m²/s for discharge per unit width. Relative L2 values are percentages. The table contains every numerical grid and every trained model. N=400 is the predeclared primary comparator.

| Run | h RMSE (m) | h relative L2 (%) | q RMSE (m²/s) | q relative L2 (%) |
|---|---:|---:|---:|---:|
| E1 Lax–Wendroff N=100 | 0.024052 | 3.477 | 0.074620 | 22.976 |
| E1 Lax–Wendroff N=200 | 0.015776 | 2.281 | 0.053978 | 16.620 |
| E1 Lax–Wendroff N=400 | 0.009937 | 1.436 | 0.036397 | 11.207 |
| E1 Lax–Wendroff N=800 | 0.007940 | 1.148 | 0.029314 | 9.026 |
| E2 Lax–Wendroff N=100 | 0.016564 | 2.421 | 0.037544 | 12.546 |
| E2 Lax–Wendroff N=200 | 0.008641 | 1.263 | 0.021914 | 7.323 |
| E2 Lax–Wendroff N=400 | 0.004292 | 0.627 | 0.012156 | 4.062 |
| E2 Lax–Wendroff N=800 | 0.002975 | 0.435 | 0.008521 | 2.847 |
| E1 PINN baseline seed 11 | 0.045117 | 6.522 | 0.116536 | 35.882 |
| E1 PINN adaptive seed 11 | 0.046568 | 6.732 | 0.118281 | 36.419 |
| E1 PINN baseline seed 22 | 0.035467 | 5.127 | 0.102817 | 31.658 |
| E1 PINN adaptive seed 22 | 0.046289 | 6.691 | 0.119215 | 36.706 |
| E1 PINN baseline seed 33 | 0.044274 | 6.400 | 0.114309 | 35.196 |
| E1 PINN adaptive seed 33 | 0.041898 | 6.057 | 0.109194 | 33.621 |
| E2 PINN baseline seed 11 | 0.008948 | 1.308 | 0.015786 | 5.275 |

The complete run IDs use prefix `core-v1-pilot-a03-`; their exact identifiers, source paths, and hashes are in the metrics. Analytical self-comparison is zero by construction and does not independently verify the reference implementation. Reference verification was completed in earlier phases.

## RQ-02: Shock/front regions and velocity

The shock/front bands retain their frozen ±0.1 m half-width. At 0.5 s, E1 numerical N=400 shock-band RMSE is 0.050410 m for h and 0.207057 m²/s for q. Across all six E1 networks, the corresponding ranges are 0.146383–0.186856 m and 0.524341–0.628967 m²/s. Whole-domain and outside-band errors remain in the metrics and figures; these difficult regions were never excluded. The profiles and error maps also expose the rarefaction and reservoir regions. No second-order convergence claim is made across a discontinuity.

At 0.5 s the E2 PINN whole-domain depth RMSE is 0.004392 m and discharge RMSE is 0.011298 m²/s. Its reference-wet velocity RMSE is 0.779742 m/s and relative L2 is 29.560%, using 319 of 400 positions. Thus the smaller h/q errors do not establish comparable velocity accuracy near shallow water. The wet mask comes only from reference h>0.001 m.

The late E2 dry-front band has no reference-wet velocity points at 0.5 s, so its velocity metrics are null with `no_selected_points`. Grid alignment can leave isolated eligible earlier times; figures show those points and gaps without interpolation. Early zero reference norms elsewhere also retain null plus a reason. The exact dry reference and numerical initial state keep hR=0. The PINN retains its disclosed positive network floor and cannot represent an exactly dry state. No threshold-derived front is substituted for the analytical front.

## RQ-03: Conservation, initial bias, and boundary mismatch

PINN/reference volumes use the separately saved 2,000/4,000-cell midpoint grids at t=0 and all 50 report times. Endpoint discharge uses 1,001/2,001-node trapezoidal time grids. All 36 auxiliary records bind the evaluation configuration and source; neural records also bind the selected checkpoint. The reported primary level is 4,000/2,001. Native numerical balance instead uses cell sums and actual numerical interface fluxes over every accepted step.

Signed imbalance is M(t)−M_IC+∫[q(b,s)−q(a,s)]ds, with prescribed M_IC=5.5 m² for E1 and 5 m² for E2. The maximum normalized PINN imbalance below is over t=0 plus the 50 report times. The native maximum covers its accepted-step history. These are sampled maxima, not continuous-time bounds.

| Model | Initial volume error (m²) | Max abs normalized imbalance (%) | Final signed imbalance (m²) | Peak endpoint mismatch (m²/s) | Max base/doubled imbalance difference (m²) |
|---|---:|---:|---:|---:|---:|
| E1 PINN baseline seed 11 | -0.021883 | 0.4344 | -0.023796 | 0.003926 | 1.27e-07 |
| E1 PINN adaptive seed 11 | -0.039442 | 0.7171 | -0.036101 | 0.002200 | 2.3e-06 |
| E1 PINN baseline seed 22 | -0.075372 | 1.8073 | -0.099404 | 0.013969 | 3.3e-08 |
| E1 PINN adaptive seed 22 | -0.038896 | 0.7072 | -0.022410 | 0.001639 | 3.57e-08 |
| E1 PINN baseline seed 33 | -0.054491 | 1.2707 | -0.067020 | 0.005613 | 3.82e-08 |
| E1 PINN adaptive seed 33 | -0.031122 | 0.7714 | -0.036953 | 0.001658 | 3.02e-08 |
| E2 PINN baseline seed 11 | 0.000268 | 0.3325 | 0.012836 | 0.005029 | 3.02e-05 |

Across all eight numerical grids, maximum normalized native imbalance is at most 3.23×10⁻¹⁶, consistent with floating-point cancellation in this computation. This conservation result does not remove the saved dispersive approximation errors. All numerical runs remain finite and their stability/minimum-depth/CFL diagnostics are retained.

Initial-volume error accounts for a substantial part of several neural imbalances. Endpoint mismatch remains visible separately from the corrected balance; correction for leakage does not make the network conservative. Adaptive maximum imbalance improves for seeds 22 and 33 and worsens for seed 11.

The analytical E1 reference also shows midpoint-quadrature sensitivity at its moving shock: maximum base/doubled volume difference is 7.44×10⁻⁴ m². Its exact physical volume is constant; this is a quadrature effect. The E2 reference difference is 2.12×10⁻⁵ m². For the seven neural models, the largest base/doubled imbalance difference is 3.02×10⁻⁵ m². These are sensitivity measurements, not certified error bounds.

## RQ-04: Measured computational cost

All comparable workloads produce physical h and q at the same 400×50 coordinates. Each repeated group has ten warmups and thirty measured samples, with median/IQR and every raw sample retained. Device synchronization brackets the measured interval; CPU execution is synchronous. Numerical timing includes fresh prescribed-state integration, its implemented diagnostics, and reconstruction. Network timing uses resident coordinates and the selected model; loading and copies are separate. Plotting and error calculation are excluded.

The available machine is identified in the recorded hardware clarification as the M5 Air. The benchmark records CPU execution, four PyTorch threads, ten interop threads, and ten logical CPUs; its platform CPU-model probe returned no model. A separate system-model query was unavailable. Both methods use this local CPU, with float64 numerical arrays and float32 networks. No CUDA/Victus result is inferred.

Host contention limits these timings. Study training, rendering, and reproduction were stopped, but sampled 1-minute load averages ranged from 16.95 to 20.37 on ten logical CPUs (18.27 before and 17.53 after). Load averages lag and do not identify competing processes. Other host processes were not controlled. Workloads ran sequentially, and several IQRs are large. These observations do not isolate algorithmic speed, precision effects, or a stable hardware-independent latency.

| Case/workload | Median (ms) | IQR (ms) |
|---|---:|---:|
| E1 exact-reference generation | 2.302 | 1.462 |
| E1 N=400 fresh integration + reconstruction | 111.147 | 65.758 |
| E1 N=400 resident cache reconstruction | 0.267 | 0.286 |
| E2 exact-reference generation | 4.227 | 4.081 |
| E2 N=400 fresh integration + reconstruction | 94.026 | 26.045 |
| E2 N=400 resident cache reconstruction | 0.220 | 0.144 |

| Model | Original training/save (s, one run) | Resident forward median / IQR (ms) | Forward + host export median / IQR (ms) | Validated CPU loading (ms, one observation) |
|---|---:|---:|---:|---:|
| E1 baseline seed 11 | 2226.98 | 27.413 / 3.112 | 27.597 / 11.981 | 51.154 |
| E1 adaptive seed 11 | 1492.42 | 36.468 / 22.402 | 38.149 / 13.380 | 38.470 |
| E1 baseline seed 22 | 469.74 | 40.220 / 23.633 | 65.940 / 39.670 | 53.029 |
| E1 adaptive seed 22 | 483.52 | 36.584 / 18.193 | 75.397 / 108.706 | 38.502 |
| E1 baseline seed 33 | 796.24 | 55.376 / 22.963 | 77.189 / 50.965 | 28.136 |
| E1 adaptive seed 33 | 699.00 | 53.424 / 33.969 | 49.457 / 35.066 | 32.837 |
| E2 baseline seed 11 | 762.41 | 42.352 / 20.955 | 46.314 / 20.613 | 85.980 |

The seven original training/save records total 6930.31 s (1.925 h). Training was not repeated to measure inference. The historical job durations vary substantially and do not isolate adaptive-update overhead. Loading measurements are single setup observations, not cold-start distributions. Input creation/transfer, CPU model movement, and repeated output copy costs remain explicit in the timing JSON.

The recorded fresh N=400 solver-median/network-median ratios range from 2.01 to 4.05 for E1 and equal 2.22 for the single E2 model. They describe these implementations, precisions, output locations, and contended sequential window. They must be read alongside the larger neural errors. Cached classical reconstruction is faster than all measured neural forwards for this repeated scenario/grid, so no repeated-query break-even advantage is established. A new scenario requiring retraining includes its training and setup cost.

The [cost-versus-accuracy figure](../results/figures/timing_accuracy.png) places these medians and Q1–Q3 ranges beside pooled depth RMSE. It retains all seven networks, both N=400 fresh/cache comparators, and both exact references. It combines saved measurements and adds no timing samples.

## RQ-05: Fixed/adaptive comparison across all paired seeds

E1 pooled depth RMSE is 0.041619±0.005345 m for fixed weighting and 0.044919±0.002619 m for adaptive weighting. Pooled discharge RMSE is 0.111221±0.007362 and 0.115563±0.005536 m²/s respectively. These are descriptive mean±sample SD across the three actual seeds, not confidence intervals or significance tests.

| Paired seed | Adaptive−baseline pooled h RMSE (m) | Adaptive−baseline pooled q RMSE (m²/s) | Adaptive−baseline max normalized imbalance (percentage points) |
|---|---:|---:|---:|
| 11 | +0.001451 | +0.001745 | +0.2827 |
| 22 | +0.010823 | +0.016397 | -1.1001 |
| 33 | -0.002376 | -0.005115 | -0.4993 |

The direction of the error difference changes with seed. Conservation and approximation error also move differently within some pairs. These three pairs support a bounded description of this weighting rule under this budget; they do not establish consistent adaptive superiority. No seed was dropped and no final error was used to change training or checkpoint selection.

## Verification, preservation, and remaining acceptance

The independent metric review checked all CSV/JSON rows, all 347 source hashes, all 36 auxiliary records, sampled error arithmetic, fixed masks, and all 17 mass records. Core figure QA opened all 34 final PNGs and verified their 68 PNG/PDF hashes. Separate supplemental QA inspected the timing-versus-accuracy PNG, its two artifact hashes, and all 13 plotted points. The canonical manifest therefore records 35 figures and 70 exports. The core exporter's own manifest retains its original 34 entries. The first renders and their manifests remain preserved; only display layout and marker clarity were repaired. Metrics and source solution arrays were unchanged. Those historical review records are not distributed separately; the scientific records and figure manifest linked above retain the reported values, definitions and artifact hashes.

These are idealized, source-free, flat-bed generated cases with fixed initial/boundary conditions. The network is a coordinate solution for each case. No held-out scenario, field dataset, 2D/inverse extension, operational flood forecast, collaborator rerun, human rehearsal, instructor acceptance, or grade has been established. Finite optimizer budgets do not establish convergence.

Saved-output replay and fresh full-study reproduction are separate routes in the [reproducibility guide](../REPRODUCIBILITY.md). Replay rebuilds the 34 core figures, with the supplemental timing figure rebuilt explicitly from saved timings and metrics. It does not retrain the frozen experiments or establish an independent environment or course acceptance.