# Planar E1 full-fit findings v01

The released single-seed experiment completed, but planar recovery is incomplete. The unconstrained x/y/t network retains transverse flow and localized variation across y. Longitudinal discharge and velocity errors remain substantial, especially around the shock. These are results of the frozen experiment; no settings or checkpoint were changed after final analytical errors became available. This is planar consistency evidence only, with no genuine two-dimensional, field-validation or multi-seed claim.

Run `p10_planar_e1_seed11_full_v01` completed exactly 10,000 fresh Adam updates with exit 0. The terminal audit verifies every history row from 0 through 10,000, all optimizer counters, fixed 1/10/10 weights, saved independent memberships, initialization identity and the release/source/protocol bindings. The earliest strict minimum of the separate physics/IC/BC validation objective selected step 4,538. Actual checkpoint writes were initial 1, selected 1,032 and latest 10,001. The selected checkpoint SHA256 is `0e6bddd4e352e48b3fcdb9600c1d8db86d8cc9849ab9975a5db707d4dd69c00c`.

The saved-checkpoint evaluator completed with exit 0. Final arrays contain 50 times by 21 y centres by 400 x centres, with separate SI coordinate axes and h/qx/qy fields. All 1,050 time/y-slice records are retained. The reference-wet mask covers all 420,000 final points. The table aggregates the complete saved grid with equal cell weights; it does not omit the shock.

| Field or region | RMSE | Relative L2 error |
|---|---:|---:|
| Whole h | 0.04530 m | 6.548% |
| Whole qx | 0.11964 m²/s | 36.836% |
| Shock-band h | 0.15874 m | 53.012% |
| Shock-band qx | 0.48036 m²/s | 74.393% |
| Outside-band h | 0.03974 m | 5.698% |
| Outside-band qx | 0.09948 m²/s | 31.596% |
| Reference-wet u | 0.65994 m/s | 89.980% |
| Whole qy | 0.003285 m²/s | Undefined: zero reference |
| Reference-wet v | 0.011045 m/s | Undefined: zero reference |

Sampled maximum absolute transverse discharge and velocity are 0.028606 m²/s and 0.116840 m/s. Y variation is measured as deviation from the arithmetic y mean at fixed x and t. Its global RMS/max-absolute values are 0.008275/0.462755 m for h and 0.018834/0.672119 m²/s for qx. The largest depth deviation occurs at the saved point t=0.01 s, x=0.0625 m, y=-4.7619 m. The largest qx deviation also occurs at t=0.01 s near the initial shock, at x=0.0875 m and the same y centre. The small global RMS values therefore coexist with much larger localized deviations. These maxima describe the saved samples, not continuous-time extrema.

Area-volume and four-edge calculations retain the prescribed initial 55 m³ separately from predicted initial volume. Refined predicted initial bias is −0.106192 m³. The maximum sampled absolute corrected balance is 0.242180 m³, or 0.440327% of prescribed volume; the signed value at 0.5 s is −0.242180 m³. Removing the initial bias gives −0.135988 m³ at 0.5 s. This subtraction is an additional diagnostic and does not replace the original balance.

| Edge | Signed outward contribution through 0.5 s | Integrated absolute normal-flux mismatch |
|---|---:|---:|
| Left | −0.092477 m³ | 0.092477 m³ |
| Right | −0.025888 m³ | 0.025888 m³ |
| Bottom | −0.005116 m³ | 0.007330 m³ |
| Top | +0.012863 m³ | 0.014235 m³ |

Base/refined volume grids are 800×21 and 1,600×42 at t=0 and all 50 reporting times. Tangential edge grids are x400/y21 and x800/y42, with 1,012/2,012 actual time nodes after unioning the nominal 1,001/2,001 grids with the exact reporting times. Maximum base/refined differences are 5.5064e-5 m³ for volume, 1.1077e-6 m³ for integrated outward flux and 5.5332e-5 m³ for corrected balance. This is combined spatial/time quadrature sensitivity, not an error bound or a separate-axis convergence study.

The training loop took 420.78 s; its logged wrapper took 423.17 s. Final reference/model queries, error diagnostics, all four edge queries, two quadratures and compressed raw-output I/O took 17.17 s; the evaluation wrapper took 18.68 s. Thus the measured training loop plus evaluation workload took 437.95 s, compared with the pilot projection of 384.50 s and its slowest-observed-step scenario of 596.72 s. These observations do not establish a general timing bound. Figure generation was not separately timed. Three saved-array figures were rendered as PNG/PDF and visually inspected for labels, legends, units and provenance.

The frozen protocol remains `c02dc2c456c397c215b1a1fd564bebf2e6647f61cc6809b5acb829dee17aae42`. Detailed checks and launch/terminal receipts are under `docs/evidence/phase_10/planar_v01/full_v01/`. Raw final arrays, all per-y diagnostics, both mass quadratures and actual costs are under `results/extensions/planar_2d/full_v01_evaluation/`; figures are under `results/extensions/planar_2d/full_v01_figures/`. The [reproducibility guide](../../REPRODUCIBILITY.md) describes saved-checkpoint replay from the `planar-data` bundle listed in [ARTIFACTS.json](../../ARTIFACTS.json). That route requires no new training.
