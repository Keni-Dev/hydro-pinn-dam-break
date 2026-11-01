# Genuine 2D reference matrix findings v01

All four predeclared reference cells completed through 0.25 s with 51 exact output times. Independent saved-data reviews pass the numerical, conservation, boundary, source-preservation and R6 analysis checks. The preselected reference remains `ref_rect320_c0225_v01`. Reference adequacy remains limited by the measured mesh/time sensitivity.

| Cell | Sum-CFL | Steps | Internal observation | Wrapper wall |
| --- | ---: | ---: | ---: | ---: |
| 80² | 0.45 | 50 | 1.031498 s | 1.322756 s |
| 160² | 0.45 | 100 | 1.833906 s | 2.113256 s |
| 320² | 0.45 | 176 | 6.483925 s | 6.695521 s |
| 320² | 0.225 | 321 | 11.069065 s | 11.286013 s |

The completed records total 20.418395 s of internal observations, 21.417547 s of wrapper wall, and 35,360,124 bytes. Maximum process-lifetime RSS is 762,478,592 bytes. All cells completed within their nominal 60/60/180/360 s numerical caps without recorded overshoot. The unknown first failed-launcher child's work/cost is excluded from these sums without being treated as zero. Its unknown exit/cause, original empty logs and failure receipts remain preserved.

Saved states and recorded interface depths are finite and positive. CFL values pass the frozen 64-epsilon allowance; parity defects are zero; outer-strip and Dmax<4 m diagnostics pass. Mass and both momentum budgets include signed four-wall pressure fluxes. Finer grids retain full states at the 51 output times; intermediate-step checks use saved scalar/vector diagnostics and edge fluxes. No unsaved PDE state was reconstructed.

R6 metrics use 50 equally weighted positive snapshots, with initial data separate. Relative denominators name each higher/refined member. Pooled RMSE/L2 take the square root after averaging squared integrals; pooled relative percentages use the ratio of summed squared integrals. The complete MAE/L1/RMSE/L2/relative values, units and null reasons are in `results/extensions/two_dimensional/reference_v01/r6_analysis_v01/norms.json`.

| Signed difference | Pooled RMSE h, m | Pooled RMSE qx, m²/s | Pooled RMSE qy, m²/s | Pooled relative L2% h / qx / qy |
| --- | ---: | ---: | ---: | --- |
| 160−80, coupled mesh/time | 0.005660282 | 0.008920153 | 0.014266765 | 3.767879 / 13.609638 / 13.229850 |
| 320−160, coupled mesh/time | 0.004397785 | 0.006736061 | 0.010027393 | 2.913264 / 9.700796 / 8.742646 |
| 320 half−full CFL, fixed-grid time | 0.000420904 | 0.000615556 | 0.000944205 | 0.278895 / 0.891017 / 0.827279 |

The finer coupled pair has smaller pooled SI differences in all three fields, and smaller per-time RMSE at 48/50 depth times and 49/50 times for each discharge. Half-CFL changes are smaller than the 320−160 changes. The remaining coupled discharge differences still matter for later model comparisons. These observations do not establish an asymptotic order, exact 2D truth or certified error bound.

Saved arrays retain every signed difference plus Dspace=abs(320,.45−160,.45), Dtime=abs(320,.225−320,.45), and Dsum=Dspace+Dtime at every field/time/common cell. They are sensitivity indicators without interval coverage. Initial differences are zero; six initial momentum relative norms are null with the zero-denominator reason. The independent review checks 2,808 norm scalars and all six nulls; reduction differences stay within arithmetic readback allowances.

The gradient region uses only preselected half-CFL reference depth on the 80² common grid, the frozen 10%-of-time-maximum threshold and one-cell eight-neighbour dilation. Raw coverage spans 1.4375–8.9375%; dilated coverage spans 2.9375–11.3125%. At 0.25 s, 564 raw and 724 dilated cells cover 8.8125% and 11.3125%. No gradient snapshot is flat. This numerical region may include shocks, rarefactions and corners; it does not locate an exact shock/front.

Exactly two logical figures are saved as PNG/PDF: `01_selected_reference_maps` and `02_refinement_rmse_by_time`. PNG readback confirms readable axes, units, scales and labels. Maps retain h=.1..1 and shared qx/qy limits ±0.9150600669316856 m²/s over all positive selected times. The RMSE plot distinguishes coupled mesh/time from fixed-grid time.

Saved-data analysis took 13.101455208 s of wrapper wall and 1.325461500 s from analyze() entry through its commit observation; module imports/startup sit outside the latter timer. It ran zero solver updates, area operators or neural queries. Its script was hash-frozen before use. All 533 independent analysis checks pass; input/output/figure hashes, 20 frozen bindings, 32 protected modules and five prior archives match.

The reference payload targets `results/extensions/two_dimensional/reference_payload_v01` and `results/extensions/two_dimensional/genuine_2d_reference_v01.zip`; the separate package receipt records byte verification. The layout retains executable source/CLI/config/lock, exact protocol/axes/operators, four complete records, analysis/figures and original/incident/recovery/launch/readback evidence. Packaging performs no simulation. At this v01 snapshot, a fresh-extraction saved-data replay was proposed; [findings v02](genuine_2d_reference_matrix_findings_v02.md) records its subsequent completion. These original archive paths are historical provenance; the publication supplies curated numerical inputs through the `genuine-2d-reference-data` bundle in [ARTIFACTS.json](../../ARTIFACTS.json).

This reference packet contains no genuine-2D neural model, pilot, training or inference. The neural study is documented separately in [neural full findings v01](genuine_2d_neural_full_findings_v01.md).
