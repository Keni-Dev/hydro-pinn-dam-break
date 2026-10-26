The single 10,000-update study completed, with the earliest strict validation minimum at update 4,109. The final raw pass used that selected checkpoint; the latest checkpoint remains update 10,000. Saved-data evaluation and the separate aggregate mass reports completed without extra model queries. The one extracted-checkpoint replay is pending at this archive snapshot.

Against the selected numerical approximation, the errors are material. The following Gauss3 metrics pool all 50 positive times. MAE and RMSE are area means; relative L2 is the separately defined ratio of area-weighted squared integrals.

| Field | Whole MAE | Whole RMSE | Whole relative L2 | Gradient-region RMSE | SI unit for MAE/RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| Depth h | 0.009431 | 0.026175 | 17.34% | 0.104857 | m |
| Discharge qx | 0.014785 | 0.038964 | 56.40% | 0.132720 | m²/s |
| Discharge qy | 0.021310 | 0.046004 | 40.31% | 0.150691 | m²/s |

Whole-domain cell-ratio velocity relative L2 is 103.80% for u and 115.29% for v. Ratios use conserved cell means qbar/hbar, with all cells retained. Complement transport/velocity relative percentages become extremely large against very small nonzero reference norms; the reports preserve these values alongside SI errors and null reasons without an epsilon or favorable filter.

The base quadrature starts 0.505278 m³ below the specified 11.8 m³ volume. Its final signed-flux-adjusted balance is −0.528558 m³. Subtracting the initial bias leaves −0.023280 m³ additional imbalance; that smaller value does not remove the initial deficit. Final net signed outward integral is -0.030543784 m³, while the summed absolute outward integral is 0.037108360 m³. Four-wall signed and absolute rates/integrals, spatial/temporal changes, and symmetry defects remain in the saved outputs.

The selected reference is the 320², half-CFL numerical run projected to 80² cell means. It retains material mesh/time sensitivity. Dspace, Dtime and Dsum are descriptive conserved-field sensitivity indicators. Dsum exceedance counts are not coverage probabilities or certified error bounds. This one-seed, short, idealized wet case does not establish exact truth, general superiority, operational accuracy or human/course acceptance.

Three saved-data figures retain the final conserved fields, every positive-time error series, and mass bias/leakage/quadrature differences. Their rendered PNGs passed visual readability review. No additional fit, pilot, precision comparison, reference solve or quality-driven tuning ran. The final closeout handoff will preserve the original archive and report the one fresh extracted replay, exact comparisons and actual costs separately.
