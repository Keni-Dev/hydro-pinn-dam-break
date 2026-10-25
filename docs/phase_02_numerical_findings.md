# Phase 02 numerical findings

All eight canonical E1/E2 runs completed to 0.5 s with the prescribed unlimited
two-step Lax-Wendroff method, CFL 0.45, fixed reservoir ghost states, and CPU
float64. No limiter, clipping, viscosity, positive downstream layer, or retry
changed the method. E2 did **not** fail on these selected grids. Successful
completion does not establish accurate dry-front resolution or positivity for
other inputs.

This is implementation verification dated October 25, 2026. The accepted
Phase 01 analytical references supply the comparison; the final evaluation
configuration was not used. The method follows PRD Section 4.6 and
`prompts/appendix/A_IMPLEMENTATION_CONTRACTS.md`. Mathematical reference
provenance and the independent exact-solver checks remain in
`docs/evidence/phase_01/`.

The canonical runs are `p02_solver_e1_n100`, `p02_solver_e2_n100`, and
`p02_refine_e1_n200/n400/n800`, `p02_refine_e2_n200/n400/n800` (each suffix names
its own run). All metadata/status/arrays were read back. Their physical inputs,
resolved-configuration snapshots, and recorded source hashes match the current
files. Run metadata records foundation revision `d113cae` with a dirty working
tree; per-file SHA256 values identify the actual solver source. Complete hashes,
commands/exits, and failure-interface checks are in
[`solver_checks.json`](evidence/phase_02/solver_checks.json) and
[`refinement_checks.json`](evidence/phase_02/refinement_checks.json).

Native states are cell averages on N cells with dx=10/N m. Initial averages
represent the dam-aligned jump exactly, with prescribed volumes 5.5 m² for E1
and 5 m² for E2. The comparison samples a piecewise-constant reconstruction at
129 common verification centres, selecting the right cell at an internal face.
Analytical values are pointwise at those same centres. The resulting errors
include numerical evolution and reconstruction error; they do not compare
native averages directly with point values while calling the two identical.
All six requested times, 0.037, 0.113, 0.229, 0.347, 0.463, and 0.5 s, occur as
accepted time-step endpoints without time interpolation.

The table reports whole-domain point-comparison errors at t=0.5 s. Depth RMSE
uses metres and discharge RMSE uses m²/s per unit width. Relative L2 uses the
analytical norm without an added denominator. The JSON retains all six times,
full extrema, and the inside/outside band results.

| Case | N | Accepted steps | h RMSE (m) | h relative L2 (%) | q RMSE (m²/s) | q relative L2 (%) |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 100 | 56 | 0.025651 | 3.8083 | 0.095530 | 20.8478 |
| E1 | 200 | 111 | 0.025557 | 3.7942 | 0.088491 | 19.3117 |
| E1 | 400 | 220 | 0.023143 | 3.4359 | 0.078062 | 17.0358 |
| E1 | 800 | 436 | 0.021587 | 3.2049 | 0.073637 | 16.0700 |
| E2 | 100 | 62 | 0.009699 | 1.4667 | 0.029596 | 7.0632 |
| E2 | 200 | 127 | 0.005112 | 0.7729 | 0.015758 | 3.7607 |
| E2 | 400 | 261 | 0.002795 | 0.4226 | 0.008390 | 2.0023 |
| E2 | 800 | 535 | 0.001457 | 0.2203 | 0.004110 | 0.9808 |

E1 retains dispersive oscillations near its shock. Across the six saved
snapshots, maximum native discharge decreases from 1.39046 to 1.34400 m²/s as N
increases from 100 to 800; the exact profile's global maximum is 0.928027 m²/s.
Global depth stays between 0.1 and 1 m, but this hides local oscillations around
the 0.396175 m intermediate plateau. At N800 and t=0.5 s, native cells with
centres in the intermediate region range from 0.343771 to 0.488971 m. Cells near
the shock also carry its transition, so this range includes transition and
oscillation effects.

E1's final ±0.1 m shock band contains only three verification centres. Its depth
RMSE is 0.12340, 0.15737, 0.14945, and 0.14122 m across N100–800; it does not
improve monotonically. Outside that band, depth RMSE falls from 0.01764 to
0.00150 m and discharge RMSE from 0.06968 to 0.00355 m²/s. The downstream
threshold location uses h>(h_star+hR)/2=0.248087 m and the right edge of the last
qualifying native cell. It is 1.5 m at N100 and 1.55 m at the other levels,
against the exact shock at 1.552567 m. This diagnostic has resolution dx and can
respond to dispersive peaks. These observations support a functioning E1
comparator with substantial local error; they establish no convergence order
through the discontinuity.

E2 preserves exact hR=0 in its initial/boundary input and analytical reference.
All accepted numerical depths are nonnegative and finite for this matrix, and
the minimum over each full history is zero. Small positive precursor depths
propagate into the analytically dry region. At t=0.5 s the rightmost native cell
has h≈1.12e-22 m at N100, decreasing to ≈2.49e-153 m at N800. Thus the numerical
solution does not preserve the analytical compact front even though it
completes successfully. The corresponding actual right boundary flux mismatch
peaks at ≈9.18e-23 m²/s for N100, with integrated net outflow ≈7.09e-25 m². E1's
observed numerical boundary mass fluxes are exactly zero. Boundary flux was
measured from each corrector interface, rather than inferred from prescribed
q=0.

E2's h>0.001 m location is 2.7, 2.65, 2.75, and 2.8625 m at t=0.5 s. The exact
zero-depth front is 3.132092 m, while Ritter's exact location at the *same*
0.001 m threshold is 2.983524 m. Relative to that threshold location, the
numerical values lag by 0.28352, 0.33352, 0.23352, and 0.12102 m. The ±0.1 m
band around the zero front contains two verification centres; its absolute
depth RMSE is about 3.1e-5 m, while relative error approaches 100% because the
nonzero reference depth there is very small. Whole-domain error alone would
hide this near-front behavior. E2 also shows nonmonotonic early discharge
overshoot: the maximum over saved snapshots is 1.00191 m²/s at N100 and
1.11847 m²/s at N200, falling near the exact maximum at N400/N800. No failed
canonical grid was discarded or manufactured.

Independent cell-sum readback reconciles every accepted mass with
M(t)-M_IC+Σdt(Fh_right-Fh_left). The largest absolute balance over all eight
histories is 1.78e-15 m², consistent with float64 summation roundoff. Maximum
recorded CFL is 0.4500000000000001. No accepted state contains negative depth or
a nonfinite value. These checks establish discrete mass accounting under this
run setup; they do not establish shock/front accuracy. Constructed predictor,
corrector, zero-wave-speed, and near-dry-overflow diagnostics in
`solver_checks.json` are explicitly diagnostic test inputs, not failures of the
canonical E1/E2 matrix.

The saved whole profiles were inspected at 0.113, 0.347, and 0.5 s:
[E1 profiles](evidence/phase_02/e1_verification_profiles.png),
[E2 profiles](evidence/phase_02/e2_verification_profiles.png).
Selected analytical points through the rarefaction, intermediate state, shock
sides, dry-front sides, and far fields are retained in `refinement_checks.json`.
Both references give h=4/9 m and q=0.9280272452364934 m²/s at x=0 and t=0.5 s.
The diagnostic reconstruction arrays are
[`reconstructed_verification.npz`](evidence/phase_02/reconstructed_verification.npz);
the canonical native arrays remain under their original run IDs.

The executed analysis command was:

```sh
rtk proxy env MPLCONFIGDIR=/private/tmp/aifox-p02-mpl XDG_CACHE_HOME=/private/tmp/aifox-p02-xdg uv run --frozen python docs/evidence/phase_02/analyze_refinement.py
```

To repeat the readback, pass `--output-dir` with a new directory under the
project; the helper refuses to replace existing outputs. Analysis iterations
are preserved under `docs/evidence/phase_02/analysis_attempt_01/` through
`analysis_attempt_03/`. Their changes were diagnostic wording/formatting,
plot-boundary connectors, and addition of the analytical threshold comparison;
no canonical numerical simulation was rerun or altered for those edits.

The E1 comparator is available for the next phase with its oscillations and
errors disclosed. E2 supplies an executed dry-bed diagnostic with front
limitations. No final-protocol accuracy, runtime comparison, GPU behavior,
PINN training outcome, field validation, or course acceptance is established.
