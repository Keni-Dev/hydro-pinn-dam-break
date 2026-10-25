# Phase09 depth-response sensitivity findings v01

The completed bounded gate is unresolved at this numerical resolution/operator and weak at the fixed 1 cm illustrative noise scale. Both predeclared local contrasts fail before and after the numerical adjustment. No inverse observation dataset or trainer was created, and broader Phase09 remains incomplete.

The fixed wet, flat E1 dam-break case uses T=.5 s, five positive Manning values on [.020,.040] s m^(-1/3), three .025 m interval means and five times. All twelve new CPU float64 solves completed once; the three n=.03 native records were hash-matched and reused. Each solve retained the full accepted 50-time schedule before rows 9/19/29/39/49 and common cells 220/240/260 were extracted in time-major order. No gauge, time, coefficient, noise scale, threshold or numerical level changed after results.

## Declared screen

Truth-versus-.025 has raw/adjusted aggregate distances .218164/.123087. Truth-versus-.035 has .271933/.146070. Both adjusted distances are below the fixed one-unit threshold. Their raw RMS contrasts are .563297 and .702128 mm; maximum absolute contrasts are 1.582874 and 2.086412 mm.

All ten pairs have L2(delta)≤L2(U), so measured numerical differences remain a separate limitation. This norm comparison is an indicator, not a certified error comparison. Several individual samples remain visible above their component indicators; their counts are retained below rather than discarded.

Truth-versus-.020 and truth-versus-.040 adjusted distances are .243538 and .337655. These competing responses fail the range check. The outer .020-versus-.040 pair is the only raw pass (1.025676), but its adjusted distance is .610231. It cannot rescue the failed primary local screen. Every pair fails the adjusted screen.

The assumed independent Gaussian sigma=.01 m is illustrative. Distances divide by sigma for one hypothetical noisy observation vector. No RNG draw, noisy observation, fitting partition or final holdout was generated. The one-unit threshold supplies no confidence level, recovery percentage or field identifiability claim. The heuristic adjustment supplies no guaranteed bound. This result does not prove intrinsic physical nonidentifiability.

## All unordered pairs

| n pair | Raw d | Adjusted d | Raw RMS (mm) | Raw max (mm) | Visible / 15 | L2(delta) ≤ L2(U) |
|---|---:|---:|---:|---:|---:|:---:|
| 0.020 / 0.025 | 0.175384 | 0.098961 | 0.452840 | 1.233974 | 4 | yes |
| 0.020 / 0.030 | 0.393499 | 0.243538 | 1.016010 | 2.816847 | 5 | yes |
| 0.020 / 0.035 | 0.664953 | 0.414619 | 1.716902 | 4.903259 | 6 | yes |
| 0.020 / 0.040 | 1.025676 | 0.610231 | 2.648284 | 7.962886 | 6 | yes |
| 0.025 / 0.030 | 0.218164 | 0.123087 | 0.563297 | 1.582874 | 4 | yes |
| 0.025 / 0.035 | 0.489857 | 0.291725 | 1.264805 | 3.669286 | 6 | yes |
| 0.025 / 0.040 | 0.851543 | 0.486260 | 2.198675 | 6.728912 | 6 | yes |
| 0.030 / 0.035 | 0.271933 | 0.146070 | 0.702128 | 2.086412 | 5 | yes |
| 0.030 / 0.040 | 0.634830 | 0.337655 | 1.639123 | 5.146038 | 6 | yes |
| 0.035 / 0.040 | 0.364085 | 0.167671 | 0.940062 | 3.059626 | 5 | yes |

Signed 15-value contrasts at all three levels, signed mesh/time difference vectors, their absolute components and sums e, pair sums U, adjusted vectors, full RMS/max/L2/noise summaries and flags are saved in `results/extensions/inverse_sensitivity/gate_v01/analysis.json`. The native outputs include depth, discharge, native axes and every returned actual-step budget. The gauge NPZ is a deterministic response table with axes [candidate, level, fixed sample], not an inverse observation dataset.

## Descriptive finite difference

J=(mu_.035-mu_.025)/.010 ranges from -.3669285665 to .2442921405 metres per unit n. J*.03/1 ranges from -.0110078570 to .0073287642. The central second difference has L2=.0005591558 m and maximum absolute value .0005035382 m. These saved vectors describe this five-point profile; they are not an exact derivative, parameter-gradient check or confidence interval.

## Native numerical evidence and cost

All new solves retained strictly positive depth, with minimum saved/running stage h=.1 m. Actual maximum CFLs stayed at the requested levels within floating-point roundoff (maximum .4500000000000001). No floor, clipping, retry or added refinement was used. Saved numerical fluxes and source impulses independently reproduce the native mass and momentum budgets. Across the twelve new solves, maximum absolute defects are 1.63425e-13 m² and 1.14214e-14 m³/s. Final cumulative drag impulse ranges from -.05227730 to -.01337344 m³/s; mass alone was not used to verify drag.

The numerical matrix used 4.430116 elapsed seconds, including storage/extraction and reuse, against the fixed 60-second budget. The twelve solver-reported costs total 4.170546 s; native compression/storage totals 0.202962 s. Projection/extraction over all fifteen records totals 0.018088 s. Saved-data analysis and the single profile figure used 12.184366 s. Driver wall time was 17.276393 s; the subprocess wrapper measured 18.209796 s including startup. The output packet measured 6127808 bytes at terminal receipt, before this handoff.

The assigned host is the available M5 Air, and the execution receipt observes macOS ARM64, Python 3.12.14 and NumPy 2.5.4. Current CPU-brand and physical-memory sysctl probes were denied by the sandbox; their failures are retained rather than presented as measured hardware identification. vm_stat succeeded. Process peak RSS at the end was 552,828,928 bytes (about 527 MiB), including imports/reporting/plotting. It is not solve-only memory or a thermal guarantee. Matplotlib used a temporary writable font cache after its default cache path was unavailable; that warning and its cost remain in stderr.

## Verification and preservation

Five focused tests passed: conservative overlap/time-major extraction, signed distance adjustment/independent accumulation, frozen protocol/source guards, existing-root preservation and exclusive launch protection. Scoped Ruff lint and format checks passed. The independent audit reads saved arrays only; no new solve or old science suite is part of that check.

The frozen source/config/operator/axes and adopted input hashes are in `protocol.json`. Every protected pre-existing module, the CLI, four accepted executable archives and accepted reference packet matched before/after. All original results, earlier failed attempts and foundations00/01 remain preserved; this owner did not stage, commit, reset, clean, delete or edit root coordination/global status. The selected08 forward result remains accepted with incomplete flow recovery.

The saved-data profile is `results/extensions/inverse_sensitivity/gate_v01/sensitivity_profile.png`. It shows selected interval-mean responses against the known numerical generator and raw/adjusted distances to n=.03; the zero at n=.03 is tautological and is not recovery or independent validation. The figure was visually inspected for labels, sign and noise qualifications.

## Handoff boundary

This finite sensitivity gate is complete. The tested .005 increment is weak under the fixed noise assumption, and measured numerical differences leave its interpretation unresolved. Inverse observation generation, neural training, recovery, viscosity, genuine 2D, human participation and submission remain uncreated or unreleased. A later physical/sensor question requires a separate proposal that preserves this unfavorable result.
