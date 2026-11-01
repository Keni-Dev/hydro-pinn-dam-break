# Wet flat-bed Manning forward result, v01

The released 10,000-update study completed with incomplete flow recovery. Whole-domain h, q and cell-ratio velocity errors are 6.71%, 33.78% and 78.57% relative L2 against the refinement-checked numerical reference. The numerical-front band has larger errors. These unfavorable results are retained. The run establishes execution of this one fixed-budget experiment; it does not establish convergence or field validity.

The case, source, sampling and selection rules are unchanged from the frozen proposal. The domain is x=[−5,5] m with dam at zero, hL=1 m, hR=0.1 m, zero initial velocity, g=9.81 m/s², T=0.5 s and prescribed initial volume 5.5 m² per unit width. Bed and rainfall are zero; both ends use fixed reservoir targets. Manning n=0.03 s/m^(1/3) is a fixed illustrative coefficient chosen before the study. No roughness fitting, parameter profile, sensor dataset, viscosity, dry-state or topographic experiment was performed.

## Execution and selection

The full study used one fresh seed-11 fit with the preserved initialization and original coordinate multiplicities, CPU float32 with four threads, four 64-unit tanh layers, Adam at 0.001, and fixed PDE/IC/BC weights 1/10/10. The fixed sets have 10,000 PDE points, 1,000 IC points and 500 points per boundary. The two repeated training coordinates after float32 conversion remain present; cross-partition overlap remains zero. The model retains its disclosed scaled softplus depth floor 10⁻⁶.

The fit launched at 2026-10-25 20:53:42.897 UTC and its wrapper terminated successfully at 20:58:23.116 UTC. Run history contains step 0 and every update through 10,000. The separate physics/IC/BC validation objective selected the earliest strict minimum at update 9,857, score 0.00474569574. There were 10,001 validation evaluations, 1,442 selected-checkpoint writes and 10,001 latest-checkpoint writes. Initial identity, selected-score reload, latest-state prediction reload and every saved optimizer parameter membership/update count passed readback. Final reference errors did not select the checkpoint or settings.

The selected checkpoint SHA256 is `495219587a7b90f562ac66ad70a7c3ef96bd71f72c7732a720eca7e5c7c26652`; history SHA256 is `f2de20ff73709c9071c817596020100ad3cb9365622cae9268f1277b82586249`. Both pilot and full records retain protocol content digest `b49eac414c46959d4d0ff0fb376de19c1c6cca3b2dc939ec2a1420357dbb2360` and the same protocol file bytes. All pilot/reference/source bytes remain preserved.

## Comparison operator and errors

The reference is the already computed 3,200-cell HLLE/Manning run at CFL 0.225, with first-order homogeneous flux then exact fixed-h drag splitting. There is no reference depth floor or friction threshold. The 800/1,600/3,200 spatial matrix and separate CFL reduction were completed before fitting and were reused without another numerical solve.

The 400 common spatial cells and 50 times from 0.01 through 0.50 s are frozen. The numerical averages are conservatively projected onto those common cells. PINN h and q are integrated over each cell with three-point Gauss–Legendre quadrature, with five-point quadrature retained for sensitivity. Velocity means qbar/hbar for both methods. It is called **cell-ratio velocity** throughout; it is neither a point velocity nor an average of point velocities. All 20,000 common cells satisfy the fixed reference hbar>0.001 m mask.

| Region | h RMSE (m); relative L2 (%) | q RMSE (m²/s); relative L2 (%) | Cell-ratio velocity RMSE (m/s); relative L2 (%) |
|---|---:|---:|---:|
| Whole, 20,000 values | 0.0464280; 6.7124 | 0.108231; 33.7766 | 0.566894; 78.5715 |
| Numerical-front band, 398 values | 0.161783; 55.2011 | 0.448953; 73.5227 | 2.13877; 134.1854 |
| Outside band, 19,602 values | 0.0408399; 5.8559 | 0.0886533; 28.4368 | 0.484785; 70.0054 |

The front band uses the numerical front definition frozen from the reference matrix, with 0.1 m half-width and measured position uncertainty up to 0.021875 m. No analytical frictional shock trajectory is claimed. Whole-domain h relative error reaches 9.6264% at 0.50 s. Per-time whole-domain q and cell-ratio velocity relative errors reach 99.3510% and 453.0703% at 0.01 s.

Early outside-band reference q and cell-ratio velocity norms are nearly zero. At 0.01 s their recorded relative errors are approximately 1.7343×10⁸% and 2.1965×10⁹%; their corresponding SI RMSE values are 0.0436251 m²/s and 0.552512 m/s. These values remain in the raw diagnostics. Relative percentages with tiny denominators are poorly conditioned, so SI RMSE and the complete time series accompany them. Figure axes preserve those values rather than clipping them.

The combined reference refinement indicators have h RMS/max 0.00186849/0.0306021 m and q RMS/max 0.00565006/0.121311 m²/s. They are measured spatial/time differences, not rigorous error bounds. The whole-domain h/q PINN errors exceed those RMS indicators. No rigorous cell-ratio velocity uncertainty bound is derived.

Changing PINN cell quadrature from three to five points changes any common-cell h/q average by at most 0.00152233 m and 0.00458615 m²/s. Five-point whole-domain relative errors are h=6.71246%, q=33.77643% and cell-ratio velocity=78.57155%. This sensitivity check leaves the incomplete recovery finding unchanged and does not replace the frozen three-point primary comparison.

## Mass and boundaries

Both boundary discharges enter the signed correction M(t)−5.5+integral(qR−qL). Volume uses 800/1,600 midpoint cells; boundary time axes use the frozen 999/1,999 nodes after disjoint-membership filtering. Raw h/q arrays and both endpoint fluxes are saved. Quadrature results refer to sampled report times, including t=0, rather than continuous-time extrema.

At refined resolution, predicted initial volume is 5.47536175 m², a bias of −0.02463825 m² (−0.44797% of the prescribed volume). Final volume is 5.47713709 m². The final net outward-flux integral is −0.00429821178 m², so the final and largest-magnitude corrected imbalance is −0.02716113 m² (0.49384% in absolute magnitude). After subtracting predicted initial bias, the final balance is −0.002522878 m²; its largest sampled absolute magnitude over time is 0.01281596 m².

Base/refined differences are at most 0.000480805 m² for volume and corrected balance, and 1.1923×10⁻⁹ m² for integrated outward flux. The initial-volume bias differs between midpoint levels and remains separately visible. These differences are quadrature sensitivities, not error bounds or a convergence claim.

Refined left/right depth mismatch maxima are 0.000146449/0.000322738 m. Discharge mismatch maxima are 0.00751090/0.00168141 m²/s. Individual absolute flux mismatch integrals are 0.00369608 m² at the left boundary and 0.000602129 m² at the right. Both signed outward integrals are negative. Reporting them separately prevents cancellation from hiding boundary mismatches. The relatively small mass imbalance leaves the substantial front and velocity errors unresolved; no unvalidated mass target is treated as a pass criterion.

The numerical reference's separate native budgets retain actual left/right numerical mass and momentum fluxes and the performed source impulses. Maximum mass/momentum defects across its four runs are 1.62×10⁻¹³ m² and 6.59×10⁻¹⁵ m³/s. The selected reference's final integrated drag impulse is −0.0298996 m³/s. Its source-step correctness was checked independently before fitting. Mass conservation alone would not verify the drag term.

## Cost and evidence limits

Full training, validation, checkpoint I/O and history flushing took 278.346 seconds; the entire fit wrapper took 280.217 seconds. The original pilot's mean-step projection excluded history flushing. The reviewed inclusive scaled-wall-plus-setup/reference/query estimate was 285.944 seconds; neither projection was a bound. No pilot, timing sample, seed or fit was repeated to obtain a favorable rate.

The complete final query/diagnostic/output workload took 0.612192 seconds, with 160,000 common-cell quadrature queries and 128,396 mass/boundary queries. Its three/five-point model-query components took 0.034856/0.046093 seconds; mass queries, integration and raw I/O took 0.475957 seconds. The external evaluator wrapper, including startup, took 1.572256 seconds. The reference/refinement/output packet had previously taken 1.311370 seconds, including selected-reference solve time 0.650993 seconds. These stages have different workloads; no workload-matched speedup claim is made. Three initial saved-data figures took 1.611034 seconds to load/render. A versioned error-figure layout repair took 0.834130 seconds; the combined 2.445163 seconds excludes imports, font-cache startup and visual review. The canonical set selects profiles v01, errors v02 and mass/boundaries v01. The v02 symlog axes retain the early relative-error outliers and explain their near-zero denominators; every v01 output remains preserved. Archive and extracted-replay costs are recorded separately in their ledgers.

Full-fit process lifetime peak RSS reached 416,743,424 bytes (about 397 MiB). Host load averages rose from 4.22/4.56/4.73 to 7.37/5.70/5.18 over the fit. Lifetime peak RSS and host load do not measure instantaneous memory, system pressure, thermals or per-process CPU utilization.

The 17 prelaunch focused checks remain the software-verification basis; no old-study tests or fits were repeated. Independent final readback recomputed 6,549 numeric values from saved arrays with zero mismatches under rtol=10⁻¹² and atol=10⁻¹³, and verified unchanged native reference hashes. The largest absolute recomputation difference was 4.768×10⁻⁷ in the enormous early relative percentage, about 2×10⁻¹⁶ relative. It did not change any stored scientific value.

Generic reference metadata retains inherited source-free E1 analytical-reference fields and unfinished execution placeholders. Generic training metadata also inherits the E1 reference description. For this extension, the physical interpretation is positive Manning n=0.03 with numerical finite-volume HLLE, a homogeneous Euler step followed by exact fixed-depth Manning decay, no numerical depth floor or friction threshold, and measured reference uncertainty. Velocity is qbar/hbar on common cells with the fixed reference hbar>0.001 m mask. The extension protocol/status/summary/native diagnostics establish completion. The training summary's final-error flag remains false because training used no final-reference labels; the separate held-out evaluation is complete and reports neural errors and mass diagnostics. Figures and these findings use these extension records rather than the inherited analytical-reference description. The inherited metadata bytes were preserved.

Frozen final arrays and diagnostics are restored under `results/extensions/friction/full_v01_evaluation/` by the `friction-data` bundle listed in [ARTIFACTS.json](../../ARTIFACTS.json). The [reproducibility guide](../../REPRODUCIBILITY.md) describes saved-checkpoint replay. This study remains one illustrative wet flat-bed case and one seed. It provides no inverse identifiability, field calibration, dry/nonflat validation or genuine 2D result.