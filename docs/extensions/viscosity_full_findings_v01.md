# E1 artificial momentum diffusion: full result v01

The fixed positive-viscosity branch reduced pooled field errors against the inviscid E1 reference in this single case and seed. Its inviscid momentum residual, bias-subtracted mass drift and left endpoint discharge mismatch increased. Local errors and undefined front descriptors remain. These results support a bounded comparison of the two selected states; they do not establish general improvement or an exact positive-viscosity solution.

The released run completed exactly 10,000 updates, with 10,001 history rows including the initial state and all ten Adam states at update 10,000. Earliest strict modified physics/IC/BC validation selected step 4,289; the terminal state was preserved separately. The historical zero branch uses its original selected step 6,512. There was one new full fit, one positive pilot and no zero retraining. The full branch restarted from the frozen common initial tensors with fresh Adam. The fixed coefficient, source, partitions, architecture, CPU float32/four-thread settings, no-chunk arithmetic and objective weights match the accepted protocol.

The protocol SHA256 is `552481e60b63ef12c066990d844a612323437d36a0d8d00fcf8d3650bb64df17`; the full positive checkpoint SHA256 is `1eeff8b7c95e29b8bfc0d8b43b87935241a6c050ea213c1adbbfebc45ff72e52`. The historical selected checkpoint and zero-cache manifest remain unchanged. Complete per-time metrics, differences, coverage and null reasons are in `results/extensions/viscosity/full_v01_evaluation/evaluation.json`.

| Pooled relative L2 error, 50 times × 400 cells | Historical zero | ν* = 0.001 |
| --- | ---: | ---: |
| Depth, whole grid | 0.065219 | 0.019824 |
| Discharge, whole grid | 0.358815 | 0.105136 |
| Reference-wet velocity, whole grid | 0.845481 | 0.195600 |
| Depth, fixed shock band | 0.525747 | 0.243396 |
| Discharge, fixed shock band | 0.746042 | 0.334674 |
| Reference-wet velocity, fixed shock band | 1.376817 | 0.492300 |

The band contains 400 of 20,000 primary space-time points (2% coverage). Outside-band relative L2 errors for depth/discharge/velocity were 0.056831/0.303354/0.765728 for zero and 0.013101/0.048865/0.126682 for positive viscosity. These pointwise comparisons use the same inviscid analytical target, thresholds, axes and weights for both branches. The remaining positive shock-band velocity error is about 49.2%; lower pooled error does not establish complete flow or shock recovery.

The fixed ±0.1 m front window retains the original 400-cell grid. The zero branch has no valid 10–90% transition width at 0.1, 0.25 or 0.5 s. The positive branch is also null at 0.1 s because the required crossing is absent or outside the window; widths at 0.25/0.5 s are 0.097692/0.071943 m. No window expansion, extra query or replacement width was used. Maximum adjacent depth-slope magnitudes at those three times are 1.09727/0.18347/0.25786 for zero and 4.63980/4.87634/4.75477 for positive viscosity. These sampled descriptors combine amplitude, displacement and smoothing; they do not prove shock accuracy or a monotone smoothing effect.

At the same 365 diagnostic coordinates, nondimensional inviscid momentum RMS increased from 0.037830 to 0.181856, while the ν* = 0.001 operator RMS decreased from 0.063686 to 0.018952. Continuity RMS was 0.023975 for zero and 0.022289 for positive; q* second-x-derivative RMS was 61.6348 and 171.1426. Both operators were evaluated at each selected state without reselection. A smaller modified residual does not imply a smaller inviscid residual because the governing momentum equation changed.

| Continuity balance, base 2,000-cell volume / 1,001-node boundary quadrature | Historical zero | ν* = 0.001 |
| --- | ---: | ---: |
| Learned initial-volume error, m² per unit width | −0.02188285 | −0.00050642 |
| Maximum absolute normalized balance including initial bias | 0.00434439 | 0.00259624 |
| Maximum absolute normalized balance after subtracting initial bias | 0.00132797 | 0.00268831 |
| Final signed balance, m² | −0.02379568 | 0.01427929 |
| Final signed balance after subtracting initial bias, m² | −0.00191284 | 0.01478572 |

The prescribed initial volume is 5.5 m² per unit width. The smaller positive initial bias reduces the total normalized imbalance, while its subsequent bias-subtracted drift is larger. Both physical endpoint discharges enter the mass balance. Time-integrated absolute left/right discharge mismatches are 0.00098758/0.00123246 m² for zero and 0.00332483/0.00084424 m² for positive viscosity. Left/right depth mismatch integrals are 0.00130579/0.00040760 m·s for zero and 0.00121787/0.00035249 m·s for positive. Depth constraint mismatch is reported separately and is not a mass-flux term.

Doubling the volume/boundary quadrature to 4,000 volume cells / 2,001 boundary nodes gives maximum normalized balances 0.00434438 and 0.00259623. Maximum absolute base-to-doubled signed-balance differences are 1.27e-7 m² for zero and 4.32e-8 m² for positive; net-outflow differences are 1.91e-9 and 2.16e-9 m². These are quadrature sensitivities, not error bounds. Sampled extrema are not continuous-time extrema. Auxiliary probes retain the core node-replacement rule and can repeat prescribed IC/BC coordinates.

Training consumed 475.490557 s; measured fit setup and trusted terminal reload consumed 0.737862 s and 0.036549 s. The complete fit command wall was 477.807746 s. The evaluation command wall was 2.369126 s, including one fresh selected-positive query pass and saved-array analysis/copy/output work. Together the two command walls total 480.176872 s (40.0% of the 1,200 s new-compute budget). The historical zero training wall 378.367893 s and its earlier 0.320493 s query/output pass remain separate reused costs. The pilot's 4.027501 s command wall is also already paid.

The fresh positive query/output/lifecycle receipt measured 0.592022 s for 332,004 physical states and 365 dual-operator points; zero used the exact cached arrays. The evaluator's nested post-record query/analysis/copy/save stage measured 1.031408 s, of which its positive-query call occupied 0.596508 s. Their difference, 0.434899 s, is combined remaining analysis/copy/save time, not a pure arithmetic benchmark. These nested stages must not be added to the outer evaluation wall. Startup, pre-record guards/loading and final integrity work are included in the outer wall without separate attribution.

The full training wall was 19.5% above the pilot's 397.851 s scaled-wall estimate and remained within budget. Mean/median observed inclusive update times were 0.047524/0.045544 s, with range 0.035449–0.549158 s. Training-process lifetime peak RSS was 499,597,312 bytes; this is not instantaneous pressure or free memory. Figure rendering cost is recorded separately. Archive, extraction and checkpoint-to-output replay costs were not yet available at this pre-packaging snapshot; [full reproduction v01](viscosity_full_reproduction_v01.md) records their subsequent observation. There is no paired-training speedup or hardware-ordering claim.

The two saved-data figures show the profiles and mass/endpoint diagnostics. At this pre-packaging snapshot, packaging and extracted replay had not yet been executed. The planned snapshot includes the executed source, configuration, lockfile, protocol, release, initial state, historical control/cache, new checkpoint and evaluation payload. The planned replay compared extracted positive-checkpoint queries and saved diagnostics; its scope excluded a full fit or zero query. The [reproducibility guide](../../REPRODUCIBILITY.md) gives the publication route.

This is project-specified constant momentum diffusion with ν* = 0.001 (SI ν = 0.015660459763365824 m²/s). The inviscid target is not an exact viscous solution. Unchanged soft reservoir penalties do not prove a well-posed viscous boundary-value problem. Momentum diffusion alone provides no entropy-selection, energy-dissipation or unconditional-stability guarantee. No momentum budget is reported because that budget requires the diffusive boundary flux −νq_x. One coefficient, one case and one seed do not establish variability or new-case generalization.
