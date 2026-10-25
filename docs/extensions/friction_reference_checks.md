# Wet flat-bed Manning reference, v01

The four predeclared numerical solves completed on the M5 CPU in float64. This packet supports a bounded forward-model pilot with measured discretization uncertainty. It does not provide an analytical friction solution or field validation.

## Physical case and source

The case is E1 on x∈[−5,5] m: dam at zero, hL=1 m, hR=0.1 m, uL=uR=0, g=9.81 m/s², T=0.5 s, and prescribed initial volume 5.5 m² per unit width. Bed elevation and rainfall are zero. Both edges retain their E1 reservoir states. The fixed illustrative Manning coefficient is n=0.03 s/m^(1/3). It is not fitted or attributed to a field site.

The source changes only momentum: Sq=−g n²q|q|/h^(7/3). The friction slope is n²q|q|/h^(10/3). Holding h fixed gives q(t+dt)=q(t)/(1+dt g n²|q(t)|/h^(7/3)). That isolated source step preserves the sign and reduces the magnitude of q for nonnegative dt. Its closed-form integration has no explicit source stability restriction; coupled splitting accuracy remains a numerical question.

## Method and provenance

`friction_reference.py` independently implements piecewise-constant finite-volume HLLE with the Einfeldt wave estimates documented in the [Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water_approximate.html). The consulted notebook is pinned to [commit 6dd68c1819de67de8e44e141ebec666f67e2fa2e](https://github.com/clawpack/riemann_book/blob/6dd68c1819de67de8e44e141ebec666f67e2fa2e/Shallow_water_approximate.ipynb). Its code license is BSD 3-Clause; the book text has a separate license. External solver code and book text were not copied into the implementation.

The [GeoClaw Manning formulation](https://www.clawpack.org/list/manning.html) and [source routine at e2ef985e314e8609200cb67a7c15ef56b07f7828](https://github.com/clawpack/geoclaw/blob/e2ef985e314e8609200cb67a7c15ef56b07f7828/src/2d/shallow/src2.f90) were checked as source context. The routine uses depth thresholds and a friction-depth switch. This wet-only implementation applies drag at every positive depth and does not import those thresholds, dry-state rules, speed limits, spherical terms, or 2D behavior. GeoClaw's license is BSD 3-Clause. Actual consulted source and exact-version license bytes, URLs, revisions, and SHA256 hashes are in `docs/evidence/phase_08/friction_v01/reference_sources/`.

At each interface, uhat=(√hL uL+√hR uR)/(√hL+√hR) and chat=√(g(hL+hR)/2). The speed estimates are sL=min(uL−√(g hL),uhat−chat) and sR=max(uR+√(g hR),uhat+chat). The conservative HLL flux chooses FL for sL≥0, FR for sR≤0, and the two-wave flux otherwise. The code checks the intermediate Riemann state even when the interface is supersonic.

One reservoir ghost cell at each edge is reset for every homogeneous step. The flux update uses forward Euler, then the exact fixed-h drag step. This Lie split is first order in time and space. The timestep is CFL dx/max(|sL|,|sR|) over every interface, shortened to hit each requested time exactly. The implementation accepts CFL≤0.5. There is no reconstruction limiter, depth floor, positivity clipping, timestep retry, or time interpolation. Nonpositive or nonfinite initial, ghost, intermediate, homogeneous, or source states are rejected.

## Necessary verification

Five focused reference tests passed in 1.15 s on the initial check; the recorded final check is in `reference_tests_stdout.log`. The tests cover both drag signs, q=0, exact n=0 source identity, source-step composition, an independent SciPy ODE integration, flat constant-depth still water, a dam-break mass and momentum budget, refinement toward exact E1 at n=0, conservative projection, and rejection of unsupported inputs. The source-physics and PINN zero-n equivalence checks belong to the separate friction-physics test packet.

For zero-n HLLE at t=0.5 s, the mean absolute h error against 16-node Gauss approximations of exact E1 cell averages decreased from 0.00647044 to 0.00368463 to 0.00206600 m on 200/400/800 cells. The q errors decreased from 0.0142728 to 0.00812623 to 0.00461044 m²/s. This verifies approach to the source-free exact case; equality with the immutable Lax–Wendroff comparator is not required.

## Actual refinement packet

The protocol was written before the four positive-n solves and before any neural comparison. All 50 output times from 0.01 through 0.50 s are stored as cell averages. Native outputs also retain every actual dt, left/right numerical mass and momentum flux, spatially integrated actual drag impulse, volume, momentum, cumulative impulse, and budget defect. For each step, the mass defect is M−M0+Σdt(FR,h−FL,h). The momentum defect is P−P0+Σdt(FR,q−FL,q)−Σdx(qnew−qstar). The source impulse comes from the performed source update. Mass conservation alone would not verify drag.

| Cells | CFL | Steps | Solve seconds | Maximum mass defect m² | Maximum momentum defect m³/s |
|---:|---:|---:|---:|---:|---:|
| 800 | 0.45 | 399 | 0.050676 | 4.53e−14 | 4.06e−15 |
| 1600 | 0.45 | 798 | 0.094142 | 4.62e−14 | 4.87e−15 |
| 3200 | 0.45 | 1548 | 0.309249 | 1.62e−13 | 6.53e−15 |
| 3200 | 0.225 | 3046 | 0.650993 | 1.61e−13 | 6.59e−15 |

The complete packet, including native compressed arrays, metadata, and projection/refinement output, took 1.31137 s. Minimum depth was 0.1 m on every run. Both boundary cell states retained the reservoir values. The selected 3200-cell/CFL=0.225 final integrated drag impulse was −0.0298996 m³/s. Its actual boundary momentum flux contribution is preserved alongside that impulse.

## Frozen comparison operator and uncertainty

The selected reference is the predeclared 3200-cell/CFL=0.225 solution. `common_reference.npz` stores conservative integrals of its piecewise-constant native field over 400 equal common cells. The PINN must be integrated over the same cells with a separately declared quadrature rule. Comparing those cell averages with neural point values would change the operator.

On these common averages, spatial h RMSE differences decrease from 0.00275534 m (800→1600) to 0.00159750 m (1600→3200), and q RMSE differences from 0.00783633 to 0.00482085 m²/s. Halving CFL at 3200 cells changes h RMSE by 0.000314488 m and q by 0.000889279 m²/s. These differences do not establish an asymptotic order or a rigorous error bound.

The retained cellwise uncertainty indicator is abs(N3200,CFL.45−N1600,CFL.45)+abs(N3200,CFL.225−N3200,CFL.45). Its h RMS/max are 0.00186849/0.0306021 m; its q RMS/max are 0.00565006/0.121311 m²/s. The comparatively large maxima occur near steep gradients and remain in the packet.

The numerical front is the first native downstream interface x≥0 with maximum negative adjacent-cell h slope. It is a numerical diagnostic. Per-time position uncertainty is the maximum front-position difference across all four solves plus one selected dx. Its largest value is 0.021875 m. The frozen front band half-width is max(0.1 m, that uncertainty), so it stays 0.1 m for all 50 times. No analytical shock trajectory is claimed for positive friction.

Raw numerical evidence is in `results/extensions/friction/reference_v01/`. Nonflat well-balancing, dry-state validation, inverse identifiability, seed variability, and final neural accuracy remain outside this reference check.
