# Planar E1 consistency protocol v01

This is an exploratory consistency test using one independently trained seed11 model. The 25-update pilot measures feasibility. This protocol records the pilot-stage design; the subsequent 10,000-update result is in [full findings v01](planar_2d_full_findings_v01.md). No genuine two-dimensional benchmark is included.

The physical square is x,y in [-5,5] m. At t=0, depth is 1 m for x<0 and 0.1 m for x>0, with u=v=0. The horizon is 0.5 s, gravity is 9.81 m/s², and the bed is flat with no friction or rain. The prescribed initial volume is 55 m³. The existing verified E1 reference and wave-safety implementation are reused without modification. Its bibliography and acceptance evidence remain in the core records.

The x boundaries retain reservoir states (h,qx,qy)=(1,0,0) and (0.1,0,0) in SI units. At the y walls, only qy=0 is supervised. No analytical h or qx profile is supplied on these walls or in the interior.

The new network accepts (x/Lc,y/Lc,t/Tc). Four hidden layers contain 64 tanh units each. It produces depth logit and two unconstrained velocities. Depth is softplus(logit)+1e-6 in nondimensional units; qx=h*u and qy=h*v. The architecture imposes no y invariance. H=1 m, Lc=5 m for both spatial axes, Uc=sqrt(gH), and Tc=Lc/Uc.

With nondimensional states, the three residuals are

```
rh = h_t + qx_x + qy_y
rx = qx_t + (qx²/h + h²/2)_x + (qx*qy/h)_y
ry = qy_t + (qx*qy/h)_x + (qy²/h + h²/2)_y
```

Both spatial derivative chain factors are 1/Lc; the time factor is 1/Tc. Physical mass residual is (H/Tc)*rh in m/s. Both physical momentum residuals are (H*Uc/Tc) times their nondimensional values, in m²/s².

CPU float32 uses four threads. Adam uses lr 0.001 for a proposed 10,000 updates. There is no LBFGS, adaptation, artificial viscosity or interior solution supervision. The fixed objective is Lpde+10Lic+10Lbc. PDE loss averages the sum over all three squared residuals. IC loss averages the sum over all three state errors. Boundary loss is an equal-point average over the four edges, supervising all three fields at x boundaries and qy only at y boundaries.

Fixed NumPy PCG64 streams use seed 110101 for training and 110102 for validation. Training has 10,000 PDE, 1,000 IC and 500 points per edge. Validation has 2,000 PDE, 400 IC and 200 per edge. The two complete coordinate memberships are saved and checked for exact overlap. Interior times are positive; IC values retain the unsmoothed jump and exclude x=0. Validation uses the same physics/IC/BC definition with independent coordinates. The initial state and every update are candidates. The earliest strict minimum of the fixed validation objective selects the checkpoint. Analytical model errors never select checkpoints or settings.

Every step records all ten equation/field/edge terms, the three grouped losses, objective, validation values, selection flag and measured costs. Initial, selected and latest checkpoints retain step, score, initialization identity, optimizer/RNG state and protocol/membership hashes. Actual writes are counted. Reload verifies both latest predictions and selected validation score. A separate saved-history readback verifies all step numbers, selection flags, the earliest minimum and checkpoint ancestry.

Final point axes have 400 x centres, 21 y centres and 50 times 0.01 through0.50 s. Arrays are explicitly shaped(time,y,x), with separate x_m,y_m,t_s and h_m,qx_m2_s,qy_m2_s. Edges define midpoint quadrature cells, not cell averages. The verified analytical E1 h/qx is extruded in y; qy=0. All y slices are compared at every saved time. Whole-domain, shock-band |x-s*t|<=0.1 m and outside-band diagnostics retain RMSE and relative L2 percent for h/qx. Velocities use the fixed reference-wet mask h_ref>0.001 m with coverage. Absolute qy/v errors are reported; their relative errors are undefined because their reference is zero. Y variation is the maximum absolute and RMS deviation from each field's arithmetic y mean, globally and per time, for h,qx,qy,u,v.

Conservation uses volume at t=0 and all 50 reporting times. Base/refined volume midpoint grids are 800x21 and 1600x42. Tangential edge grids are x=400/y=21 and x=800/y=42. Time quadrature uses 1,001 and 2,001 nodes, unioned with the exact reporting times. Outward rates are left=-integral(qx dy), right=+integral(qx dy), bottom=-integral(qy dx), top=+integral(qy dx). Trapezoidal time integrals produce M(t)-55+integral(sum outward rates). Predicted initial bias M(0)-55, bias-subtracted balance, each signed edge contribution and integrated absolute normal mismatch are retained. The simultaneous spatial/time refinement is combined sensitivity; it cannot identify separate axis effects or establish an error bound. Sampled maxima are not continuous-time extrema.

Dense neural queries stream one time slab and at most 8,192 points per forward pass. The pilot executes the complete final/reference and base/refined volume/four-edge query and compressed-output workload. It does not compute analytical neural-model errors. Diagnostic/JSON overhead is measured separately using analytical reference against itself, with no network fields in that fixture. Raw step costs, memory/load observations, checkpoint writes and query costs support the complete-cost projection. Memory is a process-lifetime peak RSS observation; it does not measure system pressure or every allocator peak. The single pilot is not repeated to improve timing.

The machine constraint is the available M5 Air. The unavailable Victus/RTX2050, CUDA and paid cloud are not prerequisites. All existing core/hybrid modules, runs and archives remain protected. Only the named additive CLI registration changes an existing module. The full fit, any final accuracy finding, genuine 2D, friction and inverse work were outside the pilot scope.
