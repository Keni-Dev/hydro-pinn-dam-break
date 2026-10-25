# Phase09 depth-response sensitivity gate v01

This gate follows the adopted plan with SHA256 `298ebfd506094ceb42e9997d1c78602b87cadb52384e5469f4ee82eb4646a05c` and the separate release in `docs/coordination/extension_inverse_sensitivity_selection.json`. It screens numerical depth response before any inverse observation dataset or trainer. The selected08 forward experiment remains accepted with incomplete flow recovery.

The fixed wet E1 case has x=[-5,5] m, dam=0, hL=1 m, hR=.1 m, initial u=0, g=9.81 m/s² and T=.5 s. Positive constant Manning drag overrides the source-free template metadata. Bed and rain are zero; fixed reservoir ghost states refresh at every homogeneous step. The existing float64 CPU HLLE/Manning solver and conservative projection are imported unchanged. The Phase08 n=.03 wrapper remains frozen.

Five n values (.020,.025,.030,.035,.040 s m^(-1/3)) each have levels 1600/.45, 3200/.45, 3200/.225. The three accepted truth records are hash-checked and reused. Exactly twelve new solves cover the four other values. Each solve uses the exact accepted 50 positive times, including their floating-point representations, before extracting rows 9/19/29/39/49. It does not shorten the schedule to five outputs.

Native piecewise-constant depths are conservatively projected onto the accepted 400 common cells. Cells 220/240/260 are the intervals [.5,.525], [1,1.025], [1.5,1.525] m, with the stored floating-point edges retained. Flattening is time-major, then G1/G2/G3, giving 15 numerical interval means. These design-screen coordinates cannot become a final inverse holdout.

For each n, signed mesh and time differences remain separate. The indicator e is the sum of their absolute values. Each unordered pair a<b retains signed delta=mu_b-selected minus mu_a-selected, its contrasts at all three levels, U=e_a+e_b, and s=max(abs(delta)-U,0). Raw and adjusted RMS, maximum and L2, L2(U), visible-sample counts and numerical/noise flags remain in the report. This adjustment is a heuristic based on measured differences; it supplies no certified lower bound or confidence limit.

The illustrative noise assumption is independent Gaussian sigma=.01 m for each mean. There is no noise draw or averaging reduction of sigma. Distances use L2(delta)/sigma and L2(s)/sigma, comparing deterministic responses with one hypothetical noisy observation vector. Both truth-versus-.025 and truth-versus-.035 must reach adjusted distance one for the weak local screen. Outer pairs cannot rescue a failed local screen. All truth-versus-outer and nontruth pairs remain visible. Passing is neither a confidence statement nor a recovery guarantee.

The descriptive finite difference is J=(mu_.035-mu_.025)/.010, accompanied by J*.03/1 and mu_.035+mu_.025-2mu_.030. It is not a gradient check or exact derivative. No n, time, gauge, noise, threshold or refinement changes are allowed after results.

Preparation saves the source/config/operator/input hashes, full time and grid axes, candidate matrix and original preservation inventory before any solve. Execution claims an exclusive launch record and refuses a second launch. It serializes the twelve solves and saves native arrays, every actual-step flux, source impulse, mass/momentum budget, timestep, CFL and wet-state record returned by the accepted solver. It independently recomputes the saved budgets. A solver exception is retained without retry; its intermediate state is unavailable because the accepted solver raises without returning it.

The 60-second matrix budget includes native storage and extraction and is checked before each new solve. If crossed, remaining records stay explicitly unexecuted and the gate is budget-limited incomplete. Failed or missing levels remain incomplete evidence. No old result or failed attempt is replaced. Focused tests cover overlap/order extraction, pair adjustment and protection/no-overwrite guards; old science suites and neural fits are excluded.

Commands, from the project root with the existing interpreter:

```sh
rtk proxy .venv/bin/python -m pytest tests/test_inverse_sensitivity.py -q
rtk proxy .venv/bin/python -m hydro_pinn.extensions.inverse_sensitivity prepare
rtk proxy .venv/bin/python -m hydro_pinn.extensions.inverse_sensitivity execute
```

The execute command is a one-time released numerical action. Saved-data analysis and the profile figure do not generate more reference solutions. The final decision records weak/unresolved outcomes, costs, failure state and unchanged protected hashes. Any inverse observation partitions, noise draws, gradients, fitting or broader Phase09 completion require a separate frozen proposal and release.
