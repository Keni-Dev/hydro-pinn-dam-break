# Saved-output sensitivity reanalysis

This helper reproduces the numerical analysis of the accepted roughness-sensitivity screen from its saved native outputs. It does not run the original sensitivity gate or make a fresh-solve claim.

## Run the saved-data route

Place `reanalyze_inverse_sensitivity.py` and its companion `inverse_sensitivity_saved_inputs_v01.json` together under `tools/`. From the repository root:

```sh
uv run --frozen python tools/reanalyze_inverse_sensitivity.py \
  --output-dir results/local-sensitivity-reanalysis-01
```

Use a new destination for every attempt. The helper refuses an existing output directory, checks every scientific input against the companion inventory, and refuses unsafe paths. The inventory lists 30 required files: 15 native output records, six saved gate files, the sensitivity configuration, and eight unchanged source dependencies. Their combined size is about 7.55 MB. The three n = 0.030 native records remain at their original friction-reference paths.

The output contains `analysis_recomputed.json`, `gauge_means_recomputed.npz`, and `reanalysis_report.json`. The report records verified input hashes, the helper's source hash, runtime versions, exact comparisons, and the route's limits. Input files remain unchanged.

## What is recomputed

The screen compares Manning n = 0.020, 0.025, 0.030, 0.035, and 0.040 s m⁻¹ᐟ³ at three native mesh/time settings. The unchanged conservative projection extracts three interval-mean depths at each of five declared times, producing 225 response values across all 15 records. The unchanged analysis function recomputes all ten coefficient-pair comparisons. Saved native flux/source arrays also reproduce the original mass and momentum budget diagnostics.

The numerical indicator is the sum of absolute mesh and time differences for each candidate. For a pair, the two indicators are summed as U; the adjusted vector is `max(abs(delta) - U, 0)`. Its L2 norm divided by the illustrative 0.01 m noise scale is a weak screen, not a certified uncertainty bound.

The two local adjusted distances to n = 0.030 remain **0.1230865672** and **0.1460702299**, below the one-unit threshold. The accepted outcome remains `unresolved_at_this_numerical_operator`. No observation noise is drawn, no inverse dataset or trainer is created, and no parameter recovery or intrinsic nonidentifiability claim follows.

Comparisons use exact saved-value equality. If the environment produces a difference, the new report records it as a failed match rather than changing the accepted data or selecting a looser tolerance from the result.

## What this route does not reproduce

The original `prepare`/`execute` interface includes historical source-version checks, four preservation-archive checks, immutable launch rules, and the bounded numerical-solve procedure. Those guards are unchanged. This helper does not call those entrypoints or certify their preservation conditions; it implements a separately named saved-data contract.

A fresh guard-bound execution still needs its matching historical source/CLI revision and complete original preservation inputs. The curated scientific inventory is sufficient for saved-output reanalysis only. It does not make that original fresh-execution route portable, repeat PDE integration, measure new performance, retrain a model, or establish independent physical validation.
