# Reproducing the study

Run commands from the repository root. Use a new output-directory name on each attempt; the tools refuse existing destinations.

## Environment and scientific inputs

Use CPython **3.12.14** with the supplied `uv.lock`:

```sh
uv sync --frozen --python 3.12.14
uv run --frozen python --version
uv run --frozen python -m hydro_pinn --help
```

The existing `.python-version` selects the 3.12 series; the explicit sync argument selects the tested patch release. The lock resolves NumPy, SciPy, Matplotlib, and PyTorch. The original strict execution route additionally compares Python build metadata, not just the version number. A different build is refused. The portable saved-array route below records the actual runtime separately and retains all numerical-input checks; it does not claim to reproduce the historical execution environment. macOS uses the PyPI PyTorch build; other platforms use the configured CPU index. CUDA installation and execution were not validated for this study.

## Restore the scientific bundles

[ARTIFACTS.json](ARTIFACTS.json) binds the release ZIPs and the files they restore. The eight groups are:

| Group | Scientific material |
| --- | --- |
| `core-data` | Core run arrays, scientific records, and result figures |
| `hybrid-data` | Hybrid/control records, labels, queries, and results |
| `planar-data` | Planar 2D run, evaluation records, and figures |
| `friction-data` | Manning reference matrix, forward-model evaluation, and figures |
| `sensitivity-data` | Saved coefficient-response matrix and sensitivity analysis |
| `viscosity-data` | Zero/positive-viscosity comparison records and results |
| `genuine-2d-reference-data` | Genuine 2D numerical matrix and refinement analysis |
| `genuine-2d-neural-data` | Genuine 2D selected-model queries, diagnostics, and results |

For private releases, sign into GitHub CLI with an account that has repository access. Then restore every group and verify the extracted files:

```sh
gh auth login
uv run --frozen python tools/fetch_artifacts.py --groups all
uv run --frozen python tools/fetch_artifacts.py --groups all --verify-only
```

To use ZIPs already downloaded from the release, put them in a sibling directory named `hydro-pinn-bundles` and use:

```sh
uv run --frozen python tools/fetch_artifacts.py \
  --groups all \
  --local-dir ../hydro-pinn-bundles
```

A comma-separated selection restores individual groups, for example `--groups friction-data,sensitivity-data`. Use all groups for the complete set of replay examples below because studies can depend on another group's reference or historical control. The downloader checks archive and member hashes before accepting their contents at the recorded project-relative paths. `--verify-only` checks restored files without downloading them.

Checkpoint hashes alone are insufficient; the corresponding trusted checkpoint files must be present. The generated cases use SI units and retain their point-value or cell-average semantics. Keep the exactly dry analytical state separate from the network's positive depth floor.

## 1. Software checks

```sh
uv run --frozen python -m hydro_pinn validate-config --config configs/cases/e1_wet.json
uv run --frozen python -m hydro_pinn validate-config --config configs/cases/e2_dry.json
uv run --frozen pytest
uv run --frozen ruff check src tests tools
uv run --frozen ruff format --check src tests tools --exclude src/hydro_pinn/cli.py
```

Tests validate software contracts and bounded fixtures; they do not execute the frozen seven-job production matrix. Some tests read protocol and historical-control records as well as source/configuration files. Temporary test protocols explicitly bind the current runtime and identify their accepted source protocol; the supplied study protocol and strict runtime-drift rejection tests are unchanged. The one frozen CLI file is excluded from format-only rewriting to preserve its scientific source hash; lint still checks it.

## 2. Replay saved core data

This portable route recomputes the 7,650 metric rows and 34 core figures from stored arrays. It verifies the frozen protocol digest and 381 input, configuration, numerical-source and data hashes before computation. It records both the actual and historical runtimes; it does not pretend that a different Python build matches the original execution.

```sh
uv run --frozen python tools/replay_core_saved.py \
  --output results/local-replay-01

uv run --frozen python docs/evidence/phase_06/rebuild_timing_accuracy.py \
  --metrics results/local-replay-01/metrics.json \
  --timings results/timings.json \
  --output-dir results/local-replay-01/figures \
  --evidence-output results/local-replay-01/timing_accuracy_readback.json
```

The first command checks every metric row, aggregate and mass diagnostic against the published results and writes `verification.json`. The second rebuilds the 35th timing-versus-accuracy figure from the original saved timing samples. Neither command trains a model, integrates a new numerical reference, runs model prediction, or measures new performance. Use `--no-figures` on the first command for a metrics-only check.

The original `hydro_pinn evaluate` and `plot` commands remain unchanged and enforce the historical runtime contract. Use them only with a matching frozen environment. Do not edit the accepted protocol to make a different environment appear identical. A fresh full-budget study under a different environment requires a separately declared protocol.

The accepted evaluation uses 400 common centres and 50 positive times. Velocity errors use a declared reference-wet mask. Neural quadrature and the numerical solver's native cell/flux accounting remain separate diagnostics. Exact numerical equality, rendered image-byte equality and historical-runtime reproduction are different checks.

## 3. Replay selected extension outputs

Genuine 2D neural checkpoint replay uses the current retained source:

```sh
uv run --frozen python -m hydro_pinn extension-2d-evaluate \
  --run runs/p10_genuine_2d_full_v01 \
  --output-dir results/local-genuine2d-replay-01
```

This command runs neural inference. It does not train or reselect a checkpoint.

Hybrid, friction, planar, viscosity, and genuine 2D numerical-reference analysis bind an earlier CLI file. Follow the exact isolated-copy commands in [replay_runtimes/README.md](replay_runtimes/README.md). Do not replace the main checkout's CLI or edit a protocol digest to accept a different source version. Saved numerical-reference analysis repeats projection/analysis, not PDE integration.

The sensitivity route uses the separately named saved-output helper:

```sh
uv run --frozen python tools/reanalyze_inverse_sensitivity.py \
  --output-dir results/local-sensitivity-reanalysis-01
```

It re-extracts all 225 depth responses from 15 saved native outputs, checks native budgets, and recomputes all ten pair screens. The accepted unresolved decision remains unchanged. The [saved-reanalysis guide](docs/inverse_sensitivity_saved_reanalysis.md) states the exact comparison and scope.

The original sensitivity `prepare`/`execute` route requires a matching historical source/CLI revision and four original preservation archives. Those legacy archives are not included in the curated research downloads. The saved-data helper neither re-certifies those preservation conditions nor makes the original fresh-execution route portable. No source/hash guard is weakened.

## 4. Optional fresh core execution

The following commands create new runs and are not part of routine saved-output replay. Smoke mode uses explicit reductions and checks only the pipeline:

```sh
uv run --frozen python -m hydro_pinn reproduce \
  --protocol configs/final_protocol.json \
  --mode smoke \
  --output-dir results/reproduction/local-smoke-01
```

Paper mode keeps the frozen budget and performs all 24 stages, including seven 10,000-step training jobs, with fresh evaluation, figures, and timing:

```sh
uv run --frozen python -m hydro_pinn reproduce \
  --protocol configs/final_protocol.json \
  --mode paper \
  --output-dir results/reproduction/local-paper-01
```

Restore all canonical parent records first. Validation failures stop the route; do not relax its bindings. Allow a comparable training budget to the original matrix. Take new timing measurements on an otherwise idle machine and record the actual environment. The accepted timings were measured in a loaded-host window and remain a different observation.

The ordinary `run-matrix` command executes or reconciles the canonical matrix; use `reproduce` for a new execution namespace. A same-host repeat, saved-output replay, and another person's run on another machine are distinct results. Preserve failed and interrupted attempt records.
