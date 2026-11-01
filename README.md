# Shallow-water dam-break PINN study

This project compares a physics-informed neural network (PINN) with analytical shallow-water solutions and an unlimited two-step Lax–Wendroff solver. It studies water depth, discharge per unit width, mass balance, and computational cost in controlled dam-break cases.

The repository contains the research implementation, frozen configurations, tests, generated scientific data, saved results, and reproduction materials.

## Project team

| Member | Contribution |
| --- | --- |
| [Kenny Ivan Zamora](https://github.com/Keni-Dev) | Code and implementation; project setup and testing |
| [Lorenz Velasco](https://github.com/marina21-cs) | Code and implementation |
| [Laurice Ann San Jose](https://github.com/laurice15) | Research and data gathering |
| [Kristian Clint Autor](https://github.com/keyziii-phy) | Research and data gathering |
| Alexis Mae Reyes | Video production |
| Ellariecia Valencia | Video production |

[Contributions](CONTRIBUTIONS.md) describes the shared implementation work.

## Study and findings

The core study uses a flat, frictionless channel from −5 to 5 m and follows the first 0.5 s after barrier release. E1 starts with 1 m of water upstream and 0.1 m downstream. E2 uses the same upstream depth and exactly zero downstream depth. These are generated benchmarks; no field dataset is used.

The frozen protocol, `core-v1-pilot-a03`, contains 24 stages and seven 10,000-step training jobs. E1 compares fixed and adaptive weighting with paired seeds 11, 22, and 33; E2 has one fixed-weight diagnostic. Each network has four hidden layers of 64 tanh units. Training uses CPU float32; analytical and numerical calculations use CPU float64.

The accepted core results contain 7,650 metric rows, 36 auxiliary mass/boundary records, 39 timing groups, and 35 figures.

- The predeclared 400-cell numerical comparator has lower pooled depth and discharge errors than every tested PINN in both cases.
- Adaptive weighting improves pooled depth and discharge errors for seed 33 and worsens them for seeds 11 and 22. Mass-balance changes do not consistently follow approximation-error changes.
- Resident network prediction is faster than a fresh numerical solve in the recorded local timing window. Cached numerical reconstruction is faster than every measured network forward pass. Training cost, unequal precision, and high background host load prevent a general speedup claim.

Selected extensions cover synthetic solution labels, planar flow, Manning friction, roughness sensitivity, artificial momentum diffusion, and genuine two-dimensional reservoir flow. Their outcomes are mixed. One genuine 2D fit has Gauss3 pooled integral relative L2 errors of **17.34% for depth, 56.40% for x-discharge, and 40.31% for y-discharge**, against numerical cell means. Its reference retains a **9.70% x-discharge change** in the coupled mesh/time refinement comparison. These studies establish neither field validation nor an operational flood-warning system. Finite training budgets do not establish convergence.

## Get started

Use CPython **3.12.14** and `uv`. Download the scientific bundles before replaying results. Private release downloads require GitHub CLI access to this repository; the reproducibility guide also gives a local-download route. From the repository root:

```sh
uv sync --frozen --python 3.12.14
uv run --frozen python tools/fetch_artifacts.py --groups all
uv run --frozen python tools/fetch_artifacts.py --groups all --verify-only
uv run --frozen python -m hydro_pinn --help
uv run --frozen python -m hydro_pinn validate-config --config configs/cases/e1_wet.json
uv run --frozen python -m hydro_pinn validate-config --config configs/cases/e2_dry.json
uv run --frozen pytest
uv run --frozen ruff check src tests tools
uv run --frozen ruff format --check src tests tools --exclude src/hydro_pinn/cli.py
```

The formatter preserves the frozen CLI file byte for byte; its historical formatting is excluded from the format-only check, while lint still checks it.

[ARTIFACTS.json](ARTIFACTS.json) lists the eight checksum-bound release bundles and their scientific files. The downloader restores their original relative paths and verifies them. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates software checks, saved-data replay, and optional fresh full-budget reproduction. The portable saved-core helper regenerates metrics and figures without training or taking new timing measurements, while verifying source and input hashes. The original strict execution commands also bind the historical Python build; using the same version number alone does not satisfy that additional check. A previous full repeat matched the scientific outputs on the same host in a new locked environment; this does not establish independent human or cross-platform reproduction.

## Research files

| Location | Contents |
| --- | --- |
| [`src/hydro_pinn/`](src/hydro_pinn/) | Analytical references, numerical solver, PINN training, evaluation, plotting, and extension implementations |
| [`configs/`](configs/) | Cases, evaluation axes, training settings, and frozen protocols |
| [`tests/`](tests/) | Configuration, numerical, model, workflow, and regression checks |
| [`docs/final_run_manifest.json`](docs/final_run_manifest.json) | Completed core run IDs and artifact bindings |
| [`results/metrics.json`](results/metrics.json) | Accepted physical-error and conservation measurements |
| [`results/timings.json`](results/timings.json) | Raw timing samples, summaries, costs, and measurement limits |
| [`results/figures/`](results/figures/) | Accepted PNG/PDF figures |
| [`docs/figure_manifest.json`](docs/figure_manifest.json) | Complete 35-figure ledger and provenance |
| [`runs/`](runs/) | Resolved inputs, scientific arrays, checkpoints, and diagnostics required by the retained study records |
| [`docs/references.md`](docs/references.md) | Bibliography and source-use notes |
| [`ARTIFACTS.json`](ARTIFACTS.json) | Release-bundle checksums, file inventories, and original extraction paths |
| [`replay_runtimes/`](replay_runtimes/) | Historical CLI identities for isolated hybrid, planar, friction, viscosity, and numerical-reference replay |

Preserve supplied scientific files and hashes. Use a new destination for every replay or experiment; existing output directories are refused. Third-party source excerpts retain their own license notices.
