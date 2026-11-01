# Historical CLI runtimes for saved replay

Some extension protocols bind the exact CLI source that existed when their inputs were frozen. The retained implementation files otherwise match those protocols. Use the supplied historical CLI in a separate working copy; keep the main checkout unchanged.

| Study | CLI file | Required SHA-256 |
| --- | --- | --- |
| Hybrid/control pair | `hybrid/cli.py` | `b095e2e5ea0db5324e46c4f23a0167efb72d674d77ce5e076193d6b4a3d8ff16` |
| Planar 2D | `planar/cli.py` | `ef183d7152d94c7a67cb9f1f644cd7b7f444e2cfe2a25d77bcb731cedeb1d15f` |
| Manning friction | `friction/cli.py` | `0de87ee151e59fa1faf551101b6ec3aa637ce28016a555bf3424e67554de1da6` |
| Artificial viscosity | `viscosity/cli.py` | `1ac2c77f88a748891ef1c441f4b19b517aab33a5f3a6562392f1c84f7ccfa0e6` |
| Genuine 2D numerical reference | `genuine-reference/cli.py` | `35b4c1737e7f1c3f7f93c86969cd56b4677ed54a8384a8a9b7ae57df9601463d` |

The paths above are relative to `replay_runtimes/`. These are source identities, not alternate scientific configurations. Do not edit protocol hashes to admit a newer CLI.

## Restore data and make isolated replay copies

From the main repository root, first install the locked environment and restore all eight scientific groups:

```sh
uv sync --frozen --python 3.12.14
uv run --frozen python tools/fetch_artifacts.py --groups all
uv run --frozen python tools/fetch_artifacts.py --groups all --verify-only
```

The following inline launcher creates five fresh sibling directories, installs the matching CLI bytes in each copy, and runs the saved-data routes. It uses the already installed locked interpreter, loads project code from each copy's `src/`, and does not create a new dependency environment. It needs disk space for five copies of the restored research files.

Each destination is new and failures remain in their replay directory. Change `attempt` to a new suffix before a repeat. The main source CLI is checked before and after the sequence.

```sh
uv run --frozen python - <<'PYREPLAY'
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys

source = Path.cwd().resolve()
attempt = "01"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original_cli_sha = sha(source / "src/hydro_pinn/cli.py")
expected = {
    "hybrid": "b095e2e5ea0db5324e46c4f23a0167efb72d674d77ce5e076193d6b4a3d8ff16",
    "friction": "0de87ee151e59fa1faf551101b6ec3aa637ce28016a555bf3424e67554de1da6",
    "planar": "ef183d7152d94c7a67cb9f1f644cd7b7f444e2cfe2a25d77bcb731cedeb1d15f",
    "viscosity": "1ac2c77f88a748891ef1c441f4b19b517aab33a5f3a6562392f1c84f7ccfa0e6",
    "genuine-reference": "35b4c1737e7f1c3f7f93c86969cd56b4677ed54a8384a8a9b7ae57df9601463d",
}
commands = {
    "hybrid": [
        "-m", "hydro_pinn", "extension-hybrid-evaluate",
        "--manifest", "results/extensions/hybrid/p11_hybrid_full_v01.json",
        "--output-dir", "results/local-hybrid-replay-01",
    ],
    "friction": [
        "-m", "hydro_pinn", "extension-friction-evaluate",
        "--manifest", "docs/extensions/friction_full_manifest_v01.json",
        "--output-dir", "results/local-friction-replay-01",
    ],
    "planar": [
        "-c",
        "from hydro_pinn.extensions.planar_2d import evaluate_planar; "
        "evaluate_planar('runs/p10_planar_e1_seed11_full_v01', "
        "'results/local-planar-replay-01')",
    ],
    "viscosity": [
        "-m", "hydro_pinn", "extension-viscosity-evaluate",
        "--manifest", "results/extensions/viscosity/full_v01_manifest.json",
        "--output-dir", "results/local-viscosity-replay-01",
    ],
    "genuine-reference": [
        "docs/evidence/phase_10/genuine_reference_v01/remaining_matrix_v01/reference_analysis_v01.py",
        "--root", ".",
        "--output", "results/local-genuine-reference-replay-01",
        "--expected-script-sha256",
        "2246738fcedad37163f22d9efc968e23847dbbe1713a8336a2e218d2b336a5ed",
    ],
}
for study, command in commands.items():
    overlay = source / "replay_runtimes" / study / "cli.py"
    if sha(overlay) != expected[study]:
        raise ValueError(f"Historical CLI mismatch: {study}")
    target = source.parent / f"{source.name}-replay-{study}-{attempt}"
    if target.exists():
        raise FileExistsError(f"Use a new attempt suffix: {target.name}")
    shutil.copytree(
        source, target,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache"
        ),
    )
    shutil.copyfile(overlay, target / "src/hydro_pinn/cli.py")
    if sha(target / "src/hydro_pinn/cli.py") != expected[study]:
        raise ValueError(f"Copied CLI mismatch: {study}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(target / ".cache/matplotlib")
    subprocess.run(
        [sys.executable, "-B", *command], cwd=target, env=env, check=True
    )
    if sha(source / "src/hydro_pinn/cli.py") != original_cli_sha:
        raise ValueError("The main checkout CLI changed")
    print(f"Completed saved replay in {target.name}")
PYREPLAY
```

Hybrid, friction, planar, and viscosity replay selected checkpoints and may run neural forward evaluations. They do not train or reselect model parameters. The genuine-reference command recomputes analysis from saved numerical states; it does not repeat PDE integration. Existing scientific files and protocol bindings stay intact. Compare physical outputs separately from figure-file hashes, which can depend on the rendering environment.

## Legacy sensitivity boundary

The sensitivity study has a separate [saved-output reanalysis guide](../docs/inverse_sensitivity_saved_reanalysis.md). Its helper reads the 15 saved native records and recomputes the original screen without a solver, fit, or original gate execution.

The original sensitivity `prepare`/`execute` interface also requires a historical source revision and four preservation archives. Those archives are not included in the curated release bundles. None of the runtime overlays or saved-data commands above re-certifies those conditions or authorizes a fresh guard-bound execution. The original source and guards are unchanged.
