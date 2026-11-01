"""Command-line entry points for implemented research stages."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hydro_pinn import __version__


def build_parser() -> argparse.ArgumentParser:
    """Advertise only commands with implemented behavior."""
    parser = argparse.ArgumentParser(
        prog="hydro_pinn",
        description="One-dimensional dam-break research. Physical values use SI units.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-config",
        help="Validate a JSON configuration without creating a run or output files.",
        description=(
            "Validate a typed JSON case, training, evaluation, verification, or smoke "
            "configuration. Case coordinates/depths are metres, time is seconds, "
            "and discharge is m²/s per unit width. Prints validation to stdout; "
            "does not create runs/."
        ),
    )
    validate.add_argument("--config", type=Path, required=True, help="Input JSON path.")
    reference = subparsers.add_parser(
        "reference",
        help="Generate analytical wet/dry reference arrays on CPU float64.",
        description=(
            "Generate analytical point values h (m) and q (m²/s per unit width), "
            "with x (m) and t (s), using CPU float64. New evidence is written to "
            "runs/<run-id>/. Existing runs are preserved. The default grid is "
            "the separate implementation-verification profile."
        ),
    )
    reference.add_argument(
        "--config", type=Path, required=True, help="Validated physical case JSON path."
    )
    reference.add_argument(
        "--evaluation",
        type=Path,
        default=Path("configs/verification.json"),
        help="Evaluation JSON path (default: configs/verification.json).",
    )
    reference.add_argument(
        "--run-id", required=True, help="New safe identifier under runs/; no overwrite."
    )
    solve = subparsers.add_parser(
        "solve",
        help="Run unlimited two-step Lax-Wendroff on native cell averages.",
        description=(
            "Integrate flat-bed source-free SWE on CPU float64 with fixed reservoir "
            "ghost states. Save native cell-average h (m), q (m²/s), x (m), "
            "edges (m), exact requested t (s), and diagnostics under runs/<run-id>/. "
            "No clipping or limiter is applied. Numerical failure saves diagnostics "
            "and last valid state, exits 3, and creates no successful outputs.npz."
        ),
    )
    solve.add_argument(
        "--config", type=Path, required=True, help="Physical case JSON path."
    )
    solve.add_argument(
        "--evaluation",
        type=Path,
        default=Path("configs/verification.json"),
        help="Requested output times/grid JSON (default: configs/verification.json).",
    )
    solve.add_argument(
        "--cells",
        type=int,
        required=True,
        help="Even native cell count N; dx=(b-a)/N metres.",
    )
    solve.add_argument(
        "--run-id", required=True, help="New safe identifier under runs/; no overwrite."
    )
    train = subparsers.add_parser(
        "train",
        help="Train the physics-only scaled PINN with saved partitions.",
        description=(
            "Train the configured 1D PINN, preserving all inputs, loss history, "
            "separate validation selection, and checkpoint under runs/<run-id>/. "
            "Use --smoke-config for bounded pipeline checks. A finite budget stop "
            "does not establish convergence or final accuracy. "
            "No interior labels are used."
        ),
    )
    train.add_argument(
        "--case", type=Path, required=True, help="Physical SI case JSON."
    )
    train.add_argument(
        "--training", type=Path, required=True, help="Typed training JSON."
    )
    train.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Explicit shared initialization/sampling seed.",
    )
    train.add_argument(
        "--device",
        required=True,
        choices=["cpu", "cuda", "mps"],
        help="Requested backend; no silent fallback.",
    )
    train.add_argument(
        "--dtype",
        required=True,
        choices=["float32", "float64"],
        help="Explicit floating dtype; MPS float64 is invalid.",
    )
    train.add_argument(
        "--run-id", required=True, help="New safe identifier under runs/."
    )
    train.add_argument(
        "--smoke-config",
        type=Path,
        help="Declared bounded overrides and nonfinal coordinates.",
    )
    train.add_argument(
        "--resume-from",
        help="Trusted local source run ID; saves compatible continuation under new ID.",
    )
    train.add_argument(
        "--chunk-size",
        type=int,
        help="Optional fixed chunk count with exact component-mean gradients.",
    )
    predict = subparsers.add_parser(
        "predict",
        help="Reload a trusted selected checkpoint and save physical h/q.",
        description=(
            "Reload a local selected model with its recorded architecture/scales, "
            "then save pointwise x (m), t (s), h (m), and q (m²/s per unit width) "
            "to a new runs/<run-id>/ record. Typed smoke uses its evaluation section."
        ),
    )
    predict.add_argument(
        "--training-run",
        required=True,
        help="Completed trusted training run ID under runs/.",
    )
    predict.add_argument(
        "--evaluation",
        type=Path,
        required=True,
        help="Evaluation JSON or typed smoke JSON.",
    )
    predict.add_argument(
        "--run-id",
        required=True,
        help="New prediction run ID under runs/; no overwrite.",
    )
    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect observed software, devices, memory bytes and storage bytes.",
        description=(
            "Write local observed hardware/software inventory as JSON. Reported "
            "specifications and inaccessible devices remain separate from observations."
        ),
    )
    doctor.add_argument(
        "--output", type=Path, required=True, help="New JSON output path."
    )
    pilot = subparsers.add_parser(
        "pilot",
        help="Execute bounded precision, chunk and synchronized training checks.",
        description=(
            "Run the declared hardware pilot, preserving attempts under new runs/<id>/ "
            "paths. Write findings JSON with numerical tolerances, actual steps, "
            "timings in seconds and memory in bytes; no final analytical error tuning."
        ),
    )
    pilot.add_argument("--config", type=Path, required=True, help="Pilot JSON path.")
    pilot.add_argument(
        "--output", type=Path, required=True, help="New findings JSON path."
    )
    freeze = subparsers.add_parser(
        "freeze-protocol",
        help="Validate and freeze the final scientific settings and stage matrix.",
        description=(
            "Write a new immutable protocol with scientific source/input hashes, "
            "paired settings, final evaluation coordinates and conservation axes. "
            "This command creates no training runs or execution manifest."
        ),
    )
    freeze.add_argument(
        "--config", type=Path, required=True, help="Candidate JSON path."
    )
    freeze.add_argument(
        "--output", type=Path, required=True, help="New protocol JSON path."
    )
    matrix = subparsers.add_parser(
        "run-matrix",
        help="Execute or reconcile a frozen matrix with preserved run attempts.",
        description=(
            "Check frozen inputs/software/scientific sources before work. Record "
            "real progress, checkpoints and outcomes in the protocol's manifest. "
            "Compatible failed/interrupted training may resume under a new run ID. "
            "A source still marked running is refused; existing runs are preserved."
        ),
    )
    matrix.add_argument(
        "--protocol", type=Path, required=True, help="Frozen JSON path."
    )
    matrix.add_argument(
        "--resume-stage", help="Frozen training stage ID; requires both resume options."
    )
    matrix.add_argument(
        "--resume-from", help="Compatible terminal source run ID; never overwritten."
    )
    matrix.add_argument(
        "--resume-run-id", help="New attempt ID for the resumed training stage."
    )
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a complete frozen matrix and save physical error/mass metrics.",
        description=(
            "After the coordinating root releases evaluation, verify the terminal "
            "manifest and source hashes; save common-grid errors plus separate "
            "checkpoint-bound dense conservation records. No training or checkpoint "
            "selection occurs. Preserve prior JSON/CSV results."
        ),
    )
    evaluate.add_argument(
        "--manifest", type=Path, required=True, help="Complete run manifest JSON."
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New metrics JSON; sibling CSV also written.",
    )
    plot = subparsers.add_parser(
        "plot",
        help="Regenerate figures from saved metrics and hash-checked run arrays.",
        description=(
            "Save PNG/PDF figures with SI units, run provenance, failed-run status, "
            "and exact profile times. Existing figure directories/manifests are "
            "preserved. This command does not retrain or generate new model outputs."
        ),
    )
    plot.add_argument(
        "--manifest", type=Path, required=True, help="Complete run manifest JSON."
    )
    plot.add_argument(
        "--output-dir", type=Path, required=True, help="New figure export directory."
    )
    plot.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics.json"),
        help="Matching metrics JSON (default: results/metrics.json).",
    )
    plot.add_argument(
        "--figure-manifest",
        type=Path,
        default=Path("docs/figure_manifest.json"),
        help="New figure provenance JSON (default: docs/figure_manifest.json).",
    )
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Measure matched physical h/q workloads after all study training stops.",
        description=(
            "After the coordinating root releases an idle timing window, require a "
            "complete frozen manifest. Measure 400 centres by 50 times with ten "
            "warmups and thirty repetitions, device synchronization, raw samples, "
            "median/IQR, and separate setup/training/cache costs."
        ),
    )
    benchmark.add_argument(
        "--protocol", type=Path, required=True, help="Frozen protocol JSON."
    )
    benchmark.add_argument(
        "--manifest", type=Path, help="Optional namespaced replay manifest JSON."
    )
    benchmark.add_argument(
        "--output", type=Path, required=True, help="New raw timing/summary JSON."
    )
    reproduce = subparsers.add_parser(
        "reproduce",
        help="Run reduced smoke or fresh frozen-budget paper reproduction.",
        description=(
            "Validate frozen inputs, code, environment, and parent manifest. "
            "Smoke executes declared reductions and cannot establish paper "
            "reproduction. Paper requires completed parent metrics/timings and "
            "executes the full matrix under new IDs. Reports preserve ancestry, "
            "saved-output replay scope, numerical differences, and failures. "
            "Existing destinations are refused."
        ),
    )
    reproduce.add_argument(
        "--protocol", type=Path, required=True, help="Frozen parent protocol JSON."
    )
    reproduce.add_argument(
        "--mode",
        choices=["smoke", "paper"],
        required=True,
        help="Explicit reduced pipeline check or unchanged paper-budget execution.",
    )
    reproduce.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New report directory; original runs and results remain preserved.",
    )
    hybrid_labels = subparsers.add_parser(
        "extension-hybrid-labels",
        help="Save separate analytical E1 labels and held-out axes (SI units).",
        description=(
            "Generate analytical h [m] and q [m²/s] labels at declared x [m], "
            "t [s] partitions under new runs/<run-id>/. Final axes have no values."
        ),
    )
    hybrid_labels.add_argument("--config", type=Path, required=True)
    hybrid_labels.add_argument("--run-id", required=True)
    hybrid_run = subparsers.add_parser(
        "extension-hybrid-run",
        help="Run the separate E1 seed11 physics-only/hybrid matched pair.",
        description=(
            "Freeze the additive protocol and save new matched runs under runs/. "
            "Default budget is 10000 Adam steps each. --pilot-steps bounds a "
            "full-coverage feasibility run; pilots cannot be evaluated for accuracy."
        ),
    )
    hybrid_run.add_argument("--config", type=Path, required=True)
    hybrid_run.add_argument("--labels-run", required=True)
    hybrid_run.add_argument("--manifest", type=Path, required=True)
    hybrid_run.add_argument("--pilot-steps", type=int, help="1..30 steps per variant.")
    hybrid_run.add_argument(
        "--protocol", type=Path, help="Existing immutable protocol."
    )
    hybrid_evaluate = subparsers.add_parser(
        "extension-hybrid-evaluate",
        help="Evaluate a completed full pair on its frozen label-held-out grid.",
        description=(
            "Write separate final h/q and reference-wet velocity errors, shock "
            "diagnostics, mass/boundary balance, arrays and costs in a new directory. "
            "No fitting or checkpoint reselection; pilot evaluation is refused."
        ),
    )
    hybrid_evaluate.add_argument("--manifest", type=Path, required=True)
    hybrid_evaluate.add_argument("--output-dir", type=Path, required=True)
    planar = subparsers.add_parser(
        "extension-2d-planar",
        help="Run the single bounded planar E1 consistency pilot (SI x,y,t).",
        description=(
            "Freeze the proposed full protocol and save a 25-step CPU32 pilot "
            "with three residuals, four boundaries, validation and query costs. "
            "Final analytical model errors are not computed. Full fitting needs "
            "a separate hash-bound root technical release."
        ),
    )
    planar.add_argument("--config", type=Path, required=True)
    planar.add_argument("--run-id", required=True)
    friction_reference = subparsers.add_parser(
        "extension-friction-reference",
        help="Create a separate wet flat-bed HLLE/Manning refinement packet (SI).",
        description=(
            "Run the fixed CPU64 reference matrix, storing native cell averages, "
            "actual mass/momentum budgets and common-cell uncertainty under a "
            "fresh runs/<run-id>/. No clipping or core-comparator changes."
        ),
    )
    friction_reference.add_argument("--config", type=Path, required=True)
    friction_reference.add_argument("--run-id", required=True)
    friction_run = subparsers.add_parser(
        "extension-friction-run",
        help="Run the single 25-update wet flat-bed Manning cost pilot.",
        description=(
            "Freeze the additive forward protocol and measure full CPU32 query, "
            "conservation and output costs. No pilot neural accuracy screening. "
            "Full10000updates require --protocol and a hash-bound root --release."
        ),
    )
    friction_run.add_argument("--config", type=Path, required=True)
    friction_run.add_argument("--manifest", type=Path, required=True)
    friction_run.add_argument("--run-id", required=True)
    friction_run.add_argument("--protocol", type=Path)
    friction_run.add_argument("--release", type=Path)
    friction_evaluate = subparsers.add_parser(
        "extension-friction-evaluate",
        help="Evaluate a completed released friction fit using common-cell averages.",
        description=(
            "Save separate h/q/reference-wet cell-ratio velocity errors, numerical "
            "front bands, two-boundary mass correction and quadrature sensitivity. "
            "Refuses pilots and never reselects checkpoints."
        ),
    )
    friction_evaluate.add_argument("--manifest", type=Path, required=True)
    friction_evaluate.add_argument("--output-dir", type=Path, required=True)
    viscosity_run = subparsers.add_parser(
        "extension-viscosity-run",
        help="Run the single 25-update artificial momentum diffusion cost pilot.",
        description=(
            "Freeze E1 seed11/nu*=.001, verify the zero limit and precision, "
            "then measure training and complete raw queries without accuracy "
            "screening. Full10000updates require frozen --protocol and --release."
        ),
    )
    viscosity_run.add_argument("--config", type=Path, required=True)
    viscosity_run.add_argument("--manifest", type=Path, required=True)
    viscosity_run.add_argument("--protocol", type=Path)
    viscosity_run.add_argument("--release", type=Path)
    viscosity_evaluate = subparsers.add_parser(
        "extension-viscosity-evaluate",
        help="Analyze a released full fit and its cached historical zero control.",
        description=(
            "Write pointwise inviscid-reference comparisons and two-boundary "
            "mass diagnostics in a new output directory. Refuses pilots before "
            "quality analysis; no fitting or checkpoint reselection."
        ),
    )
    viscosity_evaluate.add_argument("--manifest", type=Path, required=True)
    viscosity_evaluate.add_argument("--output-dir", type=Path, required=True)
    reference_2d = subparsers.add_parser(
        "extension-2d-reference",
        help="Run one explicitly released wet rectangular 2D reference cell.",
        description=(
            "Advance one named float64 NumPy HLLE cell after a matching root "
            "protocol release. Preserve partial states on budget/failure. "
            "This command never runs the four-cell matrix automatically."
        ),
    )
    reference_2d.add_argument("--config", type=Path, required=True)
    reference_2d.add_argument("--run-id", required=True)
    reference_2d.add_argument(
        "--protocol", type=Path,
        default=Path(
            "results/extensions/two_dimensional/reference_v01/"
            "candidate_protocol_v01/protocol.json"
        ),
    )
    reference_2d.add_argument("--release", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI commands and give actionable errors without tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extension-2d-reference":
        from hydro_pinn.extensions.reference_2d import run_reference

        try:
            output = run_reference(
                args.config, args.run_id, protocol_path=args.protocol,
                release_path=args.release,
            )
            state = json.loads((output / "status.json").read_text())["state"]
        except KeyboardInterrupt:
            print(
                "2D reference interrupted; partial evidence retained.", file=sys.stderr
            )
            return 130
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            print(f"2D reference failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"stage": args.command, "state": state,
                          "output": str(output)}))
        return 0 if state == "completed" else 3
    if args.command in {"extension-viscosity-run", "extension-viscosity-evaluate"}:
        try:
            if args.command == "extension-viscosity-run":
                from hydro_pinn.extensions.viscosity_experiment import run_experiment

                output = run_experiment(
                    args.config, args.manifest, args.protocol, args.release
                )
            else:
                from hydro_pinn.extensions.viscosity_experiment import (
                    evaluate_experiment,
                )

                output = evaluate_experiment(args.manifest, args.output_dir)
        except KeyboardInterrupt:
            print("Viscosity stage interrupted; evidence retained.", file=sys.stderr)
            return 130
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            print(f"Viscosity stage failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"completed": True, "output": str(output)}))
        return 0
    if args.command in {
        "extension-friction-reference",
        "extension-friction-run",
        "extension-friction-evaluate",
    }:
        try:
            if args.command == "extension-friction-reference":
                from hydro_pinn.extensions.friction_reference import run_reference
                from hydro_pinn.records import validate_run_id

                validate_run_id(args.run_id)
                if not args.run_id.startswith("p08_friction_"):
                    raise ValueError("fresh p08_friction_ run prefix required")
                output = Path("runs") / args.run_id
                run_reference(args.config, output)
            elif args.command == "extension-friction-run":
                from hydro_pinn.extensions.friction_experiment import run_friction

                output = run_friction(
                    args.config,
                    args.manifest,
                    run_id=args.run_id,
                    protocol_path=args.protocol,
                    release_path=args.release,
                )
            else:
                from hydro_pinn.extensions.friction_experiment import evaluate_friction

                output = evaluate_friction(args.manifest, args.output_dir)
        except KeyboardInterrupt:
            print("Friction extension interrupted; evidence retained.", file=sys.stderr)
            return 130
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            print(f"Friction extension failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"completed": True, "stage": args.command,
                          "output": str(output)}))
        return 0
    if args.command == "extension-2d-planar":
        from hydro_pinn.extensions.planar_2d import run_planar

        try:
            output = run_planar(args.config, args.run_id)
        except KeyboardInterrupt:
            print("Planar extension interrupted; evidence retained.", file=sys.stderr)
            return 130
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            print(f"Planar extension failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"completed": True, "stage": args.command,
                          "output": str(output)}))
        return 0
    if args.command in {
        "extension-hybrid-labels",
        "extension-hybrid-run",
        "extension-hybrid-evaluate",
    }:
        try:
            if args.command == "extension-hybrid-labels":
                from hydro_pinn.extensions.hybrid_data import generate_labels

                output = generate_labels(args.config, args.run_id)
            elif args.command == "extension-hybrid-run":
                from hydro_pinn.extensions.hybrid_experiment import run_pair

                output = run_pair(
                    args.config,
                    args.labels_run,
                    args.manifest,
                    pilot_steps=args.pilot_steps,
                    protocol_path=args.protocol,
                )
            else:
                from hydro_pinn.extensions.hybrid_experiment import evaluate_pair

                output = evaluate_pair(args.manifest, args.output_dir)
        except KeyboardInterrupt:
            print("Hybrid extension interrupted; evidence retained.", file=sys.stderr)
            return 130
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            print(f"Hybrid extension failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"completed": True, "stage": args.command, "output": str(output)}
            )
        )
        return 0
    if args.command == "reproduce":
        from hydro_pinn.reproduce import reproduce

        try:
            report = reproduce(args.protocol, args.mode, args.output_dir)
        except KeyboardInterrupt:
            print(
                "Reproduction interrupted; report and runs retained.", file=sys.stderr
            )
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"Reproduction failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "state": report["state"],
                    "mode": report["mode"],
                    "paper_reproduction_established": report[
                        "paper_reproduction_established"
                    ],
                    "execution_namespace": report["execution_namespace"],
                    "report": str(args.output_dir / "report.json"),
                },
                allow_nan=False,
            )
        )
        return (
            0 if report["state"] in {"completed", "completed_with_diagnostics"} else 3
        )
    if args.command in {"evaluate", "plot", "benchmark"}:
        try:
            if args.command == "evaluate":
                from hydro_pinn.evaluation_workflow import evaluate_manifest

                report = evaluate_manifest(
                    args.manifest, args.output, operationally_ready=True
                )
                summary = {
                    "state": report["state"],
                    "rows": len(report["rows"]),
                    "scientific_status": report["scientific_status"],
                    "output": str(args.output),
                }
            elif args.command == "plot":
                from hydro_pinn.evaluation_workflow import plot_manifest

                report = plot_manifest(
                    args.manifest,
                    args.output_dir,
                    metrics_path=args.metrics,
                    figure_manifest_path=args.figure_manifest,
                )
                summary = {
                    "state": "completed",
                    "figures": len(report["figures"]),
                    "output_dir": str(args.output_dir),
                }
            else:
                from hydro_pinn.benchmark import benchmark_protocol

                report = benchmark_protocol(
                    args.protocol,
                    args.output,
                    manifest_path=args.manifest,
                    operationally_ready=True,
                )
                summary = {"state": report["state"], "output": str(args.output)}
        except KeyboardInterrupt:
            print(
                f"{args.command.capitalize()} interrupted; saved evidence retained.",
                file=sys.stderr,
            )
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"{args.command.capitalize()} failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, allow_nan=False))
        return 0 if summary["state"] == "completed" else 3
    if args.command == "freeze-protocol":
        from hydro_pinn.protocol import freeze_protocol

        try:
            protocol = freeze_protocol(args.config, args.output)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"Protocol freeze failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "frozen": True,
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol["protocol_sha256"],
                    "matrix_stages": len(protocol["matrix"]),
                    "output": str(args.output),
                },
                allow_nan=False,
            )
        )
        return 0
    if args.command == "run-matrix":
        from hydro_pinn.experiments import run_matrix

        resume = (args.resume_stage, args.resume_from, args.resume_run_id)
        if any(resume) and not all(resume):
            print(
                "Matrix failed: --resume-stage, --resume-from and --resume-run-id "
                "must be provided together",
                file=sys.stderr,
            )
            return 2
        try:
            manifest = run_matrix(
                args.protocol,
                resume_stage=args.resume_stage,
                resume_from=args.resume_from,
                resume_run_id=args.resume_run_id,
            )
        except KeyboardInterrupt:
            print(
                "Matrix interrupted; saved attempts remain inspectable.",
                file=sys.stderr,
            )
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"Matrix failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "protocol_id": manifest["protocol_id"],
                    "workflow_status": manifest["workflow_status"],
                    "scientific_status": manifest["scientific_status"],
                },
                allow_nan=False,
            )
        )
        return (
            0
            if manifest["workflow_status"]
            in {"completed", "completed_with_diagnostics"}
            else 3
        )
    if args.command in {"doctor", "pilot"}:
        from hydro_pinn.devices import run_pilot, write_doctor

        try:
            if args.command == "doctor":
                write_doctor(args.output)
            else:
                run_pilot(args.config, args.output)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"{args.command.capitalize()} failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"completed": True, "stage": args.command, "output": str(args.output)},
                allow_nan=False,
            )
        )
        return 0
    if args.command == "validate-config":
        from hydro_pinn.config import ConfigError, load_config

        try:
            config = load_config(args.config)
        except (ConfigError, OSError) as exc:
            print(f"Invalid configuration {args.config}: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "valid": True,
                    "config": str(args.config),
                    "type": config["config_type"],
                },
                allow_nan=False,
            )
        )
        return 0
    if args.command == "reference":
        from hydro_pinn.references import generate_reference_run

        try:
            path = generate_reference_run(args.config, args.evaluation, args.run_id)
        except (ValueError, OSError) as exc:
            print(f"Reference failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"completed": True, "provenance": "analytical", "run_path": str(path)},
                allow_nan=False,
            )
        )
        return 0
    if args.command == "solve":
        from hydro_pinn.lax_wendroff import NumericalRunFailed, generate_solver_run

        try:
            path = generate_solver_run(
                args.config, args.evaluation, args.cells, args.run_id
            )
        except NumericalRunFailed as exc:
            print(f"Numerical simulation failed: {exc}", file=sys.stderr)
            return 3
        except (ValueError, OSError) as exc:
            print(f"Solver failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "completed": True,
                    "provenance": "numerical_approximation",
                    "run_path": str(path),
                },
                allow_nan=False,
            )
        )
        return 0
    if args.command in {"train", "predict"}:
        from hydro_pinn.train import predict_run, train_run

        try:
            if args.command == "train":
                path = train_run(
                    args.case,
                    args.training,
                    args.seed,
                    args.device,
                    args.dtype,
                    args.run_id,
                    smoke_path=args.smoke_config,
                    resume_from=args.resume_from,
                    chunk_size=args.chunk_size,
                )
            else:
                path = predict_run(args.training_run, args.evaluation, args.run_id)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"{args.command.capitalize()} failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"completed": True, "stage": args.command, "run_path": str(path)},
                allow_nan=False,
            )
        )
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
