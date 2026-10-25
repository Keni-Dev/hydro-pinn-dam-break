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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI commands and give actionable errors without tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
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
