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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI commands and give actionable errors without tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
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
    parser.error(f"Unsupported command: {args.command}")
    return 2
