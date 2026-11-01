"""Restore checksum-verified saved research data from private release assets.

Requires Python 3.12. For remote downloads, GitHub CLI must be authenticated
with an account that can read this private repository. Existing different files
are refused rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {"runs", "results", "docs", "configs", "src", "replay_runtimes"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def destination(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    if pure.parts[0] not in ALLOWED_ROOTS:
        raise ValueError(f"Unexpected artifact root: {relative}")
    target = ROOT.joinpath(*pure.parts)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Artifact path leaves the project: {relative}")
    return target


def verify_existing(files: list[dict]) -> None:
    problems = []
    for item in files:
        path = destination(item["path"])
        if not path.is_file():
            problems.append(item["path"] + " (missing)")
        elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            problems.append(item["path"] + " (checksum mismatch)")
    if problems:
        raise ValueError("Saved data did not verify:\n" + "\n".join(problems))


def restore(archive: Path, spec: dict, group: str) -> None:
    if archive.stat().st_size != spec["bytes"] or digest(archive) != spec["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive.name}")
    expected = {item["path"]: item for item in spec["files"]}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        if set(names) != set(expected) | {"_artifact_manifest.json"}:
            raise ValueError("Archive entries differ from ARTIFACTS.json")
        inner = json.loads(bundle.read("_artifact_manifest.json"))
        if inner.get("group") != group or inner.get("files") != spec["files"]:
            raise ValueError("Archive manifest differs from ARTIFACTS.json")
        # Check all existing destinations and members before writing any file.
        for relative, item in expected.items():
            path = destination(relative)
            member = bundle.getinfo(relative)
            if member.is_dir() or member.file_size != item["bytes"]:
                raise ValueError(f"Unexpected member size: {relative}")
            data = bundle.read(relative)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Member checksum mismatch: {relative}")
            if path.exists() and (not path.is_file() or digest(path) != item["sha256"]):
                raise FileExistsError(
                    f"Refusing to overwrite different file: {relative}"
                )
        for relative in expected:
            path = destination(relative)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(bundle.read(relative))
    verify_existing(spec["files"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", default="all", help="all or comma-separated group names"
    )
    parser.add_argument(
        "--local-dir", type=Path, help="use already-downloaded ZIP files"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check restored data without downloading",
    )
    args = parser.parse_args()
    index = json.loads((ROOT / "ARTIFACTS.json").read_text())
    groups = list(index["bundles"]) if args.groups == "all" else args.groups.split(",")
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(g not in index["bundles"] for g in groups)
    ):
        parser.error("Choose all or known groups: " + ", ".join(index["bundles"]))
    for group in groups:
        spec = index["bundles"][group]
        if args.verify_only:
            verify_existing(spec["files"])
        elif args.local_dir:
            restore(args.local_dir / spec["filename"], spec, group)
        else:
            with tempfile.TemporaryDirectory(prefix="hydro-pinn-data-") as temporary:
                download = Path(temporary)
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        index["release_tag"],
                        "--repo",
                        index["repository"],
                        "--pattern",
                        spec["filename"],
                        "--dir",
                        str(download),
                    ],
                    check=True,
                )
                restore(download / spec["filename"], spec, group)
        print(f"{group}: {spec['file_count']} files verified", flush=True)


if __name__ == "__main__":
    main()
