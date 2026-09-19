from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "src" / "original"
DATA = ROOT / "data" / "processed" / "active_mnq.pkl"
RESULTS = ROOT / "results"
RUNS = RESULTS / "_runs"

SCRIPTS = {
    "detailed-1m": "fvg_final_fast.py",
    "multi-tf": "fvg_strength_one_tf.py",
    "midpoint": "fvg_midpoint_reaction.py",
    "ce-body": "fvg_ce_rejection_study.py",
}
SOURCE_HASHES = {
    "fvg_ce_rejection_study.py": "bd9714711a61493a9a1e207364d2b7e59d7eb5b9306526e36e2fc0e645187d41",
    "fvg_final_fast.py": "f1f896c70d222cdae6eeda3e22cf24c9778b183922a96ef454528a84de6adef0",
    "fvg_midpoint_reaction.py": "9ce25efc05b6ecb13c29102c6b50abf825a0b23d2b363d69acc346f82db1036e",
    "fvg_strength_one_tf.py": "45dd13a28bcebabaaf49320d53f2bcac2c9c816761a9cdfeba2b792bc92e752c",
}
VALID_TFS = {1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 720, 1440}


class PathRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.replacements = 0

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, str):
            return node
        legacy = "/mnt" + "/data"
        replacements = {
            legacy + "/active_mnq.pkl": str(DATA),
            legacy + "/fvg_study_outputs": str(RESULTS / "detailed_1m"),
            legacy + "/fvg_strength_project_single": str(RESULTS / "multi_tf"),
            legacy + "/fvg_ce_study": str(RESULTS / "ce_body"),
            legacy + "/midpoint_year_tf": str(RESULTS / "midpoint" / "midpoint_year_tf"),
            legacy + "/midpoint_tf": str(RESULTS / "midpoint" / "midpoint_tf"),
        }
        value = node.value
        for old, new in replacements.items():
            if value.startswith(old):
                self.replacements += 1
                return ast.copy_location(ast.Constant(new + value[len(old):]), node)
        return node


def patch_source(text: str, name: str) -> str:
    tree = ast.parse(text, filename=name)
    rewriter = PathRewriter()
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    legacy = "/mnt" + "/data"
    remaining = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and legacy in n.value]
    if remaining:
        raise ValueError(f"Unpatched canonical path(s) in {name}: {remaining}")
    if rewriter.replacements < 2:
        raise ValueError(f"Expected at least two canonical path constants in {name}; found {rewriter.replacements}")
    return ast.unparse(tree) + "\n"


def snapshot_outputs() -> dict[str, int]:
    if not RESULTS.exists():
        return {}
    return {
        str(path.relative_to(ROOT)): path.stat().st_mtime_ns
        for path in RESULTS.rglob("*") if path.is_file() and RUNS not in path.parents
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a canonical analysis script with portable local paths.")
    parser.add_argument("study", choices=SCRIPTS)
    parser.add_argument("--tf", type=int, help="Native timeframe for multi-tf or midpoint")
    parser.add_argument("--ce-tfs", help="Comma-separated CE timeframes in minutes")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[int]:
    if args.study in {"multi-tf", "midpoint"} and args.tf is None:
        raise ValueError(f"{args.study} requires --tf")
    if args.tf is not None and (args.study not in {"multi-tf", "midpoint"} or args.tf not in VALID_TFS):
        raise ValueError("--tf is only valid for multi-tf/midpoint and must be a supported minute value")
    if args.ce_tfs and args.study != "ce-body":
        raise ValueError("--ce-tfs is only valid for ce-body")
    values = [int(x.strip()) for x in args.ce_tfs.split(",") if x.strip()] if args.ce_tfs else sorted(VALID_TFS)
    if args.study == "ce-body" and (not values or len(values) != len(set(values)) or any(x not in VALID_TFS for x in values)):
        raise ValueError("--ce-tfs must contain unique supported minute values")
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ce_values = validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not DATA.is_file():
        print(f"ERROR: Missing {DATA}. Prepare the dataset first.", file=sys.stderr)
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    for folder in ("detailed_1m", "multi_tf", "midpoint", "ce_body"):
        (RESULTS / folder).mkdir(parents=True, exist_ok=True)
    source = ORIGINAL / SCRIPTS[args.study]
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_HASHES[source.name]:
        print(
            f"ERROR: Canonical source hash changed for {source.name}. Audit the research change "
            "and update the declared hash deliberately.", file=sys.stderr,
        )
        return 2
    try:
        code = patch_source(raw.decode("utf-8"), source.name)
    except Exception as exc:
        print(f"ERROR: Could not create portable source: {exc}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    if args.tf is not None:
        environment["TF"] = str(args.tf)
    if args.study == "ce-body":
        environment["CE_TFS"] = ",".join(map(str, ce_values))
    command_extra = [str(args.tf)] if args.study == "midpoint" else []
    before = snapshot_outputs()
    started = datetime.now(timezone.utc)
    status = "failed"
    exit_code = 1
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="fvg_canonical_") as temporary:
            patched = Path(temporary) / source.name
            patched.write_text(code, encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(patched), *command_extra], cwd=ROOT, env=environment
            )
            exit_code = process.returncode
            if exit_code:
                raise subprocess.CalledProcessError(exit_code, process.args)
            if args.study == "ce-body":
                suffix = "_" + "_".join(map(str, ce_values)) if len(ce_values) < len(VALID_TFS) else ""
                trade_file = RESULTS / "ce_body" / f"trades{suffix}.pkl"
                subprocess.run([
                    sys.executable, str(ROOT / "scripts" / "postprocess_body_bands.py"),
                    "--trade-file", str(trade_file),
                    "--output", str(RESULTS / "ce_body" / "body_depth_5pct.csv"),
                ], check=True, cwd=ROOT, env=environment)
            status = "success"
            exit_code = 0
    except Exception as exc:
        error = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            exit_code = exc.returncode
    finished = datetime.now(timezone.utc)
    after = snapshot_outputs()
    changed = sorted(path for path, mtime in after.items() if before.get(path) != mtime)
    manifest = {
        "study": args.study, "status": status, "exit_code": exit_code,
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "source": str(source.relative_to(ROOT)), "source_sha256": digest,
        "tf": args.tf, "ce_tfs": ce_values if args.study == "ce-body" else None,
        "data_path": str(DATA.relative_to(ROOT)), "data_mtime_ns": DATA.stat().st_mtime_ns,
        "generated_files": changed, "error": error,
    }
    run_name = f"{started.strftime('%Y%m%dT%H%M%S')}_{args.study}.json"
    (RUNS / run_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RUNS / f"latest_{args.study}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if status != "success":
        print(f"ERROR: Canonical experiment failed: {error}", file=sys.stderr)
    else:
        print(f"Run manifest: {RUNS / run_name}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
