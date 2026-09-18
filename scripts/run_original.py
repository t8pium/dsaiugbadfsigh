from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "src" / "original"
DATA = ROOT / "data" / "processed" / "active_mnq.pkl"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SCRIPTS = {
    "detailed-1m": "fvg_final_fast.py",
    "multi-tf": "fvg_strength_one_tf.py",
    "midpoint": "fvg_midpoint_reaction.py",
    "ce-body": "fvg_ce_rejection_study.py",
}

def patch_source(text: str, name: str) -> str:
    data = DATA.as_posix()
    text = text.replace("/mnt/data/active_mnq.pkl", data)
    text = text.replace("BASE='/mnt/data/active_mnq.pkl'", f"BASE={data!r}")
    text = text.replace("Path('/mnt/data/fvg_study_outputs')", f"Path({(RESULTS/'detailed_1m').as_posix()!r})")
    text = text.replace("Path('/mnt/data/fvg_strength_project_single')", f"Path({(RESULTS/'multi_tf').as_posix()!r})")
    text = text.replace("Path('/mnt/data/fvg_ce_study')", f"Path({(RESULTS/'ce_body').as_posix()!r})")
    text = text.replace("f'/mnt/data/midpoint_tf{tf}.csv'", f"f'{(RESULTS/'midpoint').as_posix()}/midpoint_tf{{tf}}.csv'")
    text = text.replace("f'/mnt/data/midpoint_year_tf{tf}.csv'", f"f'{(RESULTS/'midpoint').as_posix()}/midpoint_year_tf{{tf}}.csv'")
    if name == "midpoint":
        out = (RESULTS / "midpoint").as_posix()
        text = f"from pathlib import Path\nPath({out!r}).mkdir(parents=True, exist_ok=True)\n" + text
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=SCRIPTS)
    p.add_argument("--tf", type=int, help="Native timeframe in minutes for multi-tf or midpoint")
    p.add_argument("--ce-tfs", help="Comma-separated CE timeframes in minutes")
    args = p.parse_args()
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Run scripts/prepare_active_contract.py first.")
    src = ORIGINAL / SCRIPTS[args.study]
    code = patch_source(src.read_text(encoding="utf-8"), args.study)
    env = os.environ.copy()
    if args.tf is not None:
        env["TF"] = str(args.tf)
    if args.ce_tfs:
        env["CE_TFS"] = args.ce_tfs
    cmd_extra = [str(args.tf)] if args.study == "midpoint" and args.tf else []
    with tempfile.TemporaryDirectory() as td:
        patched = Path(td) / src.name
        patched.write_text(code, encoding="utf-8")
        subprocess.run([sys.executable, str(patched)] + cmd_extra, check=True, cwd=ROOT, env=env)

if __name__ == "__main__":
    main()
