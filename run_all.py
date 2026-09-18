from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "scripts" / "run_original.py"

def call(*args):
    print("\n===", " ".join(map(str, args)), "===", flush=True)
    subprocess.run([sys.executable, str(RUN), *map(str, args)], check=True, cwd=ROOT)

call("detailed-1m")
for tf in (1, 5, 15, 60, 240):
    call("multi-tf", "--tf", tf)
for tf in (1, 5, 15, 60, 240):
    call("midpoint", "--tf", tf)
call("ce-body")

print("\nAll canonical experiments completed. See results/.")
