from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".fvg_venv"
REQ = ROOT / "requirements.txt"
PROJECT = ROOT / "pyproject.toml"
STAMP = VENV / ".requirements_installed"

def run(cmd):
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"

def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required.")

    if not VENV.exists():
        print("\nCreating isolated environment for the FVG study...", flush=True)
        run([sys.executable, "-m", "venv", str(VENV)])

    py = venv_python()
    needs_install = (
        (not STAMP.exists())
        or (REQ.stat().st_mtime > STAMP.stat().st_mtime)
        or (PROJECT.stat().st_mtime > STAMP.stat().st_mtime)
    )

    if needs_install:
        print("\nInstalling required libraries. This is only needed on first launch or after requirements change.", flush=True)
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "-r", str(REQ)])
        run([str(py), "-m", "pip", "install", "-e", "."])
        STAMP.touch()
    else:
        print("\nDependencies already installed.", flush=True)
        # Refresh the local editable package every launch. This is fast and
        # prevents stale extracted-ZIP environments from losing local imports.
        run([str(py), "-m", "pip", "install", "-e", ".", "--no-deps"])

    print("\nLaunching the local FVG Research Lab...", flush=True)
    run([
        str(py), "-m", "streamlit", "run", str(ROOT / "dashboard.py"),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ])

if __name__ == "__main__":
    main()
