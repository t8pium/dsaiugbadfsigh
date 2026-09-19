from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".fvg_venv"
REQ = ROOT / "requirements.txt"
PROJECT = ROOT / "pyproject.toml"
STAMP = VENV / ".fvg_environment.json"
SUPPORTED = {(3, 11), (3, 12), (3, 13)}


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def dependency_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (PROJECT, REQ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def venv_interpreter_supported(py: Path) -> bool:
    if not py.is_file():
        return False
    check = subprocess.run(
        [str(py), "-c", "import sys; raise SystemExit(0 if sys.version_info[:2] in {(3,11),(3,12),(3,13)} else 1)"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return check.returncode == 0


def installed_environment_is_healthy(py: Path, fingerprint: str) -> bool:
    if not py.is_file() or not STAMP.is_file():
        return False
    try:
        stamp = json.loads(STAMP.read_text(encoding="utf-8"))
        if stamp.get("fingerprint") != fingerprint:
            return False
        code = (
            "import pathlib,sys; "
            "import fvg_research,streamlit,databento,duckdb; "
            f"assert pathlib.Path(fvg_research.__file__).resolve().is_relative_to(pathlib.Path({str(ROOT)!r}).resolve()); "
            "assert sys.version_info[:2] in {(3,11),(3,12),(3,13)}"
        )
        probe = subprocess.run(
            [str(py), "-c", code], cwd=ROOT.parent,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return probe.returncode == 0
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    if sys.version_info[:2] not in SUPPORTED:
        raise SystemExit(
            "This research snapshot requires 64-bit Python 3.11, 3.12, or 3.13. "
            f"You are running {sys.version.split()[0]}. Install a supported version "
            "from https://www.python.org/downloads/ and relaunch."
        )
    if sys.maxsize <= 2**32:
        raise SystemExit("A 64-bit Python installation is required for the market dataset.")

    py = venv_python()
    if VENV.exists() and not venv_interpreter_supported(py):
        if VENV.resolve().parent != ROOT.resolve() or VENV.name != ".fvg_venv":
            raise SystemExit(f"Refusing to replace unexpected environment path: {VENV}")
        print("\nReplacing an incomplete or incompatible private environment...", flush=True)
        shutil.rmtree(VENV)
    if not py.is_file():
        print("\nCreating an isolated environment for the FVG study...", flush=True)
        run([sys.executable, "-m", "venv", str(VENV)])

    fingerprint = dependency_fingerprint()
    if not installed_environment_is_healthy(py, fingerprint):
        print("\nInstalling the pinned research environment (first launch can take several minutes)...", flush=True)
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "-r", str(REQ)])
        run([str(py), "-m", "pip", "check"])
        STAMP.write_text(
            json.dumps({"fingerprint": fingerprint, "python": sys.version.split()[0]}, indent=2),
            encoding="utf-8",
        )
    else:
        print("\nPinned environment already installed and healthy.", flush=True)

    print("\nLaunching the local FVG Research Lab...", flush=True)
    run([
        str(py), "-m", "streamlit", "run", str(ROOT / "dashboard.py"),
        "--server.headless=false", "--browser.gatherUsageStats=false",
    ])


if __name__ == "__main__":
    main()
