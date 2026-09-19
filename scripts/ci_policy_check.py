from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HASHES = {
    "src/original/fvg_ce_rejection_study.py": "bd9714711a61493a9a1e207364d2b7e59d7eb5b9306526e36e2fc0e645187d41",
    "src/original/fvg_final_fast.py": "f1f896c70d222cdae6eeda3e22cf24c9778b183922a96ef454528a84de6adef0",
    "src/original/fvg_midpoint_reaction.py": "9ce25efc05b6ecb13c29102c6b50abf825a0b23d2b363d69acc346f82db1036e",
    "src/original/fvg_strength_one_tf.py": "45dd13a28bcebabaaf49320d53f2bcac2c9c816761a9cdfeba2b792bc92e752c",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def main() -> int:
    reference = json.loads((ROOT / "reference_results/reference_metrics.json").read_text(encoding="utf-8"))
    assert reference["study"]["active_1m_rows"] == 2_303_483
    assert len(reference["experiments"]) == 9

    for relative, expected in ORIGINAL_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, f"Canonical source changed without audit: {relative}"

    forbidden_data = []
    forbidden_paths = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("data/raw/", "data/processed/", "data/uploads/")):
            forbidden_data.append(relative)
        if path.suffix.lower() in {".dbn", ".zst", ".pkl", ".parquet", ".pq"} and not relative.startswith("tests/"):
            forbidden_data.append(relative)
        if not relative.startswith("src/original/") and path.is_file():
            try:
                if ("/mnt" + "/data") in path.read_text(encoding="utf-8"):
                    forbidden_paths.append(relative)
            except UnicodeDecodeError:
                pass
    assert not forbidden_data, f"Licensed/generated data is tracked: {sorted(set(forbidden_data))}"
    assert not forbidden_paths, f"Noncanonical container paths found: {forbidden_paths}"
    print("Reference JSON, canonical hashes, path policy, and data-exclusion policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
