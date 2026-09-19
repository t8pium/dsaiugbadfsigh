from __future__ import annotations

import json
import re
from pathlib import Path


def safe_upload_name(name: str) -> str:
    basename = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", basename).strip(" .")
    if not cleaned or cleaned.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{x}" for x in range(1, 10)), *(f"LPT{x}" for x in range(1, 10)),
    }:
        cleaned = "databento_upload_" + (cleaned or "file")
    return cleaned[:240]


def parse_local_paths(text: str) -> list[Path]:
    paths = []
    for raw in text.splitlines():
        value = raw.strip().strip('"').strip("'")
        if value:
            paths.append(Path(value).expanduser().resolve())
    return paths


def discover_results(results: Path) -> list[Path]:
    if not results.exists():
        return []
    files = [
        path for suffix in ("*.csv", "*.json", "*.png")
        for path in results.rglob(suffix)
        if path.is_file()
    ]
    return sorted(files, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def load_latest_success(results: Path, data_file: Path, study: str) -> dict | None:
    manifest = results / "_runs" / f"latest_{study}.json"
    if not manifest.is_file() or not data_file.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "success" or payload.get("data_mtime_ns") != data_file.stat().st_mtime_ns:
        return None
    return payload
