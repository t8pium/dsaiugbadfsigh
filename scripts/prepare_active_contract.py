from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fvg_research.active_contract import build_active_contract
from fvg_research.config import ACTIVE_1M, ACTIVE_PICKLE, PROCESSED, RAW


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the volume-selected active MNQ one-minute series."
    )
    parser.add_argument(
        "--input", action="append", dest="inputs",
        help=(
            "Input file or directory; repeat for multiple inputs. Supported: "
            ".dbn, .dbn.zst, .parquet, .pq, .csv, .csv.gz, .csv.zst, .zip."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [Path(x) for x in args.inputs] if args.inputs else [RAW]
    print("Reading input(s):", flush=True)
    for path in inputs:
        print(" -", path, flush=True)
    try:
        summary = build_active_contract(
            inputs, ACTIVE_1M, ACTIVE_PICKLE, PROCESSED / "active_mnq.manifest.json"
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print(f"Wrote {ACTIVE_1M}")
    print(f"Wrote {ACTIVE_PICKLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
