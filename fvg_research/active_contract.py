from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from .io import iter_frames

# Databento has used one- and two-digit expiry years across the study. Four
# digits are accepted for ordinary tabular exports. Anchoring the entire value
# rejects calendars (e.g. MNQH6-MNQM6), continuous symbols, and option suffixes.
OUTRIGHT_PATTERN = re.compile(r"^MNQ[HMUZ](?:\d{1,2}|\d{4})$", re.IGNORECASE)


def is_quarterly_mnq_outright(symbol: object) -> bool:
    return bool(OUTRIGHT_PATTERN.fullmatch(str(symbol).strip()))


def cme_trade_date(index: pd.DatetimeIndex) -> pd.Series:
    """Published convention: 18:00 America/New_York begins the next trade date."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    eastern = timestamps.tz_convert("America/New_York")
    normalized = eastern.normalize().tz_localize(None)
    dates = normalized + pd.to_timedelta((eastern.hour >= 18).astype(int), unit="D")
    return pd.Series(dates.date, index=index, name="trade_date")


def _symbol_column(frame: pd.DataFrame) -> str:
    for name in ("symbol", "raw_symbol", "stype_out_symbol"):
        if name in frame.columns:
            return name
    raise ValueError(
        "Input has no resolved contract symbol. Native DBN is decoded with "
        "map_symbols=True; for a split Databento batch, include its symbology JSON sidecar."
    )


def build_active_contract(
    inputs: Iterable[str | Path],
    parquet_path: str | Path,
    pickle_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Build the volume-selected MNQ series while staging chunks in DuckDB."""
    sources = [Path(x).expanduser().resolve() for x in inputs]
    if not sources:
        raise ValueError("No inputs supplied.")
    missing = [str(x) for x in sources if not x.exists()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))

    parquet_path = Path(parquet_path)
    pickle_path = Path(pickle_path)
    manifest_path = Path(manifest_path) if manifest_path else parquet_path.with_suffix(".manifest.json")
    for target in (parquet_path, pickle_path, manifest_path):
        target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fvg_active_", dir=parquet_path.parent) as temp_dir:
        database = Path(temp_dir) / "stage.duckdb"
        connection = duckdb.connect(str(database))
        connection.execute("PRAGMA threads=2")
        created = False
        chunk_number = 0
        raw_rows = 0
        kept_rows = 0
        for source in sources:
            for frame in iter_frames(source):
                raw_rows += len(frame)
                symbol_col = _symbol_column(frame)
                use = frame.loc[frame[symbol_col].map(is_quarterly_mnq_outright)].copy()
                if use.empty:
                    continue
                use["symbol"] = use[symbol_col].astype(str).str.strip().str.upper()
                use["trade_date"] = cme_trade_date(use.index).to_numpy()
                use = use.reset_index()
                if "ts_event" not in use.columns:
                    use = use.rename(columns={use.columns[0]: "ts_event"})
                columns = ["ts_event", "open", "high", "low", "close", "volume", "symbol", "trade_date"]
                use = use[columns]
                kept_rows += len(use)
                chunk_number += 1
                connection.register("incoming", use)
                if not created:
                    connection.execute("CREATE TABLE raw AS SELECT *, ?::BIGINT AS chunk_no FROM incoming", [chunk_number])
                    created = True
                else:
                    connection.execute("INSERT INTO raw SELECT *, ?::BIGINT AS chunk_no FROM incoming", [chunk_number])
                connection.unregister("incoming")
        if not created:
            connection.close()
            raise ValueError(
                "No quarterly MNQ outright rows were found. Expected symbols such as "
                "MNQH6, MNQZ25, or MNQH2026; calendar spreads are intentionally rejected."
            )

        conflict = connection.execute(
            """
            SELECT ts_event, symbol, count(DISTINCT hash(open, high, low, close, volume)) AS variants
            FROM raw GROUP BY ts_event, symbol HAVING variants > 1 LIMIT 1
            """
        ).fetchone()
        if conflict:
            connection.close()
            raise ValueError(
                "Conflicting duplicate bar for timestamp/symbol "
                f"{conflict[0]} / {conflict[1]}; remove overlapping inconsistent inputs."
            )

        connection.execute(
            """
            CREATE TABLE clean AS
            SELECT * EXCLUDE (rn, chunk_no) FROM (
              SELECT *, row_number() OVER (
                PARTITION BY ts_event, symbol ORDER BY chunk_no DESC
              ) AS rn FROM raw
            ) WHERE rn = 1
            """
        )
        connection.execute(
            """
            CREATE TABLE winners AS
            SELECT trade_date, symbol AS active_symbol, daily_volume FROM (
              SELECT trade_date, symbol, sum(volume) AS daily_volume,
                     row_number() OVER (
                       PARTITION BY trade_date ORDER BY sum(volume) DESC, symbol ASC
                     ) AS rank
              FROM clean GROUP BY trade_date, symbol
            ) WHERE rank = 1
            """
        )
        active = connection.execute(
            """
            SELECT c.ts_event, c.open, c.high, c.low, c.close, c.volume,
                   c.symbol, c.trade_date, w.active_symbol
            FROM clean c JOIN winners w USING (trade_date)
            WHERE c.symbol = w.active_symbol
            ORDER BY c.ts_event
            """
        ).fetchdf()
        connection.close()

    active["ts_event"] = pd.to_datetime(active["ts_event"], utc=True)
    if active["ts_event"].duplicated().any():
        raise ValueError("Active construction produced duplicate timestamps.")
    if active[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Active construction produced missing OHLC values.")
    active["roll_date"] = active["symbol"].ne(active["symbol"].shift())
    contracts = int(active["symbol"].nunique())
    segments = int(active["roll_date"].sum())
    summary: dict[str, object] = {
        "input_paths": [str(x) for x in sources],
        "raw_rows_read": raw_rows,
        "outright_rows_staged": kept_rows,
        "active_rows": len(active),
        "contracts": contracts,
        "contract_segments": segments,
        "start": active["ts_event"].min().isoformat(),
        "end": active["ts_event"].max().isoformat(),
        "duplicate_timestamps": int(active["ts_event"].duplicated().sum()),
        "missing_ohlc": int(active[["open", "high", "low", "close"]].isna().sum().sum()),
        "selection": "highest total daily volume among strict quarterly MNQ outrights",
        "trade_date_convention": "America/New_York date; timestamps at/after 18:00 assigned to next date",
    }

    temp_parquet = parquet_path.with_name(parquet_path.name + ".partial")
    temp_pickle = pickle_path.with_name(pickle_path.name + ".partial")
    temp_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    active.to_parquet(temp_parquet, index=False)
    active.to_pickle(temp_pickle)
    temp_manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp_parquet.replace(parquet_path)
    temp_pickle.replace(pickle_path)
    temp_manifest.replace(manifest_path)
    return summary
