import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fvg_research.active_contract import (
    build_active_contract,
    cme_trade_date,
    is_quarterly_mnq_outright,
)


def row(timestamp, symbol, volume, price):
    return {
        "ts_event": timestamp, "symbol": symbol, "open": price, "high": price + 1,
        "low": price - 1, "close": price + 0.25, "volume": volume,
    }


class TestActiveContract(unittest.TestCase):
    def test_strict_outright_symbols(self):
        accepted = ["MNQH6", "MNQM26", "MNQU2026", "mnqz25"]
        rejected = ["MNQH26-MNQM26", "MNQ.FUT", "MNQH26 C100", "NQH26", "MNQG26"]
        self.assertTrue(all(is_quarterly_mnq_outright(x) for x in accepted))
        self.assertFalse(any(is_quarterly_mnq_outright(x) for x in rejected))

    def test_trade_date_1800_eastern_and_dst(self):
        timestamps = pd.DatetimeIndex([
            "2026-01-15 22:59:00+00:00", "2026-01-15 23:00:00+00:00",
            "2026-07-15 21:59:00+00:00", "2026-07-15 22:00:00+00:00",
        ])
        actual = list(cme_trade_date(timestamps))
        self.assertEqual([str(x) for x in actual], [
            "2026-01-15", "2026-01-16", "2026-07-15", "2026-07-16",
        ])

    def test_volume_roll_excludes_spread_and_deduplicates(self):
        records = []
        for timestamp in ("2026-03-02 15:00Z", "2026-03-02 15:01Z"):
            records.extend([
                row(timestamp, "MNQH26", 10, 100), row(timestamp, "MNQM26", 20, 200),
                row(timestamp, "MNQH26-MNQM26", 10_000, 300),
            ])
        for timestamp in ("2026-03-03 15:00Z", "2026-03-03 15:01Z"):
            records.extend([
                row(timestamp, "MNQH26", 30, 101), row(timestamp, "MNQM26", 5, 201),
            ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            duplicate = root / "duplicate.csv"
            pd.DataFrame(records).to_csv(first, index=False)
            pd.DataFrame(records).to_csv(duplicate, index=False)
            parquet, pickle = root / "active.parquet", root / "active.pkl"
            summary = build_active_contract([first, duplicate], parquet, pickle)
            active = pd.read_pickle(pickle)
        self.assertEqual(len(active), 4)
        self.assertEqual(list(active.symbol), ["MNQM26", "MNQM26", "MNQH26", "MNQH26"])
        self.assertNotIn("-", "".join(active.symbol))
        self.assertEqual(summary["contracts"], 2)
        self.assertEqual(summary["contract_segments"], 2)

    def test_conflicting_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = root / "one.csv", root / "two.csv"
            pd.DataFrame([row("2026-01-01 15:00Z", "MNQH26", 10, 100)]).to_csv(one, index=False)
            pd.DataFrame([row("2026-01-01 15:00Z", "MNQH26", 10, 101)]).to_csv(two, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                build_active_contract([one, two], root / "a.parquet", root / "a.pkl")


if __name__ == "__main__":
    unittest.main()
