import unittest
import numpy as np
import pandas as pd

from fvg_research.fvg import detect_fvgs


class TestFVGDefinition(unittest.TestCase):
    def test_detects_bullish_and_bearish_fvgs_without_lookahead(self):
        idx = pd.date_range("2026-01-01", periods=8, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open":  [100, 101, 104, 105, 106, 103, 101,  98],
                "high":  [102, 103, 106, 107, 108, 105, 103, 100],
                "low":   [ 99, 100, 103, 104, 105, 102,  99,  96],
                "close": [101, 102, 105, 106, 107, 103, 100,  97],
                "volume":[100]*8,
            },
            index=idx,
        )

        # Force enough ATR history for this tiny synthetic example by appending a
        # stable prefix, then retain the final events.
        prefix_idx = pd.date_range("2025-12-31 23:40", periods=20, freq="1min", tz="UTC")
        prefix = pd.DataFrame(
            {
                "open": np.linspace(90, 99, 20),
                "high": np.linspace(91,100,20),
                "low":  np.linspace(89, 98, 20),
                "close":np.linspace(90.5,99.5,20),
                "volume":[100]*20,
            },
            index=prefix_idx,
        )
        x = pd.concat([prefix, df])
        events = detect_fvgs(x)

        # Candle C at idx[2]: low=103 > high[idx[0]]=102 => bullish FVG.
        bull = events.loc[idx[2]]
        self.assertEqual(int(bull["direction"]), 1)
        self.assertAlmostEqual(float(bull["lower"]), 102.0)
        self.assertAlmostEqual(float(bull["upper"]), 103.0)

        # Candle C at idx[7]: high=100 < low[idx[5]]=102 => bearish FVG.
        bear = events.loc[idx[7]]
        self.assertEqual(int(bear["direction"]), -1)
        self.assertAlmostEqual(float(bear["lower"]), 100.0)
        self.assertAlmostEqual(float(bear["upper"]), 102.0)

    def test_event_is_only_created_on_candle_c(self):
        idx = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "open": np.arange(20, dtype=float)+100,
            "high": np.arange(20, dtype=float)+101,
            "low": np.arange(20, dtype=float)+99,
            "close": np.arange(20, dtype=float)+100.5,
            "volume": [100]*20,
        }, index=idx)

        # Construct a bullish gap at the final candle only.
        df.loc[idx[-3], "high"] = 110
        df.loc[idx[-1], "low"] = 112
        df.loc[idx[-1], "high"] = 114
        df.loc[idx[-1], "open"] = 112
        df.loc[idx[-1], "close"] = 113

        events = detect_fvgs(df)
        self.assertIn(idx[-1], events.index)
        self.assertNotIn(idx[-2], events.index)


if __name__ == "__main__":
    unittest.main()
