import unittest

import pandas as pd

from scripts.postprocess_body_bands import summarize_bands


class TestBodyBands(unittest.TestCase):
    def test_45_inclusive_50_exclusive_and_ambiguity_is_loss(self):
        trades = pd.DataFrame({
            "mode": ["qualifying"] * 5,
            "outcome_code": [1, -1, 2, 1, 1],
            "depth": [.45, .499999, .47, .50, 1.0],
            "timeframe": ["4H"] * 5,
            "realized_R_conservative": [1.0, -1.0, -1.0, 1.0, 1.0],
            "rr": [1.] * 5, "width_ticks": [4.] * 5,
        })
        output = summarize_bands(trades)
        target = output.loc[(output.timeframe == "4H") & (output.depth_band == "45-50%")].iloc[0]
        self.assertEqual(int(target.N), 3)
        self.assertEqual(int(target.wins), 1)
        self.assertEqual(int(target.ambiguous), 1)
        self.assertAlmostEqual(target.win_rate, 1 / 3)
        self.assertAlmostEqual(target.mean_R_conservative, -1 / 3)
        next_band = output.loc[output.depth_band == "50-55%"].iloc[0]
        self.assertEqual(int(next_band.N), 1)


if __name__ == "__main__":
    unittest.main()
