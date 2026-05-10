import unittest
from unittest.mock import patch

import pandas as pd

from engine import ASSET_CLASS_UNIVERSES, recommend_asset_class_tickers, recommend_cross_asset_tickers


def close_series(start=100, step=1, periods=30):
    return pd.Series([start + i * step for i in range(periods)], dtype=float)


class CrossAssetRecommendationTests(unittest.TestCase):
    @patch("engine._download_close")
    def test_crypto_recommendation_ranks_specific_tickers(self, mock_download_close):
        def fake_download(ticker, period="6mo"):
            if ticker == "SOL-USD":
                return close_series(start=40, step=4)
            if ticker == "ETH-USD":
                return close_series(start=100, step=0.1)
            return close_series(start=100, step=1)

        mock_download_close.side_effect = fake_download

        rows = recommend_asset_class_tickers(
            "Crypto",
            macro_score=2.0,
            regime="RISK_ON",
            class_conviction=2.0,
            top_n=3,
        )

        self.assertEqual(rows[0]["ticker"], "SOL-USD")
        self.assertEqual(rows[0]["asset_class"], "Crypto")
        self.assertIn(rows[0]["action"], {"BUY", "STRONG BUY"})
        self.assertLessEqual(len(rows), 3)

    @patch("engine._download_close", return_value=close_series())
    def test_cross_asset_recommendations_returns_every_supported_bucket(self, _mock_download_close):
        output = recommend_cross_asset_tickers(
            class_convictions={asset_class: 1.0 for asset_class in ASSET_CLASS_UNIVERSES},
            macro_score=1.0,
            regime="RISK_ON",
            top_n=2,
        )

        self.assertEqual(set(output.keys()), set(ASSET_CLASS_UNIVERSES.keys()))
        self.assertEqual(len(output["Crypto"]), 2)
        self.assertEqual(len(output["EM Equities"]), 2)
        self.assertEqual(len(output["Commodities"]), 2)
        self.assertEqual(len(output["High Yield Bonds"]), 2)
        self.assertEqual(len(output["Developed Equities"]), 2)

    @patch("engine._download_close", return_value=close_series())
    def test_gold_ig_bonds_and_cash_are_available_as_proxy_tickers(self, _mock_download_close):
        for asset_class in ["Gold", "IG Bonds", "USD Cash"]:
            rows = recommend_asset_class_tickers(
                asset_class,
                macro_score=-1.0,
                regime="RISK_OFF",
                class_conviction=1.0,
                top_n=2,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["asset_class"], asset_class)
            self.assertIn("ticker", rows[0])


if __name__ == "__main__":
    unittest.main()
