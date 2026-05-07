import unittest

from engine import risk_based_sizing


class RiskBasedSizingTests(unittest.TestCase):
    def test_pct_raw_uses_same_currency_cost_not_missing_idr_variable(self):
        sizing = risk_based_sizing(
            entry_price=100,
            stop_loss_price=90,
            portfolio_value=10_000,
            risk_pct=0.01,
            regime="RISK_ON",
            trade_type="SWING",
            max_position_pct=0.05,
            is_fractional=False,
            currency_symbol="$",
            lot_size=1,
        )

        self.assertEqual(sizing["lots"], 5)
        self.assertEqual(sizing["pct_raw"], 0.05)
        self.assertEqual(sizing["amount_idr"], "$500.00")

    def test_pct_raw_handles_fractional_us_sizing_against_usd_portfolio(self):
        sizing = risk_based_sizing(
            entry_price=250,
            stop_loss_price=240,
            portfolio_value=1_000,
            risk_pct=0.01,
            regime="RISK_ON",
            trade_type="SWING",
            max_position_pct=0.05,
            is_fractional=True,
            currency_symbol="$",
            lot_size=1,
        )

        self.assertEqual(sizing["lots"], 0.2)
        self.assertEqual(sizing["pct_raw"], 0.05)

    def test_rejects_non_positive_portfolio_value(self):
        sizing = risk_based_sizing(
            entry_price=100,
            stop_loss_price=90,
            portfolio_value=0,
            risk_pct=0.01,
            regime="RISK_ON",
            trade_type="SWING",
        )

        self.assertEqual(sizing["lots"], 0)
        self.assertEqual(sizing["pct_raw"], 0.0)
        self.assertEqual(sizing["label"], "Invalid portfolio")


if __name__ == "__main__":
    unittest.main()
