import unittest
from unittest.mock import patch

from engine import build_execution_plan, risk_based_sizing


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


    def test_build_execution_plan_uses_risk_sizing_without_name_error(self):
        results = [
            {
                "ticker": "TEST.JK",
                "trade_type": "SWING",
                "macro": 1.0,
                "technical": 1.0,
                "sentiment": 1.0,
                "fundamental": 0.0,
                "composite": 1.0,
                "breakdown": {},
                "playbook": {"action": "BUY", "strategy": "Test", "emoji": "✅", "reason": "Regression"},
                "sharia": True,
                "high_beta": False,
            }
        ]
        trade = {
            "price": 100,
            "entry_limit": 100,
            "stop_loss": 90,
            "stop_pct": "10.0%",
            "take_profit": 120,
            "tp_pct": "20.0%",
            "hold_days": 5,
            "order_expiry": "GTC",
            "entry_type": "LIMIT ORDER",
        }

        with patch("engine.get_trade_setup", return_value=trade):
            plan = build_execution_plan(
                results=results,
                macro_score=1.0,
                regime="RISK_ON",
                allocation={"cash": 0.0},
                portfolio_value=10_000_000,
                rr_ratio=2.0,
                raw_data={},
                high_beta_plays=[],
                threshold=0.25,
                usd_idr_rate=16_000,
            )

        self.assertEqual(len(plan["SWING"]), 1)
        self.assertGreater(plan["SWING"][0]["pct_raw"], 0)

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
