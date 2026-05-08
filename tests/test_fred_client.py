import unittest
from unittest.mock import Mock, patch

import requests

from utils.fred_client import (
    FRED_GRAPH_CSV_URL,
    fetch_fred,
    fetch_fred_csv,
    fetch_macro_observations,
    parse_latest_value,
)


class FredClientTests(unittest.TestCase):
    @patch("utils.fred_client.requests.get")
    def test_fetch_fred_returns_empty_without_api_key(self, mock_get):
        self.assertEqual(fetch_fred("DGS10", ""), [])
        mock_get.assert_not_called()

    @patch("utils.fred_client.requests.get")
    def test_fetch_fred_csv_normalizes_and_desc_sorts_public_feed(self, mock_get):
        response = Mock()
        response.text = "observation_date,DGS10\n2024-01-01,4.00\n2024-01-03,.\n2024-01-02,4.10\n"
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        observations = fetch_fred_csv("DGS10", limit=2)

        mock_get.assert_called_once_with(FRED_GRAPH_CSV_URL, params={"id": "DGS10"}, timeout=10)
        self.assertEqual(
            observations,
            [
                {"date": "2024-01-03", "value": "."},
                {"date": "2024-01-02", "value": "4.10"},
            ],
        )
        self.assertEqual(parse_latest_value(observations), 4.10)

    @patch("utils.fred_client.fetch_fred_csv")
    @patch("utils.fred_client.fetch_fred")
    def test_fetch_macro_observations_uses_api_when_available(self, mock_fetch_fred, mock_fetch_csv):
        mock_fetch_fred.return_value = [{"date": "2024-01-02", "value": "4.2"}]

        observations = fetch_macro_observations("DGS10", api_key="secret", limit=5)

        self.assertEqual(observations, [{"date": "2024-01-02", "value": "4.2"}])
        mock_fetch_fred.assert_called_once_with("DGS10", "secret", limit=5)
        mock_fetch_csv.assert_not_called()

    @patch("utils.fred_client.fetch_fred_csv")
    @patch("utils.fred_client.fetch_fred")
    def test_fetch_macro_observations_falls_back_to_csv_on_api_failure(self, mock_fetch_fred, mock_fetch_csv):
        mock_fetch_fred.side_effect = requests.RequestException("api down")
        mock_fetch_csv.return_value = [{"date": "2024-01-02", "value": "4.2"}]

        observations = fetch_macro_observations("DGS10", api_key="secret", limit=5)

        self.assertEqual(observations, [{"date": "2024-01-02", "value": "4.2"}])
        mock_fetch_csv.assert_called_once_with("DGS10", limit=5)

    @patch("utils.fred_client.fetch_fred_csv")
    @patch("utils.fred_client.fetch_fred")
    def test_fetch_macro_observations_uses_csv_without_key(self, mock_fetch_fred, mock_fetch_csv):
        mock_fetch_csv.return_value = [{"date": "2024-01-02", "value": "4.2"}]

        observations = fetch_macro_observations("DGS10", api_key=None, limit=5)

        self.assertEqual(observations, [{"date": "2024-01-02", "value": "4.2"}])
        mock_fetch_fred.assert_not_called()
        mock_fetch_csv.assert_called_once_with("DGS10", limit=5)


if __name__ == "__main__":
    unittest.main()
