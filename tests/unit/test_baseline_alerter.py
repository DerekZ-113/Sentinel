"""
Tests for fleet_data/baseline_alerter.py — the "before ML" analysis script.

DB access is mocked; these pin the FP-rate math and the returned
per-type stats structure.
"""

from unittest.mock import MagicMock, patch

from fleet_data.baseline_alerter import analyze_baseline


def _run_with_rows(overall_row, subtype_rows, type_rows):
    """Run analyze_baseline with a mocked cursor feeding the three queries."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = overall_row
    mock_cursor.fetchall.side_effect = [subtype_rows, type_rows]

    with patch("fleet_data.baseline_alerter.connect_to_database", return_value=mock_conn):
        return analyze_baseline()


class TestAnalyzeBaseline:

    def test_returns_per_type_stats_with_fp_rate(self):
        result = _run_with_rows(
            overall_row=(1000, 100, 40, 60),
            subtype_rows=[("stuck", None, 60, 25, 35)],
            type_rows=[
                ("stuck", 60, 25, 35),
                ("passenger_assist", 40, 40, 0),
            ],
        )

        assert result["stuck"]["total"] == 60
        assert result["stuck"]["fp"] == 35
        assert result["stuck"]["fp_rate"] == 35 / 60 * 100
        # A 100%-real type has a 0% baseline FP rate, not a div-by-zero
        assert result["passenger_assist"]["fp_rate"] == 0.0

    def test_handles_empty_database(self):
        result = _run_with_rows(
            overall_row=(0, 0, 0, 0),
            subtype_rows=[],
            type_rows=[],
        )
        assert result == {}

    def test_closes_connection(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0, 0, 0, 0)
        mock_cursor.fetchall.side_effect = [[], []]

        with patch("fleet_data.baseline_alerter.connect_to_database", return_value=mock_conn):
            analyze_baseline()

        mock_conn.close.assert_called_once()
