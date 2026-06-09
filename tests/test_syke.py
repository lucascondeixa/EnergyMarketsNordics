"""Tests for SYKE hydrology client and inflow forecaster."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.syke import (
    KEMIJOKI_ISOHAARA_ID,
    OULUJOKI_MERIKOSKI_ID,
    SYKEClient,
    SYKEInflowForecaster,
)
from src.utils.time_utils import build_horizon, synthetic_inflow_series

HORIZON_START = pd.Timestamp("2026-06-01", tz="UTC")


def _make_discharge_payload(values: list[float], start: str) -> dict:
    """Build a fake OData Virtaama response."""
    records = [
        {"Paikka_Id": 1388, "Aika": f"{start[:4]}-{i+1:02d}-01T00:00:00", "Arvo": v}
        for i, v in enumerate(values)
    ]
    return {"value": records}


def _mock_httpx_response(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = payload
    return mock


# ---------------------------------------------------------------------------
# SYKEClient
# ---------------------------------------------------------------------------

class TestSYKEClient:
    def test_returns_series_with_correct_length(self):
        monthly_values = [300.0, 280.0, 350.0, 800.0, 1200.0, 600.0]
        payload = _make_discharge_payload(monthly_values, "2026-01-01")

        with patch("httpx.Client.get", return_value=_mock_httpx_response(payload)):
            client = SYKEClient()
            result = client.get_discharge(1388, date(2026, 1, 1), date(2026, 6, 30))

        assert len(result) == len(monthly_values)
        assert result.name == "discharge_m3s"

    def test_returns_empty_series_on_no_data(self):
        with patch("httpx.Client.get", return_value=_mock_httpx_response({"value": []})):
            client = SYKEClient()
            result = client.get_discharge(9999, date(2026, 1, 1), date(2026, 6, 30))

        assert result.empty

    def test_handles_pagination(self):
        page1 = {"value": [{"Paikka_Id": 1388, "Aika": "2026-01-01T00:00:00", "Arvo": 300.0}],
                 "odata.nextLink": "https://example.com/page2"}
        page2 = {"value": [{"Paikka_Id": 1388, "Aika": "2026-01-02T00:00:00", "Arvo": 310.0}]}

        responses = [_mock_httpx_response(page1), _mock_httpx_response(page2)]
        with patch("httpx.Client.get", side_effect=responses):
            client = SYKEClient()
            result = client.get_discharge(1388, date(2026, 1, 1), date(2026, 1, 2))

        assert len(result) == 2
        assert list(result.values) == pytest.approx([300.0, 310.0])

    def test_discharge_values_are_positive(self):
        payload = _make_discharge_payload([100.0, 200.0, 500.0], "2026-01-01")
        with patch("httpx.Client.get", return_value=_mock_httpx_response(payload)):
            client = SYKEClient()
            result = client.get_discharge(1388, date(2026, 1, 1), date(2026, 3, 31))

        assert (result > 0).all()


# ---------------------------------------------------------------------------
# SYKEInflowForecaster
# ---------------------------------------------------------------------------

def _build_forecaster_with_history(
    station_id: int = 1388,
    n_years: int = 2,
    base_discharge: float = 400.0,
    seasonal_amplitude: float = 0.5,
) -> SYKEInflowForecaster:
    """Build a fitted SYKEInflowForecaster with synthetic discharge history."""
    end = date(2026, 6, 1)
    start = date(end.year - n_years, end.month, end.day)
    dates = pd.date_range(start, end, freq="D")
    doy = np.array([d.timetuple().tm_yday for d in dates])
    # Seasonal discharge: peak around May (day 130)
    discharge = base_discharge * (1 + seasonal_amplitude * np.cos(2 * np.pi * (doy - 130) / 365))
    history = pd.Series(discharge, index=dates, name="discharge_m3s")

    def mock_get(s_id, s, e):
        mask = (history.index.date >= s) & (history.index.date <= e)
        return history[mask]

    forecaster = SYKEInflowForecaster(station_id=station_id, lookback_years=n_years)
    forecaster._client = MagicMock()
    forecaster._client.get_discharge.side_effect = mock_get
    forecaster.fit(reference_date=end)
    return forecaster


class TestSYKEInflowForecaster:
    def test_level_correction_near_one_at_average_conditions(self):
        forecaster = _build_forecaster_with_history()
        correction = forecaster.level_correction()
        assert 0.8 <= correction <= 1.2, f"Expected ~1.0 correction, got {correction}"

    def test_level_correction_above_one_for_wet_year(self):
        """Recent discharge 50% above historical → correction > 1."""
        forecaster = _build_forecaster_with_history(base_discharge=400.0)
        # Inflate recent discharge to simulate a wet year
        if forecaster._recent is not None:
            forecaster._recent = forecaster._recent * 1.5
        correction = forecaster.level_correction()
        assert correction > 1.0

    def test_level_correction_below_one_for_dry_year(self):
        """Recent discharge 50% below historical → correction < 1."""
        forecaster = _build_forecaster_with_history(base_discharge=400.0)
        if forecaster._recent is not None:
            forecaster._recent = forecaster._recent * 0.5
        correction = forecaster.level_correction()
        assert correction < 1.0

    def test_correction_clamped_at_minimum(self):
        forecaster = _build_forecaster_with_history()
        if forecaster._recent is not None:
            forecaster._recent = forecaster._recent * 0.001  # extreme drought
        correction = forecaster.level_correction()
        assert correction >= 0.3

    def test_correction_clamped_at_maximum(self):
        forecaster = _build_forecaster_with_history()
        if forecaster._recent is not None:
            forecaster._recent = forecaster._recent * 100.0  # extreme flood
        correction = forecaster.level_correction()
        assert correction <= 3.0

    def test_apply_scales_synthetic_inflow(self):
        forecaster = _build_forecaster_with_history()
        index = build_horizon(HORIZON_START, horizon_hours=48)
        synthetic = synthetic_inflow_series(
            index, annual_avg_gwh_per_hour=0.22, seasonal_amplitude=0.75, peak_day_of_year=133
        )
        corrected = forecaster.apply(synthetic)

        factor = forecaster.level_correction()
        assert corrected.values == pytest.approx(synthetic.values * factor, rel=1e-6)

    def test_apply_preserves_series_length(self):
        forecaster = _build_forecaster_with_history()
        index = build_horizon(HORIZON_START, horizon_hours=168)
        synthetic = synthetic_inflow_series(
            index, annual_avg_gwh_per_hour=0.34, seasonal_amplitude=0.60, peak_day_of_year=140
        )
        corrected = forecaster.apply(synthetic)
        assert len(corrected) == 168

    def test_returns_one_on_empty_history(self):
        forecaster = SYKEInflowForecaster(station_id=9999)
        forecaster._client = MagicMock()
        forecaster._client.get_discharge.return_value = pd.Series(dtype=float, name="discharge_m3s")
        forecaster.fit()
        assert forecaster.level_correction() == 1.0

    def test_station_constants_defined(self):
        assert KEMIJOKI_ISOHAARA_ID == 1388
        assert OULUJOKI_MERIKOSKI_ID == 1314
