"""SYKE (Finnish Environment Institute) hydrological data client.

Fetches daily river discharge (virtaama, m³/s) from the SYKE open OData API:
    https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.2/

No API key required. Data is updated daily. Time resolution is 1 day.

SYKEInflowForecaster applies a level correction to the synthetic seasonal
inflow curve using the same pattern as SeasonalPriceForecaster:

    adjusted_inflow[t] = synthetic_inflow[t] × (recent_avg / historical_doy_avg)

This corrects for whether the current year is wetter or drier than historical
norms without requiring an exact discharge-to-energy conversion factor.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEMIJOKI_ISOHAARA_ID = 1388    # Kemijoki, Isohaara (river mouth, total catchment)
OULUJOKI_MERIKOSKI_ID = 1314   # Oulujoki, Merikoski (river mouth)

_BASE_URL = "https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.2/odata"


class SYKEClient:
    """Thin HTTP wrapper around the SYKE Hydrologiarajapinta OData API."""

    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)

    def get_discharge(
        self,
        station_id: int,
        start: date,
        end: date,
    ) -> pd.Series:
        """Return daily discharge (m³/s) for *station_id* between *start* and *end*.

        Handles OData pagination transparently. Returns an empty Series if the
        station has no data for the requested period.
        """
        url = f"{_BASE_URL}/Virtaama"
        params = {
            "$filter": (
                f"Paikka_Id eq {station_id}"
                f" and Aika ge datetime'{start.isoformat()}T00:00:00'"
                f" and Aika le datetime'{end.isoformat()}T00:00:00'"
            ),
            "$orderby": "Aika asc",
            "$select": "Aika,Arvo",
        }

        records: list[dict] = []
        next_url: str | None = url
        first_page = True

        while next_url:
            if first_page:
                resp = self._client.get(next_url, params=params)
                first_page = False
            else:
                resp = self._client.get(next_url)
            resp.raise_for_status()
            payload = resp.json()
            records.extend(payload.get("value", []))
            next_url = payload.get("odata.nextLink") or payload.get("@odata.nextLink")

        if not records:
            return pd.Series(dtype=float, name="discharge_m3s")

        df = pd.DataFrame(records)
        df["Aika"] = pd.to_datetime(df["Aika"]).dt.tz_localize("UTC")
        df = df.set_index("Aika").rename(columns={"Arvo": "discharge_m3s"})
        return df["discharge_m3s"].dropna().sort_index()


class SYKEInflowForecaster:
    """Level-correction forecaster using SYKE river discharge data.

    Computes how much wetter or drier the current season is relative to the
    historical day-of-year median, then scales the synthetic seasonal inflow
    curve accordingly.

    Analogue to SeasonalPriceForecaster for prices.
    """

    def __init__(
        self,
        station_id: int,
        lookback_years: int = 3,
        correction_window_days: int = 21,
    ):
        self.station_id = station_id
        self.lookback_years = lookback_years
        self.correction_window_days = correction_window_days
        self._client = SYKEClient()
        self._doy_median: pd.Series | None = None
        self._recent: pd.Series | None = None

    def fit(self, reference_date: date | None = None) -> "SYKEInflowForecaster":
        """Fetch discharge history and build the day-of-year median profile."""
        ref = reference_date or date.today()
        start = ref - timedelta(days=365 * self.lookback_years)

        history = self._client.get_discharge(self.station_id, start, ref)
        if history.empty:
            log.warning("SYKE station %d returned no data for %s – %s", self.station_id, start, ref)
            return self

        idx = history.index.tz_localize(None) if history.index.tz else history.index
        history = history.copy()
        history.index = idx

        self._doy_median = history.groupby(history.index.day_of_year).median()
        recent_start = ref - timedelta(days=self.correction_window_days)
        self._recent = history[history.index.date >= recent_start]

        log.info(
            "SYKE station %d fitted: %d days of history, recent window %d days",
            self.station_id, len(history), len(self._recent),
        )
        return self

    def level_correction(self) -> float:
        """Ratio of recent discharge to historical DOY median. Clamped [0.3, 3.0]."""
        if self._doy_median is None or self._recent is None or self._recent.empty:
            return 1.0

        recent_avg = float(self._recent.mean())
        end_doy = int(pd.Timestamp(self._recent.index[-1]).day_of_year)
        start_doy = max(1, end_doy - self.correction_window_days + 1)
        doys = list(range(start_doy, end_doy + 1))
        hist_avg = float(self._doy_median.reindex(doys).dropna().mean())

        if hist_avg <= 0:
            return 1.0

        correction = float(np.clip(recent_avg / hist_avg, 0.3, 3.0))
        log.info(
            "SYKE station %d: recent=%.0f m³/s, historical=%.0f m³/s, correction=%.3f",
            self.station_id, recent_avg, hist_avg, correction,
        )
        return correction

    def apply(self, synthetic_inflow: pd.Series) -> pd.Series:
        """Multiply the synthetic seasonal inflow by the level correction."""
        factor = self.level_correction()
        return synthetic_inflow * factor
