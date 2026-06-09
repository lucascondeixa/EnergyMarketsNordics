"""Seasonal price forecaster for Nordic day-ahead markets.

Fits a (hour_of_day, day_of_week) median profile on 90 days of ENTSO-E
historical data, then applies a level correction anchored to the last 7 days
of actuals so the forecast tracks the current price regime.

Usage:
    forecaster = SeasonalPriceForecaster()
    forecaster.fit(area=FINLAND_AREA_CODE)
    prices = forecaster.predict(horizon_index)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data_ingestion.nordpool import NordPoolClient

log = logging.getLogger(__name__)

_PRICE_FLOOR = -500.0
_PRICE_CAP = 4_000.0


class SeasonalPriceForecaster:
    """Calendar-median price model with recent-level correction.

    After calling fit(), predict() returns an hourly pd.Series for any
    future DatetimeIndex using the fitted (hour_of_day, day_of_week) medians
    scaled to the current price regime.
    """

    def __init__(self, api_key: str | None = None, lookback_days: int = 90):
        self._client = NordPoolClient(api_key)
        self.lookback_days = lookback_days
        self._profile: pd.Series | None = None       # median indexed by (hour, dow)
        self._level_correction: float = 1.0
        self._area: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, area: str, reference_date: datetime | None = None) -> "SeasonalPriceForecaster":
        """Fetch historical data and fit the seasonal model.

        Args:
            area: ENTSO-E area code (e.g. FINLAND_AREA_CODE).
            reference_date: anchor for "today"; defaults to now (UTC).
        """
        self._area = area
        ref = reference_date or datetime.now(tz=timezone.utc)

        end = ref.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=self.lookback_days)

        log.info(
            "SeasonalPriceForecaster: fetching %d days of ENTSO-E history for %s",
            self.lookback_days, area,
        )
        raw = self._client.get_day_ahead_prices(start, end, area=area)

        if raw.empty:
            raise RuntimeError(f"No historical price data returned for area {area}")

        # Resample 15-min -> hourly using mean (more accurate than nearest-neighbour)
        hourly = raw.resample("1h").mean().dropna()
        log.info("Fetched %d hourly observations (%.1f days)", len(hourly), len(hourly) / 24)

        self._profile = self._fit_profile(hourly)
        self._level_correction = self._compute_level_correction(hourly)

        log.info(
            "Profile fitted: mean=%.1f EUR/MWh, level_correction=%.3f",
            self._profile.mean(), self._level_correction,
        )
        return self

    def predict(self, horizon_index: pd.DatetimeIndex) -> pd.Series:
        """Return forecast prices (EUR/MWh) aligned to *horizon_index*."""
        if self._profile is None:
            raise RuntimeError("Call fit() before predict()")

        keys = list(zip(horizon_index.hour, horizon_index.day_of_week))
        values = np.array([self._profile.get(k, self._profile.mean()) for k in keys])
        corrected = np.clip(values * self._level_correction, _PRICE_FLOOR, _PRICE_CAP)
        return pd.Series(corrected, index=horizon_index, name="price_eur_mwh")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_profile(hourly: pd.Series) -> pd.Series:
        """Compute median price for each (hour_of_day, day_of_week) bin."""
        df = pd.DataFrame({
            "price": hourly.values,
            "hour": hourly.index.hour,
            "dow": hourly.index.day_of_week,
        })
        profile = df.groupby(["hour", "dow"])["price"].median()
        return profile

    @staticmethod
    def _compute_level_correction(hourly: pd.Series) -> float:
        """Scale factor = (last-7-day mean) / (profile mean for same hours).

        This anchors the forecast to the current price regime without
        distorting the intraday shape.
        """
        if len(hourly) < 24 * 7:
            log.warning("Fewer than 7 days of actuals available; level correction = 1.0")
            return 1.0

        recent = hourly.iloc[-24 * 7:]
        recent_mean = recent.mean()

        profile_df = pd.DataFrame({
            "price": hourly.values,
            "hour": hourly.index.hour,
            "dow": hourly.index.day_of_week,
        })
        profile = profile_df.groupby(["hour", "dow"])["price"].median()

        recent_keys = list(zip(recent.index.hour, recent.index.day_of_week))
        model_values = np.array([profile.get(k, profile.mean()) for k in recent_keys])
        model_mean = model_values.mean()

        if model_mean <= 0:
            return 1.0

        correction = recent_mean / model_mean
        # Clamp to [0.3, 3.0] to avoid extreme distortions from outlier weeks
        return float(np.clip(correction, 0.3, 3.0))
