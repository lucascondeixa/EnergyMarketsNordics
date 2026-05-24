"""Time series utilities for Nordic energy market scheduling."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def build_horizon(
    start: datetime,
    horizon_hours: int = 168,
    resolution_hours: int = 1,
) -> pd.DatetimeIndex:
    """Return a UTC DatetimeIndex for the optimisation horizon."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    freq = f"{resolution_hours}h"
    return pd.date_range(start=start, periods=horizon_hours, freq=freq, tz="UTC")


def next_day_ahead_gate(reference: datetime | None = None) -> datetime:
    """Return the next Nord Pool day-ahead gate closure (noon CET = 11:00 UTC in winter)."""
    ref = reference or datetime.now(tz=timezone.utc)
    # Gate closure is 12:00 CET = 11:00 UTC (standard time) / 10:00 UTC (summer)
    # Simplified: always use 11:00 UTC
    gate = ref.replace(hour=11, minute=0, second=0, microsecond=0)
    if ref >= gate:
        gate += timedelta(days=1)
    return gate


def synthetic_inflow_series(
    index: pd.DatetimeIndex,
    annual_avg_gwh_per_hour: float,
    seasonal_amplitude: float,
    peak_day_of_year: int,
) -> pd.Series:
    """Generate a synthetic seasonal inflow series (GWh/h, water-equivalent)."""
    day_of_year = index.day_of_year.to_numpy()
    angle = 2 * math.pi * (day_of_year - peak_day_of_year) / 365
    inflow = annual_avg_gwh_per_hour * (1 + seasonal_amplitude * np.cos(angle))
    inflow = np.clip(inflow, 0, None)
    return pd.Series(inflow, index=index, name="inflow_gwh_per_h")


def synthetic_price_series(
    index: pd.DatetimeIndex,
    base_price: float = 50.0,
    daily_amplitude: float = 20.0,
    noise_std: float = 5.0,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic Elspot price series (EUR/MWh) with daily pattern + noise."""
    rng = np.random.default_rng(seed)
    hour_of_day = index.hour.to_numpy()
    # Peak at hour 9 and 18, trough at 4 (simplified)
    pattern = np.sin(2 * math.pi * (hour_of_day - 4) / 24)
    prices = base_price + daily_amplitude * pattern + rng.normal(0, noise_std, len(index))
    prices = np.clip(prices, -10, 500)
    return pd.Series(prices, index=index, name="price_eur_mwh")
