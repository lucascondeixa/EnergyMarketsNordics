"""Forecast data adapter.

Provides a uniform interface for loading price, inflow, and wind forecasts
regardless of source (file, API, model output).

Supported sources:
  - CSV file with columns: timestamp, <value_col>
  - "synthetic" (for development / back-testing)
  - TODO: External forecast provider REST API
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion.fingrid import FingridClient
from src.data_ingestion.nordpool import NordPoolClient, FINLAND_AREA_CODE, SWEDEN_SE2_AREA_CODE
from src.data_ingestion.price_forecast import SeasonalPriceForecaster
from src.data_ingestion.syke import SYKEInflowForecaster
from src.utils.schema import HydroInflowConfig, WindFarmConfig
from src.utils.time_utils import synthetic_inflow_series, synthetic_price_series

log = logging.getLogger(__name__)


def _merge_real_and_synthetic(
    real: pd.Series,
    horizon_index: pd.DatetimeIndex,
    synthetic_fn,
) -> pd.Series:
    """Reindex *real* to *horizon_index*, filling gaps with *synthetic_fn(index)*."""
    aligned = real.reindex(horizon_index, method="nearest", tolerance="30min")
    n_real = int(aligned.notna().sum())
    n_total = len(horizon_index)
    if n_real < n_total:
        synthetic = synthetic_fn(horizon_index)
        aligned = aligned.fillna(synthetic)
        log.info("API data: %d/%d hours real, %d filled with synthetic", n_real, n_total, n_total - n_real)
    else:
        log.info("API data: all %d hours from live feed", n_total)
    return aligned.clip(lower=0)


def load_price_forecast(
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load a price forecast aligned to *horizon_index*.

    *source*: "synthetic" or path to CSV with columns 'timestamp', 'price_eur_mwh'.
    """
    if str(source) == "synthetic":
        log.info("Using synthetic price forecast (development mode)")
        return synthetic_price_series(horizon_index)

    if str(source) == "api":
        log.info("Fetching FI day-ahead prices from ENTSO-E")
        client = NordPoolClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = client.get_day_ahead_prices(start, end)
        except Exception as exc:
            log.error("ENTSO-E fetch failed (%s); falling back to synthetic", exc)
            return synthetic_price_series(horizon_index)
        return _merge_real_and_synthetic(raw, horizon_index, synthetic_price_series)

    if str(source) == "forecast":
        log.info("Fitting seasonal price forecaster for FI (90-day history + level correction)")
        try:
            forecaster = SeasonalPriceForecaster()
            forecaster.fit(area=FINLAND_AREA_CODE)
            forecast = forecaster.predict(horizon_index)
        except Exception as exc:
            log.error("SeasonalPriceForecaster failed (%s); falling back to synthetic", exc)
            return synthetic_price_series(horizon_index)
        # Blend: real ENTSO-E prices where available, forecaster elsewhere
        client = NordPoolClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = client.get_day_ahead_prices(start, end)
            return _merge_real_and_synthetic(raw, horizon_index, lambda idx: forecaster.predict(idx))
        except Exception as exc:
            log.warning("ENTSO-E live fetch failed (%s); using forecaster only", exc)
            return forecast

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Price forecast file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["price_eur_mwh"].reindex(horizon_index, method="nearest")

    if series.isna().any():
        log.warning("Price forecast has %d NaN values; forward-filling", series.isna().sum())
        series = series.ffill().bfill()

    return series


def load_inflow_forecast(
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
    inflow_cfg: HydroInflowConfig | None = None,
    syke_station_id: int | None = None,
) -> pd.Series:
    """Load a hydro inflow forecast (GWh/h, water-equivalent) aligned to *horizon_index*.

    *source*: "synthetic" (requires *inflow_cfg*), "api" (SYKE level correction
    if *syke_station_id* provided, else synthetic), or path to CSV with columns
    'timestamp', 'inflow_gwh_per_h'.
    """
    if str(source) == "api":
        if inflow_cfg is None:
            raise ValueError("inflow_cfg required for inflow forecast")
        synthetic = synthetic_inflow_series(
            horizon_index,
            annual_avg_gwh_per_hour=inflow_cfg.annual_avg_gwh_per_hour,
            seasonal_amplitude=inflow_cfg.seasonal_amplitude,
            peak_day_of_year=inflow_cfg.peak_day_of_year,
        )
        if syke_station_id is None:
            log.info("No SYKE station configured; using synthetic inflow")
            return synthetic
        log.info("Fetching SYKE discharge history for station %d", syke_station_id)
        try:
            forecaster = SYKEInflowForecaster(station_id=syke_station_id)
            forecaster.fit()
            return forecaster.apply(synthetic)
        except Exception as exc:
            log.error("SYKE inflow correction failed (%s); falling back to synthetic", exc)
            return synthetic

    if str(source) == "synthetic":
        if inflow_cfg is None:
            raise ValueError("inflow_cfg required for synthetic inflow generation")
        log.info("Using synthetic inflow forecast (development mode)")
        return synthetic_inflow_series(
            horizon_index,
            annual_avg_gwh_per_hour=inflow_cfg.annual_avg_gwh_per_hour,
            seasonal_amplitude=inflow_cfg.seasonal_amplitude,
            peak_day_of_year=inflow_cfg.peak_day_of_year,
        )

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Inflow forecast file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["inflow_gwh_per_h"].reindex(horizon_index, method="nearest")
    return series.ffill().bfill()


def load_wind_forecast(
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
    wind_cfgs: dict[str, WindFarmConfig] | None = None,
) -> pd.Series:
    """Load an aggregated wind power forecast (MW) aligned to *horizon_index*.

    *source*: "synthetic" (requires *wind_cfgs*) or path to CSV with columns
    'timestamp', 'wind_mw'.

    Synthetic generation uses a seasonal capacity factor model:
      - Higher output in winter/spring (Ostrobothnia wind regime)
      - Random hourly noise on top of seasonal pattern
    """
    if str(source) == "synthetic":
        if not wind_cfgs:
            log.info("No wind assets configured; wind schedule = 0")
            return pd.Series(0.0, index=horizon_index, name="wind_mw")

        total_capacity = sum(w.capacity_mw for w in wind_cfgs.values())
        avg_cf = sum(w.annual_capacity_factor for w in wind_cfgs.values()) / len(wind_cfgs)

        log.info(
            "Using synthetic wind forecast: %.0f MW capacity, %.1f%% avg CF",
            total_capacity, avg_cf * 100,
        )
        return _synthetic_wind(horizon_index, total_capacity, avg_cf)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Wind forecast file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["wind_mw"].reindex(horizon_index, method="nearest")
    return series.ffill().bfill().clip(lower=0)


def load_fcr_n_price_forecast(
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load an FCR-N capacity price forecast (EUR/MW/h) aligned to *horizon_index*.

    *source*: "synthetic" or path to CSV with columns 'timestamp', 'price_eur_mwh'.

    Fingrid dataset 317 contains historical FCR-N hourly market prices.
    Historical range (Finland): ~0.5–15 EUR/MW/h; typical average 3–6 EUR/MW/h.
    Higher in winter (tight frequency regulation periods) and during low hydro levels.
    """
    if str(source) == "synthetic":
        log.info("Using synthetic FCR-N price forecast (development mode)")
        return _synthetic_fcr_n_price(horizon_index)

    if str(source) == "api":
        log.info("Fetching FCR-N capacity prices from Fingrid (dataset 84)")
        client = FingridClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = client.get_fcr_n_prices(start, end)
        except Exception as exc:
            log.error("Fingrid FCR-N fetch failed (%s); falling back to synthetic", exc)
            return _synthetic_fcr_n_price(horizon_index)
        return _merge_real_and_synthetic(raw, horizon_index, _synthetic_fcr_n_price)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"FCR-N price file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["price_eur_mwh"].reindex(horizon_index, method="nearest")
    return series.ffill().bfill().clip(lower=0)


def _synthetic_fcr_n_price(
    index: pd.DatetimeIndex,
    base: float = 5.0,
    seed: int = 99,
) -> pd.Series:
    """Synthetic FCR-N capacity price series (EUR/MW/h).

    Seasonal pattern: higher in winter (tight Nordic hydro reserves).
    Daily pattern: slightly higher during demand peaks (tighter balance).
    """
    rng = np.random.default_rng(seed)
    n = len(index)

    day_of_year = index.day_of_year.to_numpy()
    # Winter peak (day ~15), summer trough (day ~200)
    seasonal = 1.0 + 0.4 * np.cos(2 * math.pi * (day_of_year - 15) / 365)

    hour_of_day = index.hour.to_numpy()
    daily = 1.0 + 0.1 * np.sin(2 * math.pi * (hour_of_day - 8) / 24)

    noise = rng.exponential(scale=1.5, size=n)  # right-skewed (occasional price spikes)
    prices = base * seasonal * daily + noise
    return pd.Series(np.clip(prices, 0.1, 30.0), index=index, name="fcr_n_price_eur_mw_h")


def load_fcr_d_price_forecast(
    direction: str,
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load an FCR-D capacity price forecast (EUR/MW/h) aligned to *horizon_index*.

    *direction*: "up" or "down".
    *source*: "synthetic" or path to CSV with columns 'timestamp', 'price_eur_mwh'.

    FCR-D Up activates on under-frequency (49.5-50.0 Hz droop) and typically
    commands a premium over FCR-N. FCR-D Down activates on over-frequency
    (50.0-50.5 Hz droop) and is usually priced lower than FCR-D Up.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    if str(source) == "synthetic":
        log.info("Using synthetic FCR-D %s price forecast (development mode)", direction)
        return _synthetic_fcr_d_price(horizon_index, direction)

    if str(source) == "api":
        log.info("Fetching FCR-D %s capacity prices from Fingrid", direction)
        client = FingridClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = (
                client.get_fcr_d_up_prices(start, end)
                if direction == "up"
                else client.get_fcr_d_down_prices(start, end)
            )
        except Exception as exc:
            log.error("Fingrid FCR-D %s fetch failed (%s); falling back to synthetic", direction, exc)
            return _synthetic_fcr_d_price(horizon_index, direction)
        return _merge_real_and_synthetic(raw, horizon_index, lambda idx: _synthetic_fcr_d_price(idx, direction))

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"FCR-D {direction} price file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["price_eur_mwh"].reindex(horizon_index, method="nearest")
    return series.ffill().bfill().clip(lower=0)


def _synthetic_fcr_d_price(
    index: pd.DatetimeIndex,
    direction: str,
    seed: int = 142,
) -> pd.Series:
    """Synthetic FCR-D capacity price series (EUR/MW/h).

    FCR-D Up commands a premium over FCR-N (higher activation value, scarcer
    in winter peak-demand hours). FCR-D Down is priced lower on average.
    """
    base = 10.0 if direction == "up" else 6.0
    rng = np.random.default_rng(seed if direction == "up" else seed + 1)
    n = len(index)

    day_of_year = index.day_of_year.to_numpy()
    seasonal = 1.0 + 0.4 * np.cos(2 * math.pi * (day_of_year - 15) / 365)

    hour_of_day = index.hour.to_numpy()
    daily = 1.0 + 0.1 * np.sin(2 * math.pi * (hour_of_day - 8) / 24)

    noise = rng.exponential(scale=2.0, size=n)
    prices = base * seasonal * daily + noise
    cap = 40.0 if direction == "up" else 25.0
    return pd.Series(np.clip(prices, 0.1, cap), index=index, name=f"fcr_d_{direction}_price_eur_mw_h")


def load_se2_price_forecast(
    source: str | Path,
    horizon_index: pd.DatetimeIndex,
    fi_price_series: pd.Series | None = None,
) -> pd.Series:
    """Load SE2 day-ahead price forecast aligned to *horizon_index*.

    *source*:
      - "synthetic" → correlated with FI prices (correlation ~0.85 historically);
        requires *fi_price_series* to be provided.
      - "same_as_fi" → uses identical FI prices (simplification).
      - Path to CSV with columns 'timestamp', 'price_eur_mwh'.

    FI–SE price divergence (congestion) occurs when the Fenno–Skan interconnector is
    constrained. For a planning model, the correlated-but-different synthetic is
    more realistic than assuming price equality.
    """
    if str(source) == "same_as_fi":
        if fi_price_series is None:
            raise ValueError("fi_price_series required for 'same_as_fi' source")
        log.info("SE2 price = FI price (price-convergence assumption)")
        return fi_price_series.rename("price_eur_mwh_se2")

    if str(source) == "api":
        log.info("Fetching SE2 day-ahead prices from ENTSO-E")
        client = NordPoolClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = client.get_day_ahead_prices(start, end, area=SWEDEN_SE2_AREA_CODE)
        except Exception as exc:
            log.error("ENTSO-E SE2 fetch failed (%s); falling back to synthetic SE2", exc)
            if fi_price_series is not None:
                return _synthetic_se2_price(horizon_index, fi_price_series)
            return synthetic_price_series(horizon_index)
        fallback = (lambda idx: _synthetic_se2_price(idx, fi_price_series)
                    if fi_price_series is not None else synthetic_price_series(idx))
        return _merge_real_and_synthetic(raw, horizon_index, fallback)

    if str(source) == "forecast":
        log.info("Fitting seasonal price forecaster for SE2 (90-day history + level correction)")
        try:
            forecaster = SeasonalPriceForecaster()
            forecaster.fit(area=SWEDEN_SE2_AREA_CODE)
            forecast = forecaster.predict(horizon_index)
        except Exception as exc:
            log.error("SE2 SeasonalPriceForecaster failed (%s); falling back to synthetic", exc)
            if fi_price_series is not None:
                return _synthetic_se2_price(horizon_index, fi_price_series)
            return synthetic_price_series(horizon_index)
        # Blend: real ENTSO-E prices where available, forecaster elsewhere
        client = NordPoolClient()
        start = horizon_index[0].to_pydatetime()
        end = (horizon_index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        try:
            raw = client.get_day_ahead_prices(start, end, area=SWEDEN_SE2_AREA_CODE)
            return _merge_real_and_synthetic(raw, horizon_index, lambda idx: forecaster.predict(idx))
        except Exception as exc:
            log.warning("ENTSO-E SE2 live fetch failed (%s); using forecaster only", exc)
            return forecast

    if str(source) == "synthetic":
        if fi_price_series is None:
            raise ValueError("fi_price_series required for synthetic SE2 generation")
        log.info("Generating synthetic SE2 price (correlated with FI, ~0.85 Pearson r)")
        return _synthetic_se2_price(horizon_index, fi_price_series)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"SE2 price forecast file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    series = df["price_eur_mwh"].reindex(horizon_index, method="nearest")
    return series.ffill().bfill()


def _synthetic_se2_price(
    index: pd.DatetimeIndex,
    fi_prices: pd.Series,
    seed: int = 13,
) -> pd.Series:
    """Generate SE2 prices correlated with FI but with independent spread."""
    rng = np.random.default_rng(seed)
    # SE2 historically ~1–5 EUR/MWh below FI on average (higher hydro availability)
    fi = fi_prices.values
    spread = rng.normal(-2.0, 5.0, len(fi))  # independent congestion component
    se2 = fi + spread
    se2 = np.clip(se2, -500, 4000)
    return pd.Series(se2, index=index, name="price_eur_mwh_se2")


def _synthetic_wind(
    index: pd.DatetimeIndex,
    capacity_mw: float,
    avg_cf: float,
    seed: int = 7,
) -> pd.Series:
    """Generate a synthetic wind power series with seasonal + random pattern."""
    rng = np.random.default_rng(seed)
    n = len(index)

    # Seasonal component: winter peak (day ~30), summer trough (day ~200)
    day_of_year = index.day_of_year.to_numpy()
    seasonal_cf = avg_cf * (1 + 0.4 * np.cos(2 * math.pi * (day_of_year - 30) / 365))

    # Add hourly noise (wind is highly variable)
    noise = rng.normal(0, 0.15 * avg_cf, n)
    cf = np.clip(seasonal_cf + noise, 0, 1.0)
    return pd.Series(cf * capacity_mw, index=index, name="wind_mw")
