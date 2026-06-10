"""Fingrid Open Data API client.

Fingrid exposes real-time and historical data at https://data.fingrid.fi/api
API key required (free registration at data.fingrid.fi).

Verified dataset IDs (confirmed against API v2 catalogue):
  - 188:  Nuclear power production - real-time data (MW, 3-min)
  - 191:  Hydro power production - real-time data (MW, 3-min)
  - 181:  Wind power production - real-time data (MW, 3-min)
  - 317:  FCR-N hourly market prices (EUR/MW/h)
  - 318:  FCR-D upward hourly market prices (EUR/MW/h)
  - 320:  FCR-D downward hourly market prices (EUR/MW/h) — verify against catalogue before live use

Note: the Fingrid API v2 returns the value under a key equal to the Finnish
dataset name (e.g. "Ydinvoimatuotanto - reaaliaikatieto"), not "value".
The parser extracts it by finding the key that is neither startTime nor endTime.

Full dataset list: https://data.fingrid.fi/api/datasets
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

FINGRID_BASE_URL = "https://data.fingrid.fi/api"


class FingridClient:
    """Thin async-capable client for the Fingrid Open Data API v2."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINGRID_API_KEY", "")
        if not self.api_key:
            log.warning(
                "FINGRID_API_KEY not set. Set the environment variable or pass api_key. "
                "Register for a free key at https://data.fingrid.fi"
            )
        self._headers = {"x-api-key": self.api_key, "Accept": "application/json"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_dataset(
        self,
        dataset_id: int,
        start: datetime,
        end: datetime,
        resolution: str = "PT1H",
    ) -> pd.Series:
        """Fetch a Fingrid dataset as a pandas Series.

        Parameters
        ----------
        dataset_id:  Fingrid dataset ID (integer)
        start:       Start of the time range (UTC)
        end:         End of the time range (UTC, exclusive)
        resolution:  ISO 8601 duration string (PT1H = 1 hour, PT15M = 15 min)

        Returns
        -------
        pd.Series with UTC DatetimeIndex and float values.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        params = {
            "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
            "oneRowPerTimePeriod": True,
            "page": 1,
            "pageSize": 20000,
        }

        url = f"{FINGRID_BASE_URL}/datasets/{dataset_id}/data"
        log.debug("Fetching Fingrid dataset %d from %s to %s", dataset_id, start, end)

        with httpx.Client(timeout=30) as client:
            response = client.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            data = response.json()

        records = data.get("data", [])
        if not records:
            log.warning("No data returned for dataset %d in range %s–%s", dataset_id, start, end)
            return pd.Series(dtype=float)

        # The value key is the Finnish dataset name, not a fixed "value" field.
        first = records[0]
        value_keys = [k for k in first if k not in ("startTime", "endTime")]
        if not value_keys:
            log.error("Cannot find value key in dataset %d response: %s", dataset_id, first)
            return pd.Series(dtype=float)
        value_key = value_keys[0]
        log.debug("Dataset %d value field: '%s'", dataset_id, value_key)

        timestamps = pd.to_datetime([r["startTime"] for r in records], utc=True)
        values = [r.get(value_key) for r in records]
        series = pd.Series(values, index=timestamps, dtype=float, name=f"dataset_{dataset_id}")
        return series.sort_index()

    def get_fcr_n_prices(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(317, start, end)

    def get_fcr_d_up_prices(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(318, start, end)

    def get_fcr_d_down_prices(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(320, start, end)

    def get_nuclear_production(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(188, start, end)

    def get_hydro_production(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(191, start, end)

    def get_wind_production(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_dataset(181, start, end)
