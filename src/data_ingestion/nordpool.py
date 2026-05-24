"""Nord Pool market data client.

Covers:
- Historical Elspot day-ahead prices (EUR/MWh) for the FI area
- Data pulled from the ENTSO-E Transparency Platform as a free alternative
  to Nord Pool's commercial API.

ENTSO-E API (free, requires registration at transparency.entsoe.eu):
  - Document type A44 = Day-ahead prices
  - Area code for Finland = 10YFI-1--------U

Set ENTSOE_API_KEY environment variable with your token.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
FINLAND_AREA_CODE = "10YFI-1--------U"
SWEDEN_SE1_AREA_CODE = "10Y1001A1001A44P"
SWEDEN_SE2_AREA_CODE = "10Y1001A1001A45N"
SWEDEN_SE3_AREA_CODE = "10Y1001A1001A46L"
SWEDEN_SE4_AREA_CODE = "10Y1001A1001A47J"


class NordPoolClient:
    """Client for day-ahead price data via ENTSO-E Transparency Platform."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ENTSOE_API_KEY", "")
        if not self.api_key:
            log.warning(
                "ENTSOE_API_KEY not set. Register at https://transparency.entsoe.eu "
                "to get a free API token."
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_day_ahead_prices(
        self,
        start: datetime,
        end: datetime,
        area: str = FINLAND_AREA_CODE,
    ) -> pd.Series:
        """Fetch day-ahead Elspot prices (EUR/MWh) from ENTSO-E.

        Returns a pd.Series with hourly UTC DatetimeIndex.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        params = {
            "securityToken": self.api_key,
            "documentType": "A44",
            "in_Domain": area,
            "out_Domain": area,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }

        log.debug("Fetching ENTSO-E day-ahead prices for %s from %s to %s", area, start, end)

        with httpx.Client(timeout=60) as client:
            response = client.get(ENTSOE_BASE_URL, params=params)
            response.raise_for_status()

        return _parse_entsoe_day_ahead(response.text)

    def get_se2_day_ahead_prices(self, start: datetime, end: datetime) -> pd.Series:
        return self.get_day_ahead_prices(start, end, area=SWEDEN_SE2_AREA_CODE)

    def get_latest_day_ahead_prices(self, area: str = FINLAND_AREA_CODE) -> pd.Series:
        """Convenience: fetch yesterday + today prices."""
        now = datetime.now(tz=timezone.utc)
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_day_ahead_prices(start, end, area)


def _parse_entsoe_day_ahead(xml_text: str) -> pd.Series:
    """Parse ENTSO-E A44 XML response into a UTC-indexed pd.Series."""
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    root = ElementTree.fromstring(xml_text)

    records: list[tuple[datetime, float]] = []

    for period in root.findall(".//ns:Period", ns):
        start_elem = period.find("ns:timeInterval/ns:start", ns)
        resolution_elem = period.find("ns:resolution", ns)
        if start_elem is None or resolution_elem is None:
            continue

        period_start = datetime.fromisoformat(start_elem.text.replace("Z", "+00:00"))
        resolution_text = resolution_elem.text  # e.g. "PT60M"
        resolution_minutes = int(resolution_text.replace("PT", "").replace("M", ""))

        for point in period.findall("ns:Point", ns):
            pos_elem = point.find("ns:position", ns)
            price_elem = point.find("ns:price.amount", ns)
            if pos_elem is None or price_elem is None:
                continue
            position = int(pos_elem.text)
            price = float(price_elem.text)
            ts = period_start + timedelta(minutes=resolution_minutes * (position - 1))
            records.append((ts, price))

    if not records:
        return pd.Series(dtype=float)

    idx, vals = zip(*sorted(records))
    return pd.Series(list(vals), index=pd.DatetimeIndex(idx, tz="UTC"), name="price_eur_mwh")
