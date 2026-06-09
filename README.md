# EnergyMarketsNordics

Day-ahead and ancillary services bidding optimiser for a Finnish/Swedish hydro-nuclear portfolio. Built as a decision-support tool — it recommends bids for a human trader to review and submit to Nord Pool and Fingrid.

## Portfolio

| Zone | Assets |
|------|--------|
| **FI** | Loviisa 1 & 2 nuclear (507 MW each) · Vuoksi + Oulujoki hydro (1,104 MW) · Kemijoki Oy hydro (630 MW, 57% share) · Pjelax wind (228 MW, 60% Fortum share) |
| **SE2** | Kymmen (50 MW) + Letten (34 MW) + Eggsjön (0.55 MW) pump storage — arbitrage only |

## What it does

- Solves a **7-day rolling MILP** (168 × 1h time steps) jointly optimising Elspot day-ahead bids and FCR-N ancillary service capacity
- Zone FI: nuclear unit commitment (min up/down times, ramp limits, planned outages) + hydro reservoir dynamics + wind must-take
- Zone SE2: pump storage arbitrage (independent from FI)
- Outputs hourly dispatch schedules and 10-step Elspot bid curves ready for manual review

## Price data

Day-ahead prices are fetched from the **ENTSO-E Transparency Platform** (free API, requires registration). A `SeasonalPriceForecaster` fills the horizon beyond confirmed D+1 prices using:

1. 90 days of historical ENTSO-E A44 data for the target area
2. Median profile by `(hour_of_day, day_of_week)` — 168 calendar bins
3. Level correction anchored to the last 7 days of actuals (corrects for current price regime)

FCR-N and hydro inflow forecasts remain synthetic (seasonal models); real sources are planned for Phase 2.

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API key
echo "ENTSOE_API_KEY=your-token-here" > .env

# Run with live ENTSO-E prices + seasonal forecast for remaining horizon
python main.py --price-fi-csv forecast --price-se2-csv forecast

# Run with synthetic prices (no API key needed, for development)
python main.py

# Custom horizon or output directory
python main.py --horizon 48 --output-dir data/processed/bids/
```

ENTSO-E API tokens are free — register at [transparency.entsoe.eu](https://transparency.entsoe.eu).

## Project structure

```
src/
  assets/
    hydro.py            # Turbine + reservoir dynamics (Pyomo block)
    nuclear.py          # VVER-440 unit commitment (Pyomo block)
    pump_hydro.py       # Pump storage (SE2 assets)
  data_ingestion/
    nordpool.py         # ENTSO-E Transparency API client
    fingrid.py          # Fingrid Open Data API client
    price_forecast.py   # SeasonalPriceForecaster (90-day + level correction)
    forecasts.py        # Forecast adapter (api / forecast / synthetic / CSV)
  markets/
    elspot.py           # Elspot revenue and bid curve construction
    ancillary.py        # FCR-N capacity market constraints
  optimization/
    joint_optimizer.py  # FI MILP (nuclear + hydro + wind + FCR-N)
    pump_arb_optimizer.py  # SE2 pump storage arbitrage MILP
  bidding/
    exporter.py         # CSV and bid-curve export
  reporting/
    dashboard.py        # Streamlit bid-review dashboard (stub, Phase 2)
  utils/
    schema.py           # Pydantic config and result schemas
    time_utils.py       # Horizon building, synthetic series

configs/
  plant_params.yaml     # Asset parameters (capacity, costs, reservoir, inflow)
  market_params.yaml    # Market rules (price caps, FCR-N limits, solver settings)

tests/                  # 42 unit tests (pytest)
notebooks/              # Exploratory analysis (Fingrid data, prices, results)
```

## Output

The optimiser reports **cash generation revenue** separately from the terminal water value credit (a planning parameter, not actual cash):

```
7-day Expected Revenue (cash)
  FI  Elspot generation:        EUR   18,662,763
  FI  FCR-N capacity (avg 14 MW): EUR       12,052
  FI  Total cash:               EUR   18,674,815
  SE2 Pump arb:                 EUR    1,817,651
  ─────────────────────────────────────────────
  Portfolio cash total:         EUR   20,492,465
  Terminal water credit:        EUR   23,183,927  (planning only)
```

Dispatch schedules and Elspot bid curves are written to `data/processed/bids/`.

## Configuration

Key parameters in `configs/plant_params.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hydro capacity (Vuoksi/Oulujoki) | 1,104 MW | Direct Fortum ownership (Vuoksi 254 MW + Oulujoki ~850 MW) |
| Reservoir (Vuoksi/Oulujoki) | 500 GWh | Oulujärvi usable storage |
| Kemijoki capacity | 630 MW | Fortum 57% share of Kemijoki Oy fleet (~20 plants); aggregated block |
| Kemijoki reservoir | 480 GWh | 57% share of Kemijärvi seasonal storage (~840 GWh total) |
| Terminal water value | 55 EUR/MWh | Applied to both hydro blocks; sets dispatch threshold at ~61 EUR/MWh |
| Nuclear capacity | 507 MW × 2 | Loviisa 1 & 2; turbine upgrade (+38 MW/unit) modelled from 2026 |
| Nuclear variable cost | 7 EUR/MWh | Fuel + variable O&M |

## Phase 2 roadmap

- Real hydro inflow from SYKE (Finnish Environment Institute)
- FCR-D and aFRR ancillary service markets
- Streamlit dashboard (`src/reporting/dashboard.py` stub exists)
- Intraday (Elbas) re-optimisation layer

## Requirements

Python ≥ 3.11. Core dependencies: `pyomo`, `highspy`, `pandas`, `numpy`, `pydantic`, `httpx`, `rich`, `typer`.

See `pyproject.toml` for the full list. Install dev extras (`pytest`, `ruff`, `mypy`) with `pip install -e ".[dev]"`.
