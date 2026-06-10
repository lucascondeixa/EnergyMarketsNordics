"""Main CLI entry point — Fortum Nordic portfolio optimiser.

Runs two independent zone optimisations and reports combined portfolio revenue:
  Zone FI:  Loviisa 1+2 nuclear + Vuoksi/Oulujoki hydro + Pjelax wind
  Zone SE2: Kymmen + Letten + Eggsjön pump storage (arbitrage)

Usage:
    python main.py                          # synthetic forecasts (dev mode)
    python main.py --price-fi-csv data/raw/fi_prices.csv --price-se2-csv data/raw/se2_prices.csv
    python main.py --horizon 168 --output data/processed/bids/
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; no-op if missing

import typer
import yaml
from rich.console import Console

from src.bidding.exporter import export_dispatch_schedule, export_elspot_bids, print_summary
from src.data_ingestion.forecasts import (
    load_fcr_d_price_forecast,
    load_fcr_n_price_forecast,
    load_inflow_forecast,
    load_price_forecast,
    load_se2_price_forecast,
    load_wind_forecast,
)
from src.markets.elspot import build_bid_curves
from src.optimization.joint_optimizer import build_and_solve, result_to_dataframe
from src.optimization.pump_arb_optimizer import build_and_solve_se2, se2_result_to_dataframe
from src.utils.schema import MarketConfig, PlantConfig
from src.utils.time_utils import build_horizon, next_day_ahead_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="emn", add_completion=False, help="Fortum Nordic Energy Markets Optimizer")


@app.command()
def run(
    plant_config: Path = typer.Option(Path("configs/plant_params.yaml")),
    market_config: Path = typer.Option(Path("configs/market_params.yaml")),
    price_fi_csv: str = typer.Option("synthetic", help="FI price forecast CSV or 'synthetic'"),
    price_se2_csv: str = typer.Option("synthetic", help="SE2 price forecast CSV or 'synthetic'"),
    inflow_csv: str = typer.Option("synthetic", help="Hydro inflow forecast CSV or 'synthetic'"),
    wind_csv: str = typer.Option("synthetic", help="Wind forecast CSV or 'synthetic'"),
    fcr_n_csv: str = typer.Option("synthetic", help="FCR-N capacity price CSV or 'synthetic'"),
    fcr_d_up_csv: str = typer.Option("synthetic", help="FCR-D Up capacity price CSV or 'synthetic'"),
    fcr_d_down_csv: str = typer.Option("synthetic", help="FCR-D Down capacity price CSV or 'synthetic'"),
    horizon: int = typer.Option(168, help="Optimisation horizon in hours"),
    output_dir: Path = typer.Option(Path("data/processed/bids")),
) -> None:
    """Run the joint Fortum portfolio optimisation (FI + SE2)."""

    # --- Load configs ---
    with open(plant_config) as f:
        plant_raw = yaml.safe_load(f)
    with open(market_config) as f:
        market_raw = yaml.safe_load(f)

    plant_cfg = PlantConfig(
        hydro=plant_raw["hydro"],
        nuclear=plant_raw["nuclear"],
        wind=plant_raw.get("wind", {}),
        se2_pump_storage=plant_raw.get("se2_pump_storage", {}),
        kemijoki=plant_raw.get("kemijoki"),
    )
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=market_raw.get("ancillary_services", {}),
        optimisation=market_raw["optimisation"],
    )
    market_cfg.optimisation.horizon_hours = horizon

    horizon_start = next_day_ahead_gate()
    log.info("Horizon: %s (%d hours)", horizon_start.isoformat(), horizon)
    index = build_horizon(horizon_start, horizon_hours=horizon)

    # -----------------------------------------------------------------------
    # Load forecasts
    # -----------------------------------------------------------------------
    price_fi = load_price_forecast(price_fi_csv, index)
    inflow = load_inflow_forecast(
        inflow_csv, index,
        inflow_cfg=plant_cfg.hydro.inflow,
        syke_station_id=plant_cfg.hydro.syke_station_id,
    )
    kemijoki_inflow = (
        load_inflow_forecast(
            inflow_csv, index,
            inflow_cfg=plant_cfg.kemijoki.inflow,
            syke_station_id=plant_cfg.kemijoki.syke_station_id,
        )
        if plant_cfg.kemijoki is not None else None
    )
    wind = load_wind_forecast(wind_csv, index, wind_cfgs=plant_cfg.wind)
    price_se2 = load_se2_price_forecast(price_se2_csv, index, fi_price_series=price_fi)
    fcr_n_prices = (
        load_fcr_n_price_forecast(fcr_n_csv, index)
        if market_cfg.ancillary_services.FCR_N.enabled
        else None
    )
    fcr_d_up_prices = (
        load_fcr_d_price_forecast("up", fcr_d_up_csv, index)
        if market_cfg.ancillary_services.FCR_D_UP.enabled
        else None
    )
    fcr_d_down_prices = (
        load_fcr_d_price_forecast("down", fcr_d_down_csv, index)
        if market_cfg.ancillary_services.FCR_D_DOWN.enabled
        else None
    )

    # -----------------------------------------------------------------------
    # Zone FI optimisation
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Zone FI — Nuclear + Hydro + Wind[/bold blue]")
    fi_model, fi_result = build_and_solve(
        plant_cfg, market_cfg, price_fi, inflow, horizon_start,
        wind_schedule=wind, fcr_n_prices=fcr_n_prices,
        fcr_d_up_prices=fcr_d_up_prices, fcr_d_down_prices=fcr_d_down_prices,
        kemijoki_inflow_series=kemijoki_inflow,
    )
    fi_df = result_to_dataframe(fi_result)

    # -----------------------------------------------------------------------
    # Zone SE2 optimisation (pump storage arbitrage)
    # -----------------------------------------------------------------------
    if plant_cfg.se2_pump_storage:
        console.rule("[bold green]Zone SE2 — Pump Storage Arbitrage (Kymmen/Letten/Eggsjön)[/bold green]")
        se2_model, se2_result = build_and_solve_se2(
            plant_cfg.se2_pump_storage, market_cfg, price_se2, horizon_start
        )
        se2_df = se2_result_to_dataframe(se2_result)
        se2_revenue = se2_result.objective_value_eur
    else:
        se2_result = None
        se2_df = None
        se2_revenue = 0.0

    # -----------------------------------------------------------------------
    # Build FI bid curves (Day 1)
    # -----------------------------------------------------------------------
    fi_bid_curves = build_bid_curves(
        fi_df, price_fi,
        n_steps=market_cfg.elspot.bid_steps,
        price_floor=market_cfg.elspot.price_floor_eur_mwh,
        price_cap=market_cfg.elspot.price_cap_eur_mwh,
    )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = horizon_start.strftime("%Y%m%d")

    export_dispatch_schedule(fi_df, output_dir / f"{date_str}_FI_schedule.csv")
    export_elspot_bids(fi_bid_curves, output_dir / f"{date_str}_FI_elspot_bids.csv", delivery_date=horizon_start)

    if se2_df is not None:
        export_dispatch_schedule(se2_df, output_dir / f"{date_str}_SE2_schedule.csv")
        log.info("SE2 schedule exported")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.rule("[bold]Portfolio Summary[/bold]")
    print_summary(fi_df, fi_result.objective_value_eur)

    fi_ancillary_total = (
        fi_result.fcr_n_revenue_eur
        + fi_result.fcr_d_up_revenue_eur
        + fi_result.fcr_d_down_revenue_eur
    )
    fi_cash_total = fi_result.elspot_cash_revenue_eur + fi_ancillary_total
    total_cash = fi_cash_total + se2_revenue
    console.print(f"\n[bold]7-day Expected Revenue (cash)[/bold]")
    console.print(f"  FI  Elspot generation:        EUR {fi_result.elspot_cash_revenue_eur:>12,.0f}")
    if fi_result.fcr_n_revenue_eur > 0:
        avg_fcr_n = sum(fi_result.fcr_n_total_mw) / len(fi_result.fcr_n_total_mw)
        console.print(f"  FI  FCR-N capacity (avg {avg_fcr_n:.0f} MW): EUR {fi_result.fcr_n_revenue_eur:>12,.0f}")
    if fi_result.fcr_d_up_revenue_eur > 0:
        avg_fcr_d_up = sum(fi_result.fcr_d_up_total_mw) / len(fi_result.fcr_d_up_total_mw)
        console.print(f"  FI  FCR-D Up (avg {avg_fcr_d_up:.0f} MW):    EUR {fi_result.fcr_d_up_revenue_eur:>12,.0f}")
    if fi_result.fcr_d_down_revenue_eur > 0:
        avg_fcr_d_down = sum(fi_result.fcr_d_down_total_mw) / len(fi_result.fcr_d_down_total_mw)
        console.print(f"  FI  FCR-D Down (avg {avg_fcr_d_down:.0f} MW):  EUR {fi_result.fcr_d_down_revenue_eur:>12,.0f}")
    console.print(f"  FI  Total cash:               EUR {fi_cash_total:>12,.0f}")
    if se2_result:
        console.print(f"  SE2 Pump arb:                 EUR {se2_revenue:>12,.0f}")
    console.print(f"  ─────────────────────────────────────────────")
    console.print(f"  Portfolio cash total:         EUR {total_cash:>12,.0f}")
    console.print(f"  [dim]Terminal water credit:       EUR {fi_result.terminal_water_value_eur:>12,.0f}  (planning only)[/dim]")

    typer.echo(f"\nFI solve:   {fi_result.solve_time_seconds:.1f}s  ({fi_result.status})")
    if se2_result:
        typer.echo(f"SE2 solve:  {se2_result.solve_time_seconds:.1f}s  ({se2_result.status})")


if __name__ == "__main__":
    app()
