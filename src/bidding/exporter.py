"""Bid export utilities.

Exports optimiser output to formats suitable for manual review and submission:
- CSV (human-readable)
- Nord Pool Elspot bid format (simplified; full XML schema in Phase 2)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def export_dispatch_schedule(
    result_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export the full 7-day dispatch schedule to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path)
    return path


def export_elspot_bids(
    bid_curves: pd.DataFrame,
    output_path: Path | str,
    delivery_date: datetime | None = None,
) -> Path:
    """Export Day 1 Elspot bid curves to a CSV file for manual review.

    Columns: delivery_date, hour, step, price_eur_mwh, quantity_mw
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = bid_curves.copy()
    if delivery_date is not None:
        df.insert(0, "delivery_date", delivery_date.strftime("%Y-%m-%d"))

    df.to_csv(path, index=False)
    return path


def print_summary(result_df: pd.DataFrame, objective_eur: float) -> None:
    """Print a human-readable one-page summary to stdout."""
    day1 = result_df.iloc[:24]
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold green]Optimisation Result[/bold green]")
    console.print(f"Expected revenue (7-day):  EUR {objective_eur:,.0f}")
    console.print(f"\n[bold]Day 1 summary:[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Hour", style="dim", width=6)
    table.add_column("Hydro (MW)", justify="right")
    table.add_column("Pump (MW)", justify="right")
    table.add_column("Nuclear (MW)", justify="right")
    table.add_column("Elspot bid (MW)", justify="right")
    table.add_column("Reservoir (GWh)", justify="right")

    for ts, row in day1.iterrows():
        table.add_row(
            str(getattr(ts, "hour", "-")),
            f"{row['hydro_gen_mw']:.1f}",
            f"{row['pump_cons_mw']:.1f}",
            f"{row['nuclear_gen_mw']:.1f}",
            f"{row['elspot_bid_mw']:.1f}",
            f"{row['reservoir_gwh']:.2f}",
        )

    console.print(table)
