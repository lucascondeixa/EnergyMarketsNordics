"""Nord Pool Elspot (day-ahead) market module.

Net position = hydro turbine + nuclear generation + wind (must-take) - pump consumption.

Wind is a non-dispatchable must-take asset. Its scheduled output is added to the net
position directly; the optimiser does not decide wind dispatch.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


def add_elspot_revenue(
    model: pyo.ConcreteModel,
    price_forecast: pd.Series,
    nuclear_unit_names: list[str],
    wind_schedule: pd.Series | None = None,
    hydro_block_names: list[str] | None = None,
) -> None:
    """Add Elspot net position variable and revenue expression to *model*.

    *price_forecast* must be indexed by the same integer steps as model.T.
    *wind_schedule* is optional; if provided, wind output (MW) is added as a
    Param to the net position (must-take, always dispatched at floor price).
    *hydro_block_names* lists all named hydro blocks to include in the net
    position (defaults to ["hydro"] for backwards compatibility).
    """
    T = model.T
    prices = {t: float(price_forecast.iloc[t]) for t in T}
    _hydro_names = hydro_block_names if hydro_block_names is not None else ["hydro"]

    if wind_schedule is not None:
        wind_mw = {t: float(wind_schedule.iloc[t]) for t in T}
        model.wind_schedule = pyo.Param(T, initialize=wind_mw)
    else:
        model.wind_schedule = pyo.Param(T, initialize=0.0)

    # Net position bid into Elspot (MW)
    model.elspot_bid = pyo.Var(T, domain=pyo.Reals)

    def _net_position(m, t):
        nuclear_gen = sum(getattr(m, f"{n}_gen")[t] for n in nuclear_unit_names)
        hydro_gen = sum(getattr(m, f"{n}_gen")[t] for n in _hydro_names)
        pump_cons = sum(getattr(m, f"{n}_pump_cons")[t] for n in _hydro_names)
        return m.elspot_bid[t] == hydro_gen + nuclear_gen + m.wind_schedule[t] - pump_cons

    model.elspot_balance = pyo.Constraint(T, rule=_net_position)

    def _revenue(m, t):
        return prices[t] * m.elspot_bid[t]

    model.elspot_revenue_expr = pyo.Expression(T, rule=_revenue)


def build_bid_curves(
    result_df: pd.DataFrame,
    price_forecast: pd.Series,
    n_steps: int = 10,
    price_floor: float = -500.0,
    price_cap: float = 4000.0,
) -> pd.DataFrame:
    """Convert optimal hourly dispatch into staircase bid curves for Day 1 only.

    Returns a DataFrame with columns:
        hour, step, price_eur_mwh, quantity_mw
    Suitable for manual review and submission.
    """
    day1 = result_df.iloc[:24].copy()
    forecast_day1 = price_forecast.iloc[:24]

    records = []
    for hour, (_, row) in enumerate(day1.iterrows()):
        q_opt = row["elspot_bid_mw"]
        p_opt = float(forecast_day1.iloc[hour])

        price_steps = [
            price_floor + (p_opt - price_floor) * i / (n_steps - 1)
            for i in range(n_steps)
        ]
        qty_steps = [q_opt * (1 - i / (n_steps - 1)) for i in range(n_steps)]

        for step, (price, qty) in enumerate(zip(price_steps, qty_steps)):
            records.append({
                "hour": hour,
                "delivery_hour": (day1.index[hour] if hasattr(day1.index[hour], "hour") else hour),
                "step": step,
                "price_eur_mwh": round(price, 2),
                "quantity_mw": round(qty, 2),
            })

    return pd.DataFrame(records)
