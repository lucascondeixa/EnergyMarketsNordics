"""Joint day-ahead + ancillary services optimiser (FI zone).

Objective (maximise):
    Σ_t [ elspot_revenue[t] + as_revenue[t] ]     (Elspot + FCR-N capacity)
  + terminal_water_value × reservoir[T_last]
  - Σ_t [ nuclear_cost[t] ]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import pyomo.environ as pyo

from src.assets.hydro import add_hydro_block
from src.assets.nuclear import add_nuclear_block
from src.assets.pump_hydro import add_pump_hydro_block
from src.markets.ancillary import add_ancillary_services
from src.markets.elspot import add_elspot_revenue
from src.utils.schema import MarketConfig, OptimisationResult, PlantConfig

log = logging.getLogger(__name__)


def build_and_solve(
    plant_cfg: PlantConfig,
    market_cfg: MarketConfig,
    price_forecast: pd.Series,
    inflow_series: pd.Series,
    horizon_start: datetime,
    wind_schedule: pd.Series | None = None,
    fcr_n_prices: pd.Series | None = None,
    kemijoki_inflow_series: pd.Series | None = None,
) -> tuple[pyo.ConcreteModel, OptimisationResult]:
    """Build and solve the joint FI optimisation model.

    Parameters
    ----------
    plant_cfg:        Plant configuration
    market_cfg:       Market configuration (Elspot + AS + solver)
    price_forecast:   FI Elspot price forecast, length = horizon_hours
    inflow_series:    Hydro inflow (GWh/h), length = horizon_hours
    horizon_start:    UTC datetime of the first time step
    wind_schedule:    Optional wind must-take schedule (MW)
    fcr_n_prices:     Optional FCR-N capacity price forecast (EUR/MW/h).
                      Required when market_cfg.ancillary_services.FCR_N.enabled=True.
    """
    opt_cfg = market_cfg.optimisation
    H = opt_cfg.horizon_hours
    nuclear_units = list(plant_cfg.nuclear.keys())
    pump_enabled = plant_cfg.hydro.pump_storage_enabled
    fcr_n_enabled = market_cfg.ancillary_services.FCR_N.enabled
    kemijoki_cfg = plant_cfg.kemijoki

    log.info(
        "Building FI model: %dh | %d nuclear | pump=%s | wind=%s | FCR-N=%s | kemijoki=%s",
        H, len(nuclear_units), pump_enabled,
        "yes" if wind_schedule is not None else "no",
        "yes" if fcr_n_enabled else "no",
        "yes" if kemijoki_cfg else "no",
    )

    # -----------------------------------------------------------------------
    # Model skeleton
    # -----------------------------------------------------------------------
    model = pyo.ConcreteModel(name="Fortum_FI_Joint_Opt")
    model.T = pyo.RangeSet(0, H - 1)

    # -----------------------------------------------------------------------
    # Asset blocks
    # -----------------------------------------------------------------------
    add_hydro_block(model, plant_cfg.hydro, inflow_series, name="hydro")

    if pump_enabled:
        add_pump_hydro_block(model, plant_cfg.hydro, inflow_series)

    if kemijoki_cfg is not None:
        if kemijoki_inflow_series is None:
            raise ValueError("kemijoki_inflow_series required when kemijoki config is present")
        add_hydro_block(model, kemijoki_cfg, kemijoki_inflow_series, name="kemijoki")

    for unit_name, unit_cfg in plant_cfg.nuclear.items():
        add_nuclear_block(model, unit_cfg, unit_name, horizon_start)

    # -----------------------------------------------------------------------
    # Market modules (Elspot + ancillary)
    # -----------------------------------------------------------------------
    hydro_names = ["hydro"] + (["kemijoki"] if kemijoki_cfg else [])
    hydro_blocks = [("hydro", plant_cfg.hydro)] + (
        [("kemijoki", kemijoki_cfg)] if kemijoki_cfg else []
    )

    add_elspot_revenue(
        model, price_forecast, nuclear_units,
        wind_schedule=wind_schedule,
        hydro_block_names=hydro_names,
    )

    add_ancillary_services(
        model,
        as_cfg=market_cfg.ancillary_services,
        hydro_cfg=plant_cfg.hydro,
        nuclear_unit_names=nuclear_units,
        nuclear_cfgs=plant_cfg.nuclear,
        fcr_n_prices=fcr_n_prices,
        hydro_blocks=hydro_blocks,
    )

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    T_last = H - 1

    def _objective(m):
        elspot_rev = sum(m.elspot_revenue_expr[t] for t in m.T)
        as_rev = sum(m.as_revenue_expr[t] for t in m.T)
        terminal_val = (
            getattr(m, "hydro_terminal_water_value") * getattr(m, "hydro_reservoir")[T_last]
        )
        if kemijoki_cfg is not None:
            terminal_val += (
                getattr(m, "kemijoki_terminal_water_value")
                * getattr(m, "kemijoki_reservoir")[T_last]
            )
        nuclear_cost = sum(
            getattr(m, f"{n}_cost_expr")[t] for n in nuclear_units for t in m.T
        )
        return elspot_rev + as_rev + terminal_val - nuclear_cost

    model.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    solver_name = opt_cfg.solver
    solver_opts = opt_cfg.solver_options

    solver = pyo.SolverFactory(solver_name)
    solver.options["mip_rel_gap"] = solver_opts.mip_rel_gap
    solver.options["time_limit"] = solver_opts.time_limit

    t0 = time.perf_counter()
    result_raw = solver.solve(model, tee=False)
    solve_time = time.perf_counter() - t0

    term_cond = result_raw.solver.termination_condition
    log.info("FI solve finished in %.1fs | condition=%s", solve_time, term_cond)

    if term_cond not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ):
        raise RuntimeError(f"FI solver condition: {term_cond}")

    # -----------------------------------------------------------------------
    # Extract results
    # -----------------------------------------------------------------------
    timestamps = [horizon_start + timedelta(hours=t) for t in range(H)]

    def _v(expr) -> float:
        return max(0.0, pyo.value(expr))

    hydro_gen = [_v(model.hydro_gen[t]) for t in model.T]
    pump_cons = [_v(model.hydro_pump_cons[t]) for t in model.T]
    reservoir = [pyo.value(model.hydro_reservoir[t]) / 1000 for t in model.T]
    nuclear_gen = [
        sum(_v(getattr(model, f"{n}_gen")[t]) for n in nuclear_units)
        for t in model.T
    ]
    wind_gen = (
        [float(wind_schedule.iloc[t]) for t in model.T]
        if wind_schedule is not None
        else [0.0] * H
    )
    elspot_bid = [pyo.value(model.elspot_bid[t]) for t in model.T]
    obj_val = pyo.value(model.objective)

    if kemijoki_cfg is not None:
        kemijoki_gen = [_v(model.kemijoki_gen[t]) for t in model.T]
        kemijoki_reservoir = [pyo.value(model.kemijoki_reservoir[t]) / 1000 for t in model.T]
    else:
        kemijoki_gen = [0.0] * H
        kemijoki_reservoir = [0.0] * H

    elspot_cash = sum(pyo.value(model.elspot_revenue_expr[t]) for t in model.T)
    hydro_terminal_wv = (
        pyo.value(model.hydro_terminal_water_value) * pyo.value(model.hydro_reservoir[T_last])
    )
    kemijoki_terminal_wv = (
        pyo.value(model.kemijoki_terminal_water_value) * pyo.value(model.kemijoki_reservoir[T_last])
        if kemijoki_cfg is not None else 0.0
    )
    terminal_wv = hydro_terminal_wv + kemijoki_terminal_wv

    # FCR-N results
    if fcr_n_enabled:
        fcr_n_hydro = [pyo.value(model.r_fcr_n_hydro[t]) for t in model.T]
        fcr_n_nuc = [
            sum(pyo.value(getattr(model, f"r_fcr_n_{n}")[t]) for n in nuclear_units)
            for t in model.T
        ]
        fcr_n_total = [pyo.value(model.r_fcr_n_total[t]) for t in model.T]
        fcr_n_rev = sum(pyo.value(model.fcr_n_revenue_expr[t]) for t in model.T)
    else:
        fcr_n_hydro = [0.0] * H
        fcr_n_nuc = [0.0] * H
        fcr_n_total = [0.0] * H
        fcr_n_rev = 0.0

    try:
        mip_gap = float(result_raw.problem.lower_bound - obj_val) / abs(obj_val)
    except Exception:
        mip_gap = None

    return model, OptimisationResult(
        solve_time_seconds=round(solve_time, 2),
        objective_value_eur=round(obj_val, 2),
        elspot_cash_revenue_eur=round(elspot_cash, 2),
        terminal_water_value_eur=round(terminal_wv, 2),
        mip_gap=mip_gap,
        status=str(term_cond),
        hydro_dispatch_mw=hydro_gen,
        pump_consumption_mw=pump_cons,
        reservoir_level_gwh=reservoir,
        kemijoki_dispatch_mw=kemijoki_gen,
        kemijoki_reservoir_level_gwh=kemijoki_reservoir,
        nuclear_dispatch_mw=nuclear_gen,
        wind_dispatch_mw=wind_gen,
        elspot_bid_mw=elspot_bid,
        fcr_n_hydro_mw=fcr_n_hydro,
        fcr_n_nuclear_mw=fcr_n_nuc,
        fcr_n_total_mw=fcr_n_total,
        fcr_n_revenue_eur=round(fcr_n_rev, 2),
        timestamps=timestamps,
    )


def result_to_dataframe(result: OptimisationResult) -> pd.DataFrame:
    """Convert OptimisationResult to a tidy DataFrame for analysis and export."""
    return pd.DataFrame({
        "timestamp": result.timestamps,
        "hydro_gen_mw": result.hydro_dispatch_mw,
        "pump_cons_mw": result.pump_consumption_mw,
        "reservoir_gwh": result.reservoir_level_gwh,
        "kemijoki_gen_mw": result.kemijoki_dispatch_mw,
        "kemijoki_reservoir_gwh": result.kemijoki_reservoir_level_gwh,
        "nuclear_gen_mw": result.nuclear_dispatch_mw,
        "wind_mw": result.wind_dispatch_mw,
        "elspot_bid_mw": result.elspot_bid_mw,
        "fcr_n_hydro_mw": result.fcr_n_hydro_mw,
        "fcr_n_nuclear_mw": result.fcr_n_nuclear_mw,
        "fcr_n_total_mw": result.fcr_n_total_mw,
    }).set_index("timestamp")
