"""SE2 pump storage arbitrage optimiser.

Optimises the charge/discharge schedule for Fortum's Swedish pump-storage units
(Kymmen, Letten, Eggsjön) against the SE2 day-ahead price forecast.

Each unit is modelled independently with its own reservoir, but they share the
same SE2 price signal. The aggregate net position forms the SE2 Elspot bid.

Objective (maximise):
    Σ_t price_SE2[t] × (Σ_u gen[u,t] - Σ_u pump[u,t])
  + Σ_u terminal_water_value[u] × reservoir[u, T_last]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import (
    MarketConfig,
    PumpStorageUnitConfig,
    SE2OptimisationResult,
)

log = logging.getLogger(__name__)


def build_and_solve_se2(
    units: dict[str, PumpStorageUnitConfig],
    market_cfg: MarketConfig,
    price_forecast_se2: pd.Series,
    horizon_start: datetime,
) -> tuple[pyo.ConcreteModel, SE2OptimisationResult]:
    """Build and solve the SE2 pump storage arbitrage model.

    Parameters
    ----------
    units:              Dict of unit_name -> PumpStorageUnitConfig (Kymmen, Letten, Eggsjön)
    market_cfg:         Market configuration (solver settings; Elspot config for SE2)
    price_forecast_se2: SE2 day-ahead price forecast, length = horizon_hours
    horizon_start:      UTC datetime of the first time step
    """
    opt_cfg = market_cfg.optimisation
    H = opt_cfg.horizon_hours
    unit_names = list(units.keys())

    log.info("Building SE2 pump-arb model: %d hours, %d units (%s)", H, len(unit_names), ", ".join(unit_names))

    model = pyo.ConcreteModel(name="Fortum_SE2_PumpArb")
    model.T = pyo.RangeSet(0, H - 1)
    model.U = pyo.Set(initialize=unit_names)

    prices = {t: float(price_forecast_se2.iloc[t]) for t in range(H)}

    # -----------------------------------------------------------------------
    # Per-unit variables and constraints
    # -----------------------------------------------------------------------
    model.gen = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.pump = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals, initialize=0)
    model.reservoir = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals)
    model.mode = pyo.Var(model.U, model.T, domain=pyo.Binary, initialize=1)  # 1=generate, 0=pump

    # Initialise reservoir at starting level
    for u, cfg in units.items():
        for t in range(H):
            model.reservoir[u, t].set_value(cfg.initial_level_gwh * 1000)

    # --- Turbine / pump capacity bounds ---
    def _gen_max(m, u, t):
        return m.gen[u, t] <= units[u].turbine_capacity_mw * m.mode[u, t]

    def _pump_max(m, u, t):
        return m.pump[u, t] <= units[u].pump_capacity_mw * (1 - m.mode[u, t])

    model.gen_max = pyo.Constraint(model.U, model.T, rule=_gen_max)
    model.pump_max = pyo.Constraint(model.U, model.T, rule=_pump_max)

    # --- Ramp constraints ---
    def _ramp_up(m, u, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.gen[u, t] - m.gen[u, t - 1] <= units[u].ramp_rate_mw_per_hour

    def _ramp_down(m, u, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.gen[u, t - 1] - m.gen[u, t] <= units[u].ramp_rate_mw_per_hour

    model.ramp_up = pyo.Constraint(model.U, model.T, rule=_ramp_up)
    model.ramp_down = pyo.Constraint(model.U, model.T, rule=_ramp_down)

    # --- Reservoir dynamics ---
    # V[u,t] = V[u,t-1] + inflow[u] + pump[u,t]*η_pump - gen[u,t]/η_turbine
    def _dynamics(m, u, t):
        cfg = units[u]
        v_prev = (
            cfg.initial_level_gwh * 1000
            if t == m.T.first()
            else m.reservoir[u, t - 1]
        )
        inflow_mwh = cfg.natural_inflow_gwh_per_hour * 1000
        water_in = m.pump[u, t] * cfg.pump_efficiency
        water_out = m.gen[u, t] / cfg.turbine_efficiency
        return m.reservoir[u, t] == v_prev + inflow_mwh + water_in - water_out

    model.reservoir_dynamics = pyo.Constraint(model.U, model.T, rule=_dynamics)

    def _res_max(m, u, t):
        return m.reservoir[u, t] <= units[u].reservoir_capacity_gwh * 1000

    def _res_min(m, u, t):
        return m.reservoir[u, t] >= units[u].min_level_gwh * 1000

    model.reservoir_max = pyo.Constraint(model.U, model.T, rule=_res_max)
    model.reservoir_min = pyo.Constraint(model.U, model.T, rule=_res_min)

    # -----------------------------------------------------------------------
    # Aggregate SE2 net position
    # -----------------------------------------------------------------------
    model.elspot_bid_se2 = pyo.Var(model.T, domain=pyo.Reals)

    def _net_pos(m, t):
        return m.elspot_bid_se2[t] == sum(m.gen[u, t] - m.pump[u, t] for u in unit_names)

    model.net_position = pyo.Constraint(model.T, rule=_net_pos)

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    T_last = H - 1

    def _objective(m):
        revenue = sum(prices[t] * m.elspot_bid_se2[t] for t in m.T)
        terminal = sum(
            units[u].terminal_water_value_eur_per_mwh * m.reservoir[u, T_last]
            for u in unit_names
        )
        return revenue + terminal

    model.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    solver_name = opt_cfg.solver
    solver_opts = opt_cfg.solver_options

    log.info("Solving SE2 with %s (gap=%.3f, limit=%ds)",
             solver_name, solver_opts.mip_rel_gap, solver_opts.time_limit)

    solver = pyo.SolverFactory(solver_name)
    solver.options["mip_rel_gap"] = solver_opts.mip_rel_gap
    solver.options["time_limit"] = solver_opts.time_limit

    t0 = time.perf_counter()
    result_raw = solver.solve(model, tee=False)
    solve_time = time.perf_counter() - t0

    term_cond = result_raw.solver.termination_condition
    log.info("SE2 solve finished in %.1fs | condition=%s", solve_time, term_cond)

    if term_cond not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ):
        raise RuntimeError(f"SE2 solver condition: {term_cond}")

    # -----------------------------------------------------------------------
    # Extract results
    # -----------------------------------------------------------------------
    timestamps = [horizon_start + timedelta(hours=t) for t in range(H)]
    obj_val = pyo.value(model.objective)

    try:
        mip_gap = float(result_raw.problem.lower_bound - obj_val) / abs(obj_val)
    except Exception:
        mip_gap = None

    generation_mw = {
        u: [pyo.value(model.gen[u, t]) for t in range(H)] for u in unit_names
    }
    pump_mw = {
        u: [pyo.value(model.pump[u, t]) for t in range(H)] for u in unit_names
    }
    reservoir_gwh = {
        u: [pyo.value(model.reservoir[u, t]) / 1000 for t in range(H)] for u in unit_names
    }
    elspot_bid = [pyo.value(model.elspot_bid_se2[t]) for t in range(H)]

    return model, SE2OptimisationResult(
        solve_time_seconds=round(solve_time, 2),
        objective_value_eur=round(obj_val, 2),
        mip_gap=mip_gap,
        status=str(term_cond),
        unit_names=unit_names,
        generation_mw=generation_mw,
        pump_mw=pump_mw,
        reservoir_gwh=reservoir_gwh,
        elspot_bid_mw=elspot_bid,
        timestamps=timestamps,
    )


def se2_result_to_dataframe(result: SE2OptimisationResult) -> pd.DataFrame:
    """Convert SE2OptimisationResult to a tidy per-unit DataFrame."""
    rows = []
    for i, ts in enumerate(result.timestamps):
        row = {"timestamp": ts, "elspot_bid_mw": result.elspot_bid_mw[i]}
        for u in result.unit_names:
            row[f"{u}_gen_mw"] = result.generation_mw[u][i]
            row[f"{u}_pump_mw"] = result.pump_mw[u][i]
            row[f"{u}_reservoir_gwh"] = result.reservoir_gwh[u][i]
        rows.append(row)
    return pd.DataFrame(rows).set_index("timestamp")
