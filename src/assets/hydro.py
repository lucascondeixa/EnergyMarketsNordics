"""Hydro turbine and reservoir Pyomo model block.

When pump_storage_enabled=False (Fortum's current Finnish portfolio):
  - Adds turbine variables, ramp constraints, reservoir dynamics, and
    a zero-valued pump_cons Param so downstream expressions stay consistent.

When pump_storage_enabled=True:
  - Adds only turbine variables and ramp constraints.
  - Reservoir dynamics (including pump contribution) are handled in pump_hydro.py.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import HydroConfig


def add_hydro_block(
    model: pyo.ConcreteModel,
    cfg: HydroConfig,
    inflow_series: pd.Series,
) -> None:
    """Add hydro turbine (and optionally reservoir) blocks to *model* in-place.

    *inflow_series* must be indexed 0..H-1 with values in GWh/h (water-equivalent).
    """
    T = model.T
    turbine = cfg.turbine

    # --- Turbine variables ---
    model.hydro_gen = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)

    def _max_output(m, t):
        return m.hydro_gen[t] <= turbine.capacity_mw

    def _min_output(m, t):
        return m.hydro_gen[t] >= turbine.min_output_mw

    def _ramp_up(m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.hydro_gen[t] - m.hydro_gen[T.prev(t)] <= turbine.ramp_rate_mw_per_hour

    def _ramp_down(m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.hydro_gen[T.prev(t)] - m.hydro_gen[t] <= turbine.ramp_rate_mw_per_hour

    model.hydro_max_output = pyo.Constraint(T, rule=_max_output)
    model.hydro_min_output = pyo.Constraint(T, rule=_min_output)
    model.hydro_ramp_up = pyo.Constraint(T, rule=_ramp_up)
    model.hydro_ramp_down = pyo.Constraint(T, rule=_ramp_down)

    if not cfg.pump_storage_enabled:
        _add_reservoir_no_pump(model, cfg, inflow_series)


def _add_reservoir_no_pump(
    model: pyo.ConcreteModel,
    cfg: HydroConfig,
    inflow_series: pd.Series,
) -> None:
    """Add reservoir dynamics for a turbine-only system (no pump storage).

    Also adds model.pump_cons as a zero Param so elspot.py net-position
    balance doesn't need a special case.
    """
    T = model.T
    turbine = cfg.turbine
    reservoir = cfg.reservoir

    v_max = reservoir.capacity_gwh * 1000   # GWh → MWh
    v_min = reservoir.min_level_gwh * 1000
    v_0 = reservoir.initial_level_gwh * 1000
    water_value = reservoir.terminal_water_value_eur_per_mwh

    inflow_mwh = {t: inflow_series.iloc[t] * 1000 for t in T}  # GWh/h → MWh/h

    model.reservoir = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=v_0)
    model.pump_cons = pyo.Param(T, initialize=0)  # no pump — placeholder for balance equations
    model.terminal_water_value = pyo.Param(initialize=water_value)

    def _dynamics(m, t):
        v_prev = v_0 if t == T.first() else m.reservoir[T.prev(t)]
        water_withdrawn = m.hydro_gen[t] / turbine.efficiency
        return m.reservoir[t] == v_prev + inflow_mwh[t] - water_withdrawn

    def _res_max(m, t):
        return m.reservoir[t] <= v_max

    def _res_min(m, t):
        return m.reservoir[t] >= v_min

    model.reservoir_dynamics = pyo.Constraint(T, rule=_dynamics)
    model.reservoir_max = pyo.Constraint(T, rule=_res_max)
    model.reservoir_min = pyo.Constraint(T, rule=_res_min)
