"""Hydro turbine and reservoir Pyomo model block.

When pump_storage_enabled=False (Fortum's current Finnish portfolio):
  - Adds turbine variables, ramp constraints, reservoir dynamics, and
    a zero-valued {name}_pump_cons Param so downstream expressions stay consistent.

When pump_storage_enabled=True:
  - Adds only turbine variables and ramp constraints.
  - Reservoir dynamics (including pump contribution) are handled in pump_hydro.py.

The *name* parameter allows multiple hydro blocks on the same model (e.g. "hydro"
for Vuoksi/Oulujoki and "kemijoki" for the Kemijoki Oy portfolio).  All Pyomo
attributes are prefixed with *name* to avoid collisions.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import HydroConfig


def add_hydro_block(
    model: pyo.ConcreteModel,
    cfg: HydroConfig,
    inflow_series: pd.Series,
    name: str = "hydro",
) -> None:
    """Add a named hydro turbine (and optionally reservoir) block to *model* in-place.

    *inflow_series* must be indexed 0..H-1 with values in GWh/h (water-equivalent).
    *name* prefixes every Pyomo attribute so multiple blocks can coexist.
    """
    T = model.T
    turbine = cfg.turbine

    gen_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
    setattr(model, f"{name}_gen", gen_var)

    def _max_output(m, t):
        return getattr(m, f"{name}_gen")[t] <= turbine.capacity_mw

    def _min_output(m, t):
        return getattr(m, f"{name}_gen")[t] >= turbine.min_output_mw

    def _ramp_up(m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return (
            getattr(m, f"{name}_gen")[t] - getattr(m, f"{name}_gen")[T.prev(t)]
            <= turbine.ramp_rate_mw_per_hour
        )

    def _ramp_down(m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return (
            getattr(m, f"{name}_gen")[T.prev(t)] - getattr(m, f"{name}_gen")[t]
            <= turbine.ramp_rate_mw_per_hour
        )

    setattr(model, f"{name}_max_output", pyo.Constraint(T, rule=_max_output))
    setattr(model, f"{name}_min_output", pyo.Constraint(T, rule=_min_output))
    setattr(model, f"{name}_ramp_up",    pyo.Constraint(T, rule=_ramp_up))
    setattr(model, f"{name}_ramp_down",  pyo.Constraint(T, rule=_ramp_down))

    if not cfg.pump_storage_enabled:
        _add_reservoir_no_pump(model, cfg, inflow_series, name)


def _add_reservoir_no_pump(
    model: pyo.ConcreteModel,
    cfg: HydroConfig,
    inflow_series: pd.Series,
    name: str,
) -> None:
    """Add reservoir dynamics for a turbine-only system (no pump storage).

    Adds {name}_pump_cons as a zero Param so elspot.py net-position
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

    res_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=v_0)
    setattr(model, f"{name}_reservoir", res_var)
    setattr(model, f"{name}_pump_cons", pyo.Param(T, initialize=0))
    setattr(model, f"{name}_terminal_water_value", pyo.Param(initialize=water_value))

    def _dynamics(m, t):
        v_prev = v_0 if t == T.first() else getattr(m, f"{name}_reservoir")[T.prev(t)]
        water_withdrawn = getattr(m, f"{name}_gen")[t] / turbine.efficiency
        return getattr(m, f"{name}_reservoir")[t] == v_prev + inflow_mwh[t] - water_withdrawn

    def _res_max(m, t):
        return getattr(m, f"{name}_reservoir")[t] <= v_max

    def _res_min(m, t):
        return getattr(m, f"{name}_reservoir")[t] >= v_min

    setattr(model, f"{name}_reservoir_dynamics", pyo.Constraint(T, rule=_dynamics))
    setattr(model, f"{name}_reservoir_max",      pyo.Constraint(T, rule=_res_max))
    setattr(model, f"{name}_reservoir_min",      pyo.Constraint(T, rule=_res_min))
