"""Pump-storage hydro Pyomo model block.

Only called when HydroConfig.pump_storage_enabled = True.
Adds pump variables, mutual exclusion binaries, and reservoir dynamics
(including the pump water contribution). The turbine variable model.hydro_gen
must already exist (created by hydro.py).

Current applicability: NOT active for Fortum's Finnish portfolio (no pump storage).
Preserved for future use when Kemijoki Oy's Ailangantunturi project (~550 MW)
becomes operational (earliest ~2032) or if Fortum acquires pump assets elsewhere.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import HydroConfig


def add_pump_hydro_block(
    model: pyo.ConcreteModel,
    cfg: HydroConfig,
    inflow_series: pd.Series,
) -> None:
    """Add pump-storage variables, mutual exclusion, and reservoir dynamics.

    Assumes model.hydro_gen already exists (from add_hydro_block).
    *inflow_series* indexed 0..H-1 with values in GWh/h (water-equivalent).
    """
    T = model.T
    turbine = cfg.turbine
    pump = cfg.pump
    reservoir = cfg.reservoir

    v_max = reservoir.capacity_gwh * 1000
    v_min = reservoir.min_level_gwh * 1000
    v_0 = reservoir.initial_level_gwh * 1000
    water_value = reservoir.terminal_water_value_eur_per_mwh

    inflow_mwh = {t: inflow_series.iloc[t] * 1000 for t in T}

    # --- Variables ---
    model.pump_cons = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
    model.reservoir = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=v_0)
    model.hydro_mode = pyo.Var(T, domain=pyo.Binary, initialize=1)  # 1 = turbine mode
    model.terminal_water_value = pyo.Param(initialize=water_value)

    # --- Pump capacity ---
    model.pump_max = pyo.Constraint(T, rule=lambda m, t: m.pump_cons[t] <= pump.capacity_mw)
    model.pump_min = pyo.Constraint(
        T, rule=lambda m, t: m.pump_cons[t] >= pump.min_input_mw * (1 - m.hydro_mode[t])
    )

    # --- Mutual exclusion (big-M) ---
    model.gen_when_turbine = pyo.Constraint(
        T, rule=lambda m, t: m.hydro_gen[t] <= turbine.capacity_mw * m.hydro_mode[t]
    )
    model.pump_when_not_turbine = pyo.Constraint(
        T, rule=lambda m, t: m.pump_cons[t] <= pump.capacity_mw * (1 - m.hydro_mode[t])
    )

    # --- Reservoir dynamics (turbine withdrawal + pump addition + natural inflow) ---
    def _dynamics(m, t):
        v_prev = v_0 if t == T.first() else m.reservoir[T.prev(t)]
        water_withdrawn = m.hydro_gen[t] / turbine.efficiency
        water_added = m.pump_cons[t] * pump.efficiency
        return m.reservoir[t] == v_prev + inflow_mwh[t] + water_added - water_withdrawn

    model.reservoir_dynamics = pyo.Constraint(T, rule=_dynamics)
    model.reservoir_max = pyo.Constraint(T, rule=lambda m, t: m.reservoir[t] <= v_max)
    model.reservoir_min = pyo.Constraint(T, rule=lambda m, t: m.reservoir[t] >= v_min)
