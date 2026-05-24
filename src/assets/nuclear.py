"""Nuclear unit commitment Pyomo model block.

Models a single nuclear unit with:
- Binary on/off status
- Minimum up/down time constraints
- Ramp rate limits
- Startup and shutdown costs
- Planned outage windows (unit forced off)
"""

from __future__ import annotations

from datetime import datetime

import pyomo.environ as pyo

from src.utils.schema import NuclearUnitConfig


def add_nuclear_block(
    model: pyo.ConcreteModel,
    cfg: NuclearUnitConfig,
    unit_name: str,
    horizon_start: datetime,
) -> None:
    """Add nuclear unit commitment variables and constraints to *model* in-place.

    *horizon_start* is the UTC datetime of time step 0, used to check planned outages.
    Variables are prefixed with *unit_name* to support multiple units.
    """
    T = model.T
    horizon_hours = len(list(T))

    p_max = cfg.capacity_mw
    p_min = cfg.min_output_fraction * p_max
    ramp = cfg.ramp_rate_mw_per_hour
    min_up = cfg.min_up_hours
    min_down = cfg.min_down_hours
    su_cost = cfg.startup_cost_eur
    sd_cost = cfg.shutdown_cost_eur
    vc = cfg.variable_cost_eur_per_mwh

    # Determine which time steps are within a planned outage
    from datetime import timedelta

    outage_steps: set[int] = set()
    for outage in cfg.planned_outages:
        for t in T:
            ts = horizon_start + timedelta(hours=t)
            if outage.start <= ts < outage.end:
                outage_steps.add(t)

    # --- Variables ---
    u = pyo.Var(T, domain=pyo.Binary, initialize=1 if cfg.initial_on else 0)
    su = pyo.Var(T, domain=pyo.Binary, initialize=0)  # startup indicator
    sd = pyo.Var(T, domain=pyo.Binary, initialize=0)  # shutdown indicator
    gen = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=p_max if cfg.initial_on else 0)

    setattr(model, f"{unit_name}_u", u)
    setattr(model, f"{unit_name}_su", su)
    setattr(model, f"{unit_name}_sd", sd)
    setattr(model, f"{unit_name}_gen", gen)

    # --- Planned outage: force unit off ---
    def _outage(m, t):
        if t in outage_steps:
            return getattr(m, f"{unit_name}_u")[t] == 0
        return pyo.Constraint.Skip

    setattr(model, f"{unit_name}_outage", pyo.Constraint(T, rule=_outage))

    # --- Startup / shutdown logic: u[t] - u[t-1] = su[t] - sd[t] ---
    def _su_sd_logic(m, t):
        if t == T.first():
            u_prev = 1 if cfg.initial_on else 0
        else:
            u_prev = getattr(m, f"{unit_name}_u")[T.prev(t)]
        return (
            getattr(m, f"{unit_name}_u")[t] - u_prev
            == getattr(m, f"{unit_name}_su")[t] - getattr(m, f"{unit_name}_sd")[t]
        )

    setattr(model, f"{unit_name}_su_sd_logic", pyo.Constraint(T, rule=_su_sd_logic))

    # Can't start up and shut down in the same period
    def _no_simultaneous(m, t):
        return (
            getattr(m, f"{unit_name}_su")[t] + getattr(m, f"{unit_name}_sd")[t] <= 1
        )

    setattr(model, f"{unit_name}_no_simul", pyo.Constraint(T, rule=_no_simultaneous))

    # --- Minimum up time ---
    def _min_up(m, t):
        # If the unit started at t, it must stay on until t + min_up - 1
        if t == T.first():
            return pyo.Constraint.Skip
        su_var = getattr(m, f"{unit_name}_su")[t]
        end = min(t + min_up - 1, horizon_hours - 1)
        lhs = sum(
            getattr(m, f"{unit_name}_u")[k] for k in range(t, end + 1)
        )
        return lhs >= (end - t + 1) * su_var

    setattr(model, f"{unit_name}_min_up", pyo.Constraint(T, rule=_min_up))

    # --- Minimum down time ---
    def _min_down(m, t):
        if t == T.first():
            return pyo.Constraint.Skip
        sd_var = getattr(m, f"{unit_name}_sd")[t]
        end = min(t + min_down - 1, horizon_hours - 1)
        lhs = sum(
            (1 - getattr(m, f"{unit_name}_u")[k]) for k in range(t, end + 1)
        )
        return lhs >= (end - t + 1) * sd_var

    setattr(model, f"{unit_name}_min_down", pyo.Constraint(T, rule=_min_down))

    # --- Generation limits (only when on) ---
    def _gen_max(m, t):
        return getattr(m, f"{unit_name}_gen")[t] <= p_max * getattr(m, f"{unit_name}_u")[t]

    def _gen_min(m, t):
        return getattr(m, f"{unit_name}_gen")[t] >= p_min * getattr(m, f"{unit_name}_u")[t]

    setattr(model, f"{unit_name}_gen_max", pyo.Constraint(T, rule=_gen_max))
    setattr(model, f"{unit_name}_gen_min", pyo.Constraint(T, rule=_gen_min))

    # --- Ramp rate constraints ---
    def _ramp_up(m, t):
        if t == T.first():
            return pyo.Constraint.Skip
        return (
            getattr(m, f"{unit_name}_gen")[t] - getattr(m, f"{unit_name}_gen")[T.prev(t)]
            <= ramp + p_max * getattr(m, f"{unit_name}_su")[t]
        )

    def _ramp_down(m, t):
        if t == T.first():
            return pyo.Constraint.Skip
        return (
            getattr(m, f"{unit_name}_gen")[T.prev(t)] - getattr(m, f"{unit_name}_gen")[t]
            <= ramp + p_max * getattr(m, f"{unit_name}_sd")[t]
        )

    setattr(model, f"{unit_name}_ramp_up", pyo.Constraint(T, rule=_ramp_up))
    setattr(model, f"{unit_name}_ramp_down", pyo.Constraint(T, rule=_ramp_down))

    # --- Cost contributions (stored as Expressions for use in objective) ---
    # startup + shutdown + variable costs
    def _unit_cost(m, t):
        return (
            su_cost * getattr(m, f"{unit_name}_su")[t]
            + sd_cost * getattr(m, f"{unit_name}_sd")[t]
            + vc * getattr(m, f"{unit_name}_gen")[t]
        )

    setattr(model, f"{unit_name}_cost_expr", pyo.Expression(T, rule=_unit_cost))
