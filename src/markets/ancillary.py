"""Ancillary services market module.

Implemented:
  FCR-N — Frequency Containment Reserve, Normal operation (Fingrid, FI zone)

Stub (future):
  FCR-D up/down, aFRR, mFRR

FCR-N key facts (Fingrid Nordic spec):
  - Symmetric: equal upward AND downward capacity required every hour
  - Full activation at ±0.1 Hz (linear droop between 49.9–50.1 Hz)
  - Capacity-only payment: EUR/MW/h. No separate energy component.
  - Gate closure: D-1 08:00 CET (before Elspot noon gate)
  - Max 70 MW per single point of failure (one SCADA portfolio)
  - Finnish procurement: ~126 MW total from all providers

Model coupling with Elspot:
  For symmetric FCR-N reserve r[t] from an asset:
    gen[t] + r[t] ≤ p_max          (upward headroom: room to increase output)
    gen[t] - r[t] ≥ p_min          (downward headroom: room to decrease output)
  The dispatch gen[t] therefore must stay in the band [p_min + r, p_max - r].
  Every MW reserved for FCR-N narrows the Elspot dispatch range.
"""

from __future__ import annotations

import logging

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import AncillaryServicesConfig, HydroConfig, NuclearUnitConfig

log = logging.getLogger(__name__)


def add_ancillary_services(
    model: pyo.ConcreteModel,
    as_cfg: AncillaryServicesConfig,
    hydro_cfg: HydroConfig,
    nuclear_unit_names: list[str],
    nuclear_cfgs: dict[str, NuclearUnitConfig],
    fcr_n_prices: pd.Series | None = None,
) -> None:
    """Top-level dispatcher: adds enabled ancillary service blocks to *model*.

    Adds model.as_revenue_expr (total AS revenue per hour) consumed by the objective.
    """
    T = model.T
    active_revenue_names: list[str] = []

    if as_cfg.FCR_N.enabled:
        if fcr_n_prices is None:
            raise ValueError("FCR-N is enabled but fcr_n_prices was not provided")
        add_fcr_n(
            model,
            as_cfg.FCR_N,
            hydro_cfg,
            nuclear_unit_names,
            nuclear_cfgs,
            fcr_n_prices,
        )
        active_revenue_names.append("fcr_n_revenue_expr")
        log.info("FCR-N enabled: hydro max=%.0f MW, nuclear max=%.0f MW/unit",
                 as_cfg.FCR_N.max_fcr_n_mw_hydro, as_cfg.FCR_N.max_fcr_n_mw_per_nuclear_unit)

    # Aggregate ancillary revenue (sum of all active services)
    if active_revenue_names:
        def _total_as(m, t):
            return sum(getattr(m, name)[t] for name in active_revenue_names)
        model.as_revenue_expr = pyo.Expression(T, rule=_total_as)
    else:
        model.as_revenue_expr = pyo.Expression(T, rule=lambda m, t: 0.0)


def add_fcr_n(
    model: pyo.ConcreteModel,
    cfg,  # FCRNConfig
    hydro_cfg: HydroConfig,
    nuclear_unit_names: list[str],
    nuclear_cfgs: dict[str, NuclearUnitConfig],
    fcr_n_prices: pd.Series,
) -> None:
    """Add FCR-N capacity reservation variables and constraints.

    Variables added to model:
      r_fcr_n_hydro[t]          — FCR-N capacity from hydro aggregate (MW)
      r_fcr_n_{unit}[t]         — FCR-N capacity from each nuclear unit (MW)

    Expressions added:
      fcr_n_revenue_expr[t]     — EUR/h revenue = price × total reserved capacity
      r_fcr_n_total[t]          — total FCR-N capacity bid (MW, for result extraction)
    """
    T = model.T
    prices = {t: float(fcr_n_prices.iloc[t]) for t in T}

    # -----------------------------------------------------------------------
    # Hydro FCR-N
    # -----------------------------------------------------------------------
    turbine_cap = hydro_cfg.turbine.capacity_mw
    min_gen = hydro_cfg.turbine.min_output_mw
    max_hydro_r = cfg.max_fcr_n_mw_hydro

    model.r_fcr_n_hydro = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)

    # Symmetric headroom: dispatch must stay in [min+r, cap-r]
    model.fcr_n_hydro_up = pyo.Constraint(
        T, rule=lambda m, t: m.hydro_gen[t] + m.r_fcr_n_hydro[t] <= turbine_cap
    )
    model.fcr_n_hydro_down = pyo.Constraint(
        T, rule=lambda m, t: m.hydro_gen[t] - m.r_fcr_n_hydro[t] >= min_gen
    )
    model.fcr_n_hydro_max = pyo.Constraint(
        T, rule=lambda m, t: m.r_fcr_n_hydro[t] <= max_hydro_r
    )

    # -----------------------------------------------------------------------
    # Nuclear FCR-N (per unit)
    # -----------------------------------------------------------------------
    for unit in nuclear_unit_names:
        cfg_nuc = nuclear_cfgs[unit]
        p_max = cfg_nuc.capacity_mw
        p_min = cfg_nuc.min_output_fraction * p_max
        max_nuc_r = cfg.max_fcr_n_mw_per_nuclear_unit

        r_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
        setattr(model, f"r_fcr_n_{unit}", r_var)

        # Upward: gen + r ≤ p_max × u (can increase by r when online)
        def _up(m, t, u=unit, _p_max=p_max):
            return (
                getattr(m, f"{u}_gen")[t] + getattr(m, f"r_fcr_n_{u}")[t]
                <= _p_max * getattr(m, f"{u}_u")[t]
            )

        # Downward: gen - r ≥ p_min × u (can decrease by r when online, stay above min load)
        def _down(m, t, u=unit, _p_min=p_min):
            return (
                getattr(m, f"{u}_gen")[t] - getattr(m, f"r_fcr_n_{u}")[t]
                >= _p_min * getattr(m, f"{u}_u")[t]
            )

        # Cap: can't offer more than pre-qualified capacity, and only when online
        def _cap(m, t, u=unit, _max=max_nuc_r):
            return getattr(m, f"r_fcr_n_{u}")[t] <= _max * getattr(m, f"{u}_u")[t]

        setattr(model, f"fcr_n_{unit}_up", pyo.Constraint(T, rule=_up))
        setattr(model, f"fcr_n_{unit}_down", pyo.Constraint(T, rule=_down))
        setattr(model, f"fcr_n_{unit}_cap", pyo.Constraint(T, rule=_cap))

    # -----------------------------------------------------------------------
    # Aggregate expressions
    # -----------------------------------------------------------------------
    def _r_total(m, t):
        nuc_r = sum(getattr(m, f"r_fcr_n_{u}")[t] for u in nuclear_unit_names)
        return m.r_fcr_n_hydro[t] + nuc_r

    model.r_fcr_n_total = pyo.Expression(T, rule=_r_total)

    def _revenue(m, t):
        return prices[t] * m.r_fcr_n_total[t]

    model.fcr_n_revenue_expr = pyo.Expression(T, rule=_revenue)
