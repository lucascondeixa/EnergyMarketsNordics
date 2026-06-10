"""Ancillary services market module.

Implemented:
  FCR-N      — Frequency Containment Reserve, Normal operation (Fingrid, FI zone)
  FCR-D Up   — Frequency Containment Reserve, Disturbance, upward
  FCR-D Down — Frequency Containment Reserve, Disturbance, downward

Stub (future):
  aFRR, mFRR

FCR-N key facts (Fingrid Nordic spec):
  - Symmetric: equal upward AND downward capacity required every hour
  - Full activation at ±0.1 Hz (linear droop between 49.9–50.1 Hz)
  - Capacity-only payment: EUR/MW/h. No separate energy component.
  - Gate closure: D-1 08:00 CET (before Elspot noon gate)
  - Max 70 MW per single point of failure (one SCADA portfolio)
  - Finnish procurement: ~126 MW total from all providers

FCR-D key facts (Fingrid Nordic spec):
  - Asymmetric: Up and Down are separate products, offered independently.
  - FCR-D Up activates on under-frequency (linear droop 49.5–50.0 Hz).
  - FCR-D Down activates on over-frequency (linear droop 50.0–50.5 Hz).
  - Capacity-only payment: EUR/MW/h, same gate closure as FCR-N.
  - Same 70 MW single-point-of-failure cap applies per BSP.

Model coupling with Elspot:
  For symmetric FCR-N reserve r_n[t] and asymmetric FCR-D reserves r_d_up[t] /
  r_d_down[t] from the same asset, all reserved capacity shares the same
  physical headroom:
    gen[t] + r_n[t] + r_d_up[t]   ≤ p_max   (upward headroom)
    gen[t] - r_n[t] - r_d_down[t] ≥ p_min   (downward headroom)
  Every MW reserved for any AS product narrows the Elspot dispatch range.
"""

from __future__ import annotations

import logging

import pandas as pd
import pyomo.environ as pyo

from src.utils.schema import (
    AncillaryServicesConfig,
    FCRDDownConfig,
    FCRDUpConfig,
    HydroConfig,
    NuclearUnitConfig,
)

log = logging.getLogger(__name__)


def add_ancillary_services(
    model: pyo.ConcreteModel,
    as_cfg: AncillaryServicesConfig,
    hydro_cfg: HydroConfig,
    nuclear_unit_names: list[str],
    nuclear_cfgs: dict[str, NuclearUnitConfig],
    fcr_n_prices: pd.Series | None = None,
    fcr_d_up_prices: pd.Series | None = None,
    fcr_d_down_prices: pd.Series | None = None,
    hydro_blocks: list[tuple[str, HydroConfig]] | None = None,
) -> None:
    """Top-level dispatcher: adds enabled ancillary service blocks to *model*.

    Adds model.as_revenue_expr (total AS revenue per hour) consumed by the objective.

    *hydro_blocks* is a list of (name, HydroConfig) tuples for all hydro assets that
    should participate in ancillary services.  Defaults to [("hydro", hydro_cfg)] for
    backwards compatibility.

    FCR-N must be added before FCR-D so that FCR-D headroom constraints can
    couple with the FCR-N reserve variables already present on the model.
    """
    T = model.T
    active_revenue_names: list[str] = []
    _hydro_blocks = hydro_blocks if hydro_blocks is not None else [("hydro", hydro_cfg)]

    if as_cfg.FCR_N.enabled:
        if fcr_n_prices is None:
            raise ValueError("FCR-N is enabled but fcr_n_prices was not provided")
        add_fcr_n(
            model,
            as_cfg.FCR_N,
            _hydro_blocks,
            nuclear_unit_names,
            nuclear_cfgs,
            fcr_n_prices,
        )
        active_revenue_names.append("fcr_n_revenue_expr")
        log.info("FCR-N enabled: hydro portfolio max=%.0f MW, nuclear max=%.0f MW/unit",
                 as_cfg.FCR_N.max_fcr_n_mw_hydro, as_cfg.FCR_N.max_fcr_n_mw_per_nuclear_unit)

    if as_cfg.FCR_D_UP.enabled:
        if fcr_d_up_prices is None:
            raise ValueError("FCR-D Up is enabled but fcr_d_up_prices was not provided")
        add_fcr_d(
            model,
            as_cfg.FCR_D_UP,
            "up",
            _hydro_blocks,
            nuclear_unit_names,
            nuclear_cfgs,
            fcr_d_up_prices,
        )
        active_revenue_names.append("fcr_d_up_revenue_expr")
        log.info("FCR-D Up enabled: hydro portfolio max=%.0f MW, nuclear max=%.0f MW/unit",
                 as_cfg.FCR_D_UP.max_fcr_d_up_mw_hydro, as_cfg.FCR_D_UP.max_fcr_d_up_mw_per_nuclear_unit)

    if as_cfg.FCR_D_DOWN.enabled:
        if fcr_d_down_prices is None:
            raise ValueError("FCR-D Down is enabled but fcr_d_down_prices was not provided")
        add_fcr_d(
            model,
            as_cfg.FCR_D_DOWN,
            "down",
            _hydro_blocks,
            nuclear_unit_names,
            nuclear_cfgs,
            fcr_d_down_prices,
        )
        active_revenue_names.append("fcr_d_down_revenue_expr")
        log.info("FCR-D Down enabled: hydro portfolio max=%.0f MW, nuclear max=%.0f MW/unit",
                 as_cfg.FCR_D_DOWN.max_fcr_d_down_mw_hydro, as_cfg.FCR_D_DOWN.max_fcr_d_down_mw_per_nuclear_unit)

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
    hydro_blocks: list[tuple[str, HydroConfig]],
    nuclear_unit_names: list[str],
    nuclear_cfgs: dict[str, NuclearUnitConfig],
    fcr_n_prices: pd.Series,
) -> None:
    """Add FCR-N capacity reservation variables and constraints.

    Variables added to model:
      r_fcr_n_{hydro_name}[t]   — FCR-N capacity from each hydro block (MW)
      r_fcr_n_{unit}[t]         — FCR-N capacity from each nuclear unit (MW)

    Constraints:
      Per-asset symmetric headroom (upward + downward)
      Portfolio-wide hydro FCR-N cap (sum across all hydro blocks ≤ max_fcr_n_mw_hydro)

    Expressions added:
      fcr_n_revenue_expr[t]     — EUR/h revenue = price × total reserved capacity
      r_fcr_n_total[t]          — total FCR-N capacity bid (MW, for result extraction)
      r_fcr_n_hydro[t]          — hydro-only FCR-N total (for result extraction)
    """
    T = model.T
    prices = {t: float(fcr_n_prices.iloc[t]) for t in T}
    max_hydro_portfolio_r = cfg.max_fcr_n_mw_hydro

    # -----------------------------------------------------------------------
    # Hydro FCR-N — one variable per block, shared portfolio cap
    # -----------------------------------------------------------------------
    # Per-block Var names use "blk_" prefix to avoid collision with the aggregate Expression.
    hydro_r_names: list[str] = []
    for block_name, hydro_cfg in hydro_blocks:
        turbine_cap = hydro_cfg.turbine.capacity_mw
        min_gen = hydro_cfg.turbine.min_output_mw
        r_name = f"r_fcr_n_blk_{block_name}"
        hydro_r_names.append(r_name)

        r_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
        setattr(model, r_name, r_var)

        def _up(m, t, bn=block_name, cap=turbine_cap, rn=r_name):
            return getattr(m, f"{bn}_gen")[t] + getattr(m, rn)[t] <= cap

        def _down(m, t, bn=block_name, mg=min_gen, rn=r_name):
            return getattr(m, f"{bn}_gen")[t] - getattr(m, rn)[t] >= mg

        setattr(model, f"fcr_n_{block_name}_up",   pyo.Constraint(T, rule=_up))
        setattr(model, f"fcr_n_{block_name}_down",  pyo.Constraint(T, rule=_down))

    # Portfolio-wide hydro cap (Fingrid single-point-of-failure limit across BSP)
    model.fcr_n_hydro_portfolio_cap = pyo.Constraint(
        T,
        rule=lambda m, t: sum(getattr(m, rn)[t] for rn in hydro_r_names)
        <= max_hydro_portfolio_r,
    )

    # Aggregate expression for result extraction: total hydro FCR-N across all blocks
    model.r_fcr_n_hydro = pyo.Expression(
        T, rule=lambda m, t: sum(getattr(m, rn)[t] for rn in hydro_r_names)
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

        def _up(m, t, u=unit, _p_max=p_max):
            return (
                getattr(m, f"{u}_gen")[t] + getattr(m, f"r_fcr_n_{u}")[t]
                <= _p_max * getattr(m, f"{u}_u")[t]
            )

        def _down(m, t, u=unit, _p_min=p_min):
            return (
                getattr(m, f"{u}_gen")[t] - getattr(m, f"r_fcr_n_{u}")[t]
                >= _p_min * getattr(m, f"{u}_u")[t]
            )

        def _cap(m, t, u=unit, _max=max_nuc_r):
            return getattr(m, f"r_fcr_n_{u}")[t] <= _max * getattr(m, f"{u}_u")[t]

        setattr(model, f"fcr_n_{unit}_up",   pyo.Constraint(T, rule=_up))
        setattr(model, f"fcr_n_{unit}_down",  pyo.Constraint(T, rule=_down))
        setattr(model, f"fcr_n_{unit}_cap",   pyo.Constraint(T, rule=_cap))

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


def add_fcr_d(
    model: pyo.ConcreteModel,
    cfg: FCRDUpConfig | FCRDDownConfig,
    direction: str,
    hydro_blocks: list[tuple[str, HydroConfig]],
    nuclear_unit_names: list[str],
    nuclear_cfgs: dict[str, NuclearUnitConfig],
    fcr_d_prices: pd.Series,
) -> None:
    """Add FCR-D (Up or Down) capacity reservation variables and constraints.

    *direction*: "up" or "down".

    Variables added to model:
      r_fcr_d_{direction}_blk_{hydro_name}[t]  — FCR-D capacity from each hydro block (MW)
      r_fcr_d_{direction}_{unit}[t]            — FCR-D capacity from each nuclear unit (MW)

    Headroom constraints couple with any FCR-N reserve already present on the
    model (r_fcr_n_blk_{name} / r_fcr_n_{unit}), since both products draw on
    the same physical up/down headroom. If FCR-N is not enabled, the FCR-D
    constraint reduces to the single-product headroom form.

    Constraints:
      Per-asset one-sided headroom (up: gen + r_n + r_d_up ≤ p_max;
                                     down: gen - r_n - r_d_down ≥ p_min)
      Portfolio-wide hydro FCR-D cap (sum across all hydro blocks ≤ max)

    Expressions added:
      fcr_d_{direction}_revenue_expr[t]  — EUR/h revenue = price × total reserved capacity
      r_fcr_d_{direction}_total[t]       — total FCR-D capacity bid (MW, for result extraction)
      r_fcr_d_{direction}_hydro[t]       — hydro-only FCR-D total (for result extraction)
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    T = model.T
    sfx = direction
    prices = {t: float(fcr_d_prices.iloc[t]) for t in T}

    if direction == "up":
        max_hydro_r = cfg.max_fcr_d_up_mw_hydro
        max_nuc_r = cfg.max_fcr_d_up_mw_per_nuclear_unit
    else:
        max_hydro_r = cfg.max_fcr_d_down_mw_hydro
        max_nuc_r = cfg.max_fcr_d_down_mw_per_nuclear_unit

    # -----------------------------------------------------------------------
    # Hydro FCR-D — one variable per block, shared portfolio cap
    # -----------------------------------------------------------------------
    hydro_r_names: list[str] = []
    for block_name, hydro_cfg in hydro_blocks:
        turbine_cap = hydro_cfg.turbine.capacity_mw
        min_gen = hydro_cfg.turbine.min_output_mw
        r_name = f"r_fcr_d_{sfx}_blk_{block_name}"
        hydro_r_names.append(r_name)

        r_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
        setattr(model, r_name, r_var)

        fcr_n_name = f"r_fcr_n_blk_{block_name}"

        if direction == "up":
            def _headroom(m, t, bn=block_name, cap=turbine_cap, rn=r_name, fn=fcr_n_name):
                fcr_n = getattr(m, fn)[t] if hasattr(m, fn) else 0
                return getattr(m, f"{bn}_gen")[t] + fcr_n + getattr(m, rn)[t] <= cap
        else:
            def _headroom(m, t, bn=block_name, mg=min_gen, rn=r_name, fn=fcr_n_name):
                fcr_n = getattr(m, fn)[t] if hasattr(m, fn) else 0
                return getattr(m, f"{bn}_gen")[t] - fcr_n - getattr(m, rn)[t] >= mg

        setattr(model, f"fcr_d_{sfx}_{block_name}_headroom", pyo.Constraint(T, rule=_headroom))

    # Portfolio-wide hydro cap (Fingrid single-point-of-failure limit across BSP)
    setattr(
        model,
        f"fcr_d_{sfx}_hydro_portfolio_cap",
        pyo.Constraint(
            T,
            rule=lambda m, t: sum(getattr(m, rn)[t] for rn in hydro_r_names) <= max_hydro_r,
        ),
    )

    # Aggregate expression for result extraction: total hydro FCR-D across all blocks
    setattr(
        model,
        f"r_fcr_d_{sfx}_hydro",
        pyo.Expression(T, rule=lambda m, t: sum(getattr(m, rn)[t] for rn in hydro_r_names)),
    )

    # -----------------------------------------------------------------------
    # Nuclear FCR-D (per unit)
    # -----------------------------------------------------------------------
    for unit in nuclear_unit_names:
        cfg_nuc = nuclear_cfgs[unit]
        p_max = cfg_nuc.capacity_mw
        p_min = cfg_nuc.min_output_fraction * p_max

        r_name = f"r_fcr_d_{sfx}_{unit}"
        r_var = pyo.Var(T, domain=pyo.NonNegativeReals, initialize=0)
        setattr(model, r_name, r_var)

        fcr_n_name = f"r_fcr_n_{unit}"

        if direction == "up":
            def _headroom(m, t, u=unit, _p_max=p_max, rn=r_name, fn=fcr_n_name):
                fcr_n = getattr(m, fn)[t] if hasattr(m, fn) else 0
                return (
                    getattr(m, f"{u}_gen")[t] + fcr_n + getattr(m, rn)[t]
                    <= _p_max * getattr(m, f"{u}_u")[t]
                )
        else:
            def _headroom(m, t, u=unit, _p_min=p_min, rn=r_name, fn=fcr_n_name):
                fcr_n = getattr(m, fn)[t] if hasattr(m, fn) else 0
                return (
                    getattr(m, f"{u}_gen")[t] - fcr_n - getattr(m, rn)[t]
                    >= _p_min * getattr(m, f"{u}_u")[t]
                )

        def _cap(m, t, u=unit, _max=max_nuc_r, rn=r_name):
            return getattr(m, rn)[t] <= _max * getattr(m, f"{u}_u")[t]

        setattr(model, f"fcr_d_{sfx}_{unit}_headroom", pyo.Constraint(T, rule=_headroom))
        setattr(model, f"fcr_d_{sfx}_{unit}_cap", pyo.Constraint(T, rule=_cap))

    # -----------------------------------------------------------------------
    # Aggregate expressions
    # -----------------------------------------------------------------------
    def _r_total(m, t):
        nuc_r = sum(getattr(m, f"r_fcr_d_{sfx}_{u}")[t] for u in nuclear_unit_names)
        return getattr(m, f"r_fcr_d_{sfx}_hydro")[t] + nuc_r

    setattr(model, f"r_fcr_d_{sfx}_total", pyo.Expression(T, rule=_r_total))

    def _revenue(m, t):
        return prices[t] * getattr(m, f"r_fcr_d_{sfx}_total")[t]

    setattr(model, f"fcr_d_{sfx}_revenue_expr", pyo.Expression(T, rule=_revenue))
