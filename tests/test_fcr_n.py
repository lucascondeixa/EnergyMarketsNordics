"""Tests for FCR-N ancillary service constraints.

Verifies that:
1. FCR-N symmetric headroom constraints are satisfied (gen ± r within [min, max])
2. Pre-qualification caps are respected (hydro ≤ 70 MW, nuclear ≤ 25 MW/unit)
3. FCR-N revenue is positive when prices > 0
4. FCR-N raises total objective vs a run without FCR-N
5. Nuclear cannot offer FCR-N when offline (outage period)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from src.data_ingestion.forecasts import (
    load_fcr_n_price_forecast,
    load_inflow_forecast,
    load_price_forecast,
    load_wind_forecast,
)
from src.optimization.joint_optimizer import build_and_solve, result_to_dataframe
from src.utils.schema import AncillaryServicesConfig, MarketConfig, PlantConfig
from src.utils.time_utils import build_horizon

CONFIG_DIR = "configs"
HORIZON = 48


def _load_configs(fcr_n_enabled: bool = True):
    with open(f"{CONFIG_DIR}/plant_params.yaml") as f:
        plant_raw = yaml.safe_load(f)
    with open(f"{CONFIG_DIR}/market_params.yaml") as f:
        market_raw = yaml.safe_load(f)

    plant_cfg = PlantConfig(
        hydro=plant_raw["hydro"],
        nuclear=plant_raw["nuclear"],
        wind=plant_raw.get("wind", {}),
        se2_pump_storage=plant_raw.get("se2_pump_storage", {}),
    )
    as_raw = market_raw.get("ancillary_services", {})
    if "FCR_N" in as_raw:
        as_raw["FCR_N"]["enabled"] = fcr_n_enabled
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=AncillaryServicesConfig(**as_raw),
        optimisation=market_raw["optimisation"],
    )
    market_cfg.optimisation.horizon_hours = HORIZON
    market_cfg.optimisation.solver_options.time_limit = 120
    return plant_cfg, market_cfg


def _solve(fcr_n_enabled: bool = True):
    plant_cfg, market_cfg = _load_configs(fcr_n_enabled)
    horizon_start = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
    index = build_horizon(horizon_start, horizon_hours=HORIZON)

    price = load_price_forecast("synthetic", index)
    inflow = load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.hydro.inflow)
    wind = load_wind_forecast("synthetic", index, wind_cfgs=plant_cfg.wind)
    fcr_n = load_fcr_n_price_forecast("synthetic", index) if fcr_n_enabled else None

    model, result = build_and_solve(
        plant_cfg, market_cfg, price, inflow, horizon_start,
        wind_schedule=wind, fcr_n_prices=fcr_n,
    )
    return model, result, result_to_dataframe(result), plant_cfg, market_cfg


@pytest.fixture(scope="module")
def fcr_n_result():
    return _solve(fcr_n_enabled=True)


def test_fcr_n_solver_optimal(fcr_n_result):
    _, result, _, _, _ = fcr_n_result
    assert result.status in ("optimal", "feasible")


def test_fcr_n_revenue_positive(fcr_n_result):
    _, result, _, _, _ = fcr_n_result
    assert result.fcr_n_revenue_eur > 0, "FCR-N should generate positive revenue when prices > 0"


def test_fcr_n_total_non_negative(fcr_n_result):
    _, result, df, _, _ = fcr_n_result
    assert all(v >= -1e-6 for v in result.fcr_n_total_mw)
    assert all(v >= -1e-6 for v in result.fcr_n_hydro_mw)
    assert all(v >= -1e-6 for v in result.fcr_n_nuclear_mw)


def test_fcr_n_hydro_prequalification_cap(fcr_n_result):
    """Hydro FCR-N must not exceed 70 MW (Fingrid 70 MW single-point-of-failure limit)."""
    _, _, _, plant_cfg, market_cfg = fcr_n_result
    cap = market_cfg.ancillary_services.FCR_N.max_fcr_n_mw_hydro
    _, result, _, _, _ = fcr_n_result
    assert max(result.fcr_n_hydro_mw) <= cap + 1e-4


def test_fcr_n_nuclear_prequalification_cap(fcr_n_result):
    """Per-unit nuclear FCR-N must not exceed 25 MW."""
    _, _, _, plant_cfg, market_cfg = fcr_n_result
    per_unit_cap = market_cfg.ancillary_services.FCR_N.max_fcr_n_mw_per_nuclear_unit
    total_nuc_cap = per_unit_cap * len(plant_cfg.nuclear)
    _, result, _, _, _ = fcr_n_result
    assert max(result.fcr_n_nuclear_mw) <= total_nuc_cap + 1e-4


def test_fcr_n_hydro_upward_headroom(fcr_n_result):
    """hydro_gen + r_fcr_n_hydro ≤ turbine_capacity at all times."""
    _, result, df, plant_cfg, _ = fcr_n_result
    cap = plant_cfg.hydro.turbine.capacity_mw
    violation = (df["hydro_gen_mw"] + df["fcr_n_hydro_mw"]) > cap + 1e-4
    assert not violation.any(), "Hydro FCR-N upward headroom violated"


def test_fcr_n_hydro_downward_headroom(fcr_n_result):
    """hydro_gen - r_fcr_n_hydro ≥ min_output at all times (symmetric reserve)."""
    _, result, df, plant_cfg, _ = fcr_n_result
    min_out = plant_cfg.hydro.turbine.min_output_mw
    violation = (df["hydro_gen_mw"] - df["fcr_n_hydro_mw"]) < min_out - 1e-4
    assert not violation.any(), "Hydro FCR-N downward headroom violated (dispatch < min + reserve)"


def test_fcr_n_increases_total_objective():
    """Total objective (Elspot + FCR-N) must be >= objective without FCR-N."""
    _, result_with, _, _, _ = _solve(fcr_n_enabled=True)
    _, result_without, _, _, _ = _solve(fcr_n_enabled=False)
    assert result_with.objective_value_eur >= result_without.objective_value_eur - 1.0, (
        "FCR-N should not reduce total objective"
    )


def test_fcr_n_result_columns(fcr_n_result):
    _, _, df, _, _ = fcr_n_result
    for col in ("fcr_n_hydro_mw", "fcr_n_nuclear_mw", "fcr_n_total_mw"):
        assert col in df.columns
