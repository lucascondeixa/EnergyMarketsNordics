"""Tests for FCR-D Up/Down ancillary service constraints.

Verifies that:
1. FCR-D Up/Down headroom constraints are satisfied (one-sided, asymmetric)
2. Combined FCR-N + FCR-D headroom is respected when both are enabled
3. Pre-qualification caps are respected (hydro <= 70 MW, nuclear <= 50 MW/unit)
4. FCR-D revenue is positive when prices > 0 and enabled
5. FCR-D raises total objective vs a run without FCR-D
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from src.data_ingestion.forecasts import (
    load_fcr_d_price_forecast,
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


def _load_configs(fcr_n_enabled: bool, fcr_d_up_enabled: bool, fcr_d_down_enabled: bool):
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
    as_raw.setdefault("FCR_N", {})["enabled"] = fcr_n_enabled
    as_raw.setdefault("FCR_D_UP", {})["enabled"] = fcr_d_up_enabled
    as_raw.setdefault("FCR_D_DOWN", {})["enabled"] = fcr_d_down_enabled
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=AncillaryServicesConfig(**as_raw),
        optimisation=market_raw["optimisation"],
    )
    market_cfg.optimisation.horizon_hours = HORIZON
    market_cfg.optimisation.solver_options.time_limit = 120
    return plant_cfg, market_cfg


def _solve(fcr_n_enabled: bool = False, fcr_d_up_enabled: bool = True, fcr_d_down_enabled: bool = True):
    plant_cfg, market_cfg = _load_configs(fcr_n_enabled, fcr_d_up_enabled, fcr_d_down_enabled)
    horizon_start = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
    index = build_horizon(horizon_start, horizon_hours=HORIZON)

    price = load_price_forecast("synthetic", index)
    inflow = load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.hydro.inflow)
    wind = load_wind_forecast("synthetic", index, wind_cfgs=plant_cfg.wind)
    fcr_n = load_fcr_n_price_forecast("synthetic", index) if fcr_n_enabled else None
    fcr_d_up = load_fcr_d_price_forecast("up", "synthetic", index) if fcr_d_up_enabled else None
    fcr_d_down = load_fcr_d_price_forecast("down", "synthetic", index) if fcr_d_down_enabled else None

    model, result = build_and_solve(
        plant_cfg, market_cfg, price, inflow, horizon_start,
        wind_schedule=wind, fcr_n_prices=fcr_n,
        fcr_d_up_prices=fcr_d_up, fcr_d_down_prices=fcr_d_down,
    )
    return model, result, result_to_dataframe(result), plant_cfg, market_cfg


@pytest.fixture(scope="module")
def fcr_d_result():
    return _solve(fcr_n_enabled=False, fcr_d_up_enabled=True, fcr_d_down_enabled=True)


@pytest.fixture(scope="module")
def fcr_n_and_d_result():
    return _solve(fcr_n_enabled=True, fcr_d_up_enabled=True, fcr_d_down_enabled=True)


def test_fcr_d_solver_optimal(fcr_d_result):
    _, result, _, _, _ = fcr_d_result
    assert result.status in ("optimal", "feasible")


def test_fcr_d_up_revenue_positive(fcr_d_result):
    _, result, _, _, _ = fcr_d_result
    assert result.fcr_d_up_revenue_eur > 0, "FCR-D Up should generate positive revenue when prices > 0"


def test_fcr_d_down_revenue_positive(fcr_d_result):
    _, result, _, _, _ = fcr_d_result
    assert result.fcr_d_down_revenue_eur > 0, "FCR-D Down should generate positive revenue when prices > 0"


def test_fcr_d_totals_non_negative(fcr_d_result):
    _, result, _, _, _ = fcr_d_result
    assert all(v >= -1e-6 for v in result.fcr_d_up_total_mw)
    assert all(v >= -1e-6 for v in result.fcr_d_up_hydro_mw)
    assert all(v >= -1e-6 for v in result.fcr_d_down_total_mw)
    assert all(v >= -1e-6 for v in result.fcr_d_down_hydro_mw)


def test_fcr_d_up_hydro_prequalification_cap(fcr_d_result):
    """Hydro FCR-D Up must not exceed the configured portfolio cap."""
    _, result, _, _, market_cfg = fcr_d_result
    cap = market_cfg.ancillary_services.FCR_D_UP.max_fcr_d_up_mw_hydro
    assert max(result.fcr_d_up_hydro_mw) <= cap + 1e-4


def test_fcr_d_down_hydro_prequalification_cap(fcr_d_result):
    """Hydro FCR-D Down must not exceed the configured portfolio cap."""
    _, result, _, _, market_cfg = fcr_d_result
    cap = market_cfg.ancillary_services.FCR_D_DOWN.max_fcr_d_down_mw_hydro
    assert max(result.fcr_d_down_hydro_mw) <= cap + 1e-4


def test_fcr_d_up_nuclear_prequalification_cap(fcr_d_result):
    """Aggregate nuclear FCR-D Up must not exceed per-unit cap x number of units."""
    _, result, _, plant_cfg, market_cfg = fcr_d_result
    per_unit_cap = market_cfg.ancillary_services.FCR_D_UP.max_fcr_d_up_mw_per_nuclear_unit
    total_cap = per_unit_cap * len(plant_cfg.nuclear)
    assert max(result.fcr_d_up_nuclear_mw) <= total_cap + 1e-4


def test_fcr_d_up_hydro_headroom(fcr_d_result):
    """hydro_gen + r_fcr_d_up_hydro <= turbine_capacity at all times."""
    _, _, df, plant_cfg, _ = fcr_d_result
    cap = plant_cfg.hydro.turbine.capacity_mw
    violation = (df["hydro_gen_mw"] + df["fcr_d_up_hydro_mw"]) > cap + 1e-4
    assert not violation.any(), "Hydro FCR-D Up headroom violated"


def test_fcr_d_down_hydro_headroom(fcr_d_result):
    """hydro_gen - r_fcr_d_down_hydro >= min_output at all times."""
    _, _, df, plant_cfg, _ = fcr_d_result
    min_out = plant_cfg.hydro.turbine.min_output_mw
    violation = (df["hydro_gen_mw"] - df["fcr_d_down_hydro_mw"]) < min_out - 1e-4
    assert not violation.any(), "Hydro FCR-D Down headroom violated"


def test_fcr_n_and_fcr_d_combined_headroom(fcr_n_and_d_result):
    """When FCR-N and FCR-D Up are both enabled, combined upward reserve must
    fit within turbine headroom: gen + fcr_n + fcr_d_up <= capacity."""
    _, _, df, plant_cfg, _ = fcr_n_and_d_result
    cap = plant_cfg.hydro.turbine.capacity_mw
    combined_up = df["hydro_gen_mw"] + df["fcr_n_hydro_mw"] + df["fcr_d_up_hydro_mw"]
    assert not (combined_up > cap + 1e-4).any(), "Combined FCR-N + FCR-D Up headroom violated"

    min_out = plant_cfg.hydro.turbine.min_output_mw
    combined_down = df["hydro_gen_mw"] - df["fcr_n_hydro_mw"] - df["fcr_d_down_hydro_mw"]
    assert not (combined_down < min_out - 1e-4).any(), "Combined FCR-N + FCR-D Down headroom violated"


def test_fcr_d_increases_total_objective():
    """Total objective with FCR-D enabled must be >= objective without it."""
    _, result_with, _, _, _ = _solve(fcr_n_enabled=False, fcr_d_up_enabled=True, fcr_d_down_enabled=True)
    _, result_without, _, _, _ = _solve(fcr_n_enabled=False, fcr_d_up_enabled=False, fcr_d_down_enabled=False)
    assert result_with.objective_value_eur >= result_without.objective_value_eur - 1.0, (
        "FCR-D should not reduce total objective"
    )


def test_fcr_d_disabled_gives_zero_results():
    _, result, _, _, _ = _solve(fcr_n_enabled=False, fcr_d_up_enabled=False, fcr_d_down_enabled=False)
    assert result.fcr_d_up_revenue_eur == 0.0
    assert result.fcr_d_down_revenue_eur == 0.0
    assert all(v == 0.0 for v in result.fcr_d_up_total_mw)
    assert all(v == 0.0 for v in result.fcr_d_down_total_mw)


def test_fcr_d_result_columns(fcr_d_result):
    _, _, df, _, _ = fcr_d_result
    for col in (
        "fcr_d_up_hydro_mw", "fcr_d_up_nuclear_mw", "fcr_d_up_total_mw",
        "fcr_d_down_hydro_mw", "fcr_d_down_nuclear_mw", "fcr_d_down_total_mw",
    ):
        assert col in df.columns
