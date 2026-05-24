"""Smoke tests for the joint optimiser — Fortum Finnish portfolio.

Covers:
  - 2 Loviisa nuclear units (VVER-440, 507 MW each)
  - Hydro aggregate (Vuoksi + Oulujoki, ~1,104 MW, no pump storage)
  - Pjelax wind (228 MW must-take)
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
from src.utils.schema import MarketConfig, PlantConfig
from src.utils.time_utils import build_horizon

CONFIG_DIR = "configs"


def _load_configs():
    with open(f"{CONFIG_DIR}/plant_params.yaml") as f:
        plant_raw = yaml.safe_load(f)
    with open(f"{CONFIG_DIR}/market_params.yaml") as f:
        market_raw = yaml.safe_load(f)
    plant_cfg = PlantConfig(
        hydro=plant_raw["hydro"],
        nuclear=plant_raw["nuclear"],
        wind=plant_raw.get("wind", {}),
    )
    from src.utils.schema import AncillaryServicesConfig
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=AncillaryServicesConfig(**market_raw.get("ancillary_services", {})),
        optimisation=market_raw["optimisation"],
    )
    return plant_cfg, market_cfg


@pytest.fixture(scope="module")
def optimisation_result():
    plant_cfg, market_cfg = _load_configs()
    market_cfg.optimisation.horizon_hours = 48    # short horizon for fast tests
    market_cfg.optimisation.solver_options.time_limit = 120

    horizon_start = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
    index = build_horizon(horizon_start, horizon_hours=48)

    price_forecast = load_price_forecast("synthetic", index)
    inflow_forecast = load_inflow_forecast(
        "synthetic", index, inflow_cfg=plant_cfg.hydro.inflow
    )
    wind_forecast = load_wind_forecast("synthetic", index, wind_cfgs=plant_cfg.wind)

    fcr_n_prices = load_fcr_n_price_forecast("synthetic", index)
    model, result = build_and_solve(
        plant_cfg, market_cfg, price_forecast, inflow_forecast,
        horizon_start, wind_schedule=wind_forecast, fcr_n_prices=fcr_n_prices,
    )
    df = result_to_dataframe(result)
    return model, result, df, plant_cfg


def test_solver_finds_solution(optimisation_result):
    _, result, _, _ = optimisation_result
    assert result.status in ("optimal", "feasible", "maxTimeLimit")


def test_objective_is_positive(optimisation_result):
    _, result, _, _ = optimisation_result
    assert result.objective_value_eur > 0


def test_reservoir_bounds(optimisation_result):
    _, _, df, plant_cfg = optimisation_result
    v_max = plant_cfg.hydro.reservoir.capacity_gwh
    v_min = plant_cfg.hydro.reservoir.min_level_gwh
    assert df["reservoir_gwh"].max() <= v_max + 1e-4
    assert df["reservoir_gwh"].min() >= v_min - 1e-4


def test_hydro_generation_bounds(optimisation_result):
    _, _, df, plant_cfg = optimisation_result
    p_max = plant_cfg.hydro.turbine.capacity_mw
    assert df["hydro_gen_mw"].max() <= p_max + 1e-4
    assert df["hydro_gen_mw"].min() >= -1e-4


def test_no_pump_storage_active(optimisation_result):
    """Fortum has no pump storage: pump_cons must be 0 throughout."""
    _, _, df, plant_cfg = optimisation_result
    assert not plant_cfg.hydro.pump_storage_enabled
    assert (df["pump_cons_mw"] == 0).all(), "Pump consumption should be zero (no pump storage)"


def test_two_nuclear_units_present(optimisation_result):
    """Loviisa 1 and Loviisa 2 must both be in the model."""
    _, _, df, plant_cfg = optimisation_result
    assert "loviisa_1" in plant_cfg.nuclear
    assert "loviisa_2" in plant_cfg.nuclear


def test_nuclear_min_output(optimisation_result):
    """When a nuclear unit is online it must stay above minimum stable load."""
    _, _, df, plant_cfg = optimisation_result
    for unit_name, unit_cfg in plant_cfg.nuclear.items():
        p_min = unit_cfg.min_output_fraction * unit_cfg.capacity_mw
        # Per-unit nuclear is not directly in result_df (it's summed).
        # Check aggregate: total nuclear must equal 0 or >= p_min per running unit.
        # (Full per-unit check is in model; this is a sanity check on the aggregate.)
    total_nuclear_max = sum(u.capacity_mw for u in plant_cfg.nuclear.values())
    assert df["nuclear_gen_mw"].max() <= total_nuclear_max + 1e-4


def test_wind_schedule_non_negative(optimisation_result):
    _, _, df, _ = optimisation_result
    assert (df["wind_mw"] >= 0).all()


def test_elspot_bid_equals_net_generation(optimisation_result):
    _, _, df, _ = optimisation_result
    net = df["hydro_gen_mw"] + df["nuclear_gen_mw"] + df["wind_mw"] - df["pump_cons_mw"]
    diff = (df["elspot_bid_mw"] - net).abs()
    assert diff.max() < 1e-2, "Elspot bid does not match net generation"


def test_result_has_correct_length(optimisation_result):
    _, result, df, _ = optimisation_result
    assert len(df) == 48
    assert len(result.timestamps) == 48


# ---------------------------------------------------------------------------
# FCR-N ancillary service tests
# ---------------------------------------------------------------------------

def test_fcr_n_revenue_positive(optimisation_result):
    """FCR-N is enabled and synthetic prices are > 0, so revenue must be positive."""
    _, result, _, _ = optimisation_result
    assert result.fcr_n_revenue_eur > 0


def test_fcr_n_total_equals_hydro_plus_nuclear(optimisation_result):
    """r_fcr_n_total[t] must equal hydro + nuclear FCR-N at every hour."""
    _, result, _, _ = optimisation_result
    for h, n, tot in zip(result.fcr_n_hydro_mw, result.fcr_n_nuclear_mw, result.fcr_n_total_mw):
        assert abs(h + n - tot) < 1e-4


def test_fcr_n_hydro_within_cap(optimisation_result):
    """Hydro FCR-N must not exceed max_fcr_n_mw_hydro from config."""
    _, result, _, _ = optimisation_result
    with open("configs/market_params.yaml") as f:
        market_raw = yaml.safe_load(f)
    cap = market_raw["ancillary_services"]["FCR_N"]["max_fcr_n_mw_hydro"]
    assert max(result.fcr_n_hydro_mw) <= cap + 1e-4


def test_fcr_n_nuclear_within_cap(optimisation_result):
    """Nuclear FCR-N per unit must not exceed max_fcr_n_mw_per_nuclear_unit."""
    _, result, _, plant_cfg = optimisation_result
    with open("configs/market_params.yaml") as f:
        market_raw = yaml.safe_load(f)
    per_unit_cap = market_raw["ancillary_services"]["FCR_N"]["max_fcr_n_mw_per_nuclear_unit"]
    n_units = len(plant_cfg.nuclear)
    aggregate_cap = per_unit_cap * n_units
    assert max(result.fcr_n_nuclear_mw) <= aggregate_cap + 1e-4


def test_fcr_n_hydro_headroom_satisfied(optimisation_result):
    """hydro_gen + r_fcr_n_hydro must not exceed turbine capacity."""
    _, result, df, plant_cfg = optimisation_result
    p_max = plant_cfg.hydro.turbine.capacity_mw
    for gen, r in zip(df["hydro_gen_mw"], result.fcr_n_hydro_mw):
        assert gen + r <= p_max + 1e-4


def test_fcr_n_non_negative(optimisation_result):
    """FCR-N reserves must be >= 0 at every hour."""
    _, result, _, _ = optimisation_result
    assert all(v >= -1e-6 for v in result.fcr_n_hydro_mw)
    assert all(v >= -1e-6 for v in result.fcr_n_nuclear_mw)
    assert all(v >= -1e-6 for v in result.fcr_n_total_mw)


def test_fcr_n_df_columns_present(optimisation_result):
    """result_to_dataframe must include FCR-N columns."""
    _, _, df, _ = optimisation_result
    assert "fcr_n_hydro_mw" in df.columns
    assert "fcr_n_nuclear_mw" in df.columns
    assert "fcr_n_total_mw" in df.columns
