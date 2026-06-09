"""Tests for the Kemijoki Oy hydro block.

Verifies:
  - Kemijoki block is active when config is present
  - Reservoir stays within physical bounds
  - Generation stays within turbine capacity
  - Elspot bid correctly includes Kemijoki output
  - Combined FI hydro (Vuoksi/Oulujoki + Kemijoki) outperforms either alone
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
HORIZON = 48
HORIZON_START = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)


def _load_configs(with_kemijoki: bool = True):
    with open(f"{CONFIG_DIR}/plant_params.yaml") as f:
        plant_raw = yaml.safe_load(f)
    with open(f"{CONFIG_DIR}/market_params.yaml") as f:
        market_raw = yaml.safe_load(f)

    plant_cfg = PlantConfig(
        hydro=plant_raw["hydro"],
        nuclear=plant_raw["nuclear"],
        wind=plant_raw.get("wind", {}),
        se2_pump_storage=plant_raw.get("se2_pump_storage", {}),
        kemijoki=plant_raw.get("kemijoki") if with_kemijoki else None,
    )
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=market_raw.get("ancillary_services", {}),
        optimisation=market_raw["optimisation"],
    )
    market_cfg.optimisation.horizon_hours = HORIZON
    market_cfg.optimisation.solver_options.time_limit = 120
    return plant_cfg, market_cfg


def _solve(with_kemijoki: bool = True):
    plant_cfg, market_cfg = _load_configs(with_kemijoki)
    index = build_horizon(HORIZON_START, horizon_hours=HORIZON)

    price = load_price_forecast("synthetic", index)
    inflow = load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.hydro.inflow)
    kemijoki_inflow = (
        load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.kemijoki.inflow)
        if plant_cfg.kemijoki is not None else None
    )
    wind = load_wind_forecast("synthetic", index, wind_cfgs=plant_cfg.wind)
    fcr_n = load_fcr_n_price_forecast("synthetic", index)

    model, result = build_and_solve(
        plant_cfg, market_cfg, price, inflow, HORIZON_START,
        wind_schedule=wind, fcr_n_prices=fcr_n,
        kemijoki_inflow_series=kemijoki_inflow,
    )
    return model, result, result_to_dataframe(result), plant_cfg


@pytest.fixture(scope="module")
def kemijoki_result():
    return _solve(with_kemijoki=True)


def test_solver_optimal_with_kemijoki(kemijoki_result):
    _, result, _, _ = kemijoki_result
    assert result.status in ("optimal", "feasible")


def test_kemijoki_dispatch_non_negative(kemijoki_result):
    _, result, df, _ = kemijoki_result
    assert all(v >= -1e-6 for v in result.kemijoki_dispatch_mw)
    assert (df["kemijoki_gen_mw"] >= -1e-6).all()


def test_kemijoki_within_turbine_cap(kemijoki_result):
    _, result, df, plant_cfg = kemijoki_result
    cap = plant_cfg.kemijoki.turbine.capacity_mw
    assert df["kemijoki_gen_mw"].max() <= cap + 1e-4


def test_kemijoki_reservoir_bounds(kemijoki_result):
    _, result, df, plant_cfg = kemijoki_result
    v_max = plant_cfg.kemijoki.reservoir.capacity_gwh
    v_min = plant_cfg.kemijoki.reservoir.min_level_gwh
    assert df["kemijoki_reservoir_gwh"].max() <= v_max + 1e-4
    assert df["kemijoki_reservoir_gwh"].min() >= v_min - 1e-4


def test_kemijoki_active_output(kemijoki_result):
    """With a 630 MW turbine and positive inflow, Kemijoki should generate > 0."""
    _, result, _, _ = kemijoki_result
    assert sum(result.kemijoki_dispatch_mw) > 0


def test_elspot_bid_includes_kemijoki(kemijoki_result):
    """Elspot bid must equal hydro + kemijoki + nuclear + wind - pump."""
    _, result, df, _ = kemijoki_result
    net = (
        df["hydro_gen_mw"]
        + df["kemijoki_gen_mw"]
        + df["nuclear_gen_mw"]
        + df["wind_mw"]
        - df["pump_cons_mw"]
    )
    diff = (df["elspot_bid_mw"] - net).abs()
    assert diff.max() < 1e-2, "Elspot bid does not include Kemijoki generation"


def test_kemijoki_result_columns_present(kemijoki_result):
    _, _, df, _ = kemijoki_result
    assert "kemijoki_gen_mw" in df.columns
    assert "kemijoki_reservoir_gwh" in df.columns


def test_kemijoki_raises_objective():
    """Objective with Kemijoki must be >= objective without Kemijoki."""
    _, result_with, _, _ = _solve(with_kemijoki=True)
    _, result_without, _, _ = _solve(with_kemijoki=False)
    assert result_with.objective_value_eur >= result_without.objective_value_eur - 1.0, (
        "Adding Kemijoki should not decrease the objective"
    )
