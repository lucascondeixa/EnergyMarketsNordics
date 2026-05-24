"""Smoke tests for the SE2 pump storage arbitrage optimiser.

Covers Fortum's Swedish assets: Kymmen (53 MW), Letten (36 MW), Eggsjön (0.6 MW).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from src.data_ingestion.forecasts import load_price_forecast, load_se2_price_forecast
from src.optimization.pump_arb_optimizer import build_and_solve_se2, se2_result_to_dataframe
from src.utils.schema import MarketConfig, PlantConfig, PumpStorageUnitConfig
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
        se2_pump_storage=plant_raw.get("se2_pump_storage", {}),
    )
    from src.utils.schema import AncillaryServicesConfig
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=AncillaryServicesConfig(**market_raw.get("ancillary_services", {})),
        optimisation=market_raw["optimisation"],
    )
    return plant_cfg, market_cfg


@pytest.fixture(scope="module")
def se2_result():
    plant_cfg, market_cfg = _load_configs()
    market_cfg.optimisation.horizon_hours = 48
    market_cfg.optimisation.solver_options.time_limit = 60

    horizon_start = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
    index = build_horizon(horizon_start, horizon_hours=48)

    fi_prices = load_price_forecast("synthetic", index)
    se2_prices = load_se2_price_forecast("synthetic", index, fi_price_series=fi_prices)

    model, result = build_and_solve_se2(
        plant_cfg.se2_pump_storage, market_cfg, se2_prices, horizon_start
    )
    df = se2_result_to_dataframe(result)
    return model, result, df, plant_cfg


def test_se2_three_units_configured(se2_result):
    _, _, _, plant_cfg = se2_result
    assert "kymmen" in plant_cfg.se2_pump_storage
    assert "letten" in plant_cfg.se2_pump_storage
    assert "eggsjoen" in plant_cfg.se2_pump_storage


def test_se2_solver_finds_solution(se2_result):
    _, result, _, _ = se2_result
    assert result.status in ("optimal", "feasible", "maxTimeLimit")


def test_se2_objective_non_negative(se2_result):
    """Pump arbitrage revenue can be zero (no spread) but should not be negative
    in a well-formed problem with non-negative prices."""
    _, result, _, _ = se2_result
    assert result.objective_value_eur >= -1.0  # allow tiny numerical slack


def test_se2_reservoir_bounds(se2_result):
    _, _, df, plant_cfg = se2_result
    for u, cfg in plant_cfg.se2_pump_storage.items():
        col = f"{u}_reservoir_gwh"
        assert df[col].max() <= cfg.reservoir_capacity_gwh + 1e-4
        assert df[col].min() >= cfg.min_level_gwh - 1e-4


def test_se2_no_simultaneous_pump_and_gen(se2_result):
    """Pump and generation must not be active simultaneously for any unit."""
    _, _, df, plant_cfg = se2_result
    for u in plant_cfg.se2_pump_storage:
        simultaneous = (df[f"{u}_gen_mw"] > 0.5) & (df[f"{u}_pump_mw"] > 0.5)
        assert not simultaneous.any(), f"Unit {u}: simultaneous pump and generation detected"


def test_se2_generation_capacity_bounds(se2_result):
    _, _, df, plant_cfg = se2_result
    for u, cfg in plant_cfg.se2_pump_storage.items():
        assert df[f"{u}_gen_mw"].max() <= cfg.turbine_capacity_mw + 1e-4
        assert df[f"{u}_gen_mw"].min() >= -1e-4


def test_se2_aggregate_bid_equals_net(se2_result):
    _, _, df, plant_cfg = se2_result
    net = sum(
        df[f"{u}_gen_mw"] - df[f"{u}_pump_mw"]
        for u in plant_cfg.se2_pump_storage
    )
    diff = (df["elspot_bid_mw"] - net).abs()
    assert diff.max() < 1e-2


def test_se2_result_length(se2_result):
    _, result, df, _ = se2_result
    assert len(df) == 48
    assert len(result.timestamps) == 48
