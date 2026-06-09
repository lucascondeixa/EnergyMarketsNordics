"""Pydantic schemas for configuration and data validation."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Plant configuration schemas
# ---------------------------------------------------------------------------


class HydroTurbineConfig(BaseModel):
    capacity_mw: float = Field(gt=0)
    min_output_mw: float = Field(ge=0, default=0)
    efficiency: float = Field(gt=0, le=1)
    ramp_rate_mw_per_hour: float = Field(gt=0)


class HydroPumpConfig(BaseModel):
    capacity_mw: float = Field(ge=0)
    min_input_mw: float = Field(ge=0, default=0)
    efficiency: float = Field(gt=0, le=1)


class HydroReservoirConfig(BaseModel):
    capacity_gwh: float = Field(gt=0)
    min_level_gwh: float = Field(ge=0)
    initial_level_gwh: float = Field(ge=0)
    terminal_water_value_eur_per_mwh: float = Field(ge=0)


class HydroInflowConfig(BaseModel):
    annual_avg_gwh_per_hour: float = Field(gt=0)
    seasonal_amplitude: float = Field(ge=0, le=1)
    peak_day_of_year: int = Field(ge=1, le=365)


class HydroConfig(BaseModel):
    pump_storage_enabled: bool = True
    turbine: HydroTurbineConfig
    pump: HydroPumpConfig
    reservoir: HydroReservoirConfig
    inflow: HydroInflowConfig
    syke_station_id: int | None = None


class PlannedOutage(BaseModel):
    start: datetime
    end: datetime


class NuclearUnitConfig(BaseModel):
    capacity_mw: float = Field(gt=0)
    min_output_fraction: float = Field(gt=0, le=1)
    startup_cost_eur: float = Field(ge=0)
    shutdown_cost_eur: float = Field(ge=0)
    min_up_hours: int = Field(ge=1)
    min_down_hours: int = Field(ge=1)
    ramp_rate_mw_per_hour: float = Field(gt=0)
    initial_on: bool
    initial_hours_on: int = Field(ge=0)
    variable_cost_eur_per_mwh: float = Field(ge=0)
    planned_outages: list[PlannedOutage] = []


class WindFarmConfig(BaseModel):
    capacity_mw: float = Field(gt=0)
    annual_capacity_factor: float = Field(gt=0, le=1)


class PumpStorageUnitConfig(BaseModel):
    """Configuration for a single pump-storage unit (used for SE2 Swedish assets)."""

    turbine_capacity_mw: float = Field(gt=0)
    pump_capacity_mw: float = Field(gt=0)
    turbine_efficiency: float = Field(gt=0, le=1)
    pump_efficiency: float = Field(gt=0, le=1)
    reservoir_capacity_gwh: float = Field(gt=0)
    min_level_gwh: float = Field(ge=0)
    initial_level_gwh: float = Field(ge=0)
    terminal_water_value_eur_per_mwh: float = Field(ge=0)
    ramp_rate_mw_per_hour: float = Field(gt=0)
    natural_inflow_gwh_per_hour: float = Field(ge=0, default=0.0)


class PlantConfig(BaseModel):
    hydro: HydroConfig
    nuclear: dict[str, NuclearUnitConfig]
    wind: dict[str, WindFarmConfig] = {}
    se2_pump_storage: dict[str, PumpStorageUnitConfig] = {}
    kemijoki: HydroConfig | None = None


# ---------------------------------------------------------------------------
# Market configuration schemas
# ---------------------------------------------------------------------------


class ElspotConfig(BaseModel):
    area: str
    price_cap_eur_mwh: float
    price_floor_eur_mwh: float
    bid_steps: int = Field(ge=2)
    gate_closure_hour: int = Field(ge=0, le=23)


class FCRNConfig(BaseModel):
    enabled: bool = False
    symmetric: bool = True
    min_bid_mw: float = 0.1
    activation_trigger_hz: float = 0.1
    gate_closure_hour_cet: int = 8
    max_fcr_n_mw_hydro: float = Field(gt=0, default=70.0)
    max_fcr_n_mw_per_nuclear_unit: float = Field(gt=0, default=25.0)


class AncillaryServicesConfig(BaseModel):
    FCR_N: FCRNConfig = FCRNConfig()
    # FCR_D_UP, FCR_D_DOWN, aFRR, mFRR: reserved for future implementation


class SolverOptions(BaseModel):
    mip_rel_gap: float = 0.005
    time_limit: int = 300


class OptimisationConfig(BaseModel):
    horizon_hours: int = Field(ge=24)
    time_resolution_hours: int = 1
    solver: str = "appsi_highs"
    solver_options: SolverOptions = SolverOptions()


class MarketConfig(BaseModel):
    elspot: ElspotConfig
    ancillary_services: AncillaryServicesConfig = AncillaryServicesConfig()
    optimisation: OptimisationConfig


# ---------------------------------------------------------------------------
# Time-series data schemas
# ---------------------------------------------------------------------------


class PriceForecast(BaseModel):
    """Validated price forecast for a single bidding zone."""

    timestamps: list[datetime]
    prices_eur_mwh: list[float]

    @field_validator("prices_eur_mwh")
    @classmethod
    def check_length_match(cls, v: list[float], info) -> list[float]:
        if "timestamps" in info.data and len(v) != len(info.data["timestamps"]):
            raise ValueError("prices_eur_mwh and timestamps must have equal length")
        return v

    def to_series(self) -> pd.Series:
        return pd.Series(
            self.prices_eur_mwh,
            index=pd.DatetimeIndex(self.timestamps, tz="UTC"),
            name="price_eur_mwh",
        )


class OptimisationResult(BaseModel):
    """Output of the FI joint optimiser run."""

    solve_time_seconds: float
    objective_value_eur: float          # planning objective (includes terminal water value credit)
    elspot_cash_revenue_eur: float      # actual cash Elspot revenue (excludes terminal water value)
    terminal_water_value_eur: float     # terminal reservoir credit (planning, not cash)
    mip_gap: float | None
    status: str
    hydro_dispatch_mw: list[float]
    pump_consumption_mw: list[float]
    reservoir_level_gwh: list[float]
    kemijoki_dispatch_mw: list[float]
    kemijoki_reservoir_level_gwh: list[float]
    nuclear_dispatch_mw: list[float]
    wind_dispatch_mw: list[float]
    elspot_bid_mw: list[float]
    # FCR-N ancillary service results (zeros when FCR-N disabled)
    fcr_n_hydro_mw: list[float]
    fcr_n_nuclear_mw: list[float]    # aggregate across all nuclear units
    fcr_n_total_mw: list[float]
    fcr_n_revenue_eur: float
    timestamps: list[datetime]


class SE2OptimisationResult(BaseModel):
    """Output of the SE2 pump storage arbitrage optimiser run."""

    solve_time_seconds: float
    objective_value_eur: float
    mip_gap: float | None
    status: str
    unit_names: list[str]
    generation_mw: dict[str, list[float]]   # unit_name -> hourly generation
    pump_mw: dict[str, list[float]]          # unit_name -> hourly pump consumption
    reservoir_gwh: dict[str, list[float]]    # unit_name -> hourly reservoir level
    elspot_bid_mw: list[float]               # aggregate SE2 net position
    timestamps: list[datetime]
