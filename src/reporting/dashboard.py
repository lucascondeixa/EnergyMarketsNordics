"""Streamlit dashboard for interactive bid review — Fortum Nordic Portfolio.

Run from the project root:
    streamlit run src/reporting/dashboard.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(
    page_title="Fortum Nordic Optimiser",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_ingestion.forecasts import (
    load_fcr_n_price_forecast,
    load_inflow_forecast,
    load_price_forecast,
    load_se2_price_forecast,
    load_wind_forecast,
)
from src.markets.elspot import build_bid_curves
from src.optimization.joint_optimizer import build_and_solve, result_to_dataframe
from src.optimization.pump_arb_optimizer import build_and_solve_se2, se2_result_to_dataframe
from src.utils.schema import MarketConfig, OptimisationResult, PlantConfig, SE2OptimisationResult
from src.utils.time_utils import build_horizon, next_day_ahead_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_configs(plant_path: Path, market_path: Path) -> tuple[PlantConfig, MarketConfig]:
    with open(plant_path) as f:
        plant_raw = yaml.safe_load(f)
    with open(market_path) as f:
        market_raw = yaml.safe_load(f)
    plant_cfg = PlantConfig(
        hydro=plant_raw["hydro"],
        nuclear=plant_raw["nuclear"],
        wind=plant_raw.get("wind", {}),
        se2_pump_storage=plant_raw.get("se2_pump_storage", {}),
        kemijoki=plant_raw.get("kemijoki"),
    )
    market_cfg = MarketConfig(
        elspot=market_raw["elspot"],
        ancillary_services=market_raw.get("ancillary_services", {}),
        optimisation=market_raw["optimisation"],
    )
    return plant_cfg, market_cfg


def _stacked_area(
    df: pd.DataFrame,
    cols: list[str],
    labels: list[str],
    colors: list[str],
    price: pd.Series | None = None,
    price_label: str = "Price (EUR/MWh)",
) -> go.Figure:
    fig = go.Figure()
    for col, label, color in zip(cols, labels, colors):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=label,
            stackgroup="one",
            line=dict(width=0),
            fillcolor=color,
            mode="lines",
        ))
    if price is not None:
        fig.add_trace(go.Scatter(
            x=price.index, y=price.values,
            name=price_label,
            yaxis="y2",
            line=dict(color="rgba(0,0,0,0.7)", dash="dot", width=1.5),
            mode="lines",
        ))
        fig.update_layout(
            yaxis2=dict(title="EUR/MWh", overlaying="y", side="right", showgrid=False),
        )
    fig.update_layout(
        yaxis_title="MW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _line(
    series_list: list[tuple[pd.Series, str, str]],
    yaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for series, label, color in series_list:
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=label, line=dict(color=color)))
    fig.update_layout(yaxis_title=yaxis_title, hovermode="x unified", margin=dict(t=30, b=40))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Fortum Nordic Optimiser")
st.sidebar.markdown("FI (nuclear + hydro + wind) + SE2 (pump storage)")

price_source = st.sidebar.selectbox(
    "Price source",
    options=["forecast", "synthetic"],
    help="'forecast' fetches live ENTSO-E data. 'synthetic' uses generated prices.",
)
horizon = st.sidebar.slider("Horizon (hours)", min_value=24, max_value=168, value=168, step=24)

st.sidebar.divider()
st.sidebar.markdown("**Sensitivity**")
twv = st.sidebar.slider("Terminal water value (EUR/MWh)", min_value=20, max_value=100, value=55, step=5)
fcr_n_cap_hydro = st.sidebar.slider("FCR-N hydro cap (MW)", min_value=0, max_value=200, value=70, step=10)

run_btn = st.sidebar.button("Run Optimisation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Optimisation (auto-run on first load, re-run on button click)
# ---------------------------------------------------------------------------

if run_btn or "fi_result" not in st.session_state:
    with st.spinner("Solving…"):
        plant_cfg, market_cfg = _load_configs(
            Path("configs/plant_params.yaml"),
            Path("configs/market_params.yaml"),
        )
        market_cfg.optimisation.horizon_hours = horizon
        plant_cfg.hydro.reservoir.terminal_water_value_eur_per_mwh = float(twv)
        market_cfg.ancillary_services.FCR_N.max_fcr_n_mw_hydro = float(fcr_n_cap_hydro)

        horizon_start = next_day_ahead_gate()
        index = build_horizon(horizon_start, horizon_hours=horizon)

        price_fi = load_price_forecast(price_source, index)
        inflow = load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.hydro.inflow)
        wind = load_wind_forecast("synthetic", index, wind_cfgs=plant_cfg.wind)
        price_se2 = load_se2_price_forecast(price_source, index, fi_price_series=price_fi)
        kemijoki_inflow = (
            load_inflow_forecast("synthetic", index, inflow_cfg=plant_cfg.kemijoki.inflow)
            if plant_cfg.kemijoki is not None else None
        )
        fcr_n_prices = (
            load_fcr_n_price_forecast("synthetic", index)
            if market_cfg.ancillary_services.FCR_N.enabled else None
        )

        _, fi_result = build_and_solve(
            plant_cfg, market_cfg, price_fi, inflow, horizon_start,
            wind_schedule=wind, fcr_n_prices=fcr_n_prices,
            kemijoki_inflow_series=kemijoki_inflow,
        )
        fi_df = result_to_dataframe(fi_result)
        fi_bid_curves = build_bid_curves(
            fi_df, price_fi,
            n_steps=market_cfg.elspot.bid_steps,
            price_floor=market_cfg.elspot.price_floor_eur_mwh,
            price_cap=market_cfg.elspot.price_cap_eur_mwh,
        )

        se2_result: SE2OptimisationResult | None = None
        se2_df: pd.DataFrame | None = None
        if plant_cfg.se2_pump_storage:
            _, se2_result = build_and_solve_se2(
                plant_cfg.se2_pump_storage, market_cfg, price_se2, horizon_start
            )
            se2_df = se2_result_to_dataframe(se2_result)

        st.session_state.update(
            fi_result=fi_result,
            fi_df=fi_df,
            fi_bid_curves=fi_bid_curves,
            se2_result=se2_result,
            se2_df=se2_df,
            price_fi=price_fi,
            price_se2=price_se2,
            horizon_start=horizon_start,
        )

fi_result: OptimisationResult = st.session_state.fi_result
fi_df: pd.DataFrame = st.session_state.fi_df
fi_bid_curves: pd.DataFrame = st.session_state.fi_bid_curves
se2_result: SE2OptimisationResult | None = st.session_state.se2_result
se2_df: pd.DataFrame | None = st.session_state.se2_df
price_fi: pd.Series = st.session_state.price_fi
price_se2: pd.Series = st.session_state.price_se2
horizon_start: datetime = st.session_state.horizon_start

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

se2_cash = se2_result.objective_value_eur if se2_result else 0.0
total_cash = fi_result.elspot_cash_revenue_eur + fi_result.fcr_n_revenue_eur + se2_cash

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio cash (EUR)", f"{total_cash:,.0f}")
c2.metric("FI Elspot (EUR)", f"{fi_result.elspot_cash_revenue_eur:,.0f}")
c3.metric("FI FCR-N (EUR)", f"{fi_result.fcr_n_revenue_eur:,.0f}")
c4.metric("SE2 pump arb (EUR)", f"{se2_cash:,.0f}")
c5.metric(
    "Terminal water credit (EUR)",
    f"{fi_result.terminal_water_value_eur:,.0f}",
    help="Planning credit only — not cash revenue",
)

st.caption(
    f"Horizon: {horizon_start.strftime('%Y-%m-%d %H:%M UTC')} +{len(fi_df)}h  |  "
    f"FI solve {fi_result.solve_time_seconds:.1f}s ({fi_result.status})"
    + (f"  |  SE2 solve {se2_result.solve_time_seconds:.1f}s ({se2_result.status})" if se2_result else "")
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_portfolio, tab_fi, tab_se2, tab_bids = st.tabs(
    ["Portfolio", "FI Zone", "SE2 Zone", "Bid Curves & Export"]
)

# ── Portfolio ──────────────────────────────────────────────────────────────

with tab_portfolio:
    st.subheader("Portfolio generation + FI price")

    portfolio_df = fi_df.copy()
    gen_cols = ["hydro_gen_mw", "nuclear_gen_mw", "wind_mw"]
    gen_labels = ["Hydro FI", "Nuclear", "Wind"]
    gen_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    if se2_df is not None:
        se2_net = se2_df["elspot_bid_mw"].reindex(fi_df.index, fill_value=0.0)
        # Only show positive net (generation side) in the portfolio stacked view
        portfolio_df["se2_gen_mw"] = se2_net.clip(lower=0)
        gen_cols.append("se2_gen_mw")
        gen_labels.append("SE2 net gen")
        gen_colors.append("#9C27B0")

    fig_port = _stacked_area(portfolio_df, gen_cols, gen_labels, gen_colors, price=price_fi)
    st.plotly_chart(fig_port, use_container_width=True, key="port_dispatch")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("FI reservoir (Oulujärvi)")
        fig_res = _line([(fi_df["reservoir_gwh"], "Reservoir", "#2196F3")], "GWh")
        st.plotly_chart(fig_res, use_container_width=True, key="port_fi_res")
    with col_r:
        if se2_df is not None:
            st.subheader("SE2 reservoir levels")
            unit_names = se2_result.unit_names
            se2_colors = ["#9C27B0", "#E91E63", "#3F51B5"]
            series_list = [
                (se2_df[f"{u}_reservoir_gwh"], u, c)
                for u, c in zip(unit_names, se2_colors)
            ]
            fig_se2_res_small = _line(series_list, "GWh")
            st.plotly_chart(fig_se2_res_small, use_container_width=True, key="port_se2_res")
        else:
            st.info("No SE2 pump storage configured.")

# ── FI Zone ────────────────────────────────────────────────────────────────

with tab_fi:
    st.subheader("FI dispatch breakdown")
    fig_fi = _stacked_area(
        fi_df,
        ["hydro_gen_mw", "nuclear_gen_mw", "wind_mw"],
        ["Hydro", "Nuclear", "Wind"],
        ["#2196F3", "#FF9800", "#4CAF50"],
        price=price_fi,
    )
    st.plotly_chart(fig_fi, use_container_width=True, key="fi_dispatch")

    has_fcr_n = bool(fi_df["fcr_n_total_mw"].any())
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Reservoir level")
        fig_res_fi = _line([(fi_df["reservoir_gwh"], "Oulujärvi", "#2196F3")], "GWh")
        st.plotly_chart(fig_res_fi, use_container_width=True, key="fi_reservoir")

    with col_b:
        if has_fcr_n:
            st.subheader("FCR-N allocation")
            fig_fcr = go.Figure()
            fig_fcr.add_trace(go.Bar(
                x=fi_df.index, y=fi_df["fcr_n_hydro_mw"],
                name="Hydro FCR-N", marker_color="#2196F3",
            ))
            fig_fcr.add_trace(go.Bar(
                x=fi_df.index, y=fi_df["fcr_n_nuclear_mw"],
                name="Nuclear FCR-N", marker_color="#FF9800",
            ))
            fig_fcr.update_layout(
                barmode="stack", yaxis_title="MW",
                margin=dict(t=30, b=40), hovermode="x unified",
            )
            st.plotly_chart(fig_fcr, use_container_width=True, key="fi_fcr_n")
        else:
            st.subheader("Pump consumption")
            fig_pump = _line([(fi_df["pump_cons_mw"], "Pump", "#9C27B0")], "MW")
            st.plotly_chart(fig_pump, use_container_width=True, key="fi_pump")

# ── SE2 Zone ───────────────────────────────────────────────────────────────

with tab_se2:
    if se2_df is None or se2_result is None:
        st.info("No SE2 pump storage units configured.")
    else:
        unit_names = se2_result.unit_names
        se2_colors = ["#9C27B0", "#E91E63", "#3F51B5"]

        st.subheader("SE2 dispatch — generation (+) and pump (−)")
        fig_se2 = go.Figure()
        for unit, color in zip(unit_names, se2_colors):
            fig_se2.add_trace(go.Scatter(
                x=se2_df.index, y=se2_df[f"{unit}_gen_mw"],
                name=f"{unit} gen",
                stackgroup="gen",
                line=dict(width=0), fillcolor=color,
                mode="lines",
            ))
        for unit, color in zip(unit_names, se2_colors):
            fig_se2.add_trace(go.Scatter(
                x=se2_df.index, y=-se2_df[f"{unit}_pump_mw"],
                name=f"{unit} pump",
                stackgroup="pump",
                line=dict(width=0),
                fillcolor=color.replace(")", ", 0.4)").replace("rgb", "rgba")
                    if color.startswith("rgb") else color,
                opacity=0.4,
                mode="lines",
            ))
        fig_se2.add_trace(go.Scatter(
            x=price_se2.index, y=price_se2.values,
            name="SE2 price",
            yaxis="y2",
            line=dict(color="rgba(0,0,0,0.6)", dash="dot", width=1.5),
            mode="lines",
        ))
        fig_se2.update_layout(
            yaxis_title="MW (gen +, pump −)",
            yaxis2=dict(title="EUR/MWh", overlaying="y", side="right", showgrid=False),
            hovermode="x unified",
            margin=dict(t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_se2, use_container_width=True, key="se2_dispatch")

        st.subheader("SE2 reservoir levels")
        fig_se2_res = _line(
            [(se2_df[f"{u}_reservoir_gwh"], u, c) for u, c in zip(unit_names, se2_colors)],
            "GWh",
        )
        st.plotly_chart(fig_se2_res, use_container_width=True, key="se2_reservoir")

# ── Bid Curves & Export ────────────────────────────────────────────────────

with tab_bids:
    st.subheader("Day 1 Elspot bid curves — FI")
    fig_bids = go.Figure()
    hours = sorted(fi_bid_curves["hour"].unique())
    for hour in hours:
        hue = int(hour * 360 / 24)
        color = f"hsl({hue}, 70%, 50%)"
        hour_df = fi_bid_curves[fi_bid_curves["hour"] == hour].sort_values("price_eur_mwh")
        fig_bids.add_trace(go.Scatter(
            x=hour_df["quantity_mw"],
            y=hour_df["price_eur_mwh"],
            name=f"H{hour:02d}",
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
        ))
    fig_bids.update_layout(
        xaxis_title="Quantity (MW)",
        yaxis_title="Price (EUR/MWh)",
        hovermode="closest",
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bids, use_container_width=True, key="bid_curves")

    st.subheader("Export")
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.download_button(
            "FI dispatch schedule (CSV)",
            fi_df.to_csv().encode(),
            file_name="FI_schedule.csv",
            mime="text/csv",
        )
    with col_e2:
        st.download_button(
            "FI Elspot bid curves (CSV)",
            fi_bid_curves.to_csv(index=False).encode(),
            file_name="FI_elspot_bids.csv",
            mime="text/csv",
        )
    with col_e3:
        if se2_df is not None:
            st.download_button(
                "SE2 dispatch schedule (CSV)",
                se2_df.to_csv().encode(),
                file_name="SE2_schedule.csv",
                mime="text/csv",
            )
        else:
            st.button("SE2 dispatch schedule (CSV)", disabled=True)
