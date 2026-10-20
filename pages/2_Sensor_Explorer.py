"""
SentinelAI - Page 2: Sensor Explorer
Deep-dive single-unit time-series telemetry analysis, anomaly overlays,
and statistical distribution inspection.
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import SENSOR_DESCRIPTIONS, UI_COLORS
from src.ui_components import apply_custom_css, render_kpi_card, render_page_header
from src.visualizations import format_sensor_label, plot_sensor_telemetry

st.set_page_config(page_title="Sensor Explorer — SentinelAI", page_icon="📡", layout="wide")
apply_custom_css()

from src.pipeline_service import ensure_session_state

ensure_session_state()

assessed_df: pd.DataFrame = st.session_state["assessed_df"]
preprocessor = st.session_state["preprocessor"]
base_model = st.session_state["base_model"]

available_engines = sorted(assessed_df["engine_id"].unique())
active_sensors = preprocessor.active_features_

render_page_header(
    title="2. Sensor Telemetry Explorer",
    subtitle="Cycle-based time-series exploration, multi-channel degradation comparison, and threshold overlays.",
)

# Control Filters Row
c1, c2, c3, c4 = st.columns([1, 1.5, 1.5, 1])

with c1:
    selected_engine = st.selectbox(
        "Select Engine ID",
        options=available_engines,
        index=0,
        help="Choose a specific engine unit to inspect.",
    )

engine_df = assessed_df[assessed_df["engine_id"] == selected_engine].sort_values("cycle")
min_c = int(engine_df["cycle"].min())
max_c = int(engine_df["cycle"].max())

with c2:
    selected_sensor = st.selectbox(
        "Primary Sensor Channel",
        options=active_sensors,
        index=active_sensors.index("sensor_2") if "sensor_2" in active_sensors else 0,
        format_func=format_sensor_label,
        help="Primary telemetry channel to plot.",
    )

with c3:
    secondary_options = ["None"] + [s for s in active_sensors if s != selected_sensor]
    sec_selection = st.selectbox(
        "Secondary Overlay Sensor (Optional)",
        options=secondary_options,
        index=0,
        format_func=lambda s: "None" if s == "None" else format_sensor_label(s),
        help="Compare a second sensor on secondary vertical axis.",
    )
    secondary_sensor = None if sec_selection == "None" else sec_selection

with c4:
    show_anomalies = st.checkbox("Overlay Anomaly Flags", value=True)
    show_thresholds = st.checkbox("Show Baseline Bounds", value=True)

# Cycle Range Slider
if min_c < max_c:
    cycle_range = st.slider(
        "Operational Flight Cycle Range",
        min_value=min_c,
        max_value=max_c,
        value=(min_c, max_c),
        step=1,
    )
    filtered_engine_df = engine_df[
        (engine_df["cycle"] >= cycle_range[0]) & (engine_df["cycle"] <= cycle_range[1])
    ]
else:
    cycle_range = (min_c, max_c)
    filtered_engine_df = engine_df

# Baseline bounds if available
bounds = None
if show_thresholds and selected_sensor in base_model.thresholds_:
    bounds = base_model.thresholds_[selected_sensor]

# Quick KPIs for the selected unit
unit_anom_count = int((filtered_engine_df["is_anomaly"] == 1).sum())
unit_max_score = float(filtered_engine_df["anomaly_score"].max()) if len(filtered_engine_df) > 0 else 0.0

uk1, uk2, uk3, uk4 = st.columns(4)
with uk1:
    render_kpi_card("Inspected Unit", f"Engine #{selected_engine}", f"{len(filtered_engine_df)} Active Cycles")
with uk2:
    render_kpi_card("Current Channel", format_sensor_label(selected_sensor).split('(')[0].strip())
with uk3:
    render_kpi_card("Unit Anomalies", f"{unit_anom_count}", f"Across selected cycle window", value_color="#F59E0B" if unit_anom_count > 0 else None)
with uk4:
    render_kpi_card("Peak Anomaly Score", f"{unit_max_score:.3f}", "Calibrated Isolation Forest", value_color="#EF4444" if unit_max_score >= 0.7 else None)

# Telemetry Time-Series Chart
fig_ts = plot_sensor_telemetry(
    df=filtered_engine_df,
    engine_id=selected_engine,
    sensor_col=selected_sensor,
    highlight_anomalies=show_anomalies,
    baseline_bounds=bounds,
    secondary_sensor=secondary_sensor,
)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# Telemetry Statistics & Point Inspector
col_stats, col_inspect = st.columns([1, 1.2])

with col_stats:
    st.markdown("#### 📊 Descriptive Statistics Comparison")
    series = filtered_engine_df[selected_sensor].dropna()
    fleet_series = assessed_df[selected_sensor].dropna()

    stats_df = pd.DataFrame({
        "Metric": ["Sample Mean", "Std Deviation", "Min Observed", "25th Percentile", "Median", "75th Percentile", "Max Observed"],
        f"Unit #{selected_engine}": [
            f"{series.mean():.3f}",
            f"{series.std():.3f}",
            f"{series.min():.3f}",
            f"{series.quantile(0.25):.3f}",
            f"{series.median():.3f}",
            f"{series.quantile(0.75):.3f}",
            f"{series.max():.3f}",
        ],
        "Fleet Baseline": [
            f"{fleet_series.mean():.3f}",
            f"{fleet_series.std():.3f}",
            f"{fleet_series.min():.3f}",
            f"{fleet_series.quantile(0.25):.3f}",
            f"{fleet_series.median():.3f}",
            f"{fleet_series.quantile(0.75):.3f}",
            f"{fleet_series.max():.3f}",
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

with col_inspect:
    st.markdown("#### 🔬 Single Flight-Cycle Forensic Inspector")
    available_cycles = sorted(filtered_engine_df["cycle"].unique())
    inspect_cycle = st.selectbox(
        "Select Flight Cycle to Inspect",
        options=available_cycles,
        index=len(available_cycles) - 1 if available_cycles else 0,
    )

    if inspect_cycle is not None:
        row = filtered_engine_df[filtered_engine_df["cycle"] == inspect_cycle].iloc[0]
        status = row.get("health_status", "UNKNOWN")
        score = float(row.get("anomaly_score", 0.0))
        triggers = str(row.get("triggering_sensors", "None"))
        is_anom = int(row.get("is_anomaly", 0))

        badge_class = "badge-healthy" if status == "HEALTHY" else ("badge-warning" if status == "DEGRADED" else "badge-critical")
        
        st.markdown(f"""
        <div class="investigation-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span class="panel-title">
                    Engine #{selected_engine} @ Flight Cycle {inspect_cycle}
                </span>
                <span class="{badge_class}">● {status}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0.4rem;">
                <b>Calibrated Anomaly Score:</b> <span style="color: #F8FAFC; font-weight: 600;">{score:.4f}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0.4rem;">
                <b>Baseline Breach Description:</b> <span style="color: #F59E0B;">{triggers}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.88rem;">
                <b>Classification:</b> {'🚨 Anomalous Telemetry' if is_anom == 1 else '✅ Nominal Inlier'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Active sensor readings at this flight cycle:")
        readings = {col: round(float(row[col]), 3) for col in active_sensors[:8] if col in row}
        st.json(readings)
