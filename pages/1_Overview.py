"""
SentinelAI - Page 1: Overview Dashboard
Fleet telemetry health, model status, anomaly counts, and recent alert triage.
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import HEALTH_THRESHOLDS, UI_COLORS
from src.reporting import generate_fleet_summary_metrics
from src.ui_components import apply_custom_css, render_kpi_card, render_page_header
from src.visualizations import (
    plot_anomaly_score_distribution,
    plot_fleet_health_breakdown,
)
from src.health_analysis import HEALTH_DISCLAIMER

st.set_page_config(page_title="Fleet Overview — SentinelAI", page_icon="🛡️", layout="wide")
apply_custom_css()

from src.pipeline_service import ensure_session_state

# Ensure session state is initialized
ensure_session_state()

assessed_df: pd.DataFrame = st.session_state["assessed_df"]
preprocessor = st.session_state["preprocessor"]
if_model = st.session_state["if_model"]
base_model = st.session_state["base_model"]
source_tag = st.session_state.get("source_tag", "official_real")
dataset_option = st.session_state.get("dataset_option", "NASA C-MAPSS FD001")
eval_split = st.session_state.get("eval_split", "Training Telemetry")
training_time = st.session_state.get("training_time", "Active")

render_page_header(
    title="1. Fleet Operational Overview & Telemetry Health",
    subtitle="High-level fleet monitoring, calibrated anomaly score distributions, and real-time alert triage.",
    badge_text=f"Partition: {eval_split.split()[0]} | Pipeline: Fitted & Monitored",
    badge_type="healthy",
)

# Scientific Context Expander
with st.expander("ℹ️ NASA C-MAPSS Telemetry Context & Architecture Details", expanded=False):
    st.markdown(
        """
        **About NASA C-MAPSS FD001:**
        - **Simulated Engine Telemetry**: Generated via NASA Glenn Research Center's *Commercial Modular Aero-Propulsion System Simulation* software. Simulates realistic thermophysical degradation of a 90,000 lb thrust class turbofan engine operating under sea-level conditions.
        - **Failure Modes**: Specifically simulates High-Pressure Compressor (HPC) degradation over operating flight cycles.
        - **Sensor Channels**: Telemetry includes 21 thermo-mechanical sensors (temperatures, pressures, rotor speeds, fuel flow ratios).
        - **Distinction Notice**: C-MAPSS represents established aerospace simulation runs rather than direct streaming telemetry from a live factory floor.
        """
    )
    st.info(HEALTH_DISCLAIMER)

# Dynamic Fleet KPIs
metrics = generate_fleet_summary_metrics(assessed_df)
latest_rows = assessed_df.groupby("engine_id")["cycle"].idxmax()
latest_df = assessed_df.loc[latest_rows]
health_counts = latest_df["health_status"].value_counts().to_dict()
for st_name in ["HEALTHY", "DEGRADED", "CRITICAL"]:
    health_counts.setdefault(st_name, 0)

total_anomalies = int((assessed_df["is_anomaly"] == 1).sum())
anomaly_rate = (total_anomalies / len(assessed_df)) * 100 if len(assessed_df) > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Fleet Size", f"{metrics['total_engines']} Units", f"{metrics['avg_cycles_per_engine']} Avg Cycles per Unit")
with k2:
    render_kpi_card("Telemetry Records", f"{metrics['total_observations']:,}", f"{len(preprocessor.active_features_)} Active Channels")
with k3:
    render_kpi_card("Flagged Anomalies", f"{total_anomalies:,}", f"{anomaly_rate:.2f}% Anomaly Rate", value_color="#F59E0B")
with k4:
    render_kpi_card("Critical Alert Units", f"{health_counts['CRITICAL']}", "Immediate inspection required", value_color="#EF4444")

st.markdown("---")

# Health Breakdown and Anomaly Distribution Plots
col_chart1, col_chart2 = st.columns([1, 1.2])

with col_chart1:
    st.markdown("#### 🩺 Fleet Health Distribution")
    fig_pie = plot_fleet_health_breakdown(health_counts)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_chart2:
    st.markdown("#### 📈 Calibrated Isolation Forest Anomaly Score Distribution")
    fig_dist = plot_anomaly_score_distribution(
        assessed_df["anomaly_score"].values,
        warn_threshold=HEALTH_THRESHOLDS["warning_score"],
        crit_threshold=HEALTH_THRESHOLDS["critical_score"],
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Recent Flagged Observations Table
st.markdown("### 🚨 High-Priority Telemetry Alerts & Triage Log")

flagged_df = assessed_df[assessed_df["health_status"].isin(["CRITICAL", "DEGRADED"])].copy()
if flagged_df.empty:
    st.success("✅ All monitored fleet units are operating within nominal baseline envelopes.")
else:
    display_cols = [
        "engine_id",
        "cycle",
        "health_status",
        "anomaly_score",
        "consecutive_anomalies",
        "triggering_sensors",
        "action_recommendation",
    ]
    if "rul" in flagged_df.columns:
        display_cols.insert(2, "rul")

    recent_table = (
        flagged_df[display_cols]
        .sort_values(by=["anomaly_score", "cycle"], ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    st.dataframe(
        recent_table,
        use_container_width=True,
        column_config={
            "engine_id": st.column_config.NumberColumn("Engine ID", format="#%d"),
            "cycle": st.column_config.NumberColumn("Flight Cycle"),
            "rul": st.column_config.NumberColumn("Remaining Life (Cycles)"),
            "health_status": st.column_config.TextColumn("Health Status"),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "consecutive_anomalies": st.column_config.NumberColumn("Streak"),
            "triggering_sensors": st.column_config.TextColumn("Root-Cause Baseline Triggers"),
            "action_recommendation": st.column_config.TextColumn("Recommended Action"),
        },
    )
