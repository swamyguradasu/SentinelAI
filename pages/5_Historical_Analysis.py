"""
SentinelAI - Page 5: Historical Analysis
Longitudinal degradation analysis, fleet trajectory comparisons,
and cycle progression heatmaps.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import UI_COLORS
from src.reporting import export_dataframe_to_csv_bytes, generate_audit_report_markdown
from src.ui_components import apply_custom_css, render_kpi_card, render_page_header
from src.visualizations import (
    DARK_LAYOUT,
    format_sensor_label,
    plot_fleet_degradation_heatmap,
)

st.set_page_config(page_title="Historical Analysis — SentinelAI", page_icon="📜", layout="wide")
apply_custom_css()

from src.pipeline_service import ensure_session_state

ensure_session_state()

assessed_df: pd.DataFrame = st.session_state["assessed_df"]
preprocessor = st.session_state["preprocessor"]
available_engines = sorted(assessed_df["engine_id"].unique())
active_sensors = preprocessor.active_features_

render_page_header(
    title="5. Historical Telemetry & Longitudinal Analysis",
    subtitle="Cross-unit degradation trajectories, fleet progression heatmaps, and longitudinal audit export.",
)

# Section 1: Fleet Degradation Heatmap
st.markdown("### 🔥 Fleet Degradation Progression Heatmap")
fig_hm = plot_fleet_degradation_heatmap(assessed_df, score_col="anomaly_score", max_engines=20)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# Section 2: Multi-Engine Trajectory Comparison
st.markdown("### 📈 Multi-Engine Comparative Telemetry Trajectories")
col_e1, col_e2 = st.columns([1.5, 1])

with col_e1:
    selected_compare_engines = st.multiselect(
        "Select Units to Compare",
        options=available_engines,
        default=available_engines[:4] if len(available_engines) >= 4 else available_engines,
        help="Select multiple engines to compare degradation curves on identical axes.",
    )

with col_e2:
    compare_sensor = st.selectbox(
        "Sensor Channel",
        options=active_sensors,
        index=active_sensors.index("sensor_2") if "sensor_2" in active_sensors else 0,
        format_func=format_sensor_label,
    )

if selected_compare_engines:
    fig_comp = go.Figure()
    palette = ["#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    for idx, eng in enumerate(selected_compare_engines):
        sub = assessed_df[assessed_df["engine_id"] == eng].sort_values("cycle")
        color = palette[idx % len(palette)]
        fig_comp.add_trace(
            go.Scatter(
                x=sub["cycle"],
                y=sub[compare_sensor],
                mode="lines",
                name=f"Engine #{eng}",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>Engine #{eng}</b><br>Cycle: %{{x}}<br>{compare_sensor}: %{{y:.3f}}<extra></extra>",
            )
        )

    fig_comp.update_layout(
        **DARK_LAYOUT,
        title=f"Comparative Trajectories: {format_sensor_label(compare_sensor)}",
        xaxis_title="Operational Flight Cycles",
        yaxis_title=f"{compare_sensor} Value",
        hovermode="x unified",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Select at least one engine unit to display comparative trajectories.")

st.markdown("---")

# Section 3: Anomaly Escalation by Flight Cycle
st.markdown("### 📊 Anomaly Escalation Across Flight Cycles")
cycle_bins = pd.cut(assessed_df["cycle"], bins=range(0, int(assessed_df["cycle"].max()) + 20, 20))
bin_stats = (
    assessed_df.groupby(cycle_bins, observed=False)
    .agg(
        total_obs=("is_anomaly", "count"),
        anom_count=("is_anomaly", "sum"),
        mean_score=("anomaly_score", "mean"),
    )
    .reset_index()
)
bin_stats["anomaly_rate"] = (bin_stats["anom_count"] / bin_stats["total_obs"]) * 100
bin_stats["cycle_range_str"] = bin_stats["cycle"].astype(str)

fig_esc = go.Figure()
fig_esc.add_trace(
    go.Bar(
        x=bin_stats["cycle_range_str"],
        y=bin_stats["anomaly_rate"],
        marker=dict(color=UI_COLORS["accent_cyan"], line=dict(color=UI_COLORS["border"], width=1)),
        name="Anomaly Rate (%)",
        hovertemplate="Cycles: %{x}<br>Anomaly Rate: %{y:.1f}%<extra></extra>",
    )
)
fig_esc.update_layout(
    **DARK_LAYOUT,
    title="Fleet Anomaly Rate Escalation by Operating Cycle Interval",
    xaxis_title="Flight Cycle Window",
    yaxis_title="Fleet Anomaly Prevalence (%)",
)
st.plotly_chart(fig_esc, use_container_width=True)

st.markdown("---")

# Section 4: Export Summary Audit Report
st.markdown("### 📝 Export Comprehensive Fleet Telemetry Audit Report")
dataset_name = st.session_state.get("dataset_option", "NASA C-MAPSS")
fleet_metrics = {
    "total_engines": int(assessed_df["engine_id"].nunique()),
    "total_observations": int(len(assessed_df)),
}
latest_rows = assessed_df.groupby("engine_id")["cycle"].idxmax()
health_counts = assessed_df.loc[latest_rows, "health_status"].value_counts().to_dict()

report_md = generate_audit_report_markdown(
    dataset_name=dataset_name,
    fleet_metrics=fleet_metrics,
    health_counts=health_counts,
)

r_col1, r_col2 = st.columns([1, 1])
with r_col1:
    st.download_button(
        label="📄 Download Markdown Audit Report",
        data=report_md.encode("utf-8"),
        file_name="sentinelai_telemetry_audit_report.md",
        mime="text/markdown",
    )
with r_col2:
    csv_bytes = export_dataframe_to_csv_bytes(assessed_df)
    st.download_button(
        label="📥 Download Complete Evaluated Telemetry (CSV)",
        data=csv_bytes,
        file_name="sentinelai_full_assessed_telemetry.csv",
        mime="text/csv",
    )
