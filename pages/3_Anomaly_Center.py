"""
SentinelAI - Page 3: Anomaly Center
Triage flagged telemetry, investigate root-cause sensor triggers,
and export audit-compliant anomaly datasets to CSV.
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import UI_COLORS
from src.reporting import export_dataframe_to_csv_bytes, filter_anomaly_records
from src.ui_components import apply_custom_css, render_kpi_card, render_page_header

st.set_page_config(page_title="Anomaly Center — SentinelAI", page_icon="🚨", layout="wide")
apply_custom_css()

from src.pipeline_service import ensure_session_state

ensure_session_state()

assessed_df: pd.DataFrame = st.session_state["assessed_df"]
preprocessor = st.session_state["preprocessor"]

render_page_header(
    title="3. Anomaly Triage & Forensic Audit Center",
    subtitle="Multi-dimensional alert filtering, root-cause sensor attribution, and forensic CSV export.",
)

available_engines = ["All"] + sorted(assessed_df["engine_id"].unique())

# Multi-Filter Triage Bar
f1, f2, f3, f4 = st.columns(4)

with f1:
    selected_eng = st.selectbox("Filter by Engine Unit", options=available_engines, index=0)

with f2:
    selected_method = st.selectbox(
        "Detection Method",
        options=["Any", "Isolation Forest", "Quantile Baseline", "Both"],
        index=0,
        help="Filter records flagged by specific anomaly detection approaches.",
    )

with f3:
    selected_health = st.selectbox(
        "Equipment Health Status",
        options=["All", "CRITICAL", "DEGRADED", "HEALTHY"],
        index=0,
    )

with f4:
    min_score_filter = st.slider(
        "Minimum Anomaly Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter records with calibrated Isolation Forest score >= threshold.",
    )

search_kw = st.text_input(
    "Search Triggering Sensors or Directions (e.g. 'sensor_2', 'sensor_3', 'HIGH', 'LOW')",
    value="",
    placeholder="Type sensor name or trigger direction...",
)

# Apply Filters
filtered_records = filter_anomaly_records(
    df=assessed_df,
    engine_id=None if selected_eng == "All" else int(selected_eng),
    min_score=min_score_filter,
    health_status=None if selected_health == "All" else selected_health,
    detection_method=selected_method,
    search_keyword=search_kw,
)

# Triage KPI Row
t1, t2, t3, t4 = st.columns(4)
crit_count = int((filtered_records["health_status"] == "CRITICAL").sum()) if len(filtered_records) > 0 else 0
warn_count = int((filtered_records["health_status"] == "DEGRADED").sum()) if len(filtered_records) > 0 else 0

with t1:
    render_kpi_card("Matching Records", f"{len(filtered_records):,}", f"Out of {len(assessed_df):,} total")
with t2:
    render_kpi_card("Critical Alerts", f"{crit_count:,}", "Requires immediate overhaul", value_color="#EF4444")
with t3:
    render_kpi_card("Degraded Warnings", f"{warn_count:,}", "Monitored telemetry drift", value_color="#F59E0B")
with t4:
    if not filtered_records.empty:
        csv_bytes = export_dataframe_to_csv_bytes(filtered_records)
        st.download_button(
            label="📥 Export Triage CSV",
            data=csv_bytes,
            file_name=f"sentinelai_anomalies_{selected_eng}_{selected_method}.csv",
            mime="text/csv",
            help="Download filtered telemetry records for external reporting.",
        )
    else:
        st.caption("No records available to export.")

st.markdown("---")

# Filtered Data Table
st.markdown("### 📋 Anomaly Log & Root-Cause Audit Trail")

if filtered_records.empty:
    st.info("No telemetry records match the current filter criteria.")
else:
    display_columns = [
        "engine_id",
        "cycle",
        "health_status",
        "anomaly_score",
        "baseline_score",
        "is_anomaly_if",
        "is_anomaly_base",
        "triggering_sensors",
    ]
    if "rul" in filtered_records.columns:
        display_columns.insert(2, "rul")

    st.dataframe(
        filtered_records[display_columns].sort_values(by=["anomaly_score", "cycle"], ascending=False),
        use_container_width=True,
        column_config={
            "engine_id": st.column_config.NumberColumn("Engine ID", format="#%d"),
            "cycle": st.column_config.NumberColumn("Flight Cycle"),
            "rul": st.column_config.NumberColumn("Remaining Life (Cycles)"),
            "health_status": st.column_config.TextColumn("Health Status"),
            "anomaly_score": st.column_config.ProgressColumn(
                "Isolation Forest Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "baseline_score": st.column_config.NumberColumn("Baseline Severity", format="%.3f"),
            "is_anomaly_if": st.column_config.CheckboxColumn("IF Alert"),
            "is_anomaly_base": st.column_config.CheckboxColumn("Baseline Alert"),
            "triggering_sensors": st.column_config.TextColumn("Triggering Sensor Channels"),
        },
    )

st.markdown("---")

# Forensic Root-Cause Breakdown
st.markdown("### 🔍 Root-Cause Channel Frequency Analysis")
triggers_list = filtered_records["triggering_sensors"].dropna()
sensor_counts = {}
for item in triggers_list:
    if item and item != "None":
        for part in item.split(";"):
            clean_part = part.strip().split()[0]
            if clean_part.startswith("sensor_"):
                sensor_counts[clean_part] = sensor_counts.get(clean_part, 0) + 1

if sensor_counts:
    counts_df = pd.DataFrame(
        list(sensor_counts.items()),
        columns=["Sensor Channel", "Breach Incident Count"],
    ).sort_values(by="Breach Incident Count", ascending=False)
    
    st.dataframe(counts_df, hide_index=True, use_container_width=True)
else:
    st.caption("No individual sensor baseline threshold breaches in the filtered subset.")
