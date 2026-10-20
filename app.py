"""
SentinelAI - Predictive Maintenance & Sensor Anomaly Detection
Main Application Entrypoint.
Industrial IoT and telemetry intelligence platform.
"""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    HEALTH_THRESHOLDS,
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    UI_COLORS,
)
from src.anomaly_detection import train_anomaly_pipeline
from src.data_loader import (
    generate_realistic_cmapss_sample,
    load_train_and_test_datasets,
    parse_cmapss_file,
    validate_telemetry_schema,
)
from src.health_analysis import HEALTH_DISCLAIMER, assess_fleet_telemetry, get_latest_engine_health_summaries
from src.preprocessing import generate_data_quality_report
from src.reporting import generate_fleet_summary_metrics
from src.ui_components import apply_custom_css, get_status_badge_html, render_kpi_card, render_page_header
from src.visualizations import plot_fleet_health_breakdown

# Page configuration
st.set_page_config(
    page_title="SentinelAI — Predictive Maintenance & Telemetry Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark industrial styling
apply_custom_css()


from src.pipeline_service import (
    get_trained_models,
    load_telemetry_dataset,
    process_and_assess_telemetry,
)


# -----------------------------------------------------------------------------
# Sidebar Navigation and Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SentinelAI Controls")
    
    dataset_option = st.selectbox(
        "Telemetry Data Source",
        [
            "NASA Official C-MAPSS FD001 (Real Historical Data)",
            "Simulated Demonstration Benchmark (Synthetic)",
            "Custom Upload (CSV)",
        ],
        index=0,
        help="Select the telemetry source. Real NASA data represents authentic run-to-failure engine degradation.",
    )

    uploaded_csv = None
    if dataset_option == "Custom Upload (CSV)":
        uploaded_csv = st.file_uploader(
            "Upload Validated CSV",
            type=["csv"],
            help="CSV must contain engine_id, cycle, and at least 5 sensor_* columns.",
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Parameters")
    contamination_param = st.slider(
        "Isolation Forest Contamination",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Proportion of anticipated anomalous observations in training data.",
    )

    n_estimators_param = st.select_slider(
        "Tree Count (n_estimators)",
        options=[50, 100, 150, 200],
        value=100,
    )

    alpha_baseline = st.slider(
        "Baseline Quantile Alpha",
        min_value=0.005,
        max_value=0.05,
        value=0.02,
        step=0.005,
        help="Lower/Upper quantile cutoffs for Model B statistical envelope.",
    )

    st.markdown("---")
    eval_split = st.radio(
        "Active Telemetry Evaluation Split",
        ["Training Telemetry (Run-to-Failure)", "Test Telemetry (In-Service)"],
        index=0,
        help="Switch between training degradation trajectories and held-out test sequences.",
    )


# -----------------------------------------------------------------------------
# Data Loading & Model Execution
# -----------------------------------------------------------------------------
train_df, test_df, rul_df, err, source_tag = load_telemetry_dataset(dataset_option, uploaded_csv)

if err:
    st.error(f"⚠️ {err}")
    st.stop()

# Train models with caching
preprocessor, if_model, base_model, training_time = get_trained_models(
    train_df,
    contamination=contamination_param,
    n_estimators=n_estimators_param,
    lower_quantile=alpha_baseline,
    upper_quantile=1.0 - alpha_baseline,
)

# Store pipeline state into session_state for subpages
st.session_state["dataset_option"] = dataset_option
st.session_state["source_tag"] = source_tag
st.session_state["train_df"] = train_df
st.session_state["test_df"] = test_df
st.session_state["rul_df"] = rul_df
st.session_state["preprocessor"] = preprocessor
st.session_state["if_model"] = if_model
st.session_state["base_model"] = base_model
st.session_state["training_time"] = training_time
st.session_state["eval_split"] = eval_split

# Evaluate on selected split
eval_df = train_df if "Training" in eval_split else test_df
assessed_df, base_audit = process_and_assess_telemetry(eval_df, preprocessor, if_model, base_model)
st.session_state["assessed_df"] = assessed_df
st.session_state["base_audit"] = base_audit

# -----------------------------------------------------------------------------
# Main Landing View
# -----------------------------------------------------------------------------
badge_text = (
    "● Official NASA C-MAPSS FD001 (Real Historical Data)"
    if source_tag == "official_real"
    else ("▲ Synthetic Benchmark Demonstration Sample" if source_tag == "simulated_benchmark" else "● Custom Uploaded Telemetry")
)
badge_type = "healthy" if source_tag == "official_real" else ("warning" if source_tag == "simulated_benchmark" else "healthy")

render_page_header(
    title="🛡️ SentinelAI — Predictive Maintenance & Sensor Anomaly Detection",
    subtitle="Enterprise multi-sensor telemetry intelligence, unsupervised degradation tracking, and explainable equipment health analysis.",
    badge_text=badge_text,
    badge_type=badge_type,
)

st.markdown(f'<div class="disclaimer-banner">{HEALTH_DISCLAIMER}</div>', unsafe_allow_html=True)

# System Status Bar
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.caption(f"**Dataset Source**: `{dataset_option.split('(')[0].strip()}`")
with s2:
    st.caption(f"**Model Status**: `FITTED & ACTIVE`")
with s3:
    st.caption(f"**Last Model Training**: `{training_time}`")
with s4:
    has_labels = "rul" in assessed_df.columns
    st.caption(f"**Ground-Truth Status**: `{'Proxy Horizon Available' if has_labels else 'Unsupervised Only'}`")

st.markdown("---")

# Dynamic Fleet KPIs
metrics = generate_fleet_summary_metrics(assessed_df)
health_summaries = get_latest_engine_health_summaries(assessed_df)
health_counts = {"HEALTHY": 0, "DEGRADED": 0, "CRITICAL": 0}
for summary in health_summaries.values():
    health_counts[summary.health_status] = health_counts.get(summary.health_status, 0) + 1

anom_total = int((assessed_df["is_anomaly"] == 1).sum())
anom_pct = (anom_total / len(assessed_df)) * 100 if len(assessed_df) > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Fleet Monitored", f"{metrics['total_engines']} Units", f"{metrics['avg_cycles_per_engine']} Avg Cycles")
with k2:
    render_kpi_card("Telemetry Cycles", f"{metrics['total_observations']:,}", f"Span: {metrics['min_cycle']} - {metrics['max_cycle']}")
with k3:
    render_kpi_card("Active Channels", f"{len(preprocessor.active_features_)} / 21", f"Filtered {len(preprocessor.dropped_zero_var_features_)} zero-var")
with k4:
    render_kpi_card("Flagged Anomalies", f"{anom_total:,}", f"{anom_pct:.2f}% anomaly rate", value_color="#F59E0B")
with k5:
    render_kpi_card("Critical Units", f"{health_counts['CRITICAL']}", "Inspection recommended", value_color="#EF4444")

# Overview Charts Row
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown("### 📊 Fleet Operational Health Breakdown")
    fig_pie = plot_fleet_health_breakdown(health_counts)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("### 🧭 Control-Room Navigation & Module Overview")
    st.info(
        """
        Navigate through the specialized analytical views:
        - **1_Overview**: High-level fleet telemetry integrity and recent critical triage alerts.
        - **2_Sensor_Explorer**: Deep-dive single-engine time-series telemetry with anomaly overlays and point inspection.
        - **3_Anomaly_Center**: Forensic multi-criteria triage table with root-cause sensor breach explanations and CSV export.
        - **4_Model_Lab**: Side-by-side Isolation Forest vs Quantile Baseline benchmarking and proxy/labeled evaluation.
        - **5_Historical_Analysis**: Multi-engine trajectory comparisons and longitudinal degradation progression heatmaps.
        - **6_Stream_Simulator**: Isolated demonstration test harness replaying historical cycles or synthetic drift with live per-observation anomaly scoring.
        """
    )
    
    # Data Quality summary expander
    with st.expander("🔍 Telemetry Data Quality Summary", expanded=False):
        dq = generate_data_quality_report(eval_df, dataset_name=dataset_option)
        st.write(f"- **Data Quality Status**: `{dq['status']}`")
        st.write(f"- **Active Sensor Channels**: `{', '.join(preprocessor.active_features_)}`")
        st.write(f"- **Dropped Zero-Variance Channels**: `{', '.join(preprocessor.dropped_zero_var_features_) if preprocessor.dropped_zero_var_features_ else 'None'}`")

# Footer Citation
st.markdown("---")
st.caption(
    "**SentinelAI Telemetry Suite** | Dataset Attribution: NASA Ames Prognostics Center of Excellence (PCoE) Turbofan Engine Degradation Simulation Dataset (C-MAPSS). "
    "Saxena et al. (2008)."
)
