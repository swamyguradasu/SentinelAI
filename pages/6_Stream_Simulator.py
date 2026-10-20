"""
SentinelAI - Page 6: Sensor Stream Simulator (Demonstration Mode)
Demonstrates live telemetry ingestion, observation-by-observation model scoring,
and real-time alert triage without claiming to interface with live factory PLCs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import SENSOR_COLUMNS, UI_COLORS
from src.pipeline_service import ensure_session_state
from src.reporting import export_dataframe_to_csv_bytes
from src.simulator import SIMULATOR_DISCLAIMER, SensorStreamSimulator
from src.ui_components import apply_custom_css, get_status_badge_html, render_kpi_card, render_page_header
from src.visualizations import DARK_LAYOUT, format_sensor_label

st.set_page_config(page_title="Stream Simulator — SentinelAI", page_icon="⚡", layout="wide")
apply_custom_css()

# Auto-initialize global session state if accessed directly
ensure_session_state()

train_df = st.session_state["train_df"]
test_df = st.session_state["test_df"]
preprocessor = st.session_state["preprocessor"]
if_model = st.session_state["if_model"]
base_model = st.session_state["base_model"]
active_sensors = preprocessor.active_features_
available_engines = sorted(train_df["engine_id"].unique())

render_page_header(
    title="6. Sensor Stream Simulator (Demonstration Harness)",
    subtitle="Simulated per-cycle observation ingestion, online anomaly scoring, and dynamic health assessment.",
    badge_text="⚡ DEMO SIMULATION MODE — ISOLATED TEST HARNESS",
    badge_type="warning",
)

st.markdown(f'<div class="disclaimer-banner"><b>NOTICE:</b> {SIMULATOR_DISCLAIMER}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Simulator Configuration Controls
# -----------------------------------------------------------------------------
st.markdown("### 🛠️ Stream Configuration & Data Mode")
col_m1, col_m2, col_m3 = st.columns([1.5, 1, 1.5])

with col_m1:
    stream_mode = st.radio(
        "Simulation Telemetry Mode",
        [
            "Historical Engine Replay (NASA C-MAPSS)",
            "Synthetic Demonstration Stream (Controllable Drift / Anomaly Injection)",
        ],
        index=0,
        help="Choose between authentic historical engine degradation replay or synthetic signal generator.",
    )

with col_m2:
    if "Historical" in stream_mode:
        selected_sim_engine = st.selectbox(
            "Select Replay Engine ID",
            options=available_engines,
            index=0,
            help="Engine trajectory to replay in chronological cycle order.",
        )
        sim_target_sensor = None
        sim_noise = 0.02
        sim_inject_cycle = 30
        sim_shift = 3.0
    else:
        selected_sim_engine = 999  # Virtual engine ID for synthetic demonstration
        sim_target_sensor = st.selectbox(
            "Perturbation Target Sensor",
            options=active_sensors,
            index=0,
            format_func=format_sensor_label,
            help="Channel to inject abnormal thermal or pressure drift into.",
        )
        sim_noise = st.slider("Gaussian Noise Level (%)", 0.0, 5.0, 1.0, 0.5) / 100.0
        sim_inject_cycle = st.slider("Anomaly Injection Start Cycle", 10, 80, 30, 5)
        sim_shift = st.slider("Anomaly Drift Magnitude (%)", 1.0, 15.0, 4.0, 0.5)

with col_m3:
    display_sensor = st.selectbox(
        "Live Monitored Telemetry Channel",
        options=active_sensors,
        index=active_sensors.index("sensor_2") if "sensor_2" in active_sensors else 0,
        format_func=format_sensor_label,
        help="Sensor channel displayed on the dynamic chart.",
    )

# -----------------------------------------------------------------------------
# Simulator Instance State Management
# -----------------------------------------------------------------------------
mode_key = "HISTORICAL_REPLAY" if "Historical" in stream_mode else "SYNTHETIC_INJECTION"
sim_instance_key = f"sim_{mode_key}_{selected_sim_engine}_{sim_target_sensor}_{sim_inject_cycle}_{sim_shift}"

if "current_sim_key" not in st.session_state or st.session_state["current_sim_key"] != sim_instance_key:
    st.session_state["current_sim_key"] = sim_instance_key
    st.session_state["simulator"] = SensorStreamSimulator(
        mode=mode_key,
        engine_id=selected_sim_engine,
        historical_df=train_df if mode_key == "HISTORICAL_REPLAY" else None,
        noise_level=sim_noise,
        anomaly_injection_cycle=sim_inject_cycle,
        target_sensor=sim_target_sensor if sim_target_sensor else "sensor_2",
        anomaly_shift=sim_shift,
    )

simulator: SensorStreamSimulator = st.session_state["simulator"]

# -----------------------------------------------------------------------------
# Stream Playback Control Buttons
# -----------------------------------------------------------------------------
st.markdown("---")
b1, b2, b3, b4 = st.columns([1, 1, 1, 1.5])

with b1:
    if st.button("▶️ Step +1 Cycle", use_container_width=True, disabled=simulator.is_finished):
        simulator.step(preprocessor, if_model, base_model)

with b2:
    if st.button("⏩ Advance +10 Cycles", use_container_width=True, disabled=simulator.is_finished):
        for _ in range(10):
            if simulator.is_finished:
                break
            simulator.step(preprocessor, if_model, base_model)

with b3:
    if st.button("🔄 Reset Stream", use_container_width=True):
        simulator.reset()
        st.rerun()

with b4:
    if simulator.is_finished:
        st.warning("🏁 Replay Completed: All available engine cycles processed.")
    else:
        st.caption(f"Stream Status: `Active` | Pointer at step `{simulator.current_step}` of `{simulator.total_available_cycles}`")

# -----------------------------------------------------------------------------
# Live Dynamic KPIs
# -----------------------------------------------------------------------------
summary = simulator.get_summary_metrics()
latest = simulator.history[-1] if simulator.history else None

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Current Cycle", f"Cycle #{latest.cycle}" if latest else "Cycle #0", f"Total: {simulator.total_available_cycles}")
with k2:
    badge_html = get_status_badge_html(summary["current_health"])
    st.markdown(f"""
    <div class="kpi-metric-card">
        <div class="kpi-metric-label">Simulated Health</div>
        <div style="margin-top: 0.4rem;">{badge_html}</div>
        <div class="kpi-metric-sub">Streak: {latest.consecutive_anomalies if latest else 0}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    render_kpi_card("Live Anomaly Score", f"{latest.anomaly_score:.3f}" if latest else "0.000", "Calibrated [0, 1]")
with k4:
    render_kpi_card("Anomalies Flagged", f"{summary['total_anomalies']}", f"{summary['anomaly_rate']:.1f}% of stream", value_color="#F59E0B" if summary['total_anomalies'] > 0 else None)
with k5:
    render_kpi_card("Critical Alerts", f"{summary['critical_events']}", "Inspection threshold", value_color="#EF4444" if summary['critical_events'] > 0 else None)

# -----------------------------------------------------------------------------
# Dynamic Plotly Sensor Telemetry Chart
# -----------------------------------------------------------------------------
st.markdown("### 📈 Live Telemetry Stream Buffer")
history_df = simulator.get_history_df()

if history_df.empty:
    st.info("Stream buffer is empty. Click **'▶️ Step +1 Cycle'** or **'⏩ Advance +10 Cycles'** to begin processing observations.")
else:
    fig_stream = go.Figure()
    
    # Plot sensor trace
    fig_stream.add_trace(
        go.Scatter(
            x=history_df["cycle"],
            y=history_df[display_sensor],
            mode="lines+markers",
            line=dict(color=UI_COLORS["accent_cyan"], width=2),
            marker=dict(size=4),
            name=f"Live {format_sensor_label(display_sensor).split('(')[0].strip()}",
            hovertemplate=f"Cycle: %{{x}}<br>{display_sensor}: %{{y:.3f}}<extra></extra>",
        )
    )

    # Highlight anomalies
    anom_pts = history_df[history_df["is_anomaly"] == 1]
    if not anom_pts.empty:
        fig_stream.add_trace(
            go.Scatter(
                x=anom_pts["cycle"],
                y=anom_pts[display_sensor],
                mode="markers",
                marker=dict(color=UI_COLORS["accent_red"], size=9, symbol="diamond"),
                name="Flagged Anomaly Alert",
                hovertemplate=f"Cycle: %{{x}}<br>Anomaly Score: %{{customdata:.3f}}<extra>ALERT</extra>",
                customdata=anom_pts["anomaly_score"],
            )
        )

    # Plot baseline bounds if available
    if display_sensor in base_model.thresholds_:
        low_b, high_b = base_model.thresholds_[display_sensor]
        fig_stream.add_hline(
            y=high_b,
            line=dict(color=UI_COLORS["accent_amber"], dash="dash", width=1.5),
            annotation_text=f"Upper Baseline ({high_b:.2f})",
            annotation_position="top right",
        )
        fig_stream.add_hline(
            y=low_b,
            line=dict(color=UI_COLORS["accent_amber"], dash="dash", width=1.5),
            annotation_text=f"Lower Baseline ({low_b:.2f})",
            annotation_position="bottom right",
        )

    fig_stream.update_layout(
        **DARK_LAYOUT,
        title=f"Live Stream Observation Trajectory: {format_sensor_label(display_sensor)}",
        xaxis_title="Operational Flight Cycles (Stream Sequence)",
        yaxis_title=f"{display_sensor} Value",
        hovermode="x unified",
    )
    st.plotly_chart(fig_stream, use_container_width=True)

# -----------------------------------------------------------------------------
# Live Alert Feed & Session Export
# -----------------------------------------------------------------------------
st.markdown("### 🚨 Stream Observation Event Feed")
if not history_df.empty:
    col_t, col_exp = st.columns([3, 1])
    with col_exp:
        stream_csv = export_dataframe_to_csv_bytes(history_df)
        st.download_button(
            label="📥 Export Stream Session (CSV)",
            data=stream_csv,
            file_name=f"sentinelai_stream_session_eng_{selected_sim_engine}.csv",
            mime="text/csv",
            help="Download recorded stream observations and pipeline scores.",
        )

    display_cols = [
        "timestamp",
        "cycle",
        "health_status",
        "anomaly_score",
        "baseline_score",
        "is_anomaly",
        "triggering_sensors",
        "action_recommendation",
    ]
    st.dataframe(
        history_df[display_cols].sort_values("cycle", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Received At"),
            "cycle": st.column_config.NumberColumn("Cycle"),
            "health_status": st.column_config.TextColumn("Simulated Health"),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "baseline_score": st.column_config.NumberColumn("Baseline Severity", format="%.3f"),
            "is_anomaly": st.column_config.CheckboxColumn("Flagged"),
            "triggering_sensors": st.column_config.TextColumn("Triggering Sensors"),
            "action_recommendation": st.column_config.TextColumn("Action"),
        },
    )
else:
    st.caption("No stream observations recorded yet.")
