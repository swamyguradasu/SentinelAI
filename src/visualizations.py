"""
Visualization Module for SentinelAI.
Creates interactive, publication-quality Plotly figures styled
with a bespoke dark industrial control-room aesthetic.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SENSOR_DESCRIPTIONS, UI_COLORS


DARK_LAYOUT = dict(
    paper_bgcolor=UI_COLORS["background"],
    plot_bgcolor=UI_COLORS["card_bg"],
    font=dict(color=UI_COLORS["text_primary"], family="Inter, system-ui, sans-serif", size=12),
    xaxis=dict(
        gridcolor=UI_COLORS["border"],
        zerolinecolor=UI_COLORS["border"],
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor=UI_COLORS["border"],
        zerolinecolor=UI_COLORS["border"],
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(19, 28, 49, 0.7)",
        bordercolor=UI_COLORS["border"],
        borderwidth=1,
    ),
    margin=dict(l=40, r=40, t=50, b=40),
)


def format_sensor_label(sensor_col: str) -> str:
    """Returns formatted sensor name with engineering symbol and units."""
    meta = SENSOR_DESCRIPTIONS.get(sensor_col)
    if meta:
        return f"{meta['symbol']} - {meta['name']} ({meta['units']})"
    return sensor_col


def plot_sensor_telemetry(
    df: pd.DataFrame,
    engine_id: int,
    sensor_col: str,
    highlight_anomalies: bool = True,
    baseline_bounds: Optional[Tuple[float, float]] = None,
    secondary_sensor: Optional[str] = None,
) -> go.Figure:
    """
    Renders time-series telemetry curve for an engine with optional anomaly overlays
    and statistical baseline threshold bounds.
    """
    engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
    if engine_data.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title=f"No telemetry records found for Engine #{engine_id}")
        return fig

    fig = go.Figure()

    # Primary sensor trace
    label = format_sensor_label(sensor_col)
    fig.add_trace(
        go.Scatter(
            x=engine_data["cycle"],
            y=engine_data[sensor_col],
            mode="lines",
            name=label,
            line=dict(color=UI_COLORS["accent_cyan"], width=2.5),
            hovertemplate=(
                f"<b>Engine #{engine_id}</b><br>"
                "Flight Cycle: %{x}<br>"
                f"{sensor_col}: %{{y:.3f}}<extra></extra>"
            ),
        )
    )

    # Optional secondary sensor
    if secondary_sensor and secondary_sensor in engine_data.columns:
        sec_label = format_sensor_label(secondary_sensor)
        fig.add_trace(
            go.Scatter(
                x=engine_data["cycle"],
                y=engine_data[secondary_sensor],
                mode="lines",
                name=sec_label,
                yaxis="y2",
                line=dict(color="#A855F7", width=2, dash="dot"),
                hovertemplate=f"Cycle: %{{x}}<br>{secondary_sensor}: %{{y:.3f}}<extra></extra>",
            )
        )

    # Highlight Anomalous observations
    if highlight_anomalies and "is_anomaly" in engine_data.columns:
        anom = engine_data[engine_data["is_anomaly"] == 1]
        if not anom.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom["cycle"],
                    y=anom[sensor_col],
                    mode="markers",
                    name="Flagged Anomaly",
                    marker=dict(
                        color=UI_COLORS["status_critical"],
                        size=9,
                        symbol="circle-open",
                        line=dict(width=2, color=UI_COLORS["status_critical"]),
                    ),
                    hovertemplate=(
                        "<b>ANOMALY FLAGGED</b><br>"
                        "Cycle: %{x}<br>"
                        f"{sensor_col}: %{{y:.3f}}<extra></extra>"
                    ),
                )
            )

    # Draw Baseline Threshold Lines
    if baseline_bounds:
        low_b, high_b = baseline_bounds
        fig.add_hline(
            y=high_b,
            line=dict(color=UI_COLORS["status_warning"], width=1.5, dash="dash"),
            annotation_text=f"Baseline Upper: {high_b:.2f}",
            annotation_position="top right",
        )
        fig.add_hline(
            y=low_b,
            line=dict(color=UI_COLORS["status_warning"], width=1.5, dash="dash"),
            annotation_text=f"Baseline Lower: {low_b:.2f}",
            annotation_position="bottom right",
        )

    layout_updates = dict(
        DARK_LAYOUT,
        title=dict(
            text=f"Engine #{engine_id} — Telemetry Trend: {label}",
            font=dict(size=14, color=UI_COLORS["text_primary"]),
        ),
        xaxis_title="Operational Flight Cycles",
        yaxis_title=f"{sensor_col} Value",
        hovermode="x unified",
    )

    if secondary_sensor:
        layout_updates["yaxis2"] = dict(
            title=secondary_sensor,
            overlaying="y",
            side="right",
            showgrid=False,
            font=dict(color="#A855F7"),
        )

    fig.update_layout(**layout_updates)
    return fig


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    warn_threshold: float = 0.55,
    crit_threshold: float = 0.70,
) -> go.Figure:
    """
    Renders histogram and density of continuous anomaly scores
    with industrial warning and critical thresholds clearly marked.
    """
    if len(scores) == 0:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No scores available.")
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=40,
            marker=dict(
                color="rgba(6, 182, 212, 0.6)",
                line=dict(color=UI_COLORS["accent_cyan"], width=1),
            ),
            name="Score Distribution",
            hovertemplate="Score Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    # Threshold markers
    fig.add_vline(
        x=warn_threshold,
        line=dict(color=UI_COLORS["status_warning"], width=2, dash="dash"),
        annotation_text=f"Warning ({warn_threshold:.2f})",
        annotation_position="top left",
    )
    fig.add_vline(
        x=crit_threshold,
        line=dict(color=UI_COLORS["status_critical"], width=2, dash="dash"),
        annotation_text=f"Critical ({crit_threshold:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Calibrated Isolation Forest Anomaly Score Distribution",
        xaxis_title="Normalized Anomaly Score (0.0 = Nominal, 1.0 = Severe Anomaly)",
        yaxis_title="Observation Count",
        bargap=0.05,
    )
    return fig


def plot_fleet_health_breakdown(health_counts: Dict[str, int]) -> go.Figure:
    """
    Renders donut chart showing fleet proportion across HEALTHY, DEGRADED, and CRITICAL.
    """
    labels = list(health_counts.keys())
    values = list(health_counts.values())

    color_map = {
        "HEALTHY": UI_COLORS["status_normal"],
        "DEGRADED": UI_COLORS["status_warning"],
        "CRITICAL": UI_COLORS["status_critical"],
    }
    colors = [color_map.get(k, "#64748B") for k in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=UI_COLORS["background"], width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b>: %{value} engines (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Fleet Operational Health Status Breakdown",
        showlegend=True,
    )
    return fig


def plot_confusion_matrix_card(cm_dict: Dict[str, int]) -> go.Figure:
    """
    Renders clean, styled 2x2 confusion matrix heatmap.
    """
    tp = cm_dict.get("tp", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tn = cm_dict.get("tn", 0)

    z = [[tn, fp], [fn, tp]]
    annotations = [
        [f"True Negative (TN)<br><b>{tn:,}</b>", f"False Positive (FP)<br><b>{fp:,}</b>"],
        [f"False Negative (FN)<br><b>{fn:,}</b>", f"True Positive (TP)<br><b>{tp:,}</b>"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Normal (0)", "Predicted Anomaly (1)"],
            y=["Actual Normal (0)", "Actual Anomaly (1)"],
            colorscale=[[0, "#0F172A"], [1, "#0284C7"]],
            showscale=False,
        )
    )

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j,
                y=i,
                text=annotations[i][j],
                showarrow=False,
                font=dict(color="#FFFFFF", size=13),
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Supervised Confusion Matrix",
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_fleet_degradation_heatmap(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    max_engines: int = 25,
) -> go.Figure:
    """
    Renders multi-engine degradation heatmap across flight cycles.
    Visualizes anomaly progression over time.
    """
    engines = df["engine_id"].unique()[:max_engines]
    subset = df[df["engine_id"].isin(engines)]

    pivot = subset.pivot_table(index="engine_id", columns="cycle", values=score_col)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[f"Eng #{e}" for e in pivot.index],
            colorscale=[
                [0.0, "#064E3B"],   # Dark Teal (healthy)
                [0.55, "#D97706"],  # Amber (warning)
                [1.0, "#DC2626"],   # Bright Red (critical)
            ],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Anomaly Score"),
            hovertemplate="Engine: %{y}<br>Cycle: %{x}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Fleet Anomaly Progression Heatmap (Sample of {len(engines)} Engines)",
        xaxis_title="Operational Cycles",
        yaxis_title="Unit Identifier",
    )
    return fig


def plot_sensor_boxplots(df: pd.DataFrame, sensor_cols: List[str]) -> go.Figure:
    """
    Renders comparative boxplots for selected sensor channels to observe spread & outliers.
    """
    fig = go.Figure()
    for col in sensor_cols:
        if col in df.columns:
            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    boxpoints="outliers",
                    marker=dict(size=3),
                )
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Sensor Telemetry Spread & Outlier Distributions",
        yaxis_title="Observed Value",
        showlegend=False,
    )
    return fig
