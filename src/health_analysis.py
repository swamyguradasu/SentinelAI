"""
Equipment Health Analysis and Explainability Module for SentinelAI.
Implements rule-based, domain-transparent health indexing (Normal, Degraded, Critical).
Explicitly distinguishes telemetry anomalies from verified mechanical failures,
avoiding invented probabilities or unsubstantiated RUL claims.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import HEALTH_THRESHOLDS


HEALTH_DISCLAIMER = (
    "INDUSTRIAL TELEMETRY NOTICE: SentinelAI's health assessment indexes statistical sensor "
    "divergence and anomaly persistence. A 'Critical' status reflects significant multi-channel "
    "telemetry departure from nominal operating envelopes. It serves as an alert for maintenance "
    "inspection and does NOT guarantee immediate physical hardware failure or exact failure timing."
)


@dataclass
class EngineHealthSummary:
    """Detailed health diagnosis summary for a single equipment unit."""
    engine_id: int
    current_cycle: int
    health_status: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    status_color: str   # Hex color code for UI
    anomaly_streak: int
    latest_score: float
    trigger_summary: str
    action_recommendation: str
    audit_notes: List[str]


def assess_observation_health(
    anomaly_score: float,
    baseline_triggers: int,
    consecutive_high_scores: int = 0,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    """
    Evaluates health status for a single observation based on transparent rules.
    Returns: (health_status, status_color, action_recommendation)
    """
    cfg = thresholds or HEALTH_THRESHOLDS
    warn_score = cfg.get("warning_score", 0.55)
    crit_score = cfg.get("critical_score", 0.70)
    persist_win = cfg.get("persistent_window", 3)
    multi_trig = cfg.get("baseline_multi_trigger", 2)

    if (
        (anomaly_score >= crit_score and consecutive_high_scores >= persist_win)
        or (baseline_triggers >= multi_trig)
    ):
        return (
            "CRITICAL",
            "#EF4444",  # Red
            "Immediate engineering inspection recommended. Multi-channel divergence detected.",
        )
    elif (
        (anomaly_score >= warn_score)
        or (baseline_triggers >= 1)
        or (consecutive_high_scores >= 2)
    ):
        return (
            "DEGRADED",
            "#F59E0B",  # Amber
            "Elevated telemetry drift observed. Schedule diagnostic monitoring and check sensor calibration.",
        )
    else:
        return (
            "HEALTHY",
            "#10B981",  # Teal/Emerald
            "Operating within nominal statistical bounds. Regular telemetry cadence maintained.",
        )


def assess_fleet_telemetry(
    df: pd.DataFrame,
    if_scores: np.ndarray,
    baseline_audit: pd.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Computes health statuses, anomaly streaks, and explainability summaries
    across the entire telemetry DataFrame.
    """
    if df.empty:
        return df.copy()

    df_out = df.copy()
    df_out["anomaly_score"] = if_scores
    df_out["baseline_trigger_count"] = baseline_audit["trigger_count"].values
    df_out["triggering_sensors"] = baseline_audit["triggering_sensors"].values

    cfg = thresholds or HEALTH_THRESHOLDS
    warn_score = cfg.get("warning_score", 0.55)

    # Compute consecutive high anomaly streak per engine
    df_out = df_out.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)
    df_out["is_above_warn"] = (df_out["anomaly_score"] >= warn_score).astype(int)

    # Rolling sum of consecutive 1s per engine
    def calc_streaks(s: pd.Series) -> pd.Series:
        streak = []
        curr = 0
        for val in s:
            if val == 1:
                curr += 1
            else:
                curr = 0
            streak.append(curr)
        return pd.Series(streak, index=s.index)

    df_out["consecutive_anomalies"] = (
        df_out.groupby("engine_id")["is_above_warn"]
        .transform(calc_streaks)
    )

    statuses = []
    colors = []
    actions = []

    for _, row in df_out.iterrows():
        st, col, act = assess_observation_health(
            anomaly_score=row["anomaly_score"],
            baseline_triggers=int(row["baseline_trigger_count"]),
            consecutive_high_scores=int(row["consecutive_anomalies"]),
            thresholds=cfg,
        )
        statuses.append(st)
        colors.append(col)
        actions.append(act)

    df_out["health_status"] = statuses
    df_out["status_color"] = colors
    df_out["action_recommendation"] = actions

    return df_out


def get_latest_engine_health_summaries(
    df_assessed: pd.DataFrame,
) -> Dict[int, EngineHealthSummary]:
    """
    Extracts the latest health status and diagnostic reasoning for each engine.
    """
    if df_assessed.empty:
        return {}

    # Get row with maximum cycle per engine
    idx_latest = df_assessed.groupby("engine_id")["cycle"].idxmax()
    latest_rows = df_assessed.loc[idx_latest]

    summaries = {}
    for _, row in latest_rows.iterrows():
        eng_id = int(row["engine_id"])
        cyc = int(row["cycle"])
        st = str(row["health_status"])
        color = str(row["status_color"])
        streak = int(row["consecutive_anomalies"])
        score = float(row["anomaly_score"])
        trig = str(row["triggering_sensors"])
        rec = str(row["action_recommendation"])

        audit_notes = [
            f"Evaluated at flight cycle {cyc}.",
            f"Current calibrated Isolation Forest anomaly score: {score:.3f}.",
            f"Consecutive high-anomaly cycles: {streak}.",
            f"Baseline statistical breaches: {trig}.",
        ]

        summaries[eng_id] = EngineHealthSummary(
            engine_id=eng_id,
            current_cycle=cyc,
            health_status=st,
            status_color=color,
            anomaly_streak=streak,
            latest_score=score,
            trigger_summary=trig,
            action_recommendation=rec,
            audit_notes=audit_notes,
        )

    return summaries
