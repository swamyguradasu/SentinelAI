"""
Unit tests for src/health_analysis.py.
Verifies transparent health rule indexing, consecutive streak accumulation,
and diagnostic explainability summaries.
"""
import numpy as np
import pandas as pd
import pytest

from src.health_analysis import (
    assess_fleet_telemetry,
    assess_observation_health,
    get_latest_engine_health_summaries,
)


def test_assess_observation_health_thresholds():
    """Verifies observation classification into HEALTHY, DEGRADED, and CRITICAL."""
    # Healthy: low score, no triggers
    status, color, act = assess_observation_health(anomaly_score=0.20, baseline_triggers=0, consecutive_high_scores=0)
    assert status == "HEALTHY"
    assert color == "#10B981"

    # Degraded: warning score or single trigger
    status_w, color_w, _ = assess_observation_health(anomaly_score=0.58, baseline_triggers=0, consecutive_high_scores=1)
    assert status_w == "DEGRADED"
    assert color_w == "#F59E0B"

    # Critical: persistent high score (>=3 consecutive) or multiple triggers (>=2)
    status_c1, color_c1, _ = assess_observation_health(anomaly_score=0.75, baseline_triggers=0, consecutive_high_scores=3)
    assert status_c1 == "CRITICAL"
    assert color_c1 == "#EF4444"

    status_c2, color_c2, _ = assess_observation_health(anomaly_score=0.40, baseline_triggers=2, consecutive_high_scores=0)
    assert status_c2 == "CRITICAL"


def test_assess_fleet_telemetry_consecutive_streaks():
    """Verifies that consecutive high anomalies streak correctly counts."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1],
        "cycle": [1, 2, 3, 4],
    })
    # Scores: low, high (warn), high (warn), low
    scores = np.array([0.2, 0.6, 0.65, 0.3])
    audit = pd.DataFrame({
        "trigger_count": [0, 0, 0, 0],
        "triggering_sensors": ["None", "None", "None", "None"],
    })

    assessed = assess_fleet_telemetry(df, scores, audit)
    streaks = list(assessed["consecutive_anomalies"])
    assert streaks == [0, 1, 2, 0]


def test_get_latest_engine_health_summaries():
    """Verifies extraction of latest cycle summary for each engine."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 3],
        "anomaly_score": [0.3, 0.75, 0.2, 0.58],
        "consecutive_anomalies": [0, 3, 0, 1],
        "baseline_trigger_count": [0, 2, 0, 0],
        "triggering_sensors": ["None", "sensor_2 (HIGH)", "None", "None"],
        "health_status": ["HEALTHY", "CRITICAL", "HEALTHY", "DEGRADED"],
        "status_color": ["#10B981", "#EF4444", "#10B981", "#F59E0B"],
        "action_recommendation": ["A", "B", "C", "D"],
    })

    summaries = get_latest_engine_health_summaries(df)
    assert len(summaries) == 2
    assert summaries[1].current_cycle == 2
    assert summaries[1].health_status == "CRITICAL"
    assert summaries[2].current_cycle == 3
    assert summaries[2].health_status == "DEGRADED"
