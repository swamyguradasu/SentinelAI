"""
Unit tests for src/feature_engineering.py.
Verifies rolling statistics, deltas, baseline deviation, and cumulative cycle features
across engine boundaries.
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    compute_baseline_deviation,
    compute_cumulative_cycle_features,
    compute_rolling_features,
    compute_sensor_deltas,
    engineer_all_telemetry_features,
)


@pytest.fixture
def multi_engine_df() -> pd.DataFrame:
    """Creates deterministic 2-engine time-series data."""
    return pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "sensor_2": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })


def test_compute_rolling_features_engine_isolation(multi_engine_df: pd.DataFrame):
    """
    CRITICAL TIME-SERIES TEST:
    Verifies that rolling mean resets at engine boundaries and does not leak from Engine 1 to Engine 2.
    """
    res = compute_rolling_features(multi_engine_df, sensor_cols=["sensor_2"], window_size=2)
    roll_means = list(res["sensor_2_roll_mean"])
    assert roll_means[0] == 10.0
    assert roll_means[1] == 15.0
    assert roll_means[3] == 100.0  # Reset for engine 2


def test_compute_sensor_deltas_engine_isolation(multi_engine_df: pd.DataFrame):
    """Verifies that first-order difference resets at engine boundaries."""
    res = compute_sensor_deltas(multi_engine_df, sensor_cols=["sensor_2"], delta_period=1)
    deltas = list(res["sensor_2_delta"])
    assert deltas[0] == 0.0  # Initial cycle
    assert deltas[1] == 10.0 # 20 - 10
    assert deltas[3] == 0.0  # Reset for engine 2 cycle 1 (MUST NOT be 100 - 30 = 70!)


def test_compute_baseline_deviation(multi_engine_df: pd.DataFrame):
    """Verifies deviation from initial baseline cycles."""
    res = compute_baseline_deviation(multi_engine_df, baseline_cycles=1, sensor_cols=["sensor_2"])
    devs = list(res["sensor_2_dev"])
    assert devs[0] == 0.0
    assert devs[1] == 10.0
    assert devs[2] == 20.0
    assert devs[3] == 0.0
    assert devs[4] == 100.0
    assert devs[5] == 200.0


def test_compute_cumulative_cycle_features(multi_engine_df: pd.DataFrame):
    """Verifies normalized cycle progression ratio computation."""
    res = compute_cumulative_cycle_features(multi_engine_df)
    assert "cycle_progression_ratio" in res.columns
    # Max cycle for both engines is 3
    assert list(res["cycle_progression_ratio"]) == [1/3, 2/3, 1.0, 1/3, 2/3, 1.0]


def test_engineer_all_telemetry_features(multi_engine_df: pd.DataFrame):
    """Verifies master pipeline executes all feature steps."""
    res = engineer_all_telemetry_features(multi_engine_df, sensor_cols=["sensor_2"], window_size=2, baseline_cycles=1)
    assert "sensor_2_roll_mean" in res.columns
    assert "sensor_2_roll_std" in res.columns
    assert "sensor_2_delta" in res.columns
    assert "sensor_2_dev" in res.columns
