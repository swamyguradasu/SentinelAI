"""
Unit and integration tests for SentinelAI Sensor Stream Simulator module.
Verifies cycle ordering, synthetic injection, state reset, and pipeline integration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.anomaly_detection import train_anomaly_pipeline
from src.simulator import SensorStreamSimulator, SIMULATOR_DISCLAIMER

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def fitted_pipeline_and_data():
    """Provides a synthetic telemetry dataset and fitted pipeline for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "engine_id": np.repeat([1, 2], n // 2),
        "cycle": np.tile(np.arange(1, (n // 2) + 1), 2),
        "sensor_1": np.full(n, 518.67),             # Zero variance
        "sensor_2": np.random.normal(642.0, 1.0, n), # Active channel
        "sensor_3": np.random.normal(1590.0, 5.0, n),# Active channel
        "sensor_7": np.random.normal(553.0, 1.5, n), # Active channel
        "sensor_11": np.random.normal(47.3, 0.2, n),
    })
    preprocessor, if_model, base_model = train_anomaly_pipeline(df, contamination=0.05, random_state=42)
    return df, preprocessor, if_model, base_model


def test_simulator_historical_replay_cycle_order(fitted_pipeline_and_data):
    """Verifies that historical replay strictly maintains chronological cycle order."""
    df, preprocessor, if_model, base_model = fitted_pipeline_and_data
    sim = SensorStreamSimulator(mode="HISTORICAL_REPLAY", engine_id=1, historical_df=df)

    assert sim.total_available_cycles > 0
    assert not sim.is_finished

    cycles_seen = []
    for _ in range(5):
        res = sim.step(preprocessor, if_model, base_model)
        assert res is not None
        cycles_seen.append(res.cycle)

    assert cycles_seen == [1, 2, 3, 4, 5], "Cycles must be sequential and start at cycle 1."
    assert sim.current_step == 5
    assert len(sim.history) == 5


def test_simulator_synthetic_anomaly_injection(fitted_pipeline_and_data):
    """Verifies synthetic stream injects anomaly flags when passing the threshold cycle."""
    df, preprocessor, if_model, base_model = fitted_pipeline_and_data
    sim = SensorStreamSimulator(
        mode="SYNTHETIC_INJECTION",
        engine_id=999,
        noise_level=0.01,
        anomaly_injection_cycle=4,
        target_sensor="sensor_2",
        anomaly_shift=15.0,  # Large enough shift to trigger alert
    )

    results = []
    for _ in range(6):
        res = sim.step(preprocessor, if_model, base_model)
        results.append(res)

    # First 3 cycles should be normal
    assert not results[0].is_injected_anomaly
    assert not results[1].is_injected_anomaly
    assert not results[2].is_injected_anomaly

    # Cycles 4+ should have injection flag True
    assert results[3].is_injected_anomaly
    assert results[4].is_injected_anomaly
    assert results[5].is_injected_anomaly


def test_simulator_reset_behavior(fitted_pipeline_and_data):
    """Verifies that reset clears step count, streak, and history list."""
    df, preprocessor, if_model, base_model = fitted_pipeline_and_data
    sim = SensorStreamSimulator(mode="HISTORICAL_REPLAY", engine_id=1, historical_df=df)

    sim.step(preprocessor, if_model, base_model)
    sim.step(preprocessor, if_model, base_model)
    assert sim.current_step == 2
    assert len(sim.history) == 2

    sim.reset()
    assert sim.current_step == 0
    assert len(sim.history) == 0
    assert sim._consecutive_streak == 0
    assert not sim.is_running


def test_simulator_frozen_models_no_retraining(fitted_pipeline_and_data):
    """Verifies that running stream steps does not modify model parameters or retrain."""
    df, preprocessor, if_model, base_model = fitted_pipeline_and_data
    sim = SensorStreamSimulator(mode="HISTORICAL_REPLAY", engine_id=1, historical_df=df)

    # Record initial model states
    init_estimator_count = len(if_model.model_.estimators_)
    init_base_thresholds = dict(base_model.thresholds_)

    for _ in range(3):
        sim.step(preprocessor, if_model, base_model)

    assert len(if_model.model_.estimators_) == init_estimator_count
    assert base_model.thresholds_ == init_base_thresholds


def test_simulator_summary_metrics_and_dataframe(fitted_pipeline_and_data):
    """Verifies that summary metrics and dataframe representations are accurate."""
    df, preprocessor, if_model, base_model = fitted_pipeline_and_data
    sim = SensorStreamSimulator(mode="HISTORICAL_REPLAY", engine_id=1, historical_df=df)

    sim.step(preprocessor, if_model, base_model)
    sim.step(preprocessor, if_model, base_model)

    summary = sim.get_summary_metrics()
    assert summary["cycles_processed"] == 2
    assert "current_health" in summary
    assert "anomaly_rate" in summary

    hist_df = sim.get_history_df()
    assert isinstance(hist_df, pd.DataFrame)
    assert len(hist_df) == 2
    assert "cycle" in hist_df.columns
    assert "anomaly_score" in hist_df.columns


def test_stream_simulator_page_renders():
    """Verifies that pages/6_Stream_Simulator.py renders without exceptions."""
    page_file = str(ROOT_DIR / "pages" / "6_Stream_Simulator.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Simulator page failed with exception: {at.exception}"
    assert len(at.radio) >= 1
    assert len(at.button) >= 3
