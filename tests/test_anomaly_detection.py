"""
Unit tests for src/anomaly_detection.py.
Verifies Isolation Forest and Quantile Baseline models, score calibration,
normal observations, injected unusual values, missing columns,
consistent output schemas, and model persistence without retraining.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    IsolationForestDetector,
    QuantileThresholdDetector,
    load_anomaly_pipeline,
    save_anomaly_pipeline,
    train_anomaly_pipeline,
    UNIFIED_SCHEMA_COLUMNS,
)
from src.preprocessing import TelemetryPreprocessor


@pytest.fixture
def synthetic_train_telemetry() -> pd.DataFrame:
    """Generates synthetic multi-engine training telemetry."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "engine_id": np.repeat([1, 2], n // 2),
        "cycle": np.tile(np.arange(1, (n // 2) + 1), 2),
        "sensor_1": np.full(n, 518.67),             # Zero variance
        "sensor_2": np.random.normal(642.0, 1.0, n), # Active channel
        "sensor_3": np.random.normal(1590.0, 5.0, n),# Active channel
        "sensor_7": np.random.normal(553.0, 1.5, n), # Active channel
    })


def test_isolation_forest_prediction_and_scoring(synthetic_train_telemetry: pd.DataFrame):
    """Verifies that Isolation Forest outputs binary labels and [0, 1] calibrated scores."""
    preprocessor = TelemetryPreprocessor(scaler_type="robust").fit(synthetic_train_telemetry)
    x_train, _ = preprocessor.transform(synthetic_train_telemetry)

    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(x_train, feature_names=preprocessor.active_features_)

    # 1. Normal observation (at the center of the training distribution)
    normal_obs = pd.DataFrame({
        "engine_id": [1],
        "cycle": [50],
        "sensor_1": [518.67],
        "sensor_2": [642.0],
        "sensor_3": [1590.0],
        "sensor_7": [553.0],
    })
    x_normal, _ = preprocessor.transform(normal_obs)
    pred_normal, score_normal = detector.predict(x_normal)

    assert pred_normal[0] == 0
    assert score_normal[0] < 0.5  # Normal observations have low anomaly score

    # 2. Injected extreme unusual observation (anomalous spike)
    anom_obs = pd.DataFrame({
        "engine_id": [1],
        "cycle": [99],
        "sensor_1": [518.67],
        "sensor_2": [750.0],   # Huge spike: +108 std
        "sensor_3": [1800.0],  # Huge spike: +42 std
        "sensor_7": [400.0],   # Severe drop
    })
    x_anom, _ = preprocessor.transform(anom_obs)
    pred_anom, score_anom = detector.predict(x_anom)

    assert pred_anom[0] == 1
    assert score_anom[0] > 0.7  # Injected anomaly must have high score
    assert score_anom[0] > score_normal[0]  # Higher score indicates greater abnormality


def test_isolation_forest_unified_schema(synthetic_train_telemetry: pd.DataFrame):
    """Verifies predict_telemetry produces the exact standard SentinelAI unified schema."""
    preprocessor, if_model, _ = train_anomaly_pipeline(
        synthetic_train_telemetry,
        contamination=0.05,
        n_estimators=50,
        random_state=42,
    )

    out_df = if_model.predict_telemetry(synthetic_train_telemetry, preprocessor)

    assert list(out_df.columns) == UNIFIED_SCHEMA_COLUMNS
    assert len(out_df) == len(synthetic_train_telemetry)
    assert out_df["detection_method"].iloc[0] == "Isolation Forest"
    assert set(out_df["anomaly_label"].unique()).issubset({0, 1})
    assert (out_df["anomaly_score"] >= 0.0).all() and (out_df["anomaly_score"] <= 1.0).all()


def test_quantile_baseline_triggers_and_schema(synthetic_train_telemetry: pd.DataFrame):
    """Verifies baseline threshold fitting, triggering sensors explanation, and unified schema."""
    _, _, base_model = train_anomaly_pipeline(
        synthetic_train_telemetry,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )

    assert base_model.is_fitted_
    assert "sensor_2" in base_model.thresholds_
    low_2, high_2 = base_model.thresholds_["sensor_2"]
    assert low_2 < 642.0 < high_2

    # Injected test data: 1 normal, 1 high on sensor_2, 1 low on sensor_7
    test_df = pd.DataFrame({
        "engine_id": [1, 1, 2],
        "cycle": [1, 2, 3],
        "sensor_1": [518.67, 518.67, 518.67],
        "sensor_2": [642.0, 700.0, 642.0],  # row 1 breaches high
        "sensor_3": [1590.0, 1590.0, 1590.0],
        "sensor_7": [553.0, 553.0, 500.0],  # row 2 breaches low
    })

    out_df = base_model.predict_telemetry(test_df)

    assert list(out_df.columns) == UNIFIED_SCHEMA_COLUMNS
    assert out_df["detection_method"].iloc[0] == "Quantile Baseline"
    
    # Row 0: Normal
    assert out_df.loc[0, "anomaly_label"] == 0
    assert out_df.loc[0, "triggering_sensors"] == "None"

    # Row 1: High trigger on sensor_2
    assert out_df.loc[1, "anomaly_label"] == 1
    assert "sensor_2 (HIGH" in out_df.loc[1, "triggering_sensors"]
    assert out_df.loc[1, "anomaly_score"] > 0.0

    # Row 2: Low trigger on sensor_7
    assert out_df.loc[2, "anomaly_label"] == 1
    assert "sensor_7 (LOW" in out_df.loc[2, "triggering_sensors"]


def test_missing_columns_error_handling(synthetic_train_telemetry: pd.DataFrame):
    """Verifies that missing required identifiers or features raise clear KeyErrors."""
    preprocessor, if_model, base_model = train_anomaly_pipeline(synthetic_train_telemetry)

    # Missing engine_id
    bad_df = pd.DataFrame({"cycle": [1], "sensor_2": [642.0], "sensor_3": [1590.0], "sensor_7": [553.0]})
    with pytest.raises(KeyError, match="engine_id"):
        if_model.predict_telemetry(bad_df, preprocessor)

    with pytest.raises(KeyError, match="engine_id"):
        base_model.predict_telemetry(bad_df)


def test_model_persistence_without_retraining(tmp_path: Path, synthetic_train_telemetry: pd.DataFrame):
    """
    CRITICAL ML TEST:
    Verifies that saving pipeline to disk via joblib and reloading produces identical
    predictions WITHOUT requiring retraining.
    """
    preprocessor, if_model, base_model = train_anomaly_pipeline(
        synthetic_train_telemetry,
        contamination=0.08,
        n_estimators=40,
        random_state=123,
    )

    # Generate initial predictions
    if_preds_orig = if_model.predict_telemetry(synthetic_train_telemetry, preprocessor)
    base_preds_orig = base_model.predict_telemetry(synthetic_train_telemetry)

    # Save complete pipeline
    save_dir = tmp_path / "models_cache"
    saved_paths = save_anomaly_pipeline(save_dir, preprocessor, if_model, base_model)
    assert saved_paths["isolation_forest"].exists()
    assert saved_paths["quantile_baseline"].exists()

    # Load back from disk
    loaded = load_anomaly_pipeline(save_dir)
    loaded_prep = loaded["preprocessor"]
    loaded_if = loaded["isolation_forest"]
    loaded_base = loaded["quantile_baseline"]

    # Verify loaded state: fitted without retraining
    assert loaded_prep.is_fitted_
    assert loaded_if.is_fitted_
    assert loaded_base.is_fitted_

    # Verify identical prediction output
    if_preds_loaded = loaded_if.predict_telemetry(synthetic_train_telemetry, loaded_prep)
    base_preds_loaded = loaded_base.predict_telemetry(synthetic_train_telemetry)

    pd.testing.assert_frame_equal(if_preds_orig, if_preds_loaded)
    pd.testing.assert_frame_equal(base_preds_orig, base_preds_loaded)


def test_reproducible_random_seed(synthetic_train_telemetry: pd.DataFrame):
    """Verifies that fixing random_state produces identical Isolation Forest models."""
    p1, m1, _ = train_anomaly_pipeline(synthetic_train_telemetry, random_state=999)
    p2, m2, _ = train_anomaly_pipeline(synthetic_train_telemetry, random_state=999)

    out1 = m1.predict_telemetry(synthetic_train_telemetry, p1)
    out2 = m2.predict_telemetry(synthetic_train_telemetry, p2)

    np.testing.assert_array_equal(out1["anomaly_label"].values, out2["anomaly_label"].values)
    np.testing.assert_allclose(out1["anomaly_score"].values, out2["anomaly_score"].values)
