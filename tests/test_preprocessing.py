"""
Unit tests for src/preprocessing.py.
Verifies zero-variance feature detection, data quality auditing,
leakage prevention, and artifact serialization.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    TelemetryPreprocessor,
    format_data_quality_report_text,
    generate_data_quality_report,
    identify_zero_variance_columns,
)


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Creates synthetic telemetry with both constant and varying sensors."""
    np.random.seed(42)
    n = 100
    data = {
        "engine_id": np.repeat([1, 2], n // 2),
        "cycle": np.tile(np.arange(1, (n // 2) + 1), 2),
        "sensor_1": np.full(n, 518.67),             # Zero variance
        "sensor_2": np.random.normal(642.0, 1.0, n), # Informative
        "sensor_3": np.random.normal(1590.0, 5.0, n),# Informative
        "sensor_5": np.full(n, 14.62),              # Zero variance
        "sensor_7": np.random.normal(553.0, 2.0, n), # Informative
    }
    return pd.DataFrame(data)


def test_identify_zero_variance_columns(sample_telemetry_df: pd.DataFrame):
    """Verifies that constant columns are correctly identified."""
    zero_vars = identify_zero_variance_columns(sample_telemetry_df)
    assert "sensor_1" in zero_vars
    assert "sensor_5" in zero_vars
    assert "sensor_2" not in zero_vars
    assert "sensor_3" not in zero_vars


def test_preprocessor_fit_transform(sample_telemetry_df: pd.DataFrame):
    """Verifies preprocessor drops zero-variance and standardizes features."""
    preprocessor = TelemetryPreprocessor(scaler_type="robust")
    x_scaled, meta = preprocessor.fit_transform(sample_telemetry_df)

    assert preprocessor.is_fitted_
    assert "sensor_1" in preprocessor.dropped_zero_var_features_
    assert "sensor_5" in preprocessor.dropped_zero_var_features_
    assert "sensor_2" in preprocessor.active_features_
    assert x_scaled.shape == (100, 3)  # sensor_2, sensor_3, sensor_7
    assert meta.shape == (100, 2)       # engine_id, cycle


def test_leakage_prevention(sample_telemetry_df: pd.DataFrame):
    """
    CRITICAL ML TEST: Ensures that test set statistics never leak into the preprocessor.
    Fitted parameters must remain strictly unchanged regardless of test data anomalies.
    """
    train_df = sample_telemetry_df.iloc[:60].copy()
    test_df = sample_telemetry_df.iloc[60:].copy()

    # Introduce extreme synthetic anomalies into the test set
    test_df.loc[:, "sensor_2"] = 999999.0

    preprocessor = TelemetryPreprocessor(scaler_type="robust")
    preprocessor.fit(train_df)

    # Record learned median before and after transforming test set
    learned_median_before = preprocessor.feature_stats_["sensor_2"]["median"]
    preprocessor.transform(test_df)
    learned_median_after = preprocessor.feature_stats_["sensor_2"]["median"]

    assert learned_median_before == learned_median_after
    assert learned_median_after < 700.0  # Must reflect train distribution, not test outlier


def test_generate_data_quality_report(sample_telemetry_df: pd.DataFrame):
    """Verifies that data quality report correctly computes metrics and descriptive stats."""
    report = generate_data_quality_report(sample_telemetry_df, dataset_name="Test Telemetry")
    assert report["dataset_name"] == "Test Telemetry"
    assert report["total_rows"] == 100
    assert report["total_cols"] == 7
    assert report["engine_count"] == 2
    assert "sensor_1" in report["zero_variance_channels"]
    assert "sensor_5" in report["zero_variance_channels"]
    assert not report["descriptive_stats"].empty

    # Check descriptive stats columns
    stats_df = report["descriptive_stats"]
    assert "Mean" in stats_df.columns
    assert "Std" in stats_df.columns
    assert "Outliers (IQR)" in stats_df.columns

    # Formatter check
    text = format_data_quality_report_text(report)
    assert "DATA QUALITY REPORT: Test Telemetry" in text
    assert "Monitored Units: 2 engines" in text


def test_save_and_load(tmp_path: Path, sample_telemetry_df: pd.DataFrame):
    """Verifies artifact persistence and round-trip fidelity."""
    preprocessor = TelemetryPreprocessor(scaler_type="standard")
    preprocessor.fit(sample_telemetry_df)

    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(save_path)
    assert save_path.exists()

    loaded = TelemetryPreprocessor.load(save_path)
    assert loaded.is_fitted_
    assert loaded.active_features_ == preprocessor.active_features_
    assert loaded.scaler_type == "standard"


def test_transform_unfitted_raises():
    """Verifies that calling transform() on an unfitted preprocessor raises RuntimeError."""
    preprocessor = TelemetryPreprocessor()
    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(pd.DataFrame({"sensor_1": [1.0]}))
