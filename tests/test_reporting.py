"""
Unit tests for src/reporting.py.
Verifies telemetry summary calculations, interactive filter queries,
and CSV export serialization.
"""
import io
import pandas as pd
import pytest

from src.reporting import (
    export_dataframe_to_csv_bytes,
    filter_anomaly_records,
    generate_audit_report_markdown,
    generate_fleet_summary_metrics,
)


@pytest.fixture
def mock_assessed_df() -> pd.DataFrame:
    """Creates mock assessed telemetry dataframe."""
    return pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 3],
        "sensor_2": [640.0, 642.0, 641.0, 650.0],
        "anomaly_score": [0.2, 0.75, 0.3, 0.85],
        "health_status": ["HEALTHY", "CRITICAL", "HEALTHY", "CRITICAL"],
        "is_anomaly_if": [0, 1, 0, 1],
        "is_anomaly_base": [0, 0, 0, 1],
        "triggering_sensors": ["None", "sensor_2 (HIGH)", "None", "sensor_3 (HIGH); sensor_2 (HIGH)"],
    })


def test_generate_fleet_summary_metrics(mock_assessed_df: pd.DataFrame):
    """Verifies fleet metric aggregation."""
    metrics = generate_fleet_summary_metrics(mock_assessed_df)
    assert metrics["total_engines"] == 2
    assert metrics["total_observations"] == 4
    assert metrics["min_cycle"] == 2
    assert metrics["max_cycle"] == 3
    assert metrics["active_sensors_count"] == 1


def test_generate_fleet_summary_metrics_empty():
    """Verifies graceful handling of empty DataFrame."""
    metrics = generate_fleet_summary_metrics(pd.DataFrame())
    assert metrics["total_engines"] == 0
    assert metrics["total_observations"] == 0


def test_filter_anomaly_records(mock_assessed_df: pd.DataFrame):
    """Verifies multi-criteria filtering."""
    # Filter by engine
    f1 = filter_anomaly_records(mock_assessed_df, engine_id=1)
    assert len(f1) == 2
    assert (f1["engine_id"] == 1).all()

    # Filter by min score
    f2 = filter_anomaly_records(mock_assessed_df, min_score=0.7)
    assert len(f2) == 2
    assert (f2["anomaly_score"] >= 0.7).all()

    # Filter by detection method
    f3 = filter_anomaly_records(mock_assessed_df, detection_method="Both")
    assert len(f3) == 1
    assert f3.iloc[0]["engine_id"] == 2

    # Filter by search keyword
    f4 = filter_anomaly_records(mock_assessed_df, search_keyword="sensor_3")
    assert len(f4) == 1


def test_export_dataframe_to_csv_bytes(mock_assessed_df: pd.DataFrame):
    """Verifies CSV byte serialization and parsability."""
    csv_bytes = export_dataframe_to_csv_bytes(mock_assessed_df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    # Parse back from bytes
    reloaded = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(reloaded) == 4
    assert list(reloaded["engine_id"]) == [1, 1, 2, 2]


def test_generate_audit_report_markdown():
    """Verifies Markdown generation."""
    md = generate_audit_report_markdown(
        dataset_name="Test Dataset",
        fleet_metrics={"total_engines": 10, "total_observations": 1500},
        health_counts={"HEALTHY": 8, "CRITICAL": 2},
    )
    assert "Test Dataset" in md
    assert "Engines Monitored" in md
    assert "**HEALTHY**: 8 units" in md
