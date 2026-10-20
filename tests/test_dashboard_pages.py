"""
Integration tests for SentinelAI Streamlit dashboard pages.
Verifies that all 5 subpages and the main entrypoint render cleanly without exceptions.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_app_main_renders():
    """Verifies that the main landing dashboard runs and renders without uncaught exceptions."""
    app_file = str(ROOT_DIR / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert not at.exception, f"App execution failed with exception: {at.exception}"
    assert len(at.markdown) > 0


def test_overview_page_renders():
    """Verifies that 1_Overview.py renders with telemetry health metrics."""
    page_file = str(ROOT_DIR / "pages" / "1_Overview.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Overview execution failed with exception: {at.exception}"
    assert len(at.dataframe) > 0 or len(at.markdown) > 0


def test_sensor_explorer_page_renders():
    """Verifies that 2_Sensor_Explorer.py renders time-series controls and charts."""
    page_file = str(ROOT_DIR / "pages" / "2_Sensor_Explorer.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Sensor Explorer execution failed with exception: {at.exception}"
    assert len(at.selectbox) >= 2


def test_anomaly_center_page_renders():
    """Verifies that 3_Anomaly_Center.py renders triage table and filters."""
    page_file = str(ROOT_DIR / "pages" / "3_Anomaly_Center.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Anomaly Center execution failed with exception: {at.exception}"
    assert len(at.selectbox) >= 3


def test_model_lab_page_renders():
    """Verifies that 4_Model_Lab.py renders comparative benchmark and metrics."""
    page_file = str(ROOT_DIR / "pages" / "4_Model_Lab.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Model Lab execution failed with exception: {at.exception}"
    assert len(at.dataframe) > 0


def test_historical_analysis_page_renders():
    """Verifies that 5_Historical_Analysis.py renders trajectories and export controls."""
    page_file = str(ROOT_DIR / "pages" / "5_Historical_Analysis.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Historical Analysis execution failed with exception: {at.exception}"
    assert len(at.download_button) >= 2


def test_stream_simulator_page_renders():
    """Verifies that 6_Stream_Simulator.py renders controls and dynamic buffer without exceptions."""
    page_file = str(ROOT_DIR / "pages" / "6_Stream_Simulator.py")
    at = AppTest.from_file(page_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Stream Simulator execution failed with exception: {at.exception}"
    assert len(at.button) >= 3
