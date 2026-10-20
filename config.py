"""
Configuration module for SentinelAI - Predictive Maintenance & Sensor Anomaly Detection.
Defines paths, NASA C-MAPSS dataset column schemas, sensor descriptions,
and default modeling hyperparameters.
"""
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"
RESULTS_DIR: Path = BASE_DIR / "results"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# NASA C-MAPSS FD001 Dataset Specification
# -----------------------------------------------------------------------------
INDEX_COLUMNS: List[str] = ["engine_id", "cycle"]
OP_SETTING_COLUMNS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS: List[str] = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS

# Documented Sensor Descriptions and Engineering Units
SENSOR_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "sensor_1": {"symbol": "T2", "name": "Total temperature at fan inlet", "units": "°R"},
    "sensor_2": {"symbol": "T24", "name": "Total temperature at LPC outlet", "units": "°R"},
    "sensor_3": {"symbol": "T30", "name": "Total temperature at HPC outlet", "units": "°R"},
    "sensor_4": {"symbol": "T50", "name": "Total temperature at LPT outlet", "units": "°R"},
    "sensor_5": {"symbol": "P2", "name": "Pressure at fan inlet", "units": "psia"},
    "sensor_6": {"symbol": "P15", "name": "Total pressure in bypass-duct", "units": "psia"},
    "sensor_7": {"symbol": "P30", "name": "Total pressure at HPC outlet", "units": "psia"},
    "sensor_8": {"symbol": "Nf", "name": "Physical fan speed", "units": "rpm"},
    "sensor_9": {"symbol": "Nc", "name": "Physical core speed", "units": "rpm"},
    "sensor_10": {"symbol": "epr", "name": "Engine pressure ratio (P50/P2)", "units": "--"},
    "sensor_11": {"symbol": "Ps30", "name": "Static pressure at HPC outlet", "units": "psia"},
    "sensor_12": {"symbol": "Phi", "name": "Ratio of fuel flow to Ps30", "units": "pps/psi"},
    "sensor_13": {"symbol": "NRf", "name": "Corrected fan speed", "units": "rpm"},
    "sensor_14": {"symbol": "NRc", "name": "Corrected core speed", "units": "rpm"},
    "sensor_15": {"symbol": "BPR", "name": "Bypass Ratio", "units": "--"},
    "sensor_16": {"symbol": "farB", "name": "Burner fuel-air ratio", "units": "--"},
    "sensor_17": {"symbol": "htBleed", "name": "Bleed Enthalpy", "units": "--"},
    "sensor_18": {"symbol": "Nf_dmd", "name": "Demanded fan speed", "units": "rpm"},
    "sensor_19": {"symbol": "PCNfR_dmd", "name": "Demanded corrected fan speed", "units": "rpm"},
    "sensor_20": {"symbol": "W31", "name": "HPT coolant bleed", "units": "lbm/s"},
    "sensor_21": {"symbol": "W32", "name": "LPT coolant bleed", "units": "lbm/s"},
}

# Sensors known to have near-zero or strictly zero variance in FD001 (sea-level test)
DEFAULT_ZERO_VARIANCE_SENSORS: List[str] = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Primary informative degradation sensors in FD001
DEFAULT_FEATURE_SENSORS: List[str] = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

# -----------------------------------------------------------------------------
# Default Anomaly Model Configurations
# -----------------------------------------------------------------------------
ISOLATION_FOREST_DEFAULTS = {
    "n_estimators": 100,
    "contamination": 0.05,
    "max_samples": "auto",
    "random_state": 42,
}

QUANTILE_BASELINE_DEFAULTS = {
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "baseline_cycle_limit": 30,  # Cycles considered initial healthy baseline
}

# -----------------------------------------------------------------------------
# Industrial Health Classification Thresholds
# -----------------------------------------------------------------------------
HEALTH_THRESHOLDS = {
    "warning_score": 0.55,       # Isolation Forest normalized anomaly score threshold for warning
    "critical_score": 0.70,      # Isolation Forest normalized score threshold for critical
    "persistent_window": 3,      # Consecutive cycles above warning threshold to trigger alert
    "baseline_multi_trigger": 2, # Number of simultaneous baseline sensor violations for critical
}

# -----------------------------------------------------------------------------
# UI Palette
# -----------------------------------------------------------------------------
UI_COLORS = {
    "background": "#0B1120",
    "card_bg": "#131C31",
    "sidebar_bg": "#080D1A",
    "accent_cyan": "#06B6D4",
    "accent_teal": "#14B8A6",
    "status_normal": "#10B981",    # Teal/Emerald
    "status_warning": "#F59E0B",   # Amber
    "status_critical": "#EF4444",  # Red
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "border": "#1E293B",
}
