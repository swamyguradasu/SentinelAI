"""
Telemetry Preprocessing and Data Quality Module for SentinelAI.
Implements zero-variance sensor filtering, data quality auditing,
descriptive statistics generation, and leakage-safe scaling strictly on training data.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import DEFAULT_FEATURE_SENSORS, DEFAULT_ZERO_VARIANCE_SENSORS, MODELS_DIR


def identify_zero_variance_columns(
    df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
    variance_threshold: float = 1e-6,
) -> List[str]:
    """
    Identifies numeric columns with variance below a minimum threshold.
    Zero-variance columns convey no degradation signal, corrupt scaling matrices,
    and must be filtered out prior to statistical modeling.
    """
    if candidate_cols is None:
        candidate_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]

    zero_var_cols: List[str] = []
    for col in candidate_cols:
        if col in df.columns:
            val = df[col].dropna()
            if len(val) == 0 or val.nunique() <= 1 or val.var() < variance_threshold:
                zero_var_cols.append(col)
    return zero_var_cols


def generate_data_quality_report(
    df: pd.DataFrame,
    dataset_name: str = "Telemetry Dataset",
) -> Dict[str, Any]:
    """
    Generates a comprehensive data quality audit and descriptive statistics summary.
    
    Examines:
    - Row and column dimensional integrity
    - Number of unique monitored equipment units (engines)
    - Operational cycle bounds and distribution
    - Missing value rates and non-finite numbers (NaN, Inf)
    - Zero-variance / constant channels
    - Distributional skewness and outlier frequency (via IQR envelope)
    - Descriptive statistics table
    """
    if df.empty:
        return {
            "dataset_name": dataset_name,
            "status": "EMPTY",
            "total_rows": 0,
            "total_cols": 0,
            "issues": ["Dataset contains zero records."],
            "descriptive_stats": pd.DataFrame(),
        }

    total_rows = len(df)
    total_cols = df.shape[1]
    issues: List[str] = []

    # 1. Engine & Cycle Integrity
    engine_count = df["engine_id"].nunique() if "engine_id" in df.columns else 0
    cycle_min = int(df["cycle"].min()) if "cycle" in df.columns else 0
    cycle_max = int(df["cycle"].max()) if "cycle" in df.columns else 0
    cycle_mean = float(df.groupby("engine_id")["cycle"].max().mean()) if "cycle" in df.columns and "engine_id" in df.columns else 0.0

    if engine_count == 0:
        issues.append("Missing or empty 'engine_id' identifier.")
    if "cycle" in df.columns and (df["cycle"] <= 0).any():
        issues.append("Invalid non-positive flight cycles detected.")

    # 2. Missing & Non-finite Values
    nan_counts = df.isna().sum()
    missing_dict = nan_counts[nan_counts > 0].to_dict()
    if missing_dict:
        issues.append(f"Missing values detected across {len(missing_dict)} columns.")

    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        infs = int(np.isinf(df[col]).sum())
        if infs > 0:
            inf_counts[col] = infs
    if inf_counts:
        issues.append(f"Infinite numeric values detected: {inf_counts}")

    # 3. Zero-variance channels
    zero_vars = identify_zero_variance_columns(df)
    if zero_vars:
        issues.append(f"Identified {len(zero_vars)} constant/zero-variance channels: {zero_vars}")

    # 4. Descriptive Statistics & Outlier Identification
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
    stats_records = []
    outlier_counts = {}

    for col in sensor_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_counts[col] = outliers

        stats_records.append({
            "Channel": col,
            "Mean": round(float(series.mean()), 4),
            "Std": round(float(series.std()), 4),
            "Min": round(float(series.min()), 4),
            "25%": round(q25, 4),
            "50% (Median)": round(float(series.median()), 4),
            "75%": round(q75, 4),
            "Max": round(float(series.max()), 4),
            "Variance": round(float(series.var()), 6),
            "Skewness": round(float(series.skew()), 3) if not np.isclose(series.var(), 0) else 0.0,
            "Outliers (IQR)": outliers,
        })

    stats_df = pd.DataFrame(stats_records)

    # Determine Overall Quality Status
    if len(missing_dict) > 0 or len(inf_counts) > 0:
        status = "NEEDS_CLEANING"
    elif len(zero_vars) > 0:
        status = "HEALTHY_WITH_CONSTANTS"  # Expected in C-MAPSS FD001
    else:
        status = "HEALTHY"

    return {
        "dataset_name": dataset_name,
        "status": status,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "engine_count": engine_count,
        "cycle_min": cycle_min,
        "cycle_max": cycle_max,
        "avg_cycles_per_engine": round(cycle_mean, 1),
        "zero_variance_channels": zero_vars,
        "missing_values": missing_dict,
        "infinite_values": inf_counts,
        "issues": issues,
        "descriptive_stats": stats_df,
    }


def format_data_quality_report_text(report: Dict[str, Any]) -> str:
    """Formats the data quality audit dictionary into a readable report."""
    lines = [
        f"===========================================================",
        f"  DATA QUALITY REPORT: {report.get('dataset_name', 'Telemetry')}",
        f"  Overall Status: {report.get('status', 'UNKNOWN')}",
        f"===========================================================",
        f"Total Records: {report.get('total_rows', 0):,} rows | {report.get('total_cols', 0)} fields",
        f"Monitored Units: {report.get('engine_count', 0)} engines",
        f"Cycle Span: Min = {report.get('cycle_min', 0)}, Max = {report.get('cycle_max', 0)} (Mean per unit: {report.get('avg_cycles_per_engine', 0)})",
    ]
    if report.get("zero_variance_channels"):
        lines.append(f"Constant/Zero-Variance Channels: {report['zero_variance_channels']}")
    if report.get("missing_values"):
        lines.append(f"Missing Values: {report['missing_values']}")
    if report.get("issues"):
        lines.append("\nAudit Findings:")
        for issue in report["issues"]:
            lines.append(f"  - {issue}")
    else:
        lines.append("\nAudit Findings: No data quality defects detected.")
    lines.append("===========================================================")
    return "\n".join(lines)


@dataclass
class TelemetryPreprocessor:
    """
    Fitted preprocessor for engine sensor telemetry.
    STRICT LEAKAGE CONTROL:
    All transformations (scalers, quantiles, feature selections) are fitted
    EXCLUSIVELY on training data. Test sequences are transformed strictly using
    the parameters learned on the training split.
    """
    scaler_type: str = "robust"  # "robust" or "standard"
    variance_threshold: float = 1e-6
    active_features_: List[str] = field(default_factory=list)
    dropped_zero_var_features_: List[str] = field(default_factory=list)
    scaler_: Optional[Union[RobustScaler, StandardScaler]] = None
    feature_stats_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    is_fitted_: bool = False

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> "TelemetryPreprocessor":
        """
        Fits scaler and identifies informative features strictly on training data.
        """
        if df_train.empty:
            raise ValueError("Cannot fit TelemetryPreprocessor on an empty DataFrame.")

        # Determine initial candidate features
        if feature_columns is None:
            candidate_features = [c for c in df_train.columns if c.startswith("sensor_")]
            if not candidate_features:
                raise ValueError("No sensor columns found in training data.")
        else:
            candidate_features = [c for c in feature_columns if c in df_train.columns]
            if not candidate_features:
                raise ValueError("None of the specified feature columns are present in the training data.")

        # Identify and drop zero-variance channels
        self.dropped_zero_var_features_ = identify_zero_variance_columns(
            df_train,
            candidate_cols=candidate_features,
            variance_threshold=self.variance_threshold,
        )
        self.active_features_ = [
            col for col in candidate_features if col not in self.dropped_zero_var_features_
        ]

        if not self.active_features_:
            raise ValueError(
                "All candidate sensor columns exhibited zero variance. "
                "No informative features remain for modeling."
            )

        # Initialize and fit the chosen scaler strictly on train data
        if self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler_ = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type '{self.scaler_type}'. Choose 'robust' or 'standard'.")

        x_train = df_train[self.active_features_].values
        self.scaler_.fit(x_train)

        # Record training distribution statistics for auditability and validation
        self.feature_stats_ = {}
        for col in self.active_features_:
            series = df_train[col].dropna()
            self.feature_stats_[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
            }

        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transforms input telemetry using parameters fitted on the training split.
        Does NOT refit or modify any learned parameters.
        Returns (scaled_numpy_features, aligned_metadata_dataframe).
        """
        if not self.is_fitted_ or self.scaler_ is None:
            raise RuntimeError("TelemetryPreprocessor must be fitted on training data before calling transform().")

        if df.empty:
            return np.empty((0, len(self.active_features_))), pd.DataFrame()

        # Check for presence of required active features
        missing = [c for c in self.active_features_ if c not in df.columns]
        if missing:
            raise KeyError(f"Input DataFrame is missing required features: {missing}")

        feature_df = df[self.active_features_].copy()
        
        # Impute missing values with learned training median
        if feature_df.isna().any().any():
            for col in self.active_features_:
                med = self.feature_stats_[col]["median"]
                feature_df[col] = feature_df[col].fillna(med)

        x_scaled = self.scaler_.transform(feature_df.values)

        # Preserve index and identifiers
        meta_cols = [c for c in ["engine_id", "cycle", "rul"] if c in df.columns]
        meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

        return x_scaled, meta_df

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fits on training data and returns transformed arrays."""
        self.fit(df_train, feature_columns=feature_columns)
        return self.transform(df_train)

    def save(self, filepath: Union[Path, str]) -> None:
        """Serializes preprocessor artifact to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(filepath: Union[Path, str]) -> "TelemetryPreprocessor":
        """Loads serialized preprocessor artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, TelemetryPreprocessor):
            raise TypeError(f"Loaded object is of type {type(obj)}, expected TelemetryPreprocessor.")
        return obj
