"""
Anomaly Detection Models for SentinelAI.
Contains:
1. Model A: Isolation Forest with normalized continuous anomaly scoring.
2. Model B: Quantile Threshold Baseline fitted strictly on training data.

SCORE DIRECTION CONVENTION:
A higher `anomaly_score` indicates GREATER abnormality.
- 0.0 represents completely nominal operating behavior.
- 1.0 represents severe statistical divergence.

DOMAIN DISCLAIMER:
Detected anomalies represent statistical telemetry divergence from nominal envelopes;
they do not constitute confirmed mechanical hardware failures, and ground-truth
failure labels are never fabricated.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    INDEX_COLUMNS,
    ISOLATION_FOREST_DEFAULTS,
    MODELS_DIR,
    QUANTILE_BASELINE_DEFAULTS,
    SENSOR_COLUMNS,
)
from src.preprocessing import TelemetryPreprocessor

UNIFIED_SCHEMA_COLUMNS = [
    "engine_id",
    "cycle",
    "anomaly_label",
    "anomaly_score",
    "detection_method",
    "triggering_sensors",
]


@dataclass
class IsolationForestDetector:
    """
    Model A: Multi-dimensional Isolation Forest Anomaly Detector.
    
    SCORE DIRECTION:
    Higher `anomaly_score` values indicate greater abnormality.
    Scores are calibrated into a normalized [0.0, 1.0] interval.
    
    FEATURE INTEGRITY:
    Engine IDs and cycle indices are strictly excluded from feature inputs
    to ensure tree isolation focuses exclusively on sensor telemetry physics.
    """
    n_estimators: int = ISOLATION_FOREST_DEFAULTS["n_estimators"]
    contamination: float = ISOLATION_FOREST_DEFAULTS["contamination"]
    max_features: Union[int, float] = 1.0
    random_state: int = ISOLATION_FOREST_DEFAULTS["random_state"]
    model_: Optional[IsolationForest] = None
    feature_names_: List[str] = field(default_factory=list)
    feature_means_: np.ndarray = field(default_factory=lambda: np.array([]))
    score_min_: float = 0.0
    score_max_: float = 1.0
    is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "IsolationForestDetector":
        """
        Fits the Isolation Forest strictly on training feature array.
        Calibrates continuous score bounds on the training distribution.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit IsolationForestDetector on an empty array.")

        # Guard: verify feature count matches names if provided
        if feature_names is not None:
            # Exclude engine_id or cycle if accidentally passed
            for forbidden in INDEX_COLUMNS + ["rul"]:
                if forbidden in feature_names:
                    raise ValueError(f"Identifier '{forbidden}' must not be used as an anomaly feature.")
            self.feature_names_ = list(feature_names)
            if X.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"Feature dimension mismatch: X has {X.shape[1]} columns, feature_names has {len(self.feature_names_)}"
                )

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X)

        self.feature_means_ = np.mean(X, axis=0)

        # Calibrate continuous anomaly score bounds on training distribution
        # In scikit-learn, score_samples is the negative anomaly score (lower means more anomalous)
        # raw_scores = -score_samples(X) -> higher means more anomalous
        raw_train_scores = -self.model_.score_samples(X)
        self.score_min_ = float(np.min(raw_train_scores))
        self.score_max_ = float(np.max(raw_train_scores))
        if np.isclose(self.score_min_, self.score_max_):
            self.score_max_ = self.score_min_ + 1e-6

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly status and calibrated continuous anomaly scores.
        Returns:
            anomaly_labels: np.ndarray of shape (N,) containing 0 (normal) or 1 (anomaly).
            anomaly_scores: np.ndarray of shape (N,) scaled into [0.0, 1.0] where 1.0 is most anomalous.
        """
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("IsolationForestDetector must be fitted before predict().")

        if len(X) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        if len(self.feature_names_) > 0 and X.shape[1] != len(self.feature_names_):
            raise ValueError(f"Expected {len(self.feature_names_)} features, but received {X.shape[1]}.")

        # sklearn predicts -1 for anomalies and 1 for inliers. Map: -1 -> 1, 1 -> 0.
        raw_preds = self.model_.predict(X)
        binary_anomalies = np.where(raw_preds == -1, 1, 0)

        # Higher score = greater abnormality
        raw_scores = -self.model_.score_samples(X)
        normalized_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-9)
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

        return binary_anomalies, normalized_scores

    def predict_telemetry(
        self,
        df: pd.DataFrame,
        preprocessor: TelemetryPreprocessor,
    ) -> pd.DataFrame:
        """
        Transforms input telemetry via fitted preprocessor and generates
        the standard SentinelAI unified output schema:
        [engine_id, cycle, anomaly_label, anomaly_score, detection_method, triggering_sensors]
        """
        if not self.is_fitted_:
            raise RuntimeError("IsolationForestDetector must be fitted before predict_telemetry().")
        if not preprocessor.is_fitted_:
            raise RuntimeError("TelemetryPreprocessor must be fitted on training data first.")

        if df.empty:
            return pd.DataFrame(columns=UNIFIED_SCHEMA_COLUMNS)

        for col in ["engine_id", "cycle"]:
            if col not in df.columns:
                raise KeyError(f"Input DataFrame must contain identifier column '{col}'.")

        # Scale features using preprocessor (strictly sensor features, excluding engine_id and cycle)
        x_scaled, meta = preprocessor.transform(df)
        labels, scores = self.predict(x_scaled)

        # Approximate top deviating sensor channel per observation
        triggering_channels = []
        if len(self.feature_means_) == x_scaled.shape[1]:
            devs = np.abs(x_scaled - self.feature_means_)
            top_sensor_indices = np.argmax(devs, axis=1)
            for i, idx in enumerate(top_sensor_indices):
                if labels[i] == 1:
                    sensor_name = self.feature_names_[idx] if idx < len(self.feature_names_) else f"sensor_{idx}"
                    triggering_channels.append(f"{sensor_name} (Peak Deviation: {devs[i, idx]:.2f} std)")
                else:
                    triggering_channels.append("None")
        else:
            triggering_channels = ["None" if l == 0 else "Multivariate Deviation" for l in labels]

        out_df = pd.DataFrame({
            "engine_id": df["engine_id"].values,
            "cycle": df["cycle"].values,
            "anomaly_label": labels,
            "anomaly_score": np.round(scores, 4),
            "detection_method": "Isolation Forest",
            "triggering_sensors": triggering_channels,
        })
        return out_df

    def save(self, filepath: Union[Path, str]) -> None:
        """Serializes detector to disk via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(filepath: Union[Path, str]) -> "IsolationForestDetector":
        """Loads serialized detector from disk. Does not require retraining."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, IsolationForestDetector):
            raise TypeError(f"Loaded artifact is {type(obj)}, expected IsolationForestDetector.")
        return obj


@dataclass
class QuantileThresholdDetector:
    """
    Model B: Transparent Statistical Process Baseline.
    Learns per-sensor empirical lower and upper quantile thresholds
    exclusively on training data (e.g. early nominal baseline cycles).
    
    SCORE DIRECTION:
    Higher `anomaly_score` values indicate greater abnormality.
    Continuous score reflects maximum normalized violation severity.
    
    ROOT CAUSE:
    Exposes exact triggering sensors and breach directions (HIGH or LOW).
    """
    lower_quantile: float = QUANTILE_BASELINE_DEFAULTS["lower_quantile"]
    upper_quantile: float = QUANTILE_BASELINE_DEFAULTS["upper_quantile"]
    min_breaching_sensors: int = 1
    thresholds_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)
    feature_stats_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    is_fitted_: bool = False

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        baseline_cycles: Optional[int] = None,
    ) -> "QuantileThresholdDetector":
        """
        Learns lower and upper quantile thresholds per feature on training data only.
        Optionally restricts baseline fitting to initial healthy cycles (e.g. cycle <= baseline_cycles).
        """
        if df_train.empty:
            raise ValueError("Cannot fit QuantileThresholdDetector on empty DataFrame.")

        # Determine feature names, strictly excluding engine_id and cycle
        if feature_names is None:
            self.feature_names_ = [c for c in df_train.columns if c.startswith("sensor_")]
        else:
            self.feature_names_ = [
                c for c in feature_names if c in df_train.columns and c not in INDEX_COLUMNS + ["rul"]
            ]

        if not self.feature_names_:
            raise ValueError("No valid sensor features found to fit QuantileThresholdDetector.")

        # Subset to baseline healthy cycles if requested and available
        fit_df = df_train
        if baseline_cycles is not None and "cycle" in df_train.columns:
            early = df_train[df_train["cycle"] <= baseline_cycles]
            if len(early) >= 30:
                fit_df = early

        self.thresholds_ = {}
        self.feature_stats_ = {}

        for col in self.feature_names_:
            series = fit_df[col].dropna()
            if series.empty or series.nunique() <= 1:
                continue

            low_thresh = float(series.quantile(self.lower_quantile))
            high_thresh = float(series.quantile(self.upper_quantile))
            self.thresholds_[col] = (low_thresh, high_thresh)
            
            iqr = float(series.quantile(0.75) - series.quantile(0.25))
            self.feature_stats_[col] = {
                "median": float(series.median()),
                "iqr": iqr if iqr > 1e-6 else 1.0,
            }

        self.is_fitted_ = True
        return self

    def predict(
        self,
        df_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Flags observations that breach learned thresholds and returns audit DataFrame.
        Returns:
            binary_anomalies: np.ndarray of shape (N,) with 0 (normal) or 1 (anomaly).
            continuous_scores: np.ndarray of shape (N,) maximum normalized violation severity.
            audit_df: pd.DataFrame detailing triggering sensors and violation magnitude.
        """
        if not self.is_fitted_:
            raise RuntimeError("QuantileThresholdDetector must be fitted before predict().")

        if df_features.empty:
            return (
                np.array([], dtype=int),
                np.array([], dtype=float),
                pd.DataFrame(columns=["trigger_count", "triggering_sensors", "severity"]),
            )

        n_rows = len(df_features)
        trigger_counts = np.zeros(n_rows, dtype=int)
        max_severities = np.zeros(n_rows, dtype=float)
        trigger_descriptions: List[str] = ["" for _ in range(n_rows)]

        audit_records = []

        for i, (_, row) in enumerate(df_features.iterrows()):
            breached_channels = []
            row_max_sev = 0.0

            for col in self.feature_names_:
                if col not in self.thresholds_:
                    continue
                low_val, high_val = self.thresholds_[col]
                val = row.get(col, np.nan)
                if pd.isna(val):
                    continue

                iqr = self.feature_stats_[col]["iqr"]

                if val > high_val:
                    sev = (val - high_val) / iqr
                    breached_channels.append(f"{col} (HIGH: {val:.2f} > {high_val:.2f})")
                    if sev > row_max_sev:
                        row_max_sev = sev
                elif val < low_val:
                    sev = (low_val - val) / iqr
                    breached_channels.append(f"{col} (LOW: {val:.2f} < {low_val:.2f})")
                    if sev > row_max_sev:
                        row_max_sev = sev

            num_breaches = len(breached_channels)
            trigger_counts[i] = num_breaches
            # Normalizing severity with a non-linear saturation curve into [0, 1]
            max_severities[i] = float(np.tanh(row_max_sev / 2.0)) if num_breaches > 0 else 0.0
            trigger_descriptions[i] = "; ".join(breached_channels) if breached_channels else "None"

            audit_records.append({
                "trigger_count": num_breaches,
                "triggering_sensors": trigger_descriptions[i],
                "severity": max_severities[i],
            })

        binary_anomalies = np.where(trigger_counts >= self.min_breaching_sensors, 1, 0)
        audit_df = pd.DataFrame(audit_records)

        return binary_anomalies, max_severities, audit_df

    def predict_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates standard SentinelAI unified output schema:
        [engine_id, cycle, anomaly_label, anomaly_score, detection_method, triggering_sensors]
        """
        if not self.is_fitted_:
            raise RuntimeError("QuantileThresholdDetector must be fitted before predict_telemetry().")

        if df.empty:
            return pd.DataFrame(columns=UNIFIED_SCHEMA_COLUMNS)

        for col in ["engine_id", "cycle"]:
            if col not in df.columns:
                raise KeyError(f"Input DataFrame must contain identifier column '{col}'.")

        labels, scores, audit_df = self.predict(df)

        out_df = pd.DataFrame({
            "engine_id": df["engine_id"].values,
            "cycle": df["cycle"].values,
            "anomaly_label": labels,
            "anomaly_score": np.round(scores, 4),
            "detection_method": "Quantile Baseline",
            "triggering_sensors": audit_df["triggering_sensors"].values,
        })
        return out_df

    def save(self, filepath: Union[Path, str]) -> None:
        """Serializes baseline detector to disk via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(filepath: Union[Path, str]) -> "QuantileThresholdDetector":
        """Loads serialized baseline detector from disk. Does not require retraining."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, QuantileThresholdDetector):
            raise TypeError(f"Loaded artifact is {type(obj)}, expected QuantileThresholdDetector.")
        return obj


def train_anomaly_pipeline(
    train_df: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    scaler_type: str = "robust",
    random_state: int = 42,
) -> Tuple[TelemetryPreprocessor, IsolationForestDetector, QuantileThresholdDetector]:
    """
    Fits preprocessing scaler and anomaly models strictly on training data.
    Ensures engine_id and cycle identifiers are not used as features.
    """
    if train_df.empty:
        raise ValueError("Cannot train anomaly pipeline on an empty DataFrame.")

    # 1. Preprocessing strictly on train data
    preprocessor = TelemetryPreprocessor(scaler_type=scaler_type)
    x_train_scaled, meta_train = preprocessor.fit_transform(train_df)

    # 2. Model A: Isolation Forest
    if_model = IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    if_model.fit(x_train_scaled, feature_names=preprocessor.active_features_)

    # 3. Model B: Quantile Threshold Baseline
    base_model = QuantileThresholdDetector(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    base_model.fit(
        train_df,
        feature_names=preprocessor.active_features_,
        baseline_cycles=30,
    )

    return preprocessor, if_model, base_model


def save_anomaly_pipeline(
    output_dir: Union[Path, str],
    preprocessor: TelemetryPreprocessor,
    if_model: IsolationForestDetector,
    base_model: QuantileThresholdDetector,
) -> Dict[str, Path]:
    """
    Serializes complete anomaly pipeline artifacts (preprocessor and models) to disk.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "preprocessor": out_dir / "preprocessor.joblib",
        "isolation_forest": out_dir / "isolation_forest.joblib",
        "quantile_baseline": out_dir / "quantile_baseline.joblib",
    }

    preprocessor.save(paths["preprocessor"])
    if_model.save(paths["isolation_forest"])
    base_model.save(paths["quantile_baseline"])

    return paths


def load_anomaly_pipeline(
    artifact_dir: Union[Path, str],
) -> Dict[str, Any]:
    """
    Loads saved anomaly pipeline artifacts from disk without requiring retraining.
    """
    in_dir = Path(artifact_dir)
    prep_path = in_dir / "preprocessor.joblib"
    if_path = in_dir / "isolation_forest.joblib"
    base_path = in_dir / "quantile_baseline.joblib"

    preprocessor = TelemetryPreprocessor.load(prep_path)
    if_model = IsolationForestDetector.load(if_path)
    base_model = QuantileThresholdDetector.load(base_path)

    return {
        "preprocessor": preprocessor,
        "isolation_forest": if_model,
        "quantile_baseline": base_model,
    }
