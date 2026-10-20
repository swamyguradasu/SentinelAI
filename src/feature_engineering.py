"""
Feature Engineering Module for SentinelAI.
Computes domain-specific rolling statistics, degradation trends,
and sensor rate-of-change per engine without crossing engine boundaries.
"""
from typing import List, Optional
import numpy as np
import pandas as pd


def compute_rolling_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    window_size: int = 5,
) -> pd.DataFrame:
    """
    Computes rolling mean and rolling standard deviation per engine.
    CRITICAL: Resets at each engine boundary to eliminate inter-engine leakage.
    """
    if df.empty:
        return df.copy()

    if "engine_id" not in df.columns:
        raise KeyError("DataFrame must contain 'engine_id' to compute grouped rolling features.")

    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    df_out = df.copy()

    # Sort strictly by engine_id and cycle
    if "cycle" in df_out.columns:
        df_out = df_out.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)

    grouped = df_out.groupby("engine_id")

    for col in sensor_cols:
        if col in df_out.columns:
            roll_mean = grouped[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
            df_out[f"{col}_roll_mean"] = roll_mean

            roll_std = grouped[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            df_out[f"{col}_roll_std"] = roll_std

    return df_out


def compute_sensor_deltas(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    delta_period: int = 1,
) -> pd.DataFrame:
    """
    Calculates step-to-step rate of change (first-order difference) per engine.
    CRITICAL: Diff is isolated per engine and does not subtract across engine boundaries.
    """
    if df.empty:
        return df.copy()

    if "engine_id" not in df.columns:
        raise KeyError("DataFrame must contain 'engine_id' to compute grouped deltas.")

    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    df_out = df.copy()
    grouped = df_out.groupby("engine_id")

    for col in sensor_cols:
        if col in df_out.columns:
            diff = grouped[col].diff(periods=delta_period).reset_index(level=0, drop=True).fillna(0.0)
            df_out[f"{col}_delta"] = diff

    return df_out


def compute_baseline_deviation(
    df: pd.DataFrame,
    baseline_cycles: int = 20,
    sensor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculates the deviation of each observation from that specific engine's
    initial nominal operating state (average of first N cycles).
    """
    if df.empty:
        return df.copy()

    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise KeyError("DataFrame must contain 'engine_id' and 'cycle' to compute baseline deviations.")

    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    df_out = df.copy()

    # Compute mean for cycle <= baseline_cycles for each engine
    early_data = df_out[df_out["cycle"] <= baseline_cycles]
    baseline_means = early_data.groupby("engine_id")[sensor_cols].mean().reset_index()

    rename_dict = {col: f"{col}_base_mean" for col in sensor_cols}
    baseline_means = baseline_means.rename(columns=rename_dict)

    df_out = df_out.merge(baseline_means, on="engine_id", how="left")

    for col in sensor_cols:
        base_col = f"{col}_base_mean"
        if base_col in df_out.columns:
            df_out[base_col] = df_out[base_col].fillna(df_out[col].mean())
            df_out[f"{col}_dev"] = df_out[col] - df_out[base_col]
            df_out.drop(columns=[base_col], inplace=True)

    return df_out


def compute_cumulative_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative flight cycles and normalized cycle progression per engine.
    """
    if df.empty or "engine_id" not in df.columns or "cycle" not in df.columns:
        return df.copy()

    df_out = df.copy()
    max_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    df_out["cycle_progression_ratio"] = df_out["cycle"] / max_cycles
    return df_out


def engineer_all_telemetry_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    window_size: int = 5,
    baseline_cycles: int = 20,
) -> pd.DataFrame:
    """
    Executes complete feature engineering workflow while strictly preserving engine isolation.
    """
    df_feat = compute_rolling_features(df, sensor_cols=sensor_cols, window_size=window_size)
    df_feat = compute_sensor_deltas(df_feat, sensor_cols=sensor_cols, delta_period=1)
    df_feat = compute_baseline_deviation(df_feat, baseline_cycles=baseline_cycles, sensor_cols=sensor_cols)
    return df_feat
