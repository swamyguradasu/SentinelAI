"""
SentinelAI - Pipeline Service & Session State Management.
Provides cached ingestion, model fitting, and telemetry assessment
that can be safely imported and executed by any dashboard page.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from config import RAW_DATA_DIR, SAMPLE_DATA_DIR
from src.anomaly_detection import train_anomaly_pipeline
from src.data_loader import (
    generate_realistic_cmapss_sample,
    load_train_and_test_datasets,
    validate_telemetry_schema,
)
from src.health_analysis import assess_fleet_telemetry


@st.cache_data(show_spinner="Ingesting NASA Telemetry Engine Data...")
def load_telemetry_dataset(dataset_choice: str, uploaded_file=None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str], str]:
    """
    Loads telemetry according to user selection without mixing partitions.
    Streamlit caching avoids repeated reading of large raw text files.
    """
    train_df, test_df, rul_df = None, None, None
    source_tag = "real_nasa"
    error_msg = None

    if dataset_choice == "Custom Upload (CSV)" and uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            valid, errors = validate_telemetry_schema(raw_df)
            if not valid:
                return None, None, None, f"Schema validation failed: {'; '.join(errors)}", "custom_error"
            source_tag = "custom_uploaded"
            train_df = raw_df
            test_df = raw_df
        except Exception as e:
            return None, None, None, f"CSV parsing error: {e}", "custom_error"

    elif dataset_choice == "NASA Official C-MAPSS FD001 (Real Historical Data)":
        try:
            datasets = load_train_and_test_datasets(RAW_DATA_DIR, dataset_name="FD001")
            train_df = datasets["train"]
            test_df = datasets["test"]
            rul_df = datasets["rul"]
            source_tag = "official_real"
        except FileNotFoundError:
            return None, None, None, "Official NASA files not found in data/raw/. Please use the synthetic benchmark.", "missing_data"

    else:  # Synthetic Demonstration Benchmark
        sample_train = SAMPLE_DATA_DIR / "train_FD001.txt"
        if not sample_train.exists():
            generate_realistic_cmapss_sample(SAMPLE_DATA_DIR)
        datasets = load_train_and_test_datasets(SAMPLE_DATA_DIR, dataset_name="FD001")
        train_df = datasets["train"]
        test_df = datasets["test"]
        rul_df = datasets["rul"]
        source_tag = "simulated_benchmark"

    return train_df, test_df, rul_df, error_msg, source_tag


@st.cache_resource(show_spinner="Training Anomaly Detection Pipeline...")
def get_trained_models(
    _train_df: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
):
    """
    Fits preprocessing scaler and anomaly models strictly on training data.
    Ensures Isolation Forest and Quantile Baseline are never retrained on UI reruns.
    """
    preprocessor, if_model, base_model = train_anomaly_pipeline(
        train_df=_train_df,
        contamination=contamination,
        n_estimators=n_estimators,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        random_state=42,
    )
    training_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return preprocessor, if_model, base_model, training_time


@st.cache_data(show_spinner="Evaluating Telemetry & Scoring Health...")
def process_and_assess_telemetry(
    _eval_df: pd.DataFrame,
    _preprocessor,
    _if_model,
    _base_model,
):
    """
    Applies fitted pipeline to telemetry data and computes health classifications.
    Uses frozen training scaler and parameters to prevent data leakage.
    """
    x_scaled, _ = _preprocessor.transform(_eval_df)
    if_preds, if_scores = _if_model.predict(x_scaled)
    base_preds, base_scores, base_audit = _base_model.predict(_eval_df)

    assessed_df = _eval_df.copy()
    assessed_df["is_anomaly_if"] = if_preds
    assessed_df["is_anomaly_base"] = base_preds
    assessed_df["is_anomaly"] = np.where((if_preds == 1) | (base_preds == 1), 1, 0)
    assessed_df["anomaly_score"] = if_scores
    assessed_df["baseline_score"] = base_scores

    assessed_df = assess_fleet_telemetry(
        assessed_df,
        if_scores=if_scores,
        baseline_audit=base_audit,
    )
    return assessed_df, base_audit


def ensure_session_state():
    """
    Ensures that session state has all required dataframes and models initialized.
    Allows subpages to be directly accessed or refreshed without failing.
    """
    if "assessed_df" in st.session_state and st.session_state["assessed_df"] is not None:
        return

    # Default to official real data if available, fallback to synthetic
    dataset_option = "NASA Official C-MAPSS FD001 (Real Historical Data)"
    train_df, test_df, rul_df, err, source_tag = load_telemetry_dataset(dataset_option)
    if err or train_df is None:
        dataset_option = "Simulated Demonstration Benchmark (Synthetic)"
        train_df, test_df, rul_df, err, source_tag = load_telemetry_dataset(dataset_option)

    preprocessor, if_model, base_model, training_time = get_trained_models(train_df)
    assessed_df, base_audit = process_and_assess_telemetry(train_df, preprocessor, if_model, base_model)

    st.session_state["dataset_option"] = dataset_option
    st.session_state["source_tag"] = source_tag
    st.session_state["train_df"] = train_df
    st.session_state["test_df"] = test_df
    st.session_state["rul_df"] = rul_df
    st.session_state["preprocessor"] = preprocessor
    st.session_state["if_model"] = if_model
    st.session_state["base_model"] = base_model
    st.session_state["training_time"] = training_time
    st.session_state["eval_split"] = "Training Telemetry (Run-to-Failure)"
    st.session_state["assessed_df"] = assessed_df
    st.session_state["base_audit"] = base_audit
