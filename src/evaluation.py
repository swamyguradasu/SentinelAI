"""
Model Evaluation and Benchmark Module for SentinelAI.
Implements supervised metrics, confusion matrix calculations, proxy degradation
target handling, labeled CSV ingestion with custom mapping, class-imbalance warnings,
and leak-free side-by-side model comparison.

SCIENTIFIC DISCLAIMER:
- Supervised metrics (Precision, Recall, F1) strictly require verified ground-truth labels.
- Isolation Forest predictions must NEVER be treated as ground truth for evaluating itself.
- C-MAPSS Remaining Useful Life (RUL) values are run-to-failure degradation counters,
  NOT intrinsic sensor fault labels. RUL thresholds serve as operational proxy horizons.
- Results inherently depend on the dataset, operating conditions, and target definition.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EVALUATION_CONTEXT_NOTE = (
    "EVALUATION CONTEXT NOTE: Predictive maintenance anomaly detection metrics strongly depend on "
    "telemetry characteristics, operational regimes, environmental noise, and the exact physical "
    "definition of the failure/anomaly state. Metrics must be interpreted within this operational context."
)


def create_rul_proxy_target(
    df: pd.DataFrame,
    rul_column: str = "rul",
    critical_threshold: int = 30,
) -> Tuple[pd.Series, str]:
    """
    Constructs an operational proxy degradation target based on Remaining Useful Life (RUL).
    
    OPERATIONAL PROXY JUSTIFICATION:
    In aviation and industrial maintenance, an impending overhaul window (RUL <= K cycles)
    defines an operational threshold of severe thermo-mechanical degradation.
    While this provides a defensible surrogate to benchmark early warning detection lead times,
    it represents proximity to end-of-life overhaul rather than physical sensor telemetry faults.
    """
    if rul_column not in df.columns:
        raise KeyError(f"Column '{rul_column}' not found in DataFrame.")

    proxy_target = (df[rul_column] <= critical_threshold).astype(int)
    justification = (
        f"OPERATIONAL PROXY TARGET (RUL <= {critical_threshold} cycles): "
        "Flags observations within the final critical degradation window. "
        "Serves as an operational wear benchmark; not verified sensor hardware failure annotations."
    )
    return proxy_target, justification


def load_and_validate_labeled_csv(
    file_or_path: Any,
    target_column: str = "label",
    label_mapping: Optional[Dict[Any, int]] = None,
    expected_length: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parses and validates an uploaded labeled CSV benchmark file.
    Supports configurable target column name, custom string/numeric label mappings,
    and enforces strict binary format (0 = normal, 1 = anomaly).
    """
    if isinstance(file_or_path, (str, pd.io.common.Path)):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    if df.empty:
        raise ValueError("Uploaded labeled CSV is empty (0 rows).")

    if target_column not in df.columns:
        available = list(df.columns)
        raise KeyError(
            f"Target column '{target_column}' not found in CSV. Available columns: {available}"
        )

    target_series = df[target_column].copy()

    # Apply custom or default label mapping if provided
    if label_mapping is not None:
        target_series = target_series.map(label_mapping)
        if target_series.isna().any():
            unmapped = df[target_column][target_series.isna()].unique()
            raise ValueError(
                f"Unmapped label values found in '{target_column}': {unmapped}. "
                f"Configured mapping: {label_mapping}"
            )

    # Cast to numeric and check for invalid/missing values
    try:
        y_true = pd.to_numeric(target_series, errors="raise").values.astype(int)
    except Exception as e:
        raise ValueError(
            f"Failed to cast target column '{target_column}' to binary integers (0/1): {e}"
        )

    unique_vals = set(np.unique(y_true))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Target labels must be strictly binary {0, 1}, but found values: {unique_vals}"
        )

    if expected_length is not None and len(y_true) != expected_length:
        raise ValueError(
            f"Record count mismatch: Labeled CSV has {len(y_true)} rows, "
            f"but evaluation dataset has {expected_length} rows."
        )

    metadata = {
        "target_column": target_column,
        "total_records": len(y_true),
        "anomaly_count": int(np.sum(y_true == 1)),
        "nominal_count": int(np.sum(y_true == 0)),
        "anomaly_prevalence": float(np.mean(y_true)),
        "label_mapping_applied": label_mapping or "None (Numeric Native)",
    }
    return y_true, metadata


def evaluate_detection_performance(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_score: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    target_name: str = "Ground Truth",
    is_proxy: bool = False,
    is_self_eval: bool = False,
) -> Dict[str, Any]:
    """
    Calculates precision, recall, F1, confusion matrix, and class-imbalance metrics.
    Gracefully handles edge cases such as zero positive predictions or single-class evaluation.
    
    STRICT SCIENTIFIC RULE:
    Raises ValueError if `is_self_eval` is True (evaluating a model against its own predictions).
    """
    if is_self_eval:
        raise ValueError(
            "SCIENTIFIC VIOLATION: Anomaly detector predictions cannot be evaluated against themselves. "
            "Supervised metrics require independent ground-truth labels."
        )

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    if len(y_t) == 0 or len(y_p) == 0:
        return {"error": "Empty ground-truth or prediction array provided."}

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items.")

    total_samples = len(y_t)
    actual_positives = int(np.sum(y_t == 1))
    predicted_positives = int(np.sum(y_p == 1))
    actual_negatives = total_samples - actual_positives
    prevalence = float(actual_positives / total_samples) if total_samples > 0 else 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    # Standard metrics with robust zero_division safety
    prec = float(precision_score(y_t, y_p, zero_division=0.0))
    rec = float(recall_score(y_t, y_p, zero_division=0.0))
    f1 = float(f1_score(y_t, y_p, zero_division=0.0))
    acc = float(accuracy_score(y_t, y_p))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    warnings: List[str] = []
    if is_proxy:
        warnings.append(
            "Evaluation performed against an operational degradation proxy target (RUL threshold). "
            "Metrics reflect operational degradation sensitivity rather than physical sensor faults."
        )

    if prevalence < 0.05 and actual_positives > 0:
        warnings.append(
            f"Severe class imbalance: anomaly prevalence is only {prevalence:.1%}. "
            "High accuracy can be misleading; focus on Precision, Recall, and F1 score."
        )

    if actual_positives == 0:
        warnings.append(
            "Evaluation set contains 0 positive ground-truth anomalies. Precision and Recall cannot be established."
        )

    if predicted_positives == 0:
        warnings.append("Model predicted 0 anomalies on this evaluation partition.")

    return {
        "target_name": target_name,
        "is_proxy": is_proxy,
        "context_note": EVALUATION_CONTEXT_NOTE,
        "total_samples": total_samples,
        "actual_positives": actual_positives,
        "predicted_positives": predicted_positives,
        "prevalence": prevalence,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "warnings": warnings,
    }


def compare_models(
    y_true: Optional[Union[np.ndarray, pd.Series]],
    if_preds: np.ndarray,
    if_scores: np.ndarray,
    baseline_preds: np.ndarray,
    baseline_scores: np.ndarray,
    is_proxy: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a structured side-by-side comparison table between
    Isolation Forest and the Quantile Baseline model.
    
    If y_true is None:
        Returns honest unsupervised comparative metrics (detection counts,
        anomaly rates, score summaries, and Jaccard agreement) and explicitly marks
        supervised metrics as UNAVAILABLE.
    
    If y_true is provided:
        Computes precision, recall, F1, accuracy, and confusion matrix side-by-side
        on identical records.
    """
    total = len(if_preds)
    if_count = int(np.sum(if_preds == 1))
    base_count = int(np.sum(baseline_preds == 1))

    # Jaccard index of model agreement
    both_flagged = int(np.sum((if_preds == 1) & (baseline_preds == 1)))
    either_flagged = int(np.sum((if_preds == 1) | (baseline_preds == 1)))
    jaccard_overlap = float(both_flagged / either_flagged) if either_flagged > 0 else 1.0

    meta = {
        "total_records": total,
        "overlap_count": both_flagged,
        "jaccard_agreement": jaccard_overlap,
        "is_supervised": y_true is not None,
        "context_note": EVALUATION_CONTEXT_NOTE,
    }

    if y_true is None:
        # Honest Unsupervised State: Never fabricate precision or recall!
        comparison_df = pd.DataFrame([
            {
                "Model": "Model A (Isolation Forest)",
                "Detection Paradigm": "Multivariate Isolation Trees",
                "Total Flagged": if_count,
                "Anomaly Rate (%)": round((if_count / total) * 100, 2) if total > 0 else 0.0,
                "Mean Anomaly Score": round(float(np.mean(if_scores)), 4) if len(if_scores) > 0 else 0.0,
                "Max Anomaly Score": round(float(np.max(if_scores)), 4) if len(if_scores) > 0 else 0.0,
                "Supervised Metrics": "UNAVAILABLE (Requires Ground-Truth Labels)",
            },
            {
                "Model": "Model B (Quantile Baseline)",
                "Detection Paradigm": "Statistical Quantile Envelopes",
                "Total Flagged": base_count,
                "Anomaly Rate (%)": round((base_count / total) * 100, 2) if total > 0 else 0.0,
                "Mean Anomaly Score": round(float(np.mean(baseline_scores)), 4) if len(baseline_scores) > 0 else 0.0,
                "Max Anomaly Score": round(float(np.max(baseline_scores)), 4) if len(baseline_scores) > 0 else 0.0,
                "Supervised Metrics": "UNAVAILABLE (Requires Ground-Truth Labels)",
            },
        ])
        return comparison_df, meta

    # Supervised Evaluation on Identical Observations
    eval_if = evaluate_detection_performance(
        y_true=y_true,
        y_pred=if_preds,
        y_score=if_scores,
        target_name="Isolation Forest",
        is_proxy=is_proxy,
    )
    eval_base = evaluate_detection_performance(
        y_true=y_true,
        y_pred=baseline_preds,
        y_score=baseline_scores,
        target_name="Quantile Baseline",
        is_proxy=is_proxy,
    )

    comparison_df = pd.DataFrame([
        {
            "Model": "Model A (Isolation Forest)",
            "Flagged Count": if_count,
            "Anomaly Rate (%)": round((if_count / total) * 100, 2) if total > 0 else 0.0,
            "Precision": round(eval_if["precision"], 4),
            "Recall": round(eval_if["recall"], 4),
            "F1 Score": round(eval_if["f1_score"], 4),
            "Accuracy": round(eval_if["accuracy"], 4),
            "Specificity": round(eval_if["specificity"], 4),
            "FPR": round(eval_if["false_positive_rate"], 4),
            "TP": eval_if["confusion_matrix"]["tp"],
            "FP": eval_if["confusion_matrix"]["fp"],
            "FN": eval_if["confusion_matrix"]["fn"],
            "TN": eval_if["confusion_matrix"]["tn"],
        },
        {
            "Model": "Model B (Quantile Baseline)",
            "Flagged Count": base_count,
            "Anomaly Rate (%)": round((base_count / total) * 100, 2) if total > 0 else 0.0,
            "Precision": round(eval_base["precision"], 4),
            "Recall": round(eval_base["recall"], 4),
            "F1 Score": round(eval_base["f1_score"], 4),
            "Accuracy": round(eval_base["accuracy"], 4),
            "Specificity": round(eval_base["specificity"], 4),
            "FPR": round(eval_base["false_positive_rate"], 4),
            "TP": eval_base["confusion_matrix"]["tp"],
            "FP": eval_base["confusion_matrix"]["fp"],
            "FN": eval_base["confusion_matrix"]["fn"],
            "TN": eval_base["confusion_matrix"]["tn"],
        },
    ])

    meta["eval_if"] = eval_if
    meta["eval_base"] = eval_base
    return comparison_df, meta


def evaluate_pipeline_on_held_out_data(
    eval_df: pd.DataFrame,
    preprocessor: Any,
    if_model: Any,
    base_model: Any,
    y_true: Optional[Union[np.ndarray, pd.Series]] = None,
    is_proxy: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates both anomaly models strictly on held-out observations.
    CRITICAL: Does NOT refit the preprocessor, Isolation Forest, or Quantile Baseline.
    Applies existing frozen parameters to ensure zero data leakage.
    """
    if not preprocessor.is_fitted_ or not if_model.is_fitted_ or not base_model.is_fitted_:
        raise RuntimeError("Models and preprocessor must be pre-fitted on training data before evaluation.")

    # Apply frozen transformations to evaluation set
    x_eval_scaled, _ = preprocessor.transform(eval_df)
    if_preds, if_scores = if_model.predict(x_eval_scaled)
    base_preds, base_scores, _ = base_model.predict(eval_df)

    return compare_models(
        y_true=y_true,
        if_preds=if_preds,
        if_scores=if_scores,
        baseline_preds=base_preds,
        baseline_scores=base_scores,
        is_proxy=is_proxy,
    )
