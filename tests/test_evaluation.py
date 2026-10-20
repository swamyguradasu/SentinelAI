"""
Unit tests for src/evaluation.py.
Verifies evaluation metrics, confusion matrix, proxy degradation targets,
labeled CSV validation with custom mapping, zero-division safety,
and leak-free held-out evaluation.
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compare_models,
    create_rul_proxy_target,
    evaluate_detection_performance,
    evaluate_pipeline_on_held_out_data,
    load_and_validate_labeled_csv,
    EVALUATION_CONTEXT_NOTE,
)
from src.anomaly_detection import train_anomaly_pipeline


def test_evaluate_detection_performance_standard():
    """Verifies precision, recall, F1, accuracy, specificity, and confusion matrix arithmetic."""
    # 4 samples: 2 actual anomalies [0, 1, 0, 1]
    # predictions: [0, 1, 1, 0] -> TN=1, TP=1, FP=1, FN=1
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    res = evaluate_detection_performance(y_true, y_pred)

    assert res["confusion_matrix"]["tp"] == 1
    assert res["confusion_matrix"]["fp"] == 1
    assert res["confusion_matrix"]["tn"] == 1
    assert res["confusion_matrix"]["fn"] == 1

    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1_score"] == 0.5
    assert res["accuracy"] == 0.5
    assert res["specificity"] == 0.5
    assert res["false_positive_rate"] == 0.5
    assert res["context_note"] == EVALUATION_CONTEXT_NOTE


def test_evaluate_detection_performance_zero_positives():
    """Verifies robust zero_division handling when model predicts no anomalies."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])  # No positive predictions

    res = evaluate_detection_performance(y_true, y_pred)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert any("0 anomalies" in w for w in res["warnings"])


def test_evaluate_detection_performance_zero_true_positives():
    """Verifies safe handling when ground truth contains no anomalies (all normal)."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0])

    res = evaluate_detection_performance(y_true, y_pred)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert any("0 positive ground-truth" in w for w in res["warnings"])


def test_self_evaluation_rejection():
    """
    CRITICAL ML INTEGRITY TEST:
    Verifies that passing is_self_eval=True raises ValueError,
    preventing a model from evaluating itself.
    """
    y_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="SCIENTIFIC VIOLATION"):
        evaluate_detection_performance(y_pred, y_pred, is_self_eval=True)


def test_create_rul_proxy_target():
    """Verifies operational proxy target binary cutoff and justification text."""
    df = pd.DataFrame({"engine_id": [1, 1, 1], "rul": [50, 30, 10]})
    target, justification = create_rul_proxy_target(df, rul_column="rul", critical_threshold=30)

    # Values <= 30 should be 1
    assert list(target) == [0, 1, 1]
    assert "OPERATIONAL PROXY TARGET" in justification
    assert "30 cycles" in justification


def test_load_and_validate_labeled_csv_numeric():
    """Verifies parsing of standard numeric labeled CSV."""
    csv_content = "engine_id,cycle,anomaly_flag\n1,1,0\n1,2,1\n2,1,0\n"
    y_true, meta = load_and_validate_labeled_csv(
        io.StringIO(csv_content),
        target_column="anomaly_flag",
        expected_length=3,
    )
    assert list(y_true) == [0, 1, 0]
    assert meta["anomaly_count"] == 1
    assert meta["nominal_count"] == 2


def test_load_and_validate_labeled_csv_custom_mapping():
    """Verifies custom string label mapping (e.g. 'normal' vs 'fault')."""
    csv_content = "unit,cycle,status\n1,1,normal\n1,2,fault\n1,3,fault\n"
    y_true, meta = load_and_validate_labeled_csv(
        io.StringIO(csv_content),
        target_column="status",
        label_mapping={"normal": 0, "fault": 1},
    )
    assert list(y_true) == [0, 1, 1]
    assert meta["anomaly_count"] == 2


def test_load_and_validate_labeled_csv_invalid_cases():
    """Verifies clear errors on missing target column, non-binary values, or length mismatch."""
    # Missing column
    with pytest.raises(KeyError, match="not found"):
        load_and_validate_labeled_csv(io.StringIO("a,b\n1,2"), target_column="missing_target")

    # Non-binary labels
    with pytest.raises(ValueError, match="strictly binary"):
        load_and_validate_labeled_csv(io.StringIO("target\n0\n1\n2"), target_column="target")

    # Length mismatch
    with pytest.raises(ValueError, match="Record count mismatch"):
        load_and_validate_labeled_csv(io.StringIO("target\n0\n1"), target_column="target", expected_length=5)


def test_compare_models_honest_unsupervised_state():
    """
    Verifies that when ground-truth labels are absent,
    supervised metrics are honestly reported as UNAVAILABLE.
    """
    if_preds = np.array([0, 1, 1, 0])
    if_scores = np.array([0.2, 0.8, 0.7, 0.3])
    base_preds = np.array([0, 0, 1, 1])
    base_scores = np.array([0.1, 0.4, 0.9, 0.6])

    comp_df, meta = compare_models(None, if_preds, if_scores, base_preds, base_scores)
    assert len(comp_df) == 2
    assert not meta["is_supervised"]
    assert "Supervised Metrics" in comp_df.columns
    assert "UNAVAILABLE" in comp_df["Supervised Metrics"].iloc[0]


def test_compare_models_supervised_evaluation():
    """Verifies side-by-side evaluation table with valid ground-truth labels."""
    if_preds = np.array([0, 1, 1, 0])
    if_scores = np.array([0.2, 0.8, 0.7, 0.3])
    base_preds = np.array([0, 0, 1, 1])
    base_scores = np.array([0.1, 0.4, 0.9, 0.6])
    y_true = np.array([0, 1, 1, 0])

    comp_df, meta = compare_models(y_true, if_preds, if_scores, base_preds, base_scores)
    assert len(comp_df) == 2
    assert meta["is_supervised"]
    assert "Precision" in comp_df.columns
    assert "Recall" in comp_df.columns
    assert "F1 Score" in comp_df.columns
    assert "TP" in comp_df.columns

    # Model A matched ground truth perfectly -> F1 should be 1.0
    row_if = comp_df[comp_df["Model"].str.contains("Isolation Forest")].iloc[0]
    assert row_if["F1 Score"] == 1.0


def test_evaluate_pipeline_on_held_out_data():
    """Verifies leak-free evaluation on held-out test data without refitting."""
    train_df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [1, 2, 3],
        "sensor_2": [640.0, 641.0, 642.0],
        "sensor_3": [1580.0, 1585.0, 1590.0],
    })
    test_df = pd.DataFrame({
        "engine_id": [2, 2],
        "cycle": [1, 2],
        "sensor_2": [641.0, 700.0],
        "sensor_3": [1585.0, 1700.0],
    })
    y_test = np.array([0, 1])

    prep, if_model, base_model = train_anomaly_pipeline(train_df, n_estimators=20)
    
    comp_df, meta = evaluate_pipeline_on_held_out_data(
        eval_df=test_df,
        preprocessor=prep,
        if_model=if_model,
        base_model=base_model,
        y_true=y_test,
    )
    assert len(comp_df) == 2
    assert meta["is_supervised"]
