"""
SentinelAI - Page 4: Model Lab
Model configuration, comparative benchmarking (Isolation Forest vs Baseline),
proxy target evaluation, confusion matrices, and supervised validation.
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import HEALTH_THRESHOLDS, UI_COLORS
from src.evaluation import (
    EVALUATION_CONTEXT_NOTE,
    compare_models,
    create_rul_proxy_target,
    evaluate_detection_performance,
    load_and_validate_labeled_csv,
)
from src.visualizations import plot_confusion_matrix_card

st.set_page_config(page_title="Model Lab — SentinelAI", page_icon="🧪", layout="wide")

st.markdown("""
<div style="padding: 1rem 0; margin-bottom: 1rem;">
    <h2 style="margin: 0; color: #F8FAFC;">4. Model Lab & Comparative Benchmark</h2>
    <p style="margin: 0.25rem 0 0 0; color: #94A3B8; font-size: 0.95rem;">
        Side-by-side algorithmic evaluation: Isolation Forest vs Statistical Quantile Baseline.
    </p>
</div>
""", unsafe_allow_html=True)

from src.pipeline_service import ensure_session_state

ensure_session_state()

assessed_df: pd.DataFrame = st.session_state["assessed_df"]
train_df = st.session_state["train_df"]
preprocessor = st.session_state["preprocessor"]
if_model = st.session_state["if_model"]
base_model = st.session_state["base_model"]

if_preds = assessed_df["is_anomaly_if"].values
if_scores = assessed_df["anomaly_score"].values
base_preds = assessed_df["is_anomaly_base"].values
base_scores = assessed_df["baseline_score"].values

# Context Note Alert
st.info(f"📋 {EVALUATION_CONTEXT_NOTE}")

# Evaluation Mode Selection
st.markdown("### 🎯 Ground-Truth Evaluation Target")

eval_mode = st.radio(
    "Select Evaluation Framework",
    [
        "Unsupervised Benchmark (No Ground-Truth Labels)",
        "Operational Proxy Target (RUL <= K Cycles)",
        "Upload Custom Labeled Benchmark CSV",
    ],
    index=1 if "rul" in assessed_df.columns else 0,
    help="Define how model performance should be measured. C-MAPSS does not have native failure labels.",
)

y_true = None
is_proxy = False

if eval_mode == "Operational Proxy Target (RUL <= K Cycles)":
    if "rul" not in assessed_df.columns:
        st.error("Remaining Useful Life (RUL) column is not available in the active telemetry split.")
    else:
        col_k, _ = st.columns([1, 2])
        with col_k:
            k_cycles = st.slider(
                "Critical Degradation Horizon (K cycles)",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
                help="Defines the final K flight cycles before engine overhaul as proxy degradation state.",
            )
        y_true, justification = create_rul_proxy_target(assessed_df, rul_column="rul", critical_threshold=k_cycles)
        is_proxy = True
        st.warning(f"⚠️ {justification}")

elif eval_mode == "Upload Custom Labeled Benchmark CSV":
    uploaded_labeled_file = st.file_uploader(
        "Upload Labeled Evaluation CSV",
        type=["csv"],
        help="Upload an aligned CSV containing ground-truth anomaly annotations.",
    )
    if uploaded_labeled_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_labeled_file)
            uploaded_labeled_file.seek(0)
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                target_col = st.selectbox("Select Ground Truth Column", options=preview_df.columns)
            with c_col2:
                mapping_choice = st.selectbox(
                    "Label Encoding Mapping",
                    [
                        "Direct Binary Numeric (0 = Normal, 1 = Anomaly)",
                        "Text: 'normal' -> 0, 'fault' / 'anomaly' -> 1",
                        "Scikit-Learn Outlier: 1 -> 0, -1 -> 1",
                    ],
                )
            
            mapping_dict = None
            if "Text" in mapping_choice:
                mapping_dict = {"normal": 0, "nominal": 0, "fault": 1, "anomaly": 1, "degraded": 1}
            elif "Scikit-Learn" in mapping_choice:
                mapping_dict = {1: 0, -1: 1}

            y_true, meta_csv = load_and_validate_labeled_csv(
                uploaded_labeled_file,
                target_column=target_col,
                label_mapping=mapping_dict,
                expected_length=len(assessed_df),
            )
            is_proxy = False
            st.success(
                f"✓ Successfully validated ground truth target '{target_col}'! "
                f"Total: {meta_csv['total_records']:,} | Anomalies: {meta_csv['anomaly_count']:,} "
                f"({meta_csv['anomaly_prevalence']:.1%} prevalence)."
            )
        except Exception as e:
            st.error(f"Failed to validate labeled CSV: {e}")

st.markdown("---")

# Comparative Benchmark Table
st.markdown("### 📊 Side-by-Side Model Performance Benchmark")

comparison_table, meta = compare_models(
    y_true=y_true,
    if_preds=if_preds,
    if_scores=if_scores,
    baseline_preds=base_preds,
    baseline_scores=base_scores,
    is_proxy=is_proxy,
)

st.dataframe(comparison_table, hide_index=True, use_container_width=True)

# Supervised Confusion Matrices
if y_true is not None and "eval_if" in meta and "eval_base" in meta:
    st.markdown("### 🔲 Confusion Matrix Comparison")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("#### Model A: Isolation Forest")
        fig_cm_if = plot_confusion_matrix_card(meta["eval_if"]["confusion_matrix"])
        st.plotly_chart(fig_cm_if, use_container_width=True)
    with cm_col2:
        st.markdown("#### Model B: Quantile Baseline")
        fig_cm_base = plot_confusion_matrix_card(meta["eval_base"]["confusion_matrix"])
        st.plotly_chart(fig_cm_base, use_container_width=True)

    # Class Imbalance and Methodology Alerts
    if meta["eval_if"].get("warnings"):
        for w in meta["eval_if"]["warnings"]:
            st.caption(f"⚠️ **Isolation Forest Note:** {w}")
    if meta["eval_base"].get("warnings"):
        for w in meta["eval_base"]["warnings"]:
            st.caption(f"⚠️ **Quantile Baseline Note:** {w}")

else:
    st.info(
        "💡 **Unsupervised Mode Notice**: When ground-truth labels are not present, precision, recall, and F1 scores "
        "cannot be established and will NEVER be fabricated. The system reports genuine anomaly detection counts, "
        "calibrated score distributions, and model agreement overlaps."
    )

st.markdown("---")

# Algorithmic Consensus & Overlap Analysis
st.markdown("### 🤝 Model Agreement & Ensemble Correlation")
ov1, ov2, ov3 = st.columns(3)
with ov1:
    st.metric("Total Records", f"{meta['total_records']:,}")
with ov2:
    st.metric("Dual-Model Agreement (Both Flagged)", f"{meta['overlap_count']:,}")
with ov3:
    st.metric("Jaccard Overlap Index", f"{meta['jaccard_agreement']:.3f}")
