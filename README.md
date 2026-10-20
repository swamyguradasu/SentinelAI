# SentinelAI — Predictive Maintenance & Sensor Anomaly Detection

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit%201.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20IsolationForest-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-pytest%2060%2F60%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD001-0B3D91.svg)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

> **Industrial-grade telemetry intelligence, multi-channel degradation tracking, and explainable equipment health analysis.**  
> Engineered to showcase practical, production-ready machine learning, data engineering, and telemetry analytics skills for ML engineering and data science roles.

---

## 1. Project Overview

**SentinelAI** is an industrial telemetry intelligence platform designed to detect degradation, sensor drift, and anomalous operational patterns in complex machinery prior to catastrophic failure. Using the benchmark **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Degradation Dataset**, SentinelAI ingests high-frequency multi-sensor flight cycles, eliminates non-informative sensor channels, prevents training-test data leakage, and executes a dual-model anomaly detection engine:

1. **Model A (Isolation Forest)**: An unsupervised multi-dimensional isolation tree model producing continuous, calibrated anomaly scores ($0.0 \to 1.0$) alongside binary operational alert flags.
2. **Model B (Statistical Quantile Baseline)**: A transparent statistical process control (SPC) baseline learning upper and lower percentile envelopes per sensor strictly on initial healthy operating cycles, attributing exact root-cause breaching channels.

The platform provides a control-room-styled dark industrial dashboard with interactive Plotly visualizations, forensic triage tables, side-by-side comparative model benchmarks, and filtered CSV export capabilities.

---

## 2. System Architecture

```
SentinelAI/
├── .streamlit/
│   └── config.toml                  # Bespoke dark industrial theme configuration
├── data/
│   ├── raw/                         # Authentic NASA C-MAPSS FD001 text files
│   │   ├── train_FD001.txt          # 100 engines run to failure (20,631 cycles)
│   │   ├── test_FD001.txt           # 100 test engines truncated prior to failure (13,096 cycles)
│   │   └── RUL_FD001.txt            # Ground-truth Remaining Useful Life vector
│   ├── processed/                   # Intermediary clean telemetry cache
│   └── sample/                      # Bundled synthetic benchmark harness for offline use
├── models/                          # Serialized joblib pipelines & model artifacts
├── results/                         # Generated audit trails and reports
├── src/
│   ├── __init__.py                  # Package declaration
│   ├── data_loader.py               # Robust whitespace parsing, schema validation & mirror downloaders
│   ├── preprocessing.py             # Zero-variance filtering, robust scaling & leakage prevention
│   ├── feature_engineering.py       # Engine-isolated rolling statistics & delta drift calculations
│   ├── anomaly_detection.py         # Isolation Forest & Quantile Threshold Baseline models
│   ├── evaluation.py                # Supervised/proxy performance metrics & confusion matrix
│   ├── health_analysis.py           # Domain rule-based health classification (Healthy, Degraded, Critical)
│   ├── pipeline_service.py          # Cached data loading, model fitting & auto session state
│   ├── simulator.py                 # Historical engine replay & synthetic drift streaming simulator
│   ├── visualizations.py            # Dark industrial Plotly figures and custom UI cards
│   └── reporting.py                 # Telemetry filtering, summary KPIs & CSV/Markdown exporters
├── pages/
│   ├── 1_Overview.py                # Fleet telemetry KPIs, model status & recent alerts
│   ├── 2_Sensor_Explorer.py         # Single-engine time-series analysis & anomaly overlays
│   ├── 3_Anomaly_Center.py          # Searchable/filterable alert triage & root-cause explanations
│   ├── 4_Model_Lab.py               # Model training, parameter tuning & comparative evaluation
│   ├── 5_Historical_Analysis.py     # Fleet degradation progression & comparative trajectory analysis
│   └── 6_Stream_Simulator.py        # Isolated demonstration stream harness & online anomaly scoring
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py          # Whitespace parsing, schema checks, RUL & plausibility checks
│   ├── test_preprocessing.py        # Zero-variance detection, robust scaling & data quality audits
│   ├── test_feature_engineering.py  # Rolling stats & first-order deltas boundary isolation
│   ├── test_anomaly_detection.py    # Isolation Forest & Quantile Baseline fit/predict/score
│   ├── test_evaluation.py           # Precision, recall, F1, confusion matrix & proxy targets
│   ├── test_health_analysis.py      # Rule-based health classification & streak tracking
│   ├── test_reporting.py            # Telemetry KPI aggregation, filtering & CSV byte export
│   ├── test_simulator.py            # Streaming order, synthetic drift injection & state reset
│   └── test_dashboard_pages.py      # Automated Streamlit AppTest integration for all 6 pages
├── app.py                           # Main application entrypoint, state management & UI styling
├── config.py                        # Constants, sensor metadata, units, and directory paths
├── requirements.txt                 # Exact, stable Python dependency specifications
└── README.md                        # Project documentation, architecture & run instructions
```

---

## 3. NASA C-MAPSS Dataset Specifications & Pipeline Ingestion

The **NASA Prognostics Center of Excellence (PCoE) C-MAPSS FD001** dataset contains simulated run-to-failure trajectories of turbofan engines operating under sea-level ambient conditions.

### Telemetry Channels (26 Raw Whitespace-Delimited Fields)
- **Index Identifiers**: `engine_id` (unit number 1 to 100), `cycle` (operational flight cycles).
- **Operational Settings**: `op_setting_1` (Altitude), `op_setting_2` (Mach Number), `op_setting_3` (TRA).
- **Sensor Measurements (21 Channels)**:
  - `sensor_1`: $T2$ - Fan inlet temperature (°R) *(Constant: 518.67)*
  - `sensor_2`: $T24$ - LPC outlet temperature (°R) *(Upward drift)*
  - `sensor_3`: $T30$ - HPC outlet temperature (°R) *(Upward drift)*
  - `sensor_4`: $T50$ - LPT outlet temperature (°R) *(Upward drift)*
  - `sensor_5`: $P2$ - Fan inlet pressure (psia) *(Constant: 14.62)*
  - `sensor_6`: $P15$ - Bypass-duct total pressure (psia)
  - `sensor_7`: $P30$ - HPC outlet total pressure (psia) *(Downward drift)*
  - `sensor_8`: $Nf$ - Physical fan speed (rpm) *(Upward drift)*
  - `sensor_9`: $Nc$ - Physical core speed (rpm) *(Upward drift)*
  - `sensor_10`: $epr$ - Engine pressure ratio ($P50/P2$) *(Constant: 1.30)*
  - `sensor_11`: $Ps30$ - HPC outlet static pressure (psia) *(Upward drift)*
  - `sensor_12`: $Phi$ - Fuel flow ratio to $Ps30$ (pps/psi) *(Downward drift)*
  - `sensor_13`: $NRf$ - Corrected fan speed (rpm) *(Upward drift)*
  - `sensor_14`: $NRc$ - Corrected core speed (rpm) *(Upward drift)*
  - `sensor_15`: $BPR$ - Bypass Ratio *(Upward drift)*
  - `sensor_16`: $farB$ - Burner fuel-air ratio *(Constant: 0.03)*
  - `sensor_17`: $htBleed$ - Bleed Enthalpy *(Upward drift)*
  - `sensor_18`: $Nf\_dmd$ - Demanded fan speed (rpm) *(Constant: 2388)*
  - `sensor_19`: $PCNfR\_dmd$ - Demanded corrected fan speed (rpm) *(Constant: 100.0)*
  - `sensor_20`: $W31$ - HPT coolant bleed (lbm/s) *(Downward drift)*
  - `sensor_21`: $W32$ - LPT coolant bleed (lbm/s) *(Downward drift)*

### Dataset Ingestion & Ground-Truth RUL Association
1. **Training RUL Labels**:
   - In training engines, units run to complete failure ($RUL = 0$ at terminal cycle).
   - For engine $i$ at flight cycle $t$:
     $$RUL_i(t) = \max_{cycle}(engine\_i) - t$$
2. **Test Engine RUL Association**:
   - In test engines, operational telemetry is truncated prior to failure at $T_{max, i}$.
   - The ground-truth file `RUL_FD001.txt` provides remaining cycles at this terminal cycle ($RUL_i(T_{max, i})$).
   - SentinelAI reconstructs the ground-truth trajectory across all historical test cycles:
     $$RUL_i(t) = RUL_i(T_{max, i}) + (T_{max, i} - t)$$
3. **Train / Test Isolation & Leakage Prevention**:
   - Training and testing datasets are loaded and stored as disjoint entities without mixing.
   - All transformations (zero-variance identification, median imputations, RobustScaler center and scale) are fitted strictly on the training partition.
4. **Data Quality & Descriptive Auditing**:
   - Automatically runs sanity checks: missing value detection, non-finite checks (NaN, Inf), constant channel detection, IQR outlier detection, and distributional skewness.

---

## 4. Modeling & Scientific Methodology

### Model A: Isolation Forest (Unsupervised Tree Isolation)
- **Concept**: Partitions feature space using random orthogonal splits. Anomalous records require fewer splits to isolate and exhibit short path lengths.
- **Calibrated Scoring**: Raw Isolation Forest decision function values are inverted and scaled into a normalized $[0.0, 1.0]$ interval where $0.0$ corresponds to nominal baseline behavior and $1.0$ indicates severe multi-sensor divergence.
- **Leakage-Safe Scaling**: Fitted `RobustScaler` / `StandardScaler` transformations are learned strictly on the training partition and applied without refitting to test sequences.

### Model B: Statistical Quantile Baseline (Explainable SPC)
- **Concept**: Learns empirical lower ($\alpha$) and upper ($1 - \alpha$) bounds per sensor on early healthy cycles ($cycle \le 30$).
- **Explainability**: Identifies the exact sensor channels breaching tolerance envelopes and reports violation magnitude normalized by interquartile range (IQR).

### Transparent Health Index Synthesis
SentinelAI combines continuous anomaly score persistence with baseline multi-channel breaches:
- **`HEALTHY`** (Teal `#10B981`): Score $< 0.55$, zero baseline breaches.
- **`DEGRADED`** (Amber `#F59E0B`): Transient score elevation ($0.55 \le \text{Score} < 0.70$) or single sensor breach.
- **`CRITICAL`** (Red `#EF4444`): Persistent score $\ge 0.70$ over $\ge 3$ consecutive cycles, or $\ge 2$ simultaneous sensor breaches.

> **Scientific Notice**: Telemetry anomalies denote statistical departure from normal behavior; they do not represent instantaneous mechanical breakage or deterministic failure times.

---

## 5. Evaluation Methodology & Proxy Targets

A fundamental reality of real-world predictive maintenance is the **absence of verified binary fault labels** in raw telemetry:
- **Unsupervised Mode**: SentinelAI reports true detection rates, model overlap agreement (Jaccard index), and score distributions. It refuses to fabricate artificial precision or recall figures.
- **Operational Proxy Evaluation**: When evaluating run-to-failure degradation, the system allows configuring an operational proxy horizon (e.g. $RUL \le 30$ cycles). The UI prominently marks these as proxy indicators of severe mechanical wear.
- **Supervised Validation**: Users can upload independent labeled CSV test benchmarks to evaluate true Precision, Recall, F1 score, Specificity, and False Positive Rates alongside confusion matrices.

---

## 6. Installation, Dataset Placement & Quick Start

### Prerequisites
- Python 3.10 – 3.13
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/swamyguradasu/SentinelAI.git
cd SentinelAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement & Verification
The authentic NASA C-MAPSS FD001 dataset files must be placed in `data/raw/`:
```
data/raw/
├── train_FD001.txt   # 20,631 rows, 26 whitespace-delimited columns
├── test_FD001.txt    # 13,096 rows, 26 whitespace-delimited columns
└── RUL_FD001.txt     # 100 rows, 1 column ground-truth terminal RUL
```

**Automated Download**:
```bash
python -m src.data_loader --download
```
*Note: If the network or mirror is unreachable, this command will report failure explicitly without faking files.*

**Dataset Verification Command**:
```bash
python -m src.data_loader --verify
```
*Expected Output:*
```text
=== SentinelAI Dataset Verification Report ===
  [PASS] Train row count is plausible: 20,631 records
  [PASS] Train column count is exact: 27 fields
  [PASS] Unique training engines: 100 units
  [PASS] Test row count is plausible: 13,096 records
  [PASS] Unique test engines: 100 units
  [PASS] RUL records (100) match test engine count exactly
Overall Plausibility Verdict: PASSED
```

### 3. Run Unit & Integration Tests
```bash
pytest -v
```
*Expected: 60 tests passed across all modules and dashboard pages.*

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Dashboard Modules Walkthrough

1. **Overview Dashboard (`pages/1_Overview.py`)**:
   - Fleet KPIs (Units, Cycles, Channels, Anomaly Rate, Critical Count).
   - Donut breakdown of fleet health distribution.
   - Isolation Forest score histogram with marked industrial thresholds.
   - Interactive triage table of high-severity alerts.
2. **Sensor Explorer (`pages/2_Sensor_Explorer.py`)**:
   - Single-engine time-series telemetry with anomaly markers.
   - Secondary sensor comparison on dual axes.
   - Baseline statistical threshold bounds.
   - Unit vs Fleet descriptive statistics and single-point inspection tool.
3. **Anomaly Center (`pages/3_Anomaly_Center.py`)**:
   - Forensic triage filterable by engine, detection method, health status, and score.
   - Root-cause breakdown identifying which sensors trigger alerts most frequently.
   - Export filtered audit logs directly to CSV.
4. **Model Lab (`pages/4_Model_Lab.py`)**:
   - Side-by-side benchmark comparing Isolation Forest and Quantile Baseline.
   - Proxy target and labeled CSV validation frameworks.
   - Precision, Recall, F1, and Confusion Matrix heatmaps.
   - Model agreement overlap (Jaccard agreement).
5. **Historical Analysis (`pages/5_Historical_Analysis.py`)**:
   - Fleet degradation heatmap across all units over time.
   - Comparative multi-engine telemetry trajectory charts.
   - Anomaly escalation bar chart by flight cycle intervals.
   - Downloadable Markdown audit summary report.
6. **Sensor Stream Simulator (`pages/6_Stream_Simulator.py`)**:
   - Isolated demonstration harness simulating dynamic observation intake.
   - Replays historical NASA engine flight cycles in strict chronological order or generates synthetic telemetry with controllable Gaussian noise and abnormal drift/spikes.
   - Evaluates each incoming cycle through the frozen, fitted anomaly detection pipeline without retraining.
   - Dynamically updates Plotly sensor charts, real-time alert event feeds, and equipment health status badges.
   - Session export button providing full CSV download of recorded simulated trajectories.
   - Clearly documented disclaimer explicitly distinguishing local UI simulation from physical factory streaming infrastructure.

---

## 8. Limitations & Future Roadmap

### Current Limitations
1. **Operating Condition Uniformity**: FD001 evaluates sea-level static flight operations. Multi-regime datasets (FD002, FD004) require operating condition clustering (e.g. k-means on altitude/Mach/TRA) prior to baseline fitting.
2. **Proxy Degradation Labels**: RUL thresholding serves as an operational surrogate for late-stage degradation; it does not replace physically verified component tear-down logs.

### Future Roadmap
- [ ] Add Autoencoder (PyTorch/ONNX) deep anomaly reconstruction model.
- [ ] Support FD002 and FD004 multi-condition operating regime clustering.
- [ ] Add direct RUL regression prediction model (Weibull survival / LightGBM regressor) as an independent pipeline stage.
- [ ] Implement Kafka / MQTT streaming telemetry ingestion emulator.

---

## 9. Dataset Attribution & Citation

```bibtex
@inproceedings{saxena2008damage,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Donald and Eklund, Neil},
  booktitle={2008 International Conference on Prognostics and Health Management},
  pages={1--9},
  year={2008},
  organization={IEEE}
}
```
*Dataset sourced from NASA Ames Prognostics Center of Excellence (PCoE) Data Repository: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).*

---

## 10. Troubleshooting

- **Dataset Not Found Error**: Run `python -c "from src.data_loader import download_official_cmapss; download_official_cmapss()"` or switch to the *Simulated Demonstration Benchmark* in the sidebar.
- **Port Conflict**: Launch on a custom port with `streamlit run app.py --server.port 8502`.
- **Streamlit Cache Reset**: Press `C` in the browser or use the sidebar rerun option to clear the cache.
