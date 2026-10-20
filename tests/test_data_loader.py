"""
Unit tests for src/data_loader.py.
Verifies whitespace parsing, schema validation, RUL computation,
test engine RUL association, and plausibility verification.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import ALL_COLUMNS, SENSOR_COLUMNS
from src.data_loader import (
    associate_test_rul,
    compute_training_rul,
    parse_cmapss_file,
    parse_rul_file,
    validate_telemetry_schema,
    verify_dataset_plausibility,
)


@pytest.fixture
def synthetic_cmapss_file(tmp_path: Path) -> Path:
    """Creates a temporary whitespace-delimited C-MAPSS style text file."""
    file_path = tmp_path / "synthetic_train.txt"
    lines = []
    # 2 engines, 5 cycles each, 26 columns
    for eng in [1, 2]:
        for cyc in range(1, 6):
            settings = [0.001, 0.0002, 100.0]
            sensors = [float(500 + i * 10 + cyc) for i in range(1, 22)]
            row_vals = [eng, cyc] + settings + sensors
            # Add irregular whitespace and trailing spaces
            lines.append("   " + "   ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row_vals) + "  \n")
    
    file_path.write_text("".join(lines))
    return file_path


@pytest.fixture
def synthetic_rul_file(tmp_path: Path) -> Path:
    """Creates a temporary RUL ground truth text file."""
    file_path = tmp_path / "synthetic_rul.txt"
    file_path.write_text("112\n98\n45\n")
    return file_path


def test_parse_cmapss_file_success(synthetic_cmapss_file: Path):
    """Verifies that whitespace parsing produces correct columns, dtypes, and dimensions."""
    df = parse_cmapss_file(synthetic_cmapss_file)
    assert df.shape == (10, 26)
    assert list(df.columns) == ALL_COLUMNS
    assert df["engine_id"].dtype == np.int64
    assert df["cycle"].dtype == np.int64
    assert (df["engine_id"].unique() == [1, 2]).all()
    assert (df["cycle"].min() == 1) and (df["cycle"].max() == 5)


def test_parse_cmapss_file_not_found():
    """Verifies FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        parse_cmapss_file("non_existent_file.txt")


def test_parse_cmapss_file_invalid_column_count(tmp_path: Path):
    """Verifies error on truncated column count (< 26 fields)."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("1 1 0.0 0.0\n1 2 0.0 0.0\n")
    with pytest.raises(ValueError, match="Expected at least 26 fields"):
        parse_cmapss_file(bad_file)


def test_parse_cmapss_file_non_numeric(tmp_path: Path):
    """Verifies ValueError when corrupt non-numeric values exist."""
    bad_file = tmp_path / "bad_numeric.txt"
    row = ["1", "1", "0.0", "0.0", "100.0"] + ["CORRUPT_TOKEN"] + ["500.0"] * 20
    bad_file.write_text(" ".join(row) + "\n")
    with pytest.raises(ValueError, match="Non-numeric or corrupt values"):
        parse_cmapss_file(bad_file)


def test_parse_cmapss_file_non_positive_identifiers(tmp_path: Path):
    """Verifies error when engine_id or cycle is non-positive."""
    bad_file = tmp_path / "bad_id.txt"
    row = ["0", "1", "0.0", "0.0", "100.0"] + ["500.0"] * 21
    bad_file.write_text(" ".join(row) + "\n")
    with pytest.raises(ValueError, match="Engine IDs must be strictly positive"):
        parse_cmapss_file(bad_file)


def test_parse_cmapss_file_duplicate_cycles(tmp_path: Path):
    """Verifies error on duplicate engine/cycle records."""
    bad_file = tmp_path / "dup.txt"
    row = ["1", "1", "0.0", "0.0", "100.0"] + ["500.0"] * 21
    bad_file.write_text(" ".join(row) + "\n" + " ".join(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate engine/cycle records"):
        parse_cmapss_file(bad_file)


def test_parse_rul_file_success(synthetic_rul_file: Path):
    """Verifies RUL parsing with correct unit indexing."""
    df_rul = parse_rul_file(synthetic_rul_file)
    assert df_rul.shape == (3, 2)
    assert list(df_rul.columns) == ["engine_id", "rul"]
    assert list(df_rul["engine_id"]) == [1, 2, 3]
    assert list(df_rul["rul"]) == [112, 98, 45]


def test_compute_training_rul():
    """Verifies run-to-failure RUL calculation."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })
    result = compute_training_rul(df)
    # Engine 1 max cycle is 3: RUL = [2, 1, 0]
    # Engine 2 max cycle is 2: RUL = [1, 0]
    assert list(result["rul"]) == [2, 1, 0, 1, 0]


def test_associate_test_rul_terminal_and_trajectory():
    """
    Verifies test engine association with ground-truth RUL target records.
    """
    test_df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 10, 11],
        "sensor_2": [640.0, 641.0, 642.0, 645.0, 646.0],
    })
    rul_df = pd.DataFrame({
        "engine_id": [1, 2],
        "rul": [50, 20],  # Engine 1 has 50 cycles remaining at cycle 3; Engine 2 has 20 at cycle 11
    })

    # 1. Full trajectory mode
    res_traj = associate_test_rul(test_df, rul_df, trajectory_mode=True)
    assert len(res_traj) == 5
    # For Engine 1: cycle 3 has RUL 50; cycle 2 has RUL 51; cycle 1 has RUL 52
    eng1_ruls = list(res_traj[res_traj["engine_id"] == 1]["rul"])
    assert eng1_ruls == [52, 51, 50]

    # For Engine 2: cycle 11 has RUL 20; cycle 10 has RUL 21
    eng2_ruls = list(res_traj[res_traj["engine_id"] == 2]["rul"])
    assert eng2_ruls == [21, 20]

    # 2. Terminal mode
    res_term = associate_test_rul(test_df, rul_df, trajectory_mode=False)
    assert len(res_term) == 2
    assert list(res_term["rul"]) == [50, 20]
    assert list(res_term["cycle"]) == [3, 11]


def test_validate_telemetry_schema():
    """Verifies schema validation on valid, empty, and corrupt inputs."""
    valid, errs = validate_telemetry_schema(pd.DataFrame())
    assert not valid
    assert "empty" in errs[0].lower()

    df_missing = pd.DataFrame({"cycle": [1, 2], "sensor_1": [1.0, 2.0]})
    valid, errs = validate_telemetry_schema(df_missing)
    assert not valid
    assert any("engine_id" in e for e in errs)


def test_verify_dataset_plausibility():
    """Verifies plausibility check flags valid and corrupt datasets correctly."""
    # Plausible train dataset (e.g. 2000 rows, 26 cols, 10 engines)
    plausible_df = pd.DataFrame({
        "engine_id": np.repeat(np.arange(1, 11), 200),
        "cycle": np.tile(np.arange(1, 201), 10),
        **{f"op_setting_{i}": 0.0 for i in range(1, 4)},
        **{f"sensor_{i}": 500.0 for i in range(1, 22)},
    })
    res = verify_dataset_plausibility(plausible_df)
    assert res["is_plausible"]
    assert res["train_rows"] == 2000
    assert res["train_cols"] == 26

    # Corrupt dataset: only 5 columns
    bad_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    res_bad = verify_dataset_plausibility(bad_df)
    assert not res_bad["is_plausible"]
