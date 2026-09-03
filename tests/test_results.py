"""Results row: column schema and the hard calibration_set_id rule.

Both rejections here are about errors that are invisible later. A misspelt column reads
back as NaN and looks like a quantity that was never measured; a hardware row without a
calibration identifier cannot be repaired after the session, because a calibration
cannot be reconstructed retrospectively.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qsocket.results import RESULT_COLUMNS, append_result_row


def base_row(**overrides) -> dict:
    row = {column: "" for column in RESULT_COLUMNS}
    row.update(
        run_id="20260814T120000_test",
        timestamp_utc="2026-08-14T12:00:00Z",
        dataset="two_curves",
        dataset_hash="abc123",
        pca_hash="def456",
        arm="A",
        ansatz_level="L1",
        R=2,
        dilution="linear",
        socket_params_nominal=35,
        socket_params_effective=35,
        head_params=6,
        seed=1,
        init_seed=123456789,
        init_spec_id="U[0,2pi)",
        backend="statevector",
        n_eval=600,
        split="val",
        accuracy=0.81,
        macro_f1=0.80,
        train_accuracy=0.84,
        theta_displacement=0.12,
        grad_rms_start=0.03,
        grad_rms_end=0.004,
        epochs_run=90,
        best_epoch=60,
        wall_seconds=1234.5,
        git_commit="04a08bd",
        env_hash="fedcba",
    )
    row.update(overrides)
    return row


def test_column_order_is_exactly_the_contract():
    assert RESULT_COLUMNS[:6] == (
        "run_id",
        "timestamp_utc",
        "dataset",
        "dataset_hash",
        "pca_hash",
        "arm",
    )
    assert RESULT_COLUMNS[33:35] == ("git_commit", "env_hash")
    assert RESULT_COLUMNS[35:] == (
        "lr_selected",
        "lr_grid",
        "g1_margin",
        "patience",
        "max_epochs",
        "feature_scale",
        "rff_width",
        "rff_omega_seed",
        "used_feature_cache",
        "ridge_accuracy",
        "ridge_alpha_selected",
    )
    assert len(RESULT_COLUMNS) == len(set(RESULT_COLUMNS)) == 46


def test_a_valid_row_is_written_with_the_contract_column_order(tmp_path):
    path = tmp_path / "results.csv"
    append_result_row(path, base_row())
    frame = pd.read_csv(path)
    assert frame.columns.tolist() == list(RESULT_COLUMNS)
    assert len(frame) == 1


def test_rows_append_rather_than_overwrite(tmp_path):
    path = tmp_path / "results.csv"
    append_result_row(path, base_row(seed=1))
    append_result_row(path, base_row(seed=2))
    frame = pd.read_csv(path)
    assert frame["seed"].tolist() == [1, 2]


def test_unknown_column_is_rejected_and_nothing_is_written(tmp_path):
    path = tmp_path / "results.csv"
    row = base_row()
    row["acuracy"] = 0.9  # the typo this rule exists for
    with pytest.raises(ValueError, match="unknown result column"):
        append_result_row(path, row)
    assert not path.exists()


def test_missing_required_column_is_rejected(tmp_path):
    path = tmp_path / "results.csv"
    row = base_row()
    del row["macro_f1"]
    with pytest.raises(ValueError, match="missing required result column"):
        append_result_row(path, row)
    assert not path.exists()


@pytest.mark.parametrize("value", ["", None, "None", "nan", "N/A", "TBD", "  "])
def test_hardware_row_without_calibration_set_id_is_rejected(tmp_path, value):
    path = tmp_path / "results.csv"
    row = base_row(backend="iqm_spark", shots=4096, session_id="s1", calibration_set_id=value)
    with pytest.raises(ValueError, match="calibration_set_id"):
        append_result_row(path, row)
    assert not path.exists()


def test_hardware_row_with_a_calibration_set_id_is_accepted(tmp_path):
    path = tmp_path / "results.csv"
    append_result_row(
        path,
        base_row(
            backend="iqm_spark",
            shots=4096,
            session_id="s1",
            calibration_set_id="6f2c1e00-0000-4000-8000-000000000001",
            repeat_index=0,
        ),
    )
    assert len(pd.read_csv(path)) == 1


def test_rejection_of_a_hardware_row_leaves_earlier_rows_intact(tmp_path):
    path = tmp_path / "results.csv"
    append_result_row(path, base_row())
    with pytest.raises(ValueError):
        append_result_row(path, base_row(backend="iqm_spark", calibration_set_id=None))
    assert len(pd.read_csv(path)) == 1


def test_statevector_row_needs_no_calibration_set_id(tmp_path):
    path = tmp_path / "results.csv"
    append_result_row(path, base_row(backend="statevector", calibration_set_id=""))
    assert len(pd.read_csv(path)) == 1


def test_appending_to_a_file_with_a_foreign_schema_is_refused(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([{"foo": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match RESULT_COLUMNS"):
        append_result_row(path, base_row())
