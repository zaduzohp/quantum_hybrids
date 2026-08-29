"""Results row — one CSV table for the whole project.
Column names and order are fixed; the analysis pipeline stands on them.

Two rejections:
  * unknown or missing column
  * hardware row with an empty, None or placeholder calibration_set_id
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Fixed order — the analysis pipeline stands on it.
RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "dataset",
    "dataset_hash",
    "pca_hash",
    "arm",
    "ansatz_level",
    "R",
    "dilution",
    "socket_params_nominal",
    "socket_params_effective",
    "head_params",
    "seed",
    "init_seed",
    "init_spec_id",
    "backend",
    "shots",
    "session_id",
    "calibration_set_id",
    "repeat_index",
    "eval_subset_id",
    "n_eval",
    "split",
    "accuracy",
    "auc",
    "macro_f1",
    "train_accuracy",
    "theta_displacement",
    "grad_rms_start",
    "grad_rms_end",
    "epochs_run",
    "best_epoch",
    "wall_seconds",
    "git_commit",
    "env_hash",
    "lr_selected",
    "lr_grid",
    # Signed gate margin: a rounded headroom can read as self-contradictory next to
    # `passed = False` when the float sits ~1e-16 below the threshold.
    "g1_margin",
    "patience",
    "max_epochs",
    # Input-scale multiplier, 1.0 by default.
    "feature_scale",
    # Arm D: width M and the key of the (Omega, b) draw. Empty for every other arm.
    "rff_width",
    "rff_omega_seed",
    # Tells a run through the feature cache apart from a recomputed one.
    "used_feature_cache",
    # Closed-form readout control, part of the protocol rather than a diagnostic.
    # Empty for arms A and F, whose socket is trained.
    "ridge_accuracy",
    "ridge_alpha_selected",
)

HARDWARE_BACKEND = "iqm_spark"

# Values that look like an identifier but carry no information.
PLACEHOLDER_CALIBRATION_IDS = frozenset(
    {"", "none", "null", "nan", "n/a", "na", "unknown", "todo", "tbd", "-", "placeholder"}
)


def _calibration_is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text.lower() in PLACEHOLDER_CALIBRATION_IDS


def append_result_row(path, row: dict) -> None:
    """Append one row to the results CSV, or raise and write nothing."""
    path = Path(path)

    keys = set(row)
    unknown = sorted(keys - set(RESULT_COLUMNS))
    if unknown:
        raise ValueError(f"unknown result column(s): {unknown}")
    missing = [c for c in RESULT_COLUMNS if c not in keys]
    if missing:
        raise ValueError(f"missing required result column(s): {missing}")

    if str(row.get("backend", "")).strip() == HARDWARE_BACKEND and _calibration_is_missing(
        row.get("calibration_set_id")
    ):
        raise ValueError(
            "hardware row rejected: calibration_set_id is empty, None or a placeholder "
            f"({row.get('calibration_set_id')!r}). Without it the calibration component "
            "of sigma_hw is undecomposable (CONTRACTS section 5). Nothing was written."
        )

    ordered = {column: row[column] for column in RESULT_COLUMNS}
    frame = pd.DataFrame([ordered], columns=list(RESULT_COLUMNS))

    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return

    existing = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
    existing_columns = existing.columns.tolist()
    if existing_columns != list(RESULT_COLUMNS):
        raise ValueError(
            f"{path} has columns {existing_columns}, which do not match RESULT_COLUMNS. "
            "Appending would mix two schemas in one file."
        )
    combined = pd.concat([existing, frame], ignore_index=True)
    combined.to_csv(path, index=False)
