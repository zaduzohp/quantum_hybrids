"""The main-series driver, its acceptance criteria, and the pipeline-level tests for
pairing, reproducibility and determinism.

Arm A is the expensive part, so these tests exercise the scaffolding — configuration,
dataset resolution, task planning, resume, lr selection, row schema, predictions, the
verdict table — on synthetic data, and touch the simulator only where a test cannot mean
anything without it:

  * the cache the cost estimate stands on gives the same run as the live socket, measured
    on the driver's own code path and not only on socket.frozen_socket_features,
  * no leak: the test rows are invisible to PCA, the scaler and early stopping. Checked by
    perturbing the test split and requiring every non-test number to be unchanged,
  * determinism: the same cell computed twice, in two processes, gives identical rows.

The verdict table is tested exhaustively because it was declared before the measurement:
a table that quietly fails to fire its stop rows is worse than no table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_main_series as a7

from qsocket import contract
from qsocket.datasets import PRODUCTION_DATASET, load_manifest
from qsocket.gates import G1_LR_GRID
from qsocket.head import DILUTION_AXIS, HEAD_PARAM_COUNTS
from qsocket.results import RESULT_COLUMNS
from qsocket.socket import D_BEST_WIDTHS, D_MATCHED_WIDTH
from qsocket.training import CONTRACT_RIDGE_ALPHA_GRID

REPO = Path(__file__).resolve().parents[1]


# --- the configuration is declared, not chosen by the script ----------------


def test_the_grid_is_the_one_the_brief_declares():
    assert a7.DATASET_SEEDS == (11, 22, 33)
    assert a7.DILUTIONS == DILUTION_AXIS == ("linear", "h2", "h4", "h42")
    assert [HEAD_PARAM_COUNTS[d] for d in a7.DILUTIONS] == [6, 15, 29, 295]
    assert a7.ANSATZ_LEVELS == ("L1", "L2")
    assert a7.SEEDS == tuple(range(1, 11))
    assert a7.LR_SELECTION_SEEDS == (1, 2, 3)
    assert a7.R_CONTRACT == 2
    assert set(a7.ARMS) == {"A", "B", "E", "F", "D_matched", "D_best"}
    assert set(a7.TRAINED_ARMS) == {"A", "F"}
    assert set(a7.FROZEN_ARMS) == {"B", "E", "D_matched", "D_best"}


def test_patience_and_the_epoch_budget_are_the_contract_values():
    assert a7.PATIENCE == 30
    assert a7.MAX_EPOCHS == 300
    assert a7.BATCH_SIZE == 64


def test_the_contract_lr_grid_is_imported_and_not_redefined():
    assert a7.CONTRACT_LR_GRID == tuple(G1_LR_GRID) == (1e-3, 3e-3, 1e-2, 3e-2)


def test_the_arm_e_grid_is_the_contract_grid_plus_exactly_one_point_upwards():
    """A separate constant, NOT a modification of the gating grid. The concession is
    one point, declared, and arm E's optimum sits on the edge of this grid too."""
    assert a7.ARM_E_LR_GRID == a7.CONTRACT_LR_GRID + (1e-1,)
    assert len(a7.ARM_E_LR_GRID) == len(a7.CONTRACT_LR_GRID) + 1
    assert tuple(G1_LR_GRID) == (1e-3, 3e-3, 1e-2, 3e-2)  # untouched


def test_the_lr_criterion_averages_over_arms_a_and_b_only():
    """Passengers are measured at the selected lr and reported; they never move it."""
    assert a7.LR_SELECTION_ARMS == ("A", "B")


def test_the_ridge_alpha_grid_is_the_contract_one():
    assert CONTRACT_RIDGE_ALPHA_GRID == (1e-6, 1e-4, 1e-2, 1.0)
    assert a7.CONTRACT_RIDGE_ALPHA_GRID == CONTRACT_RIDGE_ALPHA_GRID


def test_arm_d_has_both_variants_with_the_declared_widths():
    assert D_MATCHED_WIDTH == 5
    assert D_BEST_WIDTHS == (32, 128, 512)


def test_the_script_does_not_assign_to_any_frozen_constant():
    """Standing prohibition: the axis, the gates and the lr grid are read,
    never written. A subscript READ is fine and happens."""
    import re

    source = (REPO / "scripts" / "run_main_series.py").read_text()
    for name in ("HEAD_PARAM_COUNTS", "HIDDEN_WIDTHS", "G1_LR_GRID", "G1_SVC_GRID",
                 "G1_MIN_HEADROOM", "G2_MAX_COMPONENT_SHARE", "DILUTION_AXIS",
                 "RESULT_COLUMNS", "CONTRACT_RIDGE_ALPHA_GRID"):
        assert not re.search(rf"^\s*{name}(\[[^\]]*\])?\s*(\+)?=[^=]", source, re.MULTILINE), (
            f"the driver assigns to {name}"
        )


def test_the_environment_fingerprint_separates_two_thread_counts():
    """Two runs on one stack differing only in thread count produced different
    accuracies for the trained arms. An env_hash blind to that reports one environment
    where there were two, which is why the drift was invisible in the rows."""
    import os

    baseline = a7.env_hash()
    previous = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "8"
        assert a7.env_hash() != baseline
    finally:
        if previous is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = previous
    assert a7.env_hash() == baseline


# --- datasets: hashes asserted ---------------------------


def test_generator_seed_11_resolves_to_the_registered_production_dataset():
    name, out_dir = a7.dataset_location(11)
    assert name == PRODUCTION_DATASET
    assert out_dir == a7.DEFAULT_DATA_DIR
    assert a7.frozen_name_for(11) == PRODUCTION_DATASET


def test_the_other_generator_seeds_live_outside_the_frozen_data_directory():
    """Prohibition: the frozen production dataset is never replaced or re-frozen. Seeds 22
    and 33 are generated into their own directory and hash-asserted there."""
    for seed in (22, 33):
        name, out_dir = a7.dataset_location(seed)
        assert out_dir == a7.A7_DATA_DIR != a7.DEFAULT_DATA_DIR
        assert name.endswith(f"seed{seed}")


def test_the_pinned_hash_of_seed_11_is_the_registered_one():
    """The generation path is credible for seeds 22 and 33 exactly because running it on
    seed 11 reproduces the frozen production artefact, byte for byte."""
    registered = load_manifest(PRODUCTION_DATASET)
    dataset_prefix, pca_prefix, file_prefix = a7.GENERATED_HASH_PREFIXES[11]
    assert registered["dataset_hash"].startswith(dataset_prefix)
    assert registered["pca_hash"].startswith(pca_prefix)
    assert registered["file_sha256"].startswith(file_prefix)


def test_a_dataset_present_on_disk_is_checked_and_not_regenerated(monkeypatch):
    called = []
    monkeypatch.setattr(contract, "generate_and_freeze", lambda *a, **k: called.append(k))
    for seed in a7.DATASET_SEEDS:
        name, out_dir = a7.dataset_location(seed)
        if not (out_dir / f"{name}.npz").exists():
            pytest.skip(f"{name} is not on disk yet")
        a7.ensure_dataset(seed)
    assert called == []


def test_a_missing_dataset_with_generation_switched_off_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "A7_DATA_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        a7.ensure_dataset(22, allow_generate=False)


def test_the_pca_and_the_scaler_were_fitted_on_the_training_rows_only():
    """The test rows never took part in fitting anything, and the manifest of every
    generator seed in the run records it.
    """
    for seed in a7.DATASET_SEEDS:
        name, out_dir = a7.dataset_location(seed)
        if not (out_dir / f"{name}.manifest.json").exists():
            pytest.skip(f"{name} is not on disk yet")
        manifest = load_manifest(name, out_dir=out_dir)
        assert manifest["pca"]["fitted_on"] == "train split only (4200 rows)"
        assert manifest["scaling"]["fitted_on"].startswith("train split only (4200 rows)")
        assert manifest["split_sizes"] == {"train": 4200, "val": 600, "test": 1200}


# --- task planning: the shape of the grid, and the rows missing by construction ------


CONTEXT = {"run_id": "test", "commit": "abc", "environment": "def"}


def _plan_main(**overrides):
    options = dict(
        dataset_seeds=(11,), dilutions=a7.DILUTIONS, ansatz_levels=a7.ANSATZ_LEVELS,
        seeds=a7.SEEDS, widths=D_BEST_WIDTHS, ranks={}, margins={11: 0.01},
        context=CONTEXT, done=set(),
    )
    options.update(overrides)
    cell_lr = {
        (11, dilution, ansatz): 1e-2
        for dilution in a7.DILUTIONS for ansatz in a7.ANSATZ_LEVELS
    }
    options.setdefault("lrs", {
        "cell_lr": cell_lr,
        "arm_e_lr": {(11, dilution): 1e-1 for dilution in a7.DILUTIONS},
        "d_best_lr": {(11, width): 1e-2 for width in D_BEST_WIDTHS},
    })
    return a7.plan_main_stage(**options)


def _cells(tasks, arm):
    return [
        (task["dataset_seed"], task["ansatz_level"], task["seed"], task["width"],
         cell["dilution"], cell["lr"])
        for task in tasks if task["arm"] == arm for cell in task["cells"]
    ]


def test_the_main_grid_has_the_run_counts_of_the_cost_estimate():
    """240 arm-A runs and 120 arm-F runs per the whole grid, i.e. 80
    and 40 per generator seed. Any other number means the grid changed."""
    tasks = _plan_main()
    assert len(_cells(tasks, "A")) == len(a7.DILUTIONS) * len(a7.ANSATZ_LEVELS) * len(a7.SEEDS)
    assert len(_cells(tasks, "A")) == 80  # x 3 generator seeds = 240
    assert len(_cells(tasks, "F")) == len(a7.DILUTIONS) * len(a7.SEEDS) == 40
    assert len(_cells(tasks, "B")) == 80
    assert len(_cells(tasks, "E")) == 40
    assert len(_cells(tasks, "D_matched")) == 40
    assert len(_cells(tasks, "D_best")) == len(D_BEST_WIDTHS) * len(a7.SEEDS) == 30


def test_the_ansatz_free_arms_are_computed_once_not_twice():
    tasks = _plan_main()
    for arm in a7.ANSATZ_FREE_ARMS:
        levels = {task["ansatz_level"] for task in tasks if task["arm"] == arm}
        assert levels in ({""}, {a7.PRODUCT_ANSATZ}), (arm, levels)
    assert {t["ansatz_level"] for t in tasks if t["arm"] == "F"} == {a7.PRODUCT_ANSATZ}
    assert {t["ansatz_level"] for t in tasks if t["arm"] == "A"} == {"L1", "L2"}


def test_d_best_is_off_the_dilution_axis_and_carries_its_width():
    tasks = _plan_main()
    d_best = [task for task in tasks if task["arm"] == "D_best"]
    assert {task["width"] for task in d_best} == set(D_BEST_WIDTHS)
    assert {cell["dilution"] for task in d_best for cell in task["cells"]} == {"linear"}
    # Every other arm carries no width at all.
    assert {task["width"] for task in tasks if task["arm"] != "D_best"} == {None}


def test_when_the_two_ansatz_levels_pick_different_lrs_the_ansatz_free_arms_run_at_both():
    """The only honest way to keep "one lr per cell" and "computed once" at the same time:
    an arm with no ansatz dimension is run once per DISTINCT cell lr. One run in the
    expected case, two when L1 and L2 disagreed."""
    lrs = {
        "cell_lr": {(11, d, a): (1e-2 if a == "L1" else 3e-2)
                    for d in a7.DILUTIONS for a in a7.ANSATZ_LEVELS},
        "arm_e_lr": {(11, d): 1e-1 for d in a7.DILUTIONS},
        "d_best_lr": {(11, w): 1e-2 for w in D_BEST_WIDTHS},
    }
    tasks = _plan_main(lrs=lrs, seeds=(1,))
    assert sorted({lr for *_, lr in _cells(tasks, "F")}) == [1e-2, 3e-2]
    assert len(_cells(tasks, "F")) == 2 * len(a7.DILUTIONS)
    assert sorted({lr for *_, lr in _cells(tasks, "D_matched")}) == [1e-2, 3e-2]


def test_the_lr_stage_covers_the_whole_grid_for_arms_a_and_b():
    tasks = a7.plan_lr_stage(
        dataset_seeds=(11,), dilutions=a7.DILUTIONS, ansatz_levels=a7.ANSATZ_LEVELS,
        seeds=a7.LR_SELECTION_SEEDS, widths=D_BEST_WIDTHS, ranks={}, context=CONTEXT,
        done=set(),
    )
    a_cells = _cells(tasks, "A")
    assert len(a_cells) == 4 * 2 * 3 * len(a7.CONTRACT_LR_GRID) == 96  # x3 seeds = 288
    assert {lr for *_, lr in a_cells} == set(a7.CONTRACT_LR_GRID)
    assert {lr for *_, lr in _cells(tasks, "E")} == set(a7.ARM_E_LR_GRID)
    assert {lr for *_, lr in _cells(tasks, "D_best")} == set(a7.CONTRACT_LR_GRID)
    # Arm A is one task per expensive run, so the workers can be kept busy.
    assert all(len(t["cells"]) == 1 for t in tasks if t["arm"] == "A")


def test_planning_skips_cells_whose_rows_are_already_on_disk():
    """Resume, the requirement a 9-hour run stands on."""
    tasks = _plan_main(seeds=(1,))
    done = set()
    for task in tasks:
        if task["arm"] != "A":
            continue
        for cell in task["cells"]:
            for split in a7.SPLITS_REPORTED:
                done.add(a7._key((PRODUCTION_DATASET, "A", task["ansatz_level"],
                                  cell["dilution"], task["seed"], split, "",
                                  f"{cell['lr']:g}")))
    remaining = _plan_main(seeds=(1,), done=done)
    assert _cells(remaining, "A") == []
    assert _cells(remaining, "B")  # untouched arms are still planned


def test_row_key_survives_a_csv_round_trip(tmp_path):
    import pandas as pd

    from qsocket.results import append_result_row

    row = {column: "" for column in RESULT_COLUMNS}
    row.update({
        "dataset": PRODUCTION_DATASET, "arm": "D_best", "ansatz_level": "",
        "dilution": "linear", "seed": 3, "split": "test", "rff_width": 128,
        "lr_selected": 0.01, "backend": "pennylane",
    })
    path = tmp_path / "results.csv"
    append_result_row(path, row)
    written = pd.read_csv(path, keep_default_na=False).to_dict("records")[0]
    assert a7.row_key(written) == a7.row_key(row)
    assert a7.existing_row_keys(path) == {a7.row_key(row)}


# --- lr selection ----------------------------


def _lr_row(dataset_seed, arm, ansatz, dilution, seed, lr, accuracy, width=None):
    return {
        "dataset_seed": dataset_seed, "arm": arm, "ansatz_level": ansatz,
        "dilution": dilution, "seed": seed, "lr": lr, "val_accuracy": accuracy,
        "width": width,
    }


def _selection_rows(accuracy_by_lr, *, arms=("A", "B")):
    rows = []
    for dilution in ("linear",):
        for ansatz in ("L1",):
            for arm in arms:
                for lr, accuracy in accuracy_by_lr.items():
                    for seed in (1, 2, 3):
                        rows.append(_lr_row(11, arm, ansatz, dilution, seed, lr, accuracy))
    for lr in a7.ARM_E_LR_GRID:
        for seed in (1, 2, 3):
            rows.append(_lr_row(11, "E", "", "linear", seed, lr, 0.5 + lr))
    for width in D_BEST_WIDTHS:
        for lr in a7.CONTRACT_LR_GRID:
            for seed in (1, 2, 3):
                rows.append(_lr_row(11, "D_best", "", "linear", seed, lr, 0.6, width=width))
    return rows


def test_the_selected_lr_is_the_argmax_of_the_mean_over_a_and_b():
    rows = _selection_rows({1e-3: 0.60, 3e-3: 0.70, 1e-2: 0.75, 3e-2: 0.71})
    lrs = a7.select_all_lrs(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=D_BEST_WIDTHS, seeds=(1, 2, 3),
    )
    assert lrs["cell_lr"][(11, "linear", "L1")] == 1e-2
    assert lrs["cell_selection"][(11, "linear", "L1")].selection_arms == ("A", "B")


def test_a_tie_in_the_lr_criterion_goes_to_the_lower_lr():
    rows = _selection_rows({1e-3: 0.60, 3e-3: 0.75, 1e-2: 0.75, 3e-2: 0.70})
    lrs = a7.select_all_lrs(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=D_BEST_WIDTHS, seeds=(1, 2, 3),
    )
    assert lrs["cell_lr"][(11, "linear", "L1")] == 3e-3


def test_arm_e_is_selected_on_its_own_wider_grid():
    rows = _selection_rows({lr: 0.7 for lr in a7.CONTRACT_LR_GRID})
    lrs = a7.select_all_lrs(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=D_BEST_WIDTHS, seeds=(1, 2, 3),
    )
    # accuracy 0.5 + lr is maximal at the extra point, which only arm E's grid has.
    assert lrs["arm_e_lr"][(11, "linear")] == 1e-1
    assert lrs["arm_e_selection"][(11, "linear")].grid == a7.ARM_E_LR_GRID


def test_the_selection_refuses_a_partial_grid():
    rows = [row for row in _selection_rows({lr: 0.7 for lr in a7.CONTRACT_LR_GRID})
            if not (row["arm"] == "A" and row["lr"] == 3e-2)]
    with pytest.raises(ValueError, match="missing"):
        a7.select_all_lrs(
            rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
            widths=D_BEST_WIDTHS, seeds=(1, 2, 3),
        )


# --- the row: schema, ridge, predictions ---------------------------------------------


def _toy_splits(n_train=64, n_val=48, n_test=48, seed=0):
    """A tiny separable-ish problem in FEATURE_RANGE. Small enough for a real train_model
    run in a test, and the labels depend on the features so accuracy is not degenerate."""
    rng = np.random.default_rng(seed)
    def block(n):
        X = rng.uniform(-np.pi / 4, np.pi / 4, size=(n, 5))
        y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, 1, -1)
        return X, y
    return {"train": block(n_train), "val": block(n_val), "test": block(n_test)}


def _manifest():
    return {"dataset_hash": "0" * 64, "pca_hash": "1" * 64, "frozen_name": "toy"}


def _run_cell(arm, *, monkeypatch, splits=None, dilution="linear", ansatz="L1", seed=1,
              lr=1e-2, width=None, cached=None):
    monkeypatch.setattr(a7, "MAX_EPOCHS", 4)
    monkeypatch.setattr(a7, "PATIENCE", 2)
    rows, correctness, _ = a7.run_cell(
        splits if splits is not None else _toy_splits(),
        manifest=_manifest(), dataset="toy_seed11", arm=arm,
        ansatz_level=a7.ansatz_of(arm, ansatz), dilution=dilution, seed=seed, lr=lr,
        lr_grid=a7.CONTRACT_LR_GRID, width=width, cached_features=cached,
        effective_rank=35, g1_margin=0.0123, run_id="test", commit="abc",
        environment="def",
    )
    return rows, correctness


@pytest.mark.parametrize(
    "arm,width",
    [("A", None), ("B", None), ("E", None), ("F", None), ("D_matched", None), ("D_best", 32)],
)
def test_every_arm_produces_two_complete_rows_one_per_split(arm, width, monkeypatch):
    rows, correct = _run_cell(arm, monkeypatch=monkeypatch, width=width)
    assert [row["split"] for row in rows] == list(a7.SPLITS_REPORTED) == ["test", "val"]
    for row in rows:
        assert set(row) == set(RESULT_COLUMNS)
        assert row["n_eval"] == (48 if row["split"] == "test" else 48)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["auc"] <= 1.0
        assert row["patience"] == 2 and row["max_epochs"] == 4  # the monkeypatched budget
        assert row["feature_scale"] == 1.0
        assert row["g1_margin"] == pytest.approx(0.0123)
        assert row["lr_grid"] == "0.001 0.003 0.01 0.03"
        assert row["lr_selected"] == 1e-2
    assert correct.dtype == bool and correct.size == 48


def test_the_appended_columns_are_all_filled_for_the_arm_that_needs_them(monkeypatch):
    rows, _ = _run_cell("D_best", monkeypatch=monkeypatch, width=128)
    row = rows[0]
    assert row["rff_width"] == 128
    assert str(row["rff_omega_seed"]).startswith("0x")
    assert row["used_feature_cache"] is True
    assert row["ridge_accuracy"] != ""
    assert row["ridge_alpha_selected"] in CONTRACT_RIDGE_ALPHA_GRID
    assert row["head_params"] == 129  # linear readout on M features, off the axis
    assert row["socket_params_nominal"] == 128 * 6  # (Omega, b), all frozen


def test_the_init_seed_is_written_as_hex_so_a_csv_round_trip_cannot_corrupt_it(monkeypatch):
    """derive() returns 64 unsigned bits, which does not fit int64: written as a number it
    would come back from pandas as 2.15e+18 and stop identifying the draw it names."""
    rows, _ = _run_cell("A", monkeypatch=monkeypatch)
    assert rows[0]["init_seed"].startswith("0x")
    assert int(rows[0]["init_seed"], 16) == a7.derive(1, "L1", 2)


def test_the_ridge_control_is_computed_for_frozen_sockets_and_only_for_them(monkeypatch):
    """For arms A and F the socket is TRAINED, so "the closed form on
    the socket features" has no defined argument and the number would carry no
    interpretation. The column stays empty rather than being filled with something."""
    for arm, width in (("B", None), ("E", None), ("D_matched", None), ("D_best", 32)):
        rows, _ = _run_cell(arm, monkeypatch=monkeypatch, width=width)
        assert rows[0]["ridge_accuracy"] != "", arm
        assert rows[0]["ridge_alpha_selected"] in CONTRACT_RIDGE_ALPHA_GRID, arm
    for arm in ("A", "F"):
        rows, _ = _run_cell(arm, monkeypatch=monkeypatch)
        assert rows[0]["ridge_accuracy"] == "", arm
        assert rows[0]["ridge_alpha_selected"] == "", arm


def test_the_ridge_gap_is_reported_and_never_asserted_away(monkeypatch):
    rows, _ = _run_cell("B", monkeypatch=monkeypatch)
    gap = float(rows[0]["accuracy"]) - float(rows[0]["ridge_accuracy"])
    assert np.isfinite(gap)  # a number, whatever its sign


def test_theta_moves_for_the_trained_arms_and_is_exactly_zero_for_the_frozen_ones(monkeypatch):
    for arm in ("A", "F"):
        rows, _ = _run_cell(arm, monkeypatch=monkeypatch, lr=3e-2)
        assert rows[0]["theta_displacement"] > 0.0, arm
    for arm, width in (("B", None), ("E", None), ("D_matched", None), ("D_best", 32)):
        rows, _ = _run_cell(arm, monkeypatch=monkeypatch, width=width)
        assert rows[0]["theta_displacement"] == 0.0, arm


def test_predictions_round_trip_through_packbits(tmp_path, monkeypatch):
    """McNemar works on discordant pairs, so WHICH test rows each arm
    got right cannot be recovered from an accuracy afterwards."""
    rows, correct = _run_cell("A", monkeypatch=monkeypatch)
    prediction = {
        "dataset_seed": 11, "arm": "A", "ansatz_level": "L1", "dilution": "linear",
        "seed": 1, "width": None, "lr": 0.01, "correct": correct,
    }
    path = a7.write_prediction(tmp_path, prediction)
    assert path.exists()
    restored = a7.read_prediction(path)
    assert restored.tolist() == correct.tolist()
    assert float(restored.mean()) == pytest.approx(float(rows[0]["accuracy"]))


def test_the_socket_parameter_counts_are_the_frozen_ones():
    assert a7.socket_params_nominal("A") == 35 == 15 * a7.R_CONTRACT + 5
    assert a7.socket_params_nominal("B") == 35
    assert a7.socket_params_nominal("F") == 35
    assert a7.socket_params_nominal("E") == 0
    assert a7.socket_params_nominal("D_matched") == 30  # 5x5 Omega + 5 phases, all frozen
    assert a7.socket_params_nominal("D_best", width=512) == 512 * 6


# --- the cache is the same run, on the driver's own path ------------------------


def test_TG_the_driver_cache_path_reproduces_the_live_socket_run(monkeypatch):
    """The driver's cached arm-B run must be the same run as one trained through the live
    circuit — a stronger claim than tests/test_feature_cache.py, which only compares
    frozen_socket_features against the live socket.

    The live half recomputes the socket output at every epoch, the cost the cache exists
    to avoid; affordable here only because the toy problem is tiny and the epoch budget is
    monkeypatched down.
    """
    from qsocket.training import TrainConfig, train_model

    splits = _toy_splits(seed=3)
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    # --- live: the frozen quantum socket inside the training loop --------------------
    live_socket = a7.build_socket_for("B", ansatz="L1", seed=2)
    live_head = a7.head_for("B", "linear", seed=2)
    live = train_model(
        live_socket, live_head, X_tr, y_tr, X_val, y_val,
        cfg=TrainConfig(lr=1e-2, batch_size=a7.BATCH_SIZE, max_epochs=4, patience=2),
        seed=2,
    )
    live_test = a7.metrics_from_logits(a7.logits_of(live_socket, live_head, X_te), y_te)

    # --- cached: the path the driver takes for every frozen arm ----------------------
    socket = a7.build_socket_for("B", ansatz="L1", seed=2)
    features = {
        split: a7.frozen_socket_features(socket, splits[split][0])
        for split in ("train", "val", "test")
    }
    rows, correct = _run_cell(
        "B", monkeypatch=monkeypatch, splits=splits, seed=2, cached=features
    )
    cached = {row["split"]: row for row in rows}

    # The trajectory must be identical; the two validation accuracies count the same
    # correct rows but reduce in float32 and float64 respectively, so they agree to
    # float32 resolution rather than bit for bit.
    assert cached["val"]["accuracy"] == pytest.approx(live.val_accuracy, abs=1e-6)
    assert cached["val"]["train_accuracy"] == live.train_accuracy
    assert cached["test"]["accuracy"] == live_test["accuracy"]
    assert cached["test"]["auc"] == live_test["auc"]
    assert cached["val"]["best_epoch"] == live.best_epoch
    assert cached["val"]["epochs_run"] == live.epochs_run
    assert correct.tolist() == live_test["correct"].tolist()
    assert cached["test"]["used_feature_cache"] is True


def test_TG_the_cache_refuses_a_trainable_socket():
    """The failure mode the cache CANNOT be allowed to have: a cached trainable socket is
    stale from the first optimiser step, and stale in a way nothing downstream notices."""
    trainable = a7.build_socket_for("A", ansatz="L1", seed=1)
    with pytest.raises(ValueError, match="trainable"):
        a7.frozen_socket_features(trainable, _toy_splits()["val"][0])


# --- no leak -------------------------------------------------------------------


def test_TH_train_model_cannot_even_be_given_a_test_set():
    """Enforced by the signature, not by discipline."""
    import inspect

    from qsocket.training import train_model

    parameters = list(inspect.signature(train_model).parameters)
    assert "X_te" not in parameters and "X_test" not in parameters
    assert [p for p in parameters if p.startswith("X")] == ["X_tr", "X_val"]


@pytest.mark.parametrize("arm,width", [("A", None), ("B", None), ("D_best", 32)])
def test_TH_perturbing_the_test_split_changes_nothing_but_the_test_metrics(arm, width, monkeypatch):
    """The behavioural half of the no-leak claim. If early stopping, the head, the socket
    or the ridge control had ever seen the test rows, replacing them with noise would move
    a validation number. Nothing here may move except the test row itself.
    """
    splits = _toy_splits(seed=5)
    rng = np.random.default_rng(99)
    poisoned = dict(splits)
    X_te, y_te = splits["test"]
    poisoned["test"] = (rng.uniform(-np.pi / 4, np.pi / 4, size=X_te.shape), -y_te)

    clean_rows, _ = _run_cell(arm, monkeypatch=monkeypatch, splits=splits, width=width)
    dirty_rows, _ = _run_cell(arm, monkeypatch=monkeypatch, splits=poisoned, width=width)

    clean = {row["split"]: row for row in clean_rows}
    dirty = {row["split"]: row for row in dirty_rows}
    for column in ("accuracy", "auc", "macro_f1", "train_accuracy", "best_epoch",
                   "epochs_run", "theta_displacement", "grad_rms_start", "grad_rms_end",
                   "ridge_alpha_selected"):
        assert clean["val"][column] == dirty["val"][column], column
    assert clean["val"]["ridge_accuracy"] == dirty["val"]["ridge_accuracy"]


def test_TH_the_lr_selection_never_reads_a_test_number():
    """The lr is chosen on VALIDATION accuracy. Choosing it on the test split would make
    every reported accuracy optimistically biased by the grid it was selected with."""
    import inspect

    source = inspect.getsource(a7.select_all_lrs)
    assert "val_accuracy" in source
    assert "test_accuracy" not in source


# --- determinism --------------------------------------------------------------------


def test_TI_the_same_cell_computed_twice_gives_identical_rows(monkeypatch):
    """Two runs of one cell inside one process: identical accuracies, identical epochs,
    identical predictions. Wall time and the timestamp are the only things allowed to
    differ, and they are not inputs to anything."""
    first_rows, first_correct = _run_cell("A", monkeypatch=monkeypatch, seed=4)
    second_rows, second_correct = _run_cell("A", monkeypatch=monkeypatch, seed=4)
    volatile = {"timestamp_utc", "wall_seconds"}
    for a, b in zip(first_rows, second_rows):
        assert {k: v for k, v in a.items() if k not in volatile} == \
               {k: v for k, v in b.items() if k not in volatile}
    assert first_correct.tolist() == second_correct.tolist()


def test_TI_a_worker_process_reproduces_what_the_parent_computed():
    """Determinism ACROSS PROCESSES, which is what the run does: the grid is executed in a spawn
    pool, so determinism inside one interpreter is not the claim that matters."""
    import multiprocessing as mp

    task = a7.make_task(
        stage="main", dataset_seed=11, arm="D_matched", ansatz_level="", seed=7,
        width=None, cells=[{"dilution": "linear", "lr": 1e-2}],
        lr_grid=a7.CONTRACT_LR_GRID, effective_rank=None, g1_margin=0.01,
        run_id="test", commit="abc", environment="def",
    )
    name, out_dir = a7.dataset_location(11)
    if not (out_dir / f"{name}.npz").exists():
        pytest.skip("the production dataset is not on disk")

    context = mp.get_context("spawn")
    with context.Pool(processes=2, initializer=a7._worker_init) as pool:
        first, second = pool.map(a7._worker_run, [task, task])
    volatile = {"timestamp_utc", "wall_seconds"}
    for a, b in zip(first["rows"], second["rows"]):
        assert {k: v for k, v in a.items() if k not in volatile} == \
               {k: v for k, v in b.items() if k not in volatile}
    assert first["predictions"][0]["correct"].tolist() == \
           second["predictions"][0]["correct"].tolist()


def test_TI_the_in_run_check_compares_the_overlapping_cells_and_fails_on_a_mismatch():
    """The run re-measures arms A and B at the selected lr and seeds 1-3, hours after the
    lr stage measured them in another process. The check must FAIL when they disagree,
    otherwise it is decoration."""
    lr_rows = [
        {"dataset_seed": 11, "arm": "A", "ansatz_level": "L1", "dilution": "linear",
         "seed": 1, "lr": 0.01, "val_accuracy": 0.75}
    ]
    def result_row(accuracy):
        return {
            "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": "A",
            "ansatz_level": "L1", "dilution": "linear", "seed": 1, "split": "val",
            "lr_selected": 0.01, "accuracy": accuracy,
        }
    agreeing = a7.determinism_check([result_row(0.75)], lr_rows)
    assert agreeing["passed"] and agreeing["overlapping_cells_compared"] == 1
    disagreeing = a7.determinism_check([result_row(0.75 + 1e-12)], lr_rows)
    assert not disagreeing["passed"] and len(disagreeing["mismatches"]) == 1


# --- the verdict table, declared before the measurement -----------------------------


def _summary(*, delta_ab, theta, acc, gate_passed=True, budget_fraction=0.0, sd=0.01):
    cell = "ds11|linear|L1"
    def paired(mean):
        return {
            "n": 10, "mean": mean, "sd": sd, "min": mean, "max": mean,
            "seeds": list(range(1, 11)), "unpaired_left": [], "unpaired_right": [],
            "sigma_delta": sd, "mde_from_this_series": a7.mde(sd, 10),
            "differences": [mean] * 10,
        }
    def arm(mean, displacement):
        return {
            "test": {"n": 10, "mean": mean, "sd": sd, "min": mean, "max": mean},
            "val": {"n": 10, "mean": mean, "sd": sd, "min": mean, "max": mean},
            "lr_selected": 0.01,
            "theta_displacement": {"n": 10, "mean": displacement, "sd": 0.0,
                                   "min": displacement, "max": displacement},
            "epochs_run": {"n": 10, "mean": 100.0, "sd": 0.0, "min": 100, "max": 100},
            "best_epoch": {"n": 10, "mean": 70.0, "sd": 0.0, "min": 70, "max": 70},
        }
    return {
        "per_arm": {cell: {
            "A": arm(acc["A"], theta), "B": arm(acc["B"], 0.0), "E": arm(acc["E"], 0.0),
            "F": arm(acc.get("F", acc["A"]), theta),
            "D_matched": arm(acc.get("D_matched", acc["B"]), 0.0),
        }},
        "estimands": {cell: {
            "delta_AB": paired(delta_ab),
            "delta_AE": paired(acc["A"] - acc["E"]),
            "delta_AF": paired(0.0),
            "delta_BD_matched": paired(0.0),
            "decomposition_delta_AE": {},
        }},
        "gates": {"ds11": {
            "passed": gate_passed, "g1_margin": 0.01 if gate_passed else -0.01,
            "g1": {"failures": [] if gate_passed else ["headroom"]},
            "g2": {"failures": []},
        }},
        "epoch_budget": {"ALL": {"runs": 100, "hit_budget": int(100 * budget_fraction),
                                 "fraction": budget_fraction}},
    }


def _rows_of(summary, row_name):
    return [row for row in a7.verdicts(summary) if row["row"] == row_name]


def test_the_stop_row_for_an_optimiser_failure_fires():
    """Delta ~ 0 together with theta ~ 0 is not "training is unnecessary" but an optimiser
    failure — a different claim, and one that may not be written in after the fact.
    """
    summary = _summary(delta_ab=0.0, theta=0.0, acc={"A": 0.70, "B": 0.70, "E": 0.60})
    rows = _rows_of(summary, "Delta_AB ~ 0 with theta_displacement ~ 0")
    assert [row["verdict"] for row in rows] == ["STOP"]


def test_the_optimiser_failure_row_does_not_fire_when_theta_moved():
    summary = _summary(delta_ab=0.0, theta=0.5, acc={"A": 0.70, "B": 0.70, "E": 0.60})
    assert _rows_of(summary, "Delta_AB ~ 0 with theta_displacement ~ 0") == []


def test_the_stop_row_for_the_architecture_scenario_fires():
    """acc(A) ~ acc(E) with acc(B) < acc(E): the socket adds nothing and training only
    undoes the damage of a random initialisation. That is the most likely scenario, so the
    table must catch it.
    """
    summary = _summary(delta_ab=0.08, theta=0.5, acc={"A": 0.70, "B": 0.62, "E": 0.70})
    rows = _rows_of(summary, "acc(A) ~ acc(E) with acc(B) < acc(E)")
    assert [row["verdict"] for row in rows] == ["STOP"]


def test_the_architecture_row_does_not_fire_when_the_socket_helps():
    summary = _summary(delta_ab=0.08, theta=0.5, acc={"A": 0.80, "B": 0.72, "E": 0.60})
    assert _rows_of(summary, "acc(A) ~ acc(E) with acc(B) < acc(E)") == []


def test_a_delta_below_the_mde_is_undecidable_and_never_called_absent():
    summary = _summary(delta_ab=0.001, theta=0.5, acc={"A": 0.70, "B": 0.699, "E": 0.60})
    rows = _rows_of(summary, "|Delta| below the MDE of this series")
    assert rows and rows[0]["verdict"] == "UNDECIDABLE at n=10"
    assert "do NOT write 'there is no effect'" in rows[0]["why"]


def test_a_failing_gate_is_a_stop_because_there_is_no_seed_to_substitute():
    summary = _summary(delta_ab=0.08, theta=0.5, acc={"A": 0.80, "B": 0.72, "E": 0.60},
                       gate_passed=False)
    rows = _rows_of(summary, "a generator seed fails G1/G2 at the final lr")
    assert [row["verdict"] for row in rows] == ["STOP"]
    assert "Z8" in rows[0]["why"]


def test_more_than_a_fifth_of_the_runs_hitting_the_budget_is_noted_with_its_number():
    summary = _summary(delta_ab=0.08, theta=0.5, acc={"A": 0.80, "B": 0.72, "E": 0.60},
                       budget_fraction=0.25)
    rows = _rows_of(summary, "> 20 % of runs hit the 300-epoch budget")
    assert rows and rows[0]["numbers"]["hit_budget"] == 25
    assert a7.BUDGET_HIT_NOTE_FRACTION == 0.20


def test_the_mde_is_recomputed_from_the_sigma_of_this_series_and_not_hard_coded():
    """The pilot MDE comes from one point of the axis, one ansatz and one generator seed.
    It orients; it never decides.
    """
    from scipy.stats import t

    sigma = 0.0488
    expected = (t.ppf(0.975, 9) + t.ppf(0.80, 9)) / np.sqrt(10) * sigma
    assert a7.mde(sigma, 10) == pytest.approx(expected)
    source = (REPO / "scripts" / "run_main_series.py").read_text()
    assert "0.0486" not in source.replace("0.0486 / 0.0075", "")


def test_the_binomial_se_on_the_test_split_is_the_documented_number():
    assert a7.binomial_se(1200) == pytest.approx(0.0144, abs=5e-5)


# --- statistics: what the report must contain ---------------------------------------


def test_paired_statistics_pairs_by_seed_and_reports_unpaired_seeds():
    left = {1: 0.80, 2: 0.82, 3: 0.79}
    right = {1: 0.75, 2: 0.80, 4: 0.70}
    stats = a7.paired_statistics(left, right)
    assert stats["n"] == 2 and stats["seeds"] == [1, 2]
    assert stats["mean"] == pytest.approx(np.mean([0.05, 0.02]))
    assert stats["unpaired_left"] == [3] and stats["unpaired_right"] == [4]


def test_the_decomposition_of_delta_ae_is_exact_in_every_cell():
    """Delta_AE = Delta_AB + (acc(B) - acc(E)), in every cell. It is what tells "training
    helps" apart from "the socket adds nothing".
    """
    rows = []
    accuracy = {"A": 0.80, "B": 0.70, "E": 0.74, "F": 0.78, "D_matched": 0.69}
    for arm, value in accuracy.items():
        for seed in (1, 2, 3):
            rows.append({
                "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": arm,
                "ansatz_level": a7.ansatz_of(arm, "L1"), "dilution": "linear",
                "rff_width": "", "seed": seed, "split": "test",
                "accuracy": value + 0.001 * seed, "theta_displacement": 0.0,
                "epochs_run": 100, "best_epoch": 50, "ridge_accuracy": "",
                "ridge_alpha_selected": "", "lr_selected": "",
            })
            rows.append({**rows[-1], "split": "val"})
    summary = a7.summarise(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=(), seeds=(1, 2, 3), gates={},
        lrs={"cell_lr": {}, "arm_e_lr": {}, "d_best_lr": {}},
    )
    decomposition = summary["estimands"]["ds11|linear|L1"]["decomposition_delta_AE"]
    assert decomposition["residual"] == pytest.approx(0.0, abs=1e-12)
    assert decomposition["delta_AB"] == pytest.approx(0.10)
    assert decomposition["acc_B_minus_acc_E"] == pytest.approx(-0.04)


def test_every_arm_gets_its_own_mean_and_sigma_not_only_the_differences():
    """Without per-arm means the scenario "Delta_AB > 0 while Delta_AE ~ 0" is
    indistinguishable from success.
    """
    rows = []
    for arm in ("A", "B", "E", "F", "D_matched"):
        for seed in (1, 2):
            rows.append({
                "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": arm,
                "ansatz_level": a7.ansatz_of(arm, "L1"), "dilution": "linear",
                "rff_width": "", "seed": seed, "split": "test", "accuracy": 0.7 + 0.01 * seed,
                "theta_displacement": 0.0, "epochs_run": 300, "best_epoch": 250,
                "ridge_accuracy": "", "ridge_alpha_selected": "", "lr_selected": "",
            })
            rows.append({**rows[-1], "split": "val"})
    summary = a7.summarise(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=(), seeds=(1, 2), gates={}, lrs={"cell_lr": {}, "arm_e_lr": {}, "d_best_lr": {}},
    )
    per_arm = summary["per_arm"]["ds11|linear|L1"]
    assert set(per_arm) == {"A", "B", "E", "F", "D_matched"}
    for arm, block in per_arm.items():
        assert block["test"]["n"] == 2
        assert np.isfinite(block["test"]["mean"]) and np.isfinite(block["test"]["sd"])
    # And the budget counter sees that every run hit the epoch limit.
    assert summary["epoch_budget"]["ALL"]["fraction"] == 1.0


def test_dataset_seed_is_read_back_from_the_frozen_name_and_never_stored_twice():
    assert a7.dataset_seed_of(PRODUCTION_DATASET) == 11
    assert "dataset_seed" not in RESULT_COLUMNS
    annotated = a7.annotate([{"dataset": PRODUCTION_DATASET}])
    assert annotated[0]["dataset_seed"] == 11


def test_the_dry_run_is_the_configuration_the_brief_asks_for():
    """The dry-run grid: 2 seeds x 1 dilution x 2 ansatze x every arm."""
    assert a7.DRY_RUN["seeds"] == (1, 2)
    assert len(a7.DRY_RUN["dilutions"]) == 1
    assert a7.DRY_RUN["ansatz_levels"] == a7.ANSATZ_LEVELS
    assert a7.DRY_RUN["widths"] == D_BEST_WIDTHS


# --- the third uncertainty account: discordant pairs --------------------------------


def _lrs(cell_lr, arm_e_lr=None, d_best_lr=None) -> dict:
    """The lr maps summarise() and discordant_pairs() read, in the shape run() builds."""
    return {
        "cell_lr": dict(cell_lr),
        "arm_e_lr": dict(arm_e_lr or {}),
        "d_best_lr": dict(d_best_lr or {}),
    }


def test_discordant_pair_counts_are_the_input_of_mcnemar(tmp_path):
    """Three uncertainty accounts are reported side by side. The first two come from
    accuracies; this one cannot be recovered afterwards, which is why the per-test-row
    predictions are written. The series reports the counts, the analysis runs the test.
    """
    correctness = {
        "A": np.array([True, True, True, False]),
        "B": np.array([True, False, False, False]),
        "E": np.array([True, True, False, False]),
        "F": np.array([False, True, True, True]),
        "D_matched": np.array([True, True, True, True]),
    }
    rows = []
    for arm, correct in correctness.items():
        prediction = {
            "dataset_seed": 11, "arm": arm, "ansatz_level": a7.ansatz_of(arm, "L1"),
            "dilution": "linear", "seed": 1, "width": None, "lr": 0.01,
            "correct": correct,
        }
        a7.write_prediction(tmp_path, prediction)
        rows.append({
            "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": arm,
            "ansatz_level": a7.ansatz_of(arm, "L1"), "dilution": "linear",
            "rff_width": "", "seed": 1, "split": "test", "lr_selected": 0.01,
        })

    block = a7.discordant_pairs(
        rows, tmp_path, dataset_seeds=(11,), dilutions=("linear",),
        ansatz_levels=("L1",), seeds=(1,),
        lrs=_lrs({(11, "linear", "L1"): 0.01}, {(11, "linear"): 0.01}),
    )["ds11|linear|L1"]

    # A right / B wrong on rows 2 and 3; B right / A wrong nowhere.
    assert block["delta_AB"]["b_total"] == 2
    assert block["delta_AB"]["c_total"] == 0
    assert block["delta_AB"]["per_seed"][1]["n_test"] == 4
    # A right / E wrong on row 3; E right / A wrong nowhere.
    assert (block["delta_AE"]["b_total"], block["delta_AE"]["c_total"]) == (1, 0)
    # A vs F: A right where F wrong on row 1; F right where A wrong on row 4.
    assert (block["delta_AF"]["b_total"], block["delta_AF"]["c_total"]) == (1, 1)
    # B vs D_matched: D_matched is right everywhere, B is wrong on rows 2, 3 and 4.
    assert (block["delta_BD_matched"]["b_total"], block["delta_BD_matched"]["c_total"]) == (0, 3)


def test_discordant_pairs_are_silent_about_cells_whose_predictions_are_absent(tmp_path):
    rows = [{
        "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": "A",
        "ansatz_level": "L1", "dilution": "linear", "rff_width": "", "seed": 1,
        "split": "test", "lr_selected": 0.01,
    }]
    assert a7.discordant_pairs(
        rows, tmp_path, dataset_seeds=(11,), dilutions=("linear",),
        ansatz_levels=("L1",), seeds=(1,),
        lrs=_lrs({(11, "linear", "L1"): 0.01}, {(11, "linear"): 0.01}),
    ) == {}


def test_discordant_pairs_read_arm_f_at_the_cell_lr_of_its_own_ansatz(tmp_path):
    """The D-32 failure, on the counts rather than on the accuracies.

    Arm F has no ansatz dimension while the cell lr does, so when L1 and L2 select
    different lr arm F exists TWICE and writes two prediction files. Keyed without lr the
    second overwrites the first and one of the two cells is counted against a run it was
    never paired with — silently, because both cells still produce plausible counts.

    Here arm F matches arm A exactly at the L1 lr and disagrees on every row at the L2
    lr, so a key that drops lr reports the L2 disagreement in the L1 cell too.
    """
    n = 4
    right, wrong = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    # (arm, recorded ansatz, lr, correctness). The two ansatz levels of this cell picked
    # different lr, so arm F exists once per lr while arm A exists once per level.
    runs = [
        ("A", "L1", 0.003, right),
        ("A", "L2", 0.03, right),
        ("F", a7.PRODUCT_ANSATZ, 0.003, right),
        ("F", a7.PRODUCT_ANSATZ, 0.03, wrong),
    ]
    rows = []
    for arm, recorded, lr, correct in runs:
        a7.write_prediction(tmp_path, {
            "dataset_seed": 11, "arm": arm, "ansatz_level": recorded,
            "dilution": "linear", "seed": 1, "width": None, "lr": lr, "correct": correct,
        })
        rows.append({
            "dataset": PRODUCTION_DATASET, "dataset_seed": 11, "arm": arm,
            "ansatz_level": recorded, "dilution": "linear", "rff_width": "",
            "seed": 1, "split": "test", "lr_selected": lr,
        })

    out = a7.discordant_pairs(
        rows, tmp_path, dataset_seeds=(11,), dilutions=("linear",),
        ansatz_levels=("L1", "L2"), seeds=(1,),
        lrs=_lrs({(11, "linear", "L1"): 0.003, (11, "linear", "L2"): 0.03}),
    )
    # L1: A and F agree on every row at lr = 0.003, so there is nothing discordant.
    assert (out["ds11|linear|L1"]["delta_AF"]["b_total"],
            out["ds11|linear|L1"]["delta_AF"]["c_total"]) == (0, 0)
    # L2: A right everywhere, F wrong everywhere, at lr = 0.03.
    assert (out["ds11|linear|L2"]["delta_AF"]["b_total"],
            out["ds11|linear|L2"]["delta_AF"]["c_total"]) == (n, 0)


def test_the_lr_resume_key_separates_the_three_d_best_widths(tmp_path):
    """Without rff_width in the key the three widths share one identity, and a resumed run
    would treat M=32 as covering M=128 and M=512."""
    rows = []
    for width in D_BEST_WIDTHS:
        rows.append({
            "run_id": "t", "timestamp_utc": "t", "dataset_seed": 11,
            "dataset": PRODUCTION_DATASET, "arm": "D_best", "ansatz_level": "",
            "dilution": "linear", "rff_width": width, "seed": 1, "lr": 0.01,
            "lr_grid": "", "in_selection": False, "val_accuracy": 0.7,
            "test_accuracy": 0.7, "train_accuracy": 0.7, "best_epoch": 1,
            "epochs_run": 1, "theta_displacement": 0.0, "wall_seconds": 0.0,
        })
    path = tmp_path / "lr.csv"
    a7.append_rows(path, a7.LR_TABLE_COLUMNS, rows)
    assert len(a7.existing_lr_keys(path)) == len(D_BEST_WIDTHS)

    done = a7.existing_lr_keys(path)
    tasks = a7.plan_lr_stage(
        dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",), seeds=(1,),
        widths=D_BEST_WIDTHS, ranks={}, context=CONTEXT, done=done,
    )
    remaining = [
        (task["width"], cell["lr"])
        for task in tasks if task["arm"] == "D_best" for cell in task["cells"]
    ]
    # Only the lr = 0.01 cell of each width was on disk; the other three lrs remain.
    assert set(width for width, _ in remaining) == set(D_BEST_WIDTHS)
    assert 0.01 not in {lr for _, lr in remaining}


# --- arm F's coverage of the axis is a declared knob, not a hard-wired loop ----------


def test_arm_f_runs_on_all_four_dilutions_by_default():
    """For control rather than symmetry: Delta_AF going to zero at high dilution the same
    way Delta_AB does is what shows the axis drives the zeroing, not the absence of
    entanglement.
    """
    assert a7.ARM_F_DILUTIONS == a7.DILUTIONS == ("linear", "h2", "h4", "h42")
    assert {dilution for *_, dilution, _ in _cells(_plan_main(), "F")} == set(a7.DILUTIONS)


def test_narrowing_arm_f_to_the_two_ends_of_the_axis_halves_its_runs():
    """The only cut that keeps the control argument: the ends are where the claim lives
    ("does Delta_AF vanish at h42 the way Delta_AB does"), the middle is interpolation."""
    full = _cells(_plan_main(), "F")
    ends = _cells(_plan_main(f_dilutions=("linear", "h42")), "F")
    assert len(full) == 4 * len(a7.SEEDS)
    assert len(ends) == 2 * len(a7.SEEDS) == len(full) // 2
    assert {dilution for *_, dilution, _ in ends} == {"linear", "h42"}
    # Nothing else moves: A, B, E, D_matched and D_best keep the whole axis.
    for arm in ("A", "B", "E", "D_matched"):
        assert len(_cells(_plan_main(f_dilutions=("linear", "h42")), arm)) == \
               len(_cells(_plan_main(), arm))


def test_the_architecture_row_is_judged_against_the_mde_of_its_own_contrast():
    """sigma_Delta depends on the contrast, by an order of magnitude between A-B and A-E,
    so judging acc(A) ~ acc(E) against the A-B threshold would fire this stop row whenever
    A-B is merely noisy.
    """
    summary = _summary(delta_ab=0.11, theta=0.8, acc={"A": 0.83, "B": 0.72, "E": 0.77},
                       sd=0.065)
    # Give the A-E contrast its own, much smaller sigma, as measured in the pilot.
    summary["estimands"]["ds11|linear|L1"]["delta_AE"].update(
        {"sd": 0.0075, "sigma_delta": 0.0075, "mde_from_this_series": a7.mde(0.0075, 10)}
    )
    fired = _rows_of(summary, "acc(A) ~ acc(E) with acc(B) < acc(E)")
    assert fired == [], "acc(A) - acc(E) = +0.06 is eight times its own MDE; nothing to stop for"

    # And it still fires when acc(A) really is within the A-E resolution of acc(E).
    summary["per_arm"]["ds11|linear|L1"]["A"]["test"]["mean"] = 0.7705
    assert [row["verdict"] for row in
            _rows_of(summary, "acc(A) ~ acc(E) with acc(B) < acc(E)")] == ["STOP"]


# --- the CSV round trip of an integer column that is sometimes empty -----------------


def test_optional_int_survives_the_float_round_trip_pandas_forces():
    """pandas returns an integer column as float64 as soon as one row is empty, so a width
    written as 32 comes back as "32.0": int() raises on it, and the resume key stops
    matching, which recomputes arm D_best and appends its rows a second time.
    """
    assert a7.optional_int("32.0") == 32
    assert a7.optional_int(32.0) == 32
    assert a7.optional_int(32) == 32
    assert a7.optional_int("") is None
    assert a7.optional_int(None) is None
    assert a7.optional_int(float("nan")) is None


def test_a_key_is_the_same_whether_it_came_from_a_csv_or_from_live_values():
    live = a7._key((11, "D_best", "", "linear", 2, "test", 32, "0.01"))
    from_csv = a7._key((11.0, "D_best", "", "linear", 2.0, "test", 32.0, "0.01"))
    assert live == from_csv == ("11", "D_best", "", "linear", "2", "test", "32", "0.01")


def _lrs_for_summary(accuracy_by_lr):
    rows = _selection_rows(accuracy_by_lr)
    return a7.select_all_lrs(
        rows, dataset_seeds=(11,), dilutions=("linear",), ansatz_levels=("L1",),
        widths=D_BEST_WIDTHS, seeds=(1, 2, 3),
    )


def _summarise_lr(accuracy_by_lr):
    return a7.lr_selection_summary(
        _lrs_for_summary(accuracy_by_lr), dataset_seeds=(11,), dilutions=("linear",),
        ansatz_levels=("L1",), widths=D_BEST_WIDTHS,
    )


def test_the_lr_phase_reports_whether_the_winner_was_bracketed():
    """Between the sweep and the main series what matters is not only which lr won but
    whether it is surrounded: an argmax on an edge means the optimum was not bracketed and
    the value is a bound.
    """
    interior = _summarise_lr({1e-3: 0.60, 3e-3: 0.70, 1e-2: 0.75, 3e-2: 0.71})["cell"][0]
    assert interior["lr_selected"] == 1e-2
    assert interior["on_grid_edge"] is False and interior["edge"] is None
    assert interior["margin_over_runner_up"] == pytest.approx(0.75 - 0.71)

    upper = _summarise_lr({1e-3: 0.60, 3e-3: 0.70, 1e-2: 0.75, 3e-2: 0.80})["cell"][0]
    assert upper["lr_selected"] == 3e-2
    assert upper["on_grid_edge"] is True and upper["edge"] == "upper"

    lower = _summarise_lr({1e-3: 0.80, 3e-3: 0.70, 1e-2: 0.65, 3e-2: 0.60})["cell"][0]
    assert lower["lr_selected"] == 1e-3
    assert lower["on_grid_edge"] is True and lower["edge"] == "lower"


def test_the_lr_phase_keeps_the_per_arm_curves_not_only_their_mean():
    """The lr x arm table is the whole point of writing the table out: with it, adding an arm
    to the criterion in part B is a recombination of numbers already on disk."""
    summary = _summarise_lr({1e-3: 0.60, 3e-3: 0.70, 1e-2: 0.75, 3e-2: 0.71})
    cell = summary["cell"][0]
    assert set(cell["mean_val_accuracy_per_lr_and_arm"]) == {"A", "B"}
    assert set(cell["mean_val_accuracy_per_lr"]) == {"0.001", "0.003", "0.01", "0.03"}
    assert cell["selection_arms"] == ["A", "B"]
    # Arm E is summarised on its own, wider grid; D_best per width.
    assert set(summary["arm_E"][0]["mean_val_accuracy_per_lr"]) == {
        "0.001", "0.003", "0.01", "0.03", "0.1"
    }
    assert [entry["cell"] for entry in summary["arm_D_best"]] == [
        f"ds11|M{width}" for width in D_BEST_WIDTHS
    ]


def test_every_edge_selection_is_listed_so_none_is_missed_by_reading():
    summary = _summarise_lr({1e-3: 0.60, 3e-3: 0.70, 1e-2: 0.75, 3e-2: 0.80})
    listed = summary["on_grid_edge"]
    assert "ds11|linear|L1" in listed["cell"]
    # Arm E's synthetic curve rises with lr, so it lands on its own upper edge too.
    assert listed["arm_E"] == ["ds11|linear"]
    assert a7.report_lr_selection(summary) is None  # printing must not raise on any shape


# --- lr as part of the pairing key -------------------------------


def _row(**kw):
    """Minimal result row for the accuracy index."""
    base = {
        "split": "test", "dataset_seed": "11", "arm": "F", "ansatz_level": "product",
        "dilution": "linear", "rff_width": "", "seed": "1", "lr_selected": "0.03",
        "accuracy": "0.5", "theta_displacement": "0.1", "epochs_run": "10",
        "best_epoch": "5",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_the_accuracy_index_keeps_lr_in_the_key():
    rows = [_row(lr_selected=0.01, accuracy=0.810), _row(lr_selected=0.03, accuracy=0.815)]
    index = a7._accuracy_index(rows, "test")
    assert len(index) == 2, index
    assert sorted(float(r["accuracy"]) for r in index.values()) == [0.810, 0.815]


def test_the_lr_key_is_canonical_so_one_value_is_one_cell():
    assert a7._lr_key(0.01) == a7._lr_key("0.010") == a7._lr_key("1e-2")
    assert a7._lr_key(None) == a7._lr_key("") == ""
    assert a7._lr_key(0.03) != a7._lr_key(0.01)

