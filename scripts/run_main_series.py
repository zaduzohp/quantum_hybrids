"""Experiment driver and the main-series run.

Produces the rows the study is made of, plus descriptive statistics. It does not analyse
them: tests, CI, TOST, MixedLM and figures live in run_a8_analysis.py. The grid itself is
in qsocket.contract.

Arms carry different dimensions, which is why rows are "missing" by construction:
A and B run over dataset x dilution x ansatz x seed; E, F, D_matched have no ansatz
dimension (E is the identity, F the product circuit, arm D takes Omega from the frequency
support, which depends on R only); D_best has no dilution dimension either.

Stages, in order, because each can stop the next:
    0 validate  hashes, theta pairing A<->B, h42 == mlp42, column schema
    1 lr        per (dataset x dilution x ansatz) over arms A and B, seeds 1-3; arm E on
                the grid extended by one point. The full lr x arm table goes to disk.
    2 gates     G1 and G2 per generator seed. A failure stops the run.
    3 main      the grid; test and val as separate ROWS. Per-row predictions are saved
                because McNemar cannot be computed after the fact.
    4 summary   per-arm means and sigma, the estimands, theta diagnostics, budget hits.

Every raw row hits disk before any statistics, and the run resumes off the CSV in
--out-dir, so give concurrent jobs separate directories or they interleave writes.
Cluster: --prepare-datasets once on the shared filesystem, then --no-generate in the
jobs. --workers defaults to min(10, cores - 2); arm A does not scale with threads.

    caffeinate -dimsu .venv/bin/python scripts/run_main_series.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Before numpy and torch: the BLAS pools read these at import and never again. A
# multi-threaded BLAS reduces in a split-dependent order, which Adam amplifies over 300
# epochs into a different accuracy for the TRAINED arms only.
from qsocket.core import derive, pin_blas_threads

pin_blas_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from qsocket.ansatzes import build_socket_circuit, socket_param_count
from qsocket.datasets import DEFAULT_DATA_DIR, load_manifest, load_splits  # noqa: F401
from qsocket.gates import (
    check_g1_headroom,
    check_g2_effective_dim,
    make_arm_e_linear_floor_model,
    make_svc_strong_model,
)
from qsocket.head import (
    DILUTION_AXIS,
    HEAD_PARAM_COUNTS,
    canonical_head_name,
    make_head,
    make_linear_readout,
)
from qsocket.rank import effective_dimension
from qsocket.results import RESULT_COLUMNS, append_result_row
from qsocket import stats
from qsocket.socket import (
    D_BEST_WIDTHS,
    D_MATCHED_WIDTH,
    DEFAULT_BACKEND,
    DEFAULT_N_QUBITS,
    frozen_socket_features,
    make_socket,
)
from qsocket.training import (
    CONTRACT_RIDGE_ALPHA_GRID,
    TrainConfig,
    lr_selection_from_measurements,
    macro_f1,
    ridge_control,
    train_model,
)
from qsocket.vendored.metrics_cls import accuracy_from_z

# --- fixed configuration ------------------------------------------------------------

# The contract lives in the package (qsocket.contract), not in this driver. Re-exported
# here because the analysis and the probes address it as a7.X.
from qsocket.contract import (  # noqa: F401
    A7_DATA_DIR,
    ANSATZ_FREE_ARMS,
    ANSATZ_LEVELS,
    ARMS,
    ARM_E_LR_GRID,
    ARM_F_DILUTIONS,
    BATCH_SIZE,
    BUDGET_HIT_NOTE_FRACTION,
    CONTRACT_LR_GRID,
    DATASET_SEEDS,
    DEFAULT_OUT_DIR,
    DILUTIONS,
    FEATURE_SCALE,
    FROZEN_ARMS,
    GATES_JSON,
    GENERATED_HASH_PREFIXES,
    GENERATOR,
    GENERATOR_KWARGS,
    LR_SELECTION_ARMS,
    LR_SELECTION_JSON,
    LR_SELECTION_SEEDS,
    LR_TABLE_CSV,
    MAX_EPOCHS,
    PATIENCE,
    PREDICTIONS_DIR,
    PRODUCT_ANSATZ,
    RESULTS_CSV,
    R_CONTRACT,
    SEEDS,
    SPLITS_REPORTED,
    SUMMARY_JSON,
    THETA_STILL,
    TRAINED_ARMS,
    dataset_location,
    dataset_seed_of,
    ensure_dataset,
    env_hash,
    frozen_name_for,
    git_commit,
    optional_int,
    utc_now,
)

# --- datasets ------------------------------------------------------------------------


# --- cells ---------------------------------------------------------------------------


def head_for(arm: str, dilution: str, *, seed: int, width: int | None = None):
    """The head of one run. D_best is the only arm that does not take a 5-wide input."""
    if arm == "D_best":
        return make_linear_readout(width, seed=seed)
    return make_head(dilution, seed=seed)


def head_params(arm: str, dilution: str, *, width: int | None = None) -> int:
    if arm == "D_best":
        return width + 1
    return HEAD_PARAM_COUNTS[dilution]


def build_socket_for(arm: str, *, ansatz: str | None, seed: int, width: int | None = None):
    """The socket of one arm. This is the ONLY thing that differs between arms."""
    if arm == "A":
        return make_socket("quantum", R=R_CONTRACT, ansatz=ansatz, trainable=True, seed=seed)
    if arm == "B":
        return make_socket("quantum", R=R_CONTRACT, ansatz=ansatz, trainable=False, seed=seed)
    if arm == "F":
        return make_socket(
            "quantum", R=R_CONTRACT, ansatz=PRODUCT_ANSATZ, trainable=True, seed=seed
        )
    if arm == "E":
        return make_socket("identity", R=None, ansatz=None, trainable=False, seed=seed)
    if arm in ("D_matched", "D_best"):
        return make_socket(
            "random", R=R_CONTRACT, ansatz=None, trainable=False, seed=seed,
            rff_width=D_MATCHED_WIDTH if arm == "D_matched" else width,
        )
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def assert_theta_pairing(seed: int, ansatz: str) -> None:
    """A and B start from bit-for-bit identical theta, proved for THIS seed in THIS run.

    Checked rather than cited from the seeding contract: without it Delta_AB stops
    meaning "what training did from this starting point".
    """
    a = build_socket_for("A", ansatz=ansatz, seed=seed)
    b = build_socket_for("B", ansatz=ansatz, seed=seed)
    assert torch.equal(a.theta_init, b.theta_init), (
        f"theta_init(A) != theta_init(B) at seed {seed}, ansatz {ansatz}"
    )
    assert torch.equal(a.theta(), b.theta()), f"theta(A) != theta(B) at seed {seed}"


def socket_init_seed(arm: str, *, ansatz: str | None, seed: int, width: int | None = None):
    """The key of the socket draw, recorded so a row can be replayed.

    Written as a hex string: derive() returns 64 unsigned bits, which do not fit int64,
    and a CSV round trip through pandas would turn the key into a float that no longer
    identifies the draw.
    """
    if arm in ("A", "B"):
        value = derive(seed, ansatz, R_CONTRACT)
    elif arm == "F":
        value = derive(seed, PRODUCT_ANSATZ, R_CONTRACT)
    elif arm == "D_matched":
        value = derive(seed, "RFF", R_CONTRACT, D_MATCHED_WIDTH)
    elif arm == "D_best":
        value = derive(seed, "RFF", R_CONTRACT, width)
    else:
        return ""  # arm E has no socket parameters to draw
    return f"0x{value:016x}"


def init_spec_id(arm: str) -> str:
    if arm in ("A", "B", "F"):
        return "U0-2pi"
    if arm in ("D_matched", "D_best"):
        return "RFF-int-spectrum-U0-2pi"
    return ""


# --- metrics -------------------------------------------------------------------------


def metrics_from_logits(logits: np.ndarray, y) -> dict:
    """accuracy (the metric the tests stand on), plus AUC and macro-F1 as DIAGNOSTICS.

    accuracy goes through the vendored accuracy_from_z, i.e. threshold 0 on the logit,
    the same rule as 0.5 on the sigmoid. macro_f1 is always training.macro_f1, never the
    f1_score of the vendored metrics_cls: the two disagree in the degenerate case.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y).reshape(-1)
    predicted_pm1 = np.where(logits > 0, 1.0, -1.0)
    binary_true = (y > 0).astype(np.float64)
    binary_predicted = (logits > 0).astype(np.float64)
    return {
        "accuracy": float(accuracy_from_z(logits, y)),
        # AUC on the raw logit: a threshold-free reading of the same score.
        "auc": float(roc_auc_score(binary_true, logits)),
        "macro_f1": float(macro_f1(binary_true, binary_predicted)),
        # Per-test-row correctness: what McNemar needs, and not recoverable from an
        # accuracy after the fact.
        "correct": (predicted_pm1 == np.where(y > 0, 1.0, -1.0)),
    }


@torch.no_grad()
def logits_of(socket, head, X) -> np.ndarray:
    features = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return head(socket(features)).reshape(-1).numpy()


# --- one cell ------------------------------------------------------------------------


def _cell_row(
    *,
    manifest: dict,
    dataset: str,
    arm: str,
    ansatz_level: str,
    dilution: str,
    seed: int,
    lr: float,
    lr_grid: tuple[float, ...],
    width: int | None,
    result,
    split: str,
    metrics: dict,
    wall_seconds: float,
    used_cache: bool,
    ridge: dict | None,
    effective_rank,
    g1_margin,
    run_id: str,
    commit: str,
    environment: str,
) -> dict:
    """One results row. Every column of RESULT_COLUMNS is present; "not applicable" is an
    empty value, so it stays distinguishable from "forgotten"."""
    row = {column: "" for column in RESULT_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "timestamp_utc": utc_now(),
            "dataset": dataset,
            "dataset_hash": manifest["dataset_hash"],
            "pca_hash": manifest["pca_hash"],
            "arm": arm,
            "ansatz_level": ansatz_level,
            "R": R_CONTRACT if arm != "E" else "",
            "dilution": dilution,
            "socket_params_nominal": socket_params_nominal(arm, width=width),
            "socket_params_effective": "" if effective_rank is None else effective_rank,
            "head_params": head_params(arm, dilution, width=width),
            "seed": int(seed),
            "init_seed": socket_init_seed(arm, ansatz=ansatz_level, seed=seed, width=width),
            "init_spec_id": init_spec_id(arm),
            "backend": DEFAULT_BACKEND,
            # Statevector: no shots, session or calibration. Empty, not zero.
            "shots": "",
            "session_id": "",
            "calibration_set_id": "",
            "repeat_index": 0,
            "eval_subset_id": f"{split}_full",
            "n_eval": int(len(metrics["correct"])),
            "split": split,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "macro_f1": metrics["macro_f1"],
            "train_accuracy": float(result.train_accuracy),
            "theta_displacement": float(result.theta_displacement),
            "grad_rms_start": float(result.grad_rms_start),
            "grad_rms_end": float(result.grad_rms_end),
            "epochs_run": int(result.epochs_run),
            "best_epoch": int(result.best_epoch),
            "wall_seconds": float(wall_seconds),
            "git_commit": commit,
            "env_hash": environment,
            "lr_selected": float(lr),
            "lr_grid": " ".join(f"{value:g}" for value in lr_grid),
            # Measured in stage 2 for this generator seed at the final lr and carried
            # into every row, so no join is needed to see whether the dataset cleared G1
            # and by how much.
            "g1_margin": "" if g1_margin is None else float(g1_margin),
            "patience": PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "feature_scale": FEATURE_SCALE,
            "rff_width": "" if width is None else int(width),
            "rff_omega_seed": (
                "" if arm not in ("D_matched", "D_best")
                else socket_init_seed(arm, ansatz=None, seed=seed, width=width)
            ),
            "used_feature_cache": bool(used_cache),
            "ridge_accuracy": "" if ridge is None else ridge["accuracy"][split],
            "ridge_alpha_selected": "" if ridge is None else ridge["alpha_selected"],
        }
    )
    return row


def socket_params_nominal(arm: str, *, width: int | None = None):
    if arm in ("A", "B", "F"):
        return socket_param_count(DEFAULT_N_QUBITS, R_CONTRACT)
    if arm == "E":
        return 0
    # Arm D: the frozen draw is (Omega, b), width*(n_inputs + 1) numbers held as buffers,
    # so none of them are trainable.
    size = D_MATCHED_WIDTH if arm == "D_matched" else width
    return int(size * (DEFAULT_N_QUBITS + 1))


def run_cell(
    splits: dict,
    *,
    manifest: dict,
    dataset: str,
    arm: str,
    ansatz_level: str,
    dilution: str,
    seed: int,
    lr: float,
    lr_grid: tuple[float, ...],
    width: int | None,
    cached_features: dict | None,
    effective_rank,
    g1_margin,
    run_id: str,
    commit: str,
    environment: str,
    socket_factory=None,
    head_factory=None,
    row_builder=None,
    trained_arms: tuple[str, ...] = TRAINED_ARMS,
) -> tuple[list[dict], np.ndarray, dict]:
    """Train one cell and return (its rows, the per-test-row correctness vector, the
    trained modules).

    The third element is {"socket": the ARM-DEFINING socket, "head": the trained head}.
    For a frozen arm that socket is rebuilt, not the identity pass-through train_model was
    handed, which carries no parameters.

    Trained arms (A, F) go through the live socket. Frozen arms (B, E, D_*) train the head
    on cached socket features through an identity pass-through — a frozen socket's output
    is constant, so this is the same run bit for bit at ~1/100 of the cost.

    The last four arguments let a probe reuse this function instead of copying it:
    everything arm-specific is injected, while the cache invariant and the ridge control —
    the parts a copy would put outside tests/test_feature_cache.py — are not. Defaults are
    None rather than the functions themselves, because a literal default binds at def time
    and a test monkeypatching build_socket_for would then pass while proving nothing.
    row_builder must accept the same keywords as _cell_row; that is the one injected
    interface nothing enforces.
    """
    socket_factory = build_socket_for if socket_factory is None else socket_factory
    head_factory = head_for if head_factory is None else head_factory
    row_builder = _cell_row if row_builder is None else row_builder
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    cfg = TrainConfig(lr=lr, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    started = time.perf_counter()
    ridge = None

    if arm in trained_arms:
        socket = socket_factory(arm, ansatz=ansatz_level, seed=seed, width=width)
        head = head_factory(arm, dilution, seed=seed, width=width)
        result = train_model(socket, head, X_tr, y_tr, X_val, y_val, cfg=cfg, seed=seed)
        logits = {
            "test": logits_of(socket, head, X_te),
            "val": logits_of(socket, head, X_val),
        }
        used_cache = False
    else:
        features = cached_features
        if features is None:
            socket = socket_factory(arm, ansatz=ansatz_level, seed=seed, width=width)
            features = {
                split: frozen_socket_features(socket, splits[split][0])
                for split in ("train", "val", "test")
            }
        # The identity is only a carrier for the cached features, so it comes from the
        # main-series factory whatever arm this is: arm E is the one arm whose socket is
        # a pass-through, and its width never enters the head, which was already sized
        # by head_factory from the arm itself.
        identity = build_socket_for("E", ansatz=None, seed=seed)
        head = head_factory(arm, dilution, seed=seed, width=width)
        result = train_model(
            identity, head, features["train"], y_tr, features["val"], y_val, cfg=cfg, seed=seed
        )
        logits = {
            "test": logits_of(identity, head, features["test"]),
            "val": logits_of(identity, head, features["val"]),
        }
        used_cache = True
        # Closed-form readout control, frozen sockets only.
        ridge = ridge_control(
            features["train"], y_tr, features["val"], y_val,
            grid=CONTRACT_RIDGE_ALPHA_GRID,
            evaluation={"test": (features["test"], y_te)},
        )

    wall = time.perf_counter() - started

    # Trained parameters, for hardware evaluation. `head` is always the trained one;
    # `socket` is the arm-defining socket, not the identity that train_model was handed
    # for frozen arms, which carries no parameters.
    arm_socket = socket if arm in trained_arms else socket_factory(
        arm, ansatz=ansatz_level, seed=seed, width=width
    )
    trained_state = {"socket": arm_socket, "head": head}

    rows, correctness = [], None
    for split in SPLITS_REPORTED:
        y_split = {"test": y_te, "val": y_val}[split]
        metrics = metrics_from_logits(logits[split], y_split)
        if split == "test":
            correctness = metrics["correct"]
        rows.append(
            row_builder(
                manifest=manifest, dataset=dataset, arm=arm, ansatz_level=ansatz_level,
                dilution=dilution, seed=seed, lr=lr, lr_grid=lr_grid, width=width,
                result=result, split=split, metrics=metrics, wall_seconds=wall,
                used_cache=used_cache, ridge=ridge, effective_rank=effective_rank,
                g1_margin=g1_margin, run_id=run_id, commit=commit, environment=environment,
            )
        )
    return rows, correctness, trained_state


# --- parallel execution --------------------------------------------------------------

_SPLITS: dict[str, dict] = {}
_MANIFESTS: dict[str, dict] = {}


def _worker_init() -> None:
    """One torch thread per worker.

    A run of arm A occupies effectively one core — lightning.qubit at 5 qubits does not
    scale with threads — so multi-threaded workers would only oversubscribe the machine.
    """
    torch.set_num_threads(1)


def _worker_splits(dataset_seed: int) -> tuple[str, dict, dict]:
    name, out_dir = dataset_location(dataset_seed)
    if name not in _SPLITS:
        _SPLITS[name] = load_splits(name, out_dir=out_dir)
        _MANIFESTS[name] = load_manifest(name, out_dir=out_dir)
    return name, _SPLITS[name], _MANIFESTS[name]


def _worker_run(task: dict) -> dict:
    """Execute one task: one expensive run, or one cache group of cheap runs."""
    torch.set_num_threads(1)
    dataset, splits, manifest = _worker_splits(task["dataset_seed"])
    arm, seed, width = task["arm"], task["seed"], task["width"]

    cached = None
    cache_seconds = 0.0
    if arm in FROZEN_ARMS and len(task["cells"]) > 1:
        # Build the socket's features once for the whole group; they are a constant.
        socket = build_socket_for(arm, ansatz=task["ansatz_level"], seed=seed, width=width)
        started = time.perf_counter()
        cached = {
            split: frozen_socket_features(socket, splits[split][0])
            for split in ("train", "val", "test")
        }
        cache_seconds = time.perf_counter() - started

    rows, predictions, weights = [], [], []
    for cell in task["cells"]:
        cell_rows, correctness, trained_state = run_cell(
            splits,
            manifest=manifest,
            dataset=dataset,
            arm=arm,
            ansatz_level=task["ansatz_level"],
            dilution=cell["dilution"],
            seed=seed,
            lr=cell["lr"],
            lr_grid=tuple(task["lr_grid"]),
            width=width,
            cached_features=cached,
            effective_rank=task["effective_rank"],
            g1_margin=task["g1_margin"],
            run_id=task["run_id"],
            commit=task["git_commit"],
            environment=task["env_hash"],
        )
        rows.extend(cell_rows)
        predictions.append(
            {
                "dataset_seed": task["dataset_seed"],
                "arm": arm,
                "ansatz_level": task["ansatz_level"],
                "dilution": cell["dilution"],
                "seed": seed,
                "width": width,
                "lr": cell["lr"],
                "correct": np.asarray(correctness, dtype=bool),
            }
        )
        # Arrays, not live modules: a torch Module does not belong in a multiprocessing
        # payload.
        identity = {
            "dataset_seed": task["dataset_seed"],
            "arm": arm,
            "ansatz_level": task["ansatz_level"],
            "dilution": cell["dilution"],
            "seed": seed,
            "width": width,
            "lr": cell["lr"],
        }
        socket_theta = (
            trained_state["socket"].theta()
            if trained_state["socket"] is not None and trained_state["socket"].is_quantum
            else None
        )
        theta_init = getattr(trained_state["socket"], "theta_init", None)
        arrays = {
            f"head__{k}": v.detach().cpu().numpy().astype(np.float64)
            for k, v in trained_state["head"].state_dict().items()
        }
        if socket_theta is not None:
            arrays["socket_theta"] = socket_theta.detach().cpu().numpy().astype(np.float64)
        if theta_init is not None:
            arrays["socket_theta_init"] = theta_init.detach().cpu().numpy().astype(np.float64)
        weights.append({"identity": identity, "arrays": arrays})
    return {
        "stage": task["stage"],
        "task": {k: v for k, v in task.items() if k != "cells"},
        "cells": task["cells"],
        "rows": rows,
        "predictions": predictions,
        "weights": weights,
        "cache_seconds": cache_seconds,
    }


def default_workers() -> int:
    return max(1, min(10, (os.cpu_count() or 3) - 2))


def execute(tasks: list[dict], *, workers: int, on_result) -> None:
    """Run tasks and hand every result to `on_result` as it arrives.

    The callback puts raw rows on disk the moment they exist, so an exception in the
    statistics cannot cost a finished grid.
    """
    if not tasks:
        return
    if workers <= 1:
        _worker_init()
        for task in tasks:
            on_result(_worker_run(task))
        return
    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_worker_run, tasks):
            on_result(result)


# --- resume --------------------------------------------------------------------------


def _key(values) -> tuple:
    """Identity tuple. Integers are rendered canonically (32, never 32.0) so that a key
    built from a CSV read and one built from live values are the same key."""
    out = []
    for value in values:
        if value is None or value == "":
            out.append("")
        elif isinstance(value, bool):
            out.append(str(value))
        elif isinstance(value, (int, np.integer)):
            out.append(str(int(value)))
        elif isinstance(value, (float, np.floating)):
            # An integral float is the CSV round trip of an int column.
            out.append(str(int(value)) if float(value).is_integer() else repr(float(value)))
        else:
            out.append(str(value))
    return tuple(out)


def row_key(row: dict) -> tuple:
    """Identity of a results row for resume purposes.

    The integer columns go through optional_int rather than straight into _key: read back
    from a CSV they are the string "32.0", which _key cannot canonicalise on its own.
    Getting this wrong recomputes arm D_best on every resume and duplicates its rows.
    """
    return _key(
        (
            row["dataset"], row["arm"], row["ansatz_level"], row["dilution"],
            optional_int(row["seed"]), row["split"], optional_int(row["rff_width"]),
            "" if row["lr_selected"] == "" else f"{float(row['lr_selected']):g}",
        )
    )


def existing_row_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    if frame.columns.tolist() != list(RESULT_COLUMNS):
        raise ValueError(
            f"{path} has a different column schema than RESULT_COLUMNS; a resumed run "
            "would mix two schemas in one file. Move the old file aside deliberately."
        )
    return {row_key(row) for row in frame.to_dict("records")}


def existing_lr_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    return {
        _key((r["dataset"], r["arm"], r["ansatz_level"], r["dilution"],
              optional_int(r["seed"]), optional_int(r["rff_width"]),
              f"{float(r['lr']):g}"))
        for r in frame.to_dict("records")
    }


# --- task planning -------------------------------------------------------------------


def make_task(
    *,
    stage: str,
    dataset_seed: int,
    arm: str,
    ansatz_level: str,
    seed: int,
    width: int | None,
    cells: list[dict],
    lr_grid: tuple[float, ...],
    effective_rank,
    g1_margin,
    run_id: str,
    commit: str,
    environment: str,
) -> dict:
    return {
        "stage": stage,
        "dataset_seed": int(dataset_seed),
        "arm": arm,
        "ansatz_level": ansatz_level,
        "seed": int(seed),
        "width": None if width is None else int(width),
        "cells": [{"dilution": c["dilution"], "lr": float(c["lr"])} for c in cells],
        "lr_grid": tuple(float(v) for v in lr_grid),
        "effective_rank": effective_rank,
        "g1_margin": g1_margin,
        "run_id": run_id,
        "commit": commit,
        "git_commit": commit,
        "env_hash": environment,
    }


def ansatz_of(arm: str, ansatz_level: str) -> str:
    """The ansatz recorded for an arm. F is the product circuit; the ansatz-free arms
    carry an empty value, because "not applicable" and "forgotten" must not look alike."""
    if arm in ("A", "B"):
        return ansatz_level
    if arm == "F":
        return PRODUCT_ANSATZ
    return ""


def plan_lr_stage(
    *, dataset_seeds, dilutions, ansatz_levels, seeds, widths, ranks, context, done
) -> list[dict]:
    """Stage 1: everything the lr selection needs, and nothing else.

    Arm A is one task per (cell, lr, seed), the expensive units to spread over workers.
    Arms B, E and D_best are one task per cache group, so their features are computed once
    for every lr and dilution in the group.
    """
    tasks: list[dict] = []

    def add(arm, dataset_seed, ansatz_level, seed, width, cells, grid):
        # rff_width belongs in the key, or a resumed run would treat one D_best width as
        # covering the other two.
        cells = [c for c in cells if _key((
            dataset_location(dataset_seed)[0], arm, ansatz_of(arm, ansatz_level),
            c["dilution"], seed, "" if width is None else width,
            f"{float(c['lr']):g}")) not in done]
        if not cells:
            return
        tasks.append(make_task(
            stage="lr", dataset_seed=dataset_seed, arm=arm,
            ansatz_level=ansatz_of(arm, ansatz_level), seed=seed, width=width,
            cells=cells, lr_grid=grid, effective_rank=ranks.get(ansatz_of(arm, ansatz_level)),
            g1_margin=None, **context,
        ))

    for dataset_seed in dataset_seeds:
        for seed in seeds:
            for ansatz_level in ansatz_levels:
                for dilution in dilutions:
                    for lr in CONTRACT_LR_GRID:
                        add("A", dataset_seed, ansatz_level, seed, None,
                            [{"dilution": dilution, "lr": lr}], CONTRACT_LR_GRID)
                add("B", dataset_seed, ansatz_level, seed, None,
                    [{"dilution": d, "lr": lr} for d in dilutions for lr in CONTRACT_LR_GRID],
                    CONTRACT_LR_GRID)
            # Arm E: its own grid, one point wider. It is not in the criterion, so its lr
            # is selected on arm E itself.
            add("E", dataset_seed, "", seed, None,
                [{"dilution": d, "lr": lr} for d in dilutions for lr in ARM_E_LR_GRID],
                ARM_E_LR_GRID)
            # Arm D_best: tuned on its own, on the contract grid. It asks whether a
            # properly sized classical random-feature model catches up, and a baseline
            # denied its own lr would be a straw baseline.
            for width in widths:
                add("D_best", dataset_seed, "", seed, width,
                    [{"dilution": "linear", "lr": lr} for lr in CONTRACT_LR_GRID],
                    CONTRACT_LR_GRID)
    return tasks


def select_all_lrs(lr_rows: list[dict], *, dataset_seeds, dilutions, ansatz_levels, widths,
                   seeds) -> dict:
    """Apply training.lr_selection_from_measurements to the measured table.

    Three families, three declared grids:

        (dataset x dilution x ansatz)  arms A and B, contract grid  -- the cell lr
        (dataset x dilution)           arm E alone, grid + 1e-1
        (dataset x width)              arm D_best alone, contract grid
    """
    measured: dict[tuple, dict] = defaultdict(dict)
    for row in lr_rows:
        measured[(row["dataset_seed"], row["arm"], row["ansatz_level"], row["dilution"],
                  row["width"])][(float(row["lr"]), row["arm"], optional_int(row["seed"]))] = float(
            row["val_accuracy"]
        )

    def merge(keys):
        joined: dict = {}
        for key in keys:
            joined.update(measured[key])
        return joined

    cell_lr, cell_selection = {}, {}
    for dataset_seed in dataset_seeds:
        for dilution in dilutions:
            for ansatz_level in ansatz_levels:
                table = merge([
                    (dataset_seed, "A", ansatz_level, dilution, None),
                    (dataset_seed, "B", ansatz_level, dilution, None),
                ])
                selection = lr_selection_from_measurements(
                    table, grid=CONTRACT_LR_GRID, seeds=seeds,
                    arms=LR_SELECTION_ARMS, selection_arms=LR_SELECTION_ARMS,
                )
                cell_lr[(dataset_seed, dilution, ansatz_level)] = selection.best
                cell_selection[(dataset_seed, dilution, ansatz_level)] = selection

    arm_e_lr, arm_e_selection = {}, {}
    for dataset_seed in dataset_seeds:
        for dilution in dilutions:
            selection = lr_selection_from_measurements(
                merge([(dataset_seed, "E", "", dilution, None)]),
                grid=ARM_E_LR_GRID, seeds=seeds, arms=("E",), selection_arms=("E",),
            )
            arm_e_lr[(dataset_seed, dilution)] = selection.best
            arm_e_selection[(dataset_seed, dilution)] = selection

    d_best_lr, d_best_selection = {}, {}
    for dataset_seed in dataset_seeds:
        for width in widths:
            selection = lr_selection_from_measurements(
                merge([(dataset_seed, "D_best", "", "linear", width)]),
                grid=CONTRACT_LR_GRID, seeds=seeds,
                arms=("D_best",), selection_arms=("D_best",),
            )
            d_best_lr[(dataset_seed, width)] = selection.best
            d_best_selection[(dataset_seed, width)] = selection

    return {
        "cell_lr": cell_lr,
        "cell_selection": cell_selection,
        "arm_e_lr": arm_e_lr,
        "arm_e_selection": arm_e_selection,
        "d_best_lr": d_best_lr,
        "d_best_selection": d_best_selection,
    }


def plan_main_stage(
    *, dataset_seeds, dilutions, ansatz_levels, seeds, widths, lrs, ranks, margins,
    context, done, f_dilutions=None
) -> list[dict]:
    """Stage 3: the main grid.

    Which lr each arm runs at:
      A, B          the cell lr of their own (dataset x dilution x ansatz)
      F, D_matched  the cell lr too, since they are paired against A and B and a paired
                    difference taken at two different lr values is not one. Having no
                    ansatz dimension, they run once per distinct cell lr of that
                    (dataset x dilution) — once when L1 and L2 agree, twice otherwise.
      E             its own lr, on the grid extended by one point
      D_best        its own lr, on the contract grid
    """
    tasks: list[dict] = []
    cell_lr, arm_e_lr, d_best_lr = lrs["cell_lr"], lrs["arm_e_lr"], lrs["d_best_lr"]
    f_dilutions = tuple(ARM_F_DILUTIONS if f_dilutions is None else f_dilutions)

    def add(arm, dataset_seed, ansatz_level, seed, width, cells, grid):
        recorded = ansatz_of(arm, ansatz_level)
        cells = [
            c for c in cells
            if any(
                _key((dataset_location(dataset_seed)[0], arm, recorded, c["dilution"],
                      seed, split, "" if width is None else width,
                      f"{float(c['lr']):g}")) not in done
                for split in SPLITS_REPORTED
            )
        ]
        if not cells:
            return
        tasks.append(make_task(
            stage="main", dataset_seed=dataset_seed, arm=arm, ansatz_level=recorded,
            seed=seed, width=width, cells=cells, lr_grid=grid,
            effective_rank=ranks.get(recorded), g1_margin=margins.get(dataset_seed),
            **context,
        ))

    for dataset_seed in dataset_seeds:
        for seed in seeds:
            for ansatz_level in ansatz_levels:
                for dilution in dilutions:
                    lr = cell_lr[(dataset_seed, dilution, ansatz_level)]
                    add("A", dataset_seed, ansatz_level, seed, None,
                        [{"dilution": dilution, "lr": lr}], CONTRACT_LR_GRID)
                add("B", dataset_seed, ansatz_level, seed, None,
                    [{"dilution": d, "lr": cell_lr[(dataset_seed, d, ansatz_level)]}
                     for d in dilutions],
                    CONTRACT_LR_GRID)

            # Ansatz-free trained arm F: once per (dilution x distinct cell lr), on the
            # dilutions of ARM_F_DILUTIONS only.
            for dilution in [d for d in dilutions if d in f_dilutions]:
                for lr in sorted({cell_lr[(dataset_seed, dilution, a)] for a in ansatz_levels}):
                    add("F", dataset_seed, "", seed, None,
                        [{"dilution": dilution, "lr": lr}], CONTRACT_LR_GRID)

            add("E", dataset_seed, "", seed, None,
                [{"dilution": d, "lr": arm_e_lr[(dataset_seed, d)]} for d in dilutions],
                ARM_E_LR_GRID)

            add("D_matched", dataset_seed, "", seed, None,
                [{"dilution": d, "lr": lr} for d in dilutions
                 for lr in sorted({cell_lr[(dataset_seed, d, a)] for a in ansatz_levels})],
                CONTRACT_LR_GRID)

            for width in widths:
                add("D_best", dataset_seed, "", seed, width,
                    [{"dilution": "linear", "lr": d_best_lr[(dataset_seed, width)]}],
                    CONTRACT_LR_GRID)
    return tasks


# --- writing -------------------------------------------------------------------------

LR_TABLE_COLUMNS: tuple[str, ...] = (
    "run_id", "timestamp_utc", "dataset_seed", "dataset", "arm", "ansatz_level",
    "dilution", "rff_width", "seed", "lr", "lr_grid", "in_selection",
    "val_accuracy", "test_accuracy", "train_accuracy", "best_epoch", "epochs_run",
    "theta_displacement", "wall_seconds",
)


def append_rows(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """Append rows to a CSV, creating it with a header if needed. Called the moment a
    task returns, never at the end of a stage."""
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def lr_table_rows(result: dict) -> list[dict]:
    """The lr x arm table: one row per (arm, dilution, lr, seed), both splits folded in.

    The whole table goes to disk, not just the winning lr: adding an arm to the lr
    criterion later is then a recombination of numbers already on disk rather than a
    repeat of the series. Costs no extra runs.
    """
    task = result["task"]
    grouped: dict[tuple, dict] = defaultdict(dict)
    for row in result["rows"]:
        grouped[(row["dilution"], f"{float(row['lr_selected']):g}")][row["split"]] = row

    rows = []
    for (dilution, lr), by_split in grouped.items():
        val, test = by_split["val"], by_split["test"]
        rows.append(
            {
                "run_id": val["run_id"],
                "timestamp_utc": val["timestamp_utc"],
                "dataset_seed": task["dataset_seed"],
                "dataset": val["dataset"],
                "arm": val["arm"],
                "ansatz_level": val["ansatz_level"],
                "dilution": dilution,
                "rff_width": val["rff_width"],
                "seed": val["seed"],
                "lr": float(lr),
                "lr_grid": val["lr_grid"],
                "in_selection": val["arm"] in LR_SELECTION_ARMS,
                "val_accuracy": val["accuracy"],
                "test_accuracy": test["accuracy"],
                "train_accuracy": val["train_accuracy"],
                "best_epoch": val["best_epoch"],
                "epochs_run": val["epochs_run"],
                "theta_displacement": val["theta_displacement"],
                "wall_seconds": val["wall_seconds"],
            }
        )
    return rows


def prediction_filename(prediction: dict) -> str:
    """The file one run's predictions are stored under, without any directory.

    Split out from prediction_path so a reader that already holds the predictions
    DIRECTORY can name the file without having to reconstruct the run's out-dir and let
    prediction_path re-append "predictions" to it — which only worked while the directory
    was literally named that.
    """
    parts = [
        prediction["arm"],
        prediction["ansatz_level"] or "noansatz",
        prediction["dilution"],
        f"seed{prediction['seed']}",
        f"lr{prediction['lr']:g}",
    ]
    if prediction["width"] is not None:
        parts.append(f"M{prediction['width']}")
    return "__".join(parts) + ".npz"


def prediction_path_in(predictions_dir: Path, prediction: dict) -> Path:
    """Where one run's predictions sit INSIDE a given predictions directory."""
    return Path(predictions_dir) / f"ds{prediction['dataset_seed']}" / prediction_filename(prediction)


def prediction_path(out_dir: Path, prediction: dict) -> Path:
    """Where one run's predictions sit under a RUN directory (out_dir/predictions/...)."""
    return prediction_path_in(Path(out_dir) / PREDICTIONS_DIR, prediction)


def write_prediction(out_dir: Path, prediction: dict) -> Path:
    """Per-test-row correctness, packed to bits.

    McNemar works on discordant pairs, so it needs to know which test rows each arm got
    right, and that cannot be recovered from an accuracy after the fact. One bit per test
    row, ~150 kB for the whole series.
    """
    path = prediction_path(out_dir, prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    correct = np.asarray(prediction["correct"], dtype=bool)
    np.savez_compressed(
        path,
        correct_packed=np.packbits(correct),
        n_test=np.array(correct.size),
        meta=np.array(
            json.dumps({k: v for k, v in prediction.items() if k != "correct"}, default=str)
        ),
    )
    return path


WEIGHTS_DIR = "weights"


def weights_path(out_dir: Path, prediction: dict) -> Path:
    """Same naming as prediction_path, different directory, so the two pair up by name."""
    name = prediction_path(out_dir, prediction).name
    return out_dir / WEIGHTS_DIR / f"ds{prediction['dataset_seed']}" / name


def write_weights(out_dir: Path, prediction: dict, *, arrays: dict) -> Path:
    """The trained parameters of one run: socket theta and the head's state dict.

    The analysis needs only accuracies, but hardware evaluation puts the trained model on
    the machine, and the head has to be the one actually trained. Without this file arm A's
    final theta is recoverable only by re-running in the same environment, and arm B's head
    — trained on the cached features — not at all. Under 2 kB per run.

    Frozen arms train through the identity socket, so the socket recorded here is the
    arm-defining one built in run_cell.
    """
    path = weights_path(out_dir, prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(prediction, default=str)),
        **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()},
    )
    return path


def read_weights(path) -> dict:
    """Inverse of write_weights: {name -> ndarray} plus the parsed meta dict."""
    with np.load(path, allow_pickle=False) as archive:
        out = {k: archive[k] for k in archive.files if k != "meta"}
        out["meta"] = json.loads(str(archive["meta"]))
    return out


def read_prediction(path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        n = int(archive["n_test"])
        return np.unpackbits(archive["correct_packed"])[:n].astype(bool)


# --- stage 1 output: the selection, its table, and where the argmax sits -------------


def lr_selection_summary(lrs: dict, *, dataset_seeds, dilutions, ansatz_levels, widths) -> dict:
    """Everything stage 1 decided, readable without rerunning anything.

    The main series is committed to these numbers, so what matters between the two phases
    is not only which lr won but whether the winner is surrounded. An argmax on an edge of
    the grid means the optimum was not bracketed and the reported value is a bound.
    """
    def block(selection, key):
        grid = list(selection.grid)
        position = grid.index(selection.best)
        return {
            "cell": key,
            "lr_selected": selection.best,
            "grid": grid,
            "on_grid_edge": position in (0, len(grid) - 1),
            "edge": ("lower" if position == 0 else "upper") if position in (0, len(grid) - 1) else None,
            "selection_arms": list(selection.selection_arms),
            "seeds": list(selection.seeds),
            "mean_val_accuracy_per_lr": {f"{lr:g}": selection.mean_by_lr[lr] for lr in grid},
            "mean_val_accuracy_per_lr_and_arm": {
                arm: {f"{lr:g}": selection.by_lr_arm[(lr, arm)] for lr in grid}
                for arm in selection.arms
                if all((lr, arm) in selection.by_lr_arm for lr in grid)
            },
            # What the shared lr costs each arm. One lr per cell is chosen on the mean of
            # arms A and B, so neither arm necessarily runs at its own optimum. Per arm,
            # this records its own best lr and the validation accuracy it gives up by
            # running at the selected one. A handicap far below the validation noise costs
            # that arm nothing; a large one-sided handicap is a finding, because it lands
            # inside Delta_AB.
            #
            # Switching to "every arm at its own optimum" would redefine the estimand from
            # "trained vs frozen socket at one lr" to "best-tuned A vs best-tuned B", a
            # different question. Answering it costs no extra runs, since the per-arm
            # curves are on disk.
            "per_arm_handicap": {
                arm: {
                    "own_best_lr": (own := max(
                        grid, key=lambda lr: (selection.by_lr_arm[(lr, arm)], -lr))),
                    "own_best_val_accuracy": selection.by_lr_arm[(own, arm)],
                    "val_accuracy_at_selected_lr": selection.by_lr_arm[(selection.best, arm)],
                    "handicap": selection.by_lr_arm[(own, arm)]
                    - selection.by_lr_arm[(selection.best, arm)],
                    "runs_at_its_own_optimum": own == selection.best,
                }
                for arm in selection.arms
                if all((lr, arm) in selection.by_lr_arm for lr in grid)
            },
            # Accuracy separating the winner from the runner-up. A gap far below the
            # validation noise means the choice is a coin flip.
            "margin_over_runner_up": (
                selection.mean_by_lr[selection.best]
                - max(v for lr, v in selection.mean_by_lr.items() if lr != selection.best)
                if len(grid) > 1 else float("nan")
            ),
        }

    cells = [
        block(lrs["cell_selection"][(dataset_seed, dilution, ansatz_level)],
              f"ds{dataset_seed}|{dilution}|{ansatz_level}")
        for dataset_seed in dataset_seeds
        for dilution in dilutions
        for ansatz_level in ansatz_levels
    ]
    arm_e = [
        block(lrs["arm_e_selection"][(dataset_seed, dilution)], f"ds{dataset_seed}|{dilution}")
        for dataset_seed in dataset_seeds
        for dilution in dilutions
    ]
    d_best = [
        block(lrs["d_best_selection"][(dataset_seed, width)], f"ds{dataset_seed}|M{width}")
        for dataset_seed in dataset_seeds
        for width in widths
    ]
    return {
        "rule": {
            "cell": "mean validation accuracy over arms A and B, seeds 1-3, ties to the lower lr (D-6)",
            "arm_E": "arm E alone, on the contract grid plus one point upwards (D-30)",
            "arm_D_best": "arm D_best alone, on the contract grid — it has to be a STRONG baseline",
            "contract_grid": list(CONTRACT_LR_GRID),
            "arm_E_grid": list(ARM_E_LR_GRID),
        },
        "cell": cells,
        "arm_E": arm_e,
        "arm_D_best": d_best,
        "on_grid_edge": {
            "cell": [b["cell"] for b in cells if b["on_grid_edge"]],
            "arm_E": [b["cell"] for b in arm_e if b["on_grid_edge"]],
            "arm_D_best": [b["cell"] for b in d_best if b["on_grid_edge"]],
        },
    }


def report_lr_selection(summary: dict) -> None:
    line = "=" * 100
    print(line)
    print("A7 stage 1 — lr selection. THE MAIN SERIES IS COMMITTED TO THESE VALUES.")
    print(line)
    for family, label in (("cell", "CELL (dataset seed x dilution x ansatz) — arms A and B"),
                          ("arm_E", "ARM E — own grid, one point wider (D-30)"),
                          ("arm_D_best", "ARM D_best — own grid, per width")):
        print(f"\n{label}")
        for entry in summary[family]:
            table = " ".join(f"{lr}:{value:.4f}"
                             for lr, value in entry["mean_val_accuracy_per_lr"].items())
            edge = f"  <-- ON THE {entry['edge'].upper()} EDGE" if entry["on_grid_edge"] else ""
            print(f"  {entry['cell']:<24} -> {entry['lr_selected']:<7g} | {table}"
                  f" | margin {entry['margin_over_runner_up']:+.4f}{edge}")
            for arm, per_lr in entry["mean_val_accuracy_per_lr_and_arm"].items():
                handicap = entry["per_arm_handicap"].get(arm)
                suffix = ""
                if handicap is not None:
                    suffix = (
                        "  own optimum"
                        if handicap["runs_at_its_own_optimum"]
                        else f"  own best {handicap['own_best_lr']:g}, "
                             f"handicap -{handicap['handicap']:.4f}"
                    )
                print(f"      {arm:<10} " + " ".join(f"{lr}:{v:.4f}" for lr, v in per_lr.items())
                      + suffix)

    worst = max(
        (
            (handicap["handicap"], entry["cell"], arm)
            for family in ("cell", "arm_E", "arm_D_best")
            for entry in summary[family]
            for arm, handicap in entry["per_arm_handicap"].items()
        ),
        default=(0.0, "", ""),
    )
    print(f"\nWHAT THE SHARED lr COSTS: largest per-arm handicap {worst[0]:+.4f} "
          f"({worst[2] or '-'} in {worst[1] or '-'})")
    print("  The criterion is ONE lr per cell over arms A and B (D-6), so neither arm need")
    print("  sit at its own optimum. A handicap below the validation noise costs nothing; a")
    print("  large one-sided handicap lands inside Delta_AB and is a finding. Switching to")
    print("  'every arm at its own lr' redefines the estimand and is the owner's call.")

    edges = summary["on_grid_edge"]
    total = sum(len(v) for v in edges.values())
    print(f"\nARGMAX ON A GRID EDGE: {total} selection(s)")
    for family, cells in edges.items():
        if cells:
            print(f"  {family}: {', '.join(cells)}")
    if total:
        print("  ⚠ an optimum on an edge is NOT bracketed: the value is a bound, not an optimum.")
        print("    Arm E is expected here (D-30, declared and limited). A CELL lr on the upper")
        print("    edge is the shape of D-18/D-21 and is a decision for the owner BEFORE the")
        print("    main series — widening the contract grid is not the driver's call.")
    print(line)


# --- stage 2: the gates at the final lr ----------------------------------------------


def run_gates(dataset_seed: int, *, arm_e_lr: float, seeds=LR_SELECTION_SEEDS) -> dict:
    """G1 and G2 for one generator seed, with the G1 floor at the final lr.

    A generator seed failing either gate stops the run. Exactly three seeds pass, so there
    is nothing to swap in and this cannot be a warning.

    The floor is contract arm E with a linear head, which is what makes the G1 verdict
    binding. The strong model is SVC(rbf) over the declared grid; the MLP reading is the
    ceiling and check_g1_headroom refuses it as a gate.
    """
    name, out_dir = dataset_location(dataset_seed)
    splits = load_splits(name, out_dir=out_dir)
    manifest = load_manifest(name, out_dir=out_dir)

    g2 = check_g2_effective_dim(manifest)
    g1 = check_g1_headroom(
        splits,
        strong_model=make_svc_strong_model(),
        # Single-point grid: the lr is the one stage 1 chose, not a new selection.
        floor_model=make_arm_e_linear_floor_model(lr_grid=(arm_e_lr,), seeds=tuple(seeds)),
    )
    return {
        "dataset_seed": dataset_seed,
        "dataset": name,
        "arm_e_lr_used": float(arm_e_lr),
        "g1": g1,
        "g2": g2,
        "g1_margin": g1["g1_margin"],
        "passed": bool(g1["passed"] and g2["passed"]),
    }


# --- stage 4: descriptive statistics (testing happens in the analysis script) --------

# MDE = (t_.975,n-1 + t_.80,n-1) / sqrt(n) * sigma_delta, recomputed from the sigma
# measured in this series rather than hard-coded: the pilot value covers one point of the
# axis only, and the binding MDE is the analysis script's to compute. One implementation,
# shared with the pilot and the analysis — see qsocket.stats.
mde = stats.mde
binomial_se = stats.binomial_se


def _accuracy_index(rows: list[dict], split: str = "test") -> dict:
    """(dataset_seed, arm, ansatz_level, dilution, width, seed, lr) -> row, for one split.

    `lr` is part of the key and must stay there. Arms F and D_matched have no ansatz
    dimension while the cell lr does, so when L1 and L2 select different lr those arms are
    computed twice and each paired difference has to be taken at the lr of its own cell.
    Keyed without lr, the two collide and whichever row the file happened to hold first
    wins, silently mixing lr values inside one paired difference.
    """
    index = {}
    for row in rows:
        if row["split"] != split:
            continue
        key = (
            optional_int(row["dataset_seed"]), row["arm"], row["ansatz_level"],
            row["dilution"], optional_int(row["rff_width"]),
            optional_int(row["seed"]), _lr_key(row["lr_selected"]),
        )
        # With lr in the key a duplicate can only mean the same cell computed twice, which
        # is expected by design; keeping the first is right, and the determinism check
        # below still notices any disagreement.
        index.setdefault(key, row)
    return index


def _lr_key(value) -> str:
    """lr as a canonical string, so 0.01 and "0.010" cannot become two different cells."""
    if value in (None, ""):
        return ""
    return f"{float(value):.6g}"


def paired_lr(arm, *, lrs, dataset_seed, dilution, ansatz_level, width=None) -> str:
    """The lr at which an arm is read inside one cell, as a canonical string.

    A and B take the cell lr, and so do their pair partners F and D_matched: a paired
    difference between arms trained at two different lr is not a paired difference. E has
    its own selection on the grid extended by one point, D_best its own per width.

    One definition, used by both summarise() and discordant_pairs(). A second copy is
    exactly how the accuracy table and the discordant counts would start pairing at
    different lr while both looked right.
    """
    if arm == "E":
        return _lr_key(lrs["arm_e_lr"].get((dataset_seed, dilution)))
    if arm == "D_best":
        return _lr_key(lrs["d_best_lr"].get((dataset_seed, width)))
    return _lr_key(lrs["cell_lr"].get((dataset_seed, dilution, ansatz_level)))


def arm_statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else float("nan"),
        # ddof=1: sigma_seed is an estimate of the spread over initialisations.
        "sd": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "min": float(array.min()) if array.size else float("nan"),
        "max": float(array.max()) if array.size else float("nan"),
    }


def paired_statistics(left: dict[int, float], right: dict[int, float]) -> dict:
    """Paired differences BY SEED — the estimand of this study, not a difference of means.

    Seeds present on one side only are reported as unpaired, never dropped into the mean:
    an unbalanced pairing would make sigma_Delta a different quantity.
    """
    shared = sorted(set(left) & set(right))
    differences = np.array([left[seed] - right[seed] for seed in shared], dtype=float)
    stats = arm_statistics(list(differences))
    stats.update(
        {
            "seeds": shared,
            "unpaired_left": sorted(set(left) - set(right)),
            "unpaired_right": sorted(set(right) - set(left)),
            "sigma_delta": stats["sd"],
            "mde_from_this_series": mde(stats["sd"], len(shared)) if len(shared) > 1 else float("nan"),
            "differences": [float(v) for v in differences],
        }
    )
    return stats


def summarise(rows: list[dict], *, dataset_seeds, dilutions, ansatz_levels, widths, seeds,
              gates: dict, lrs: dict) -> dict:
    """Means and sigma of every arm separately, then the four estimands.

    Per-arm means are mandatory: without them "Delta_AB > 0 while Delta_AE ~ 0" — a result
    about architecture rather than trainability — is indistinguishable from success.
    """
    test = _accuracy_index(rows, "test")
    val = _accuracy_index(rows, "val")

    def lr_of(arm, dataset_seed, dilution, ansatz_level, width=None):
        """Which lr an arm is paired at — the shared rule, never a second copy of it."""
        return paired_lr(
            arm, lrs=lrs, dataset_seed=dataset_seed, dilution=dilution,
            ansatz_level=ansatz_level, width=width,
        )

    def by_seed(dataset_seed, arm, ansatz_level, dilution, width=None, index=None, *, lr=None):
        index = test if index is None else index
        if lr is None:
            raise ValueError(
                f"by_seed({arm!r}) called without lr; the caller must DECLARE the lr at "
                "which the arm is paired. Use lr_of()."
            )
        return {
            seed: float(index[(dataset_seed, arm, ansatz_level, dilution, width, seed, lr)][
                "accuracy"
            ])
            for seed in seeds
            if (dataset_seed, arm, ansatz_level, dilution, width, seed, lr) in index
        }

    per_arm, estimands = {}, {}
    for dataset_seed in dataset_seeds:
        for dilution in dilutions:
            for ansatz_level in ansatz_levels:
                cell_lr = lr_of("A", dataset_seed, dilution, ansatz_level)
                accuracy = {
                    # A, B, F, D_matched all run at the cell lr, so every paired
                    # difference in this cell is taken at the same lr.
                    "A": by_seed(dataset_seed, "A", ansatz_level, dilution, lr=cell_lr),
                    "B": by_seed(dataset_seed, "B", ansatz_level, dilution, lr=cell_lr),
                    "E": by_seed(
                        dataset_seed, "E", "", dilution,
                        lr=lr_of("E", dataset_seed, dilution, ansatz_level),
                    ),
                    "F": by_seed(dataset_seed, "F", PRODUCT_ANSATZ, dilution, lr=cell_lr),
                    "D_matched": by_seed(dataset_seed, "D_matched", "", dilution, lr=cell_lr),
                }
                cell = f"ds{dataset_seed}|{dilution}|{ansatz_level}"
                per_arm[cell] = {
                    arm: {
                        "test": arm_statistics(list(values.values())),
                        "val": arm_statistics(
                            list(
                                by_seed(
                                    dataset_seed, arm,
                                    ansatz_of(arm, ansatz_level), dilution, index=val,
                                    lr=lr_of(arm, dataset_seed, dilution, ansatz_level),
                                ).values()
                            )
                        ),
                        "lr_selected": (
                            lrs["arm_e_lr"].get((dataset_seed, dilution))
                            if arm == "E"
                            else lrs["cell_lr"].get((dataset_seed, dilution, ansatz_level))
                        ),
                        "theta_displacement": arm_statistics([
                            float(test[key]["theta_displacement"])
                            for key in test
                            if key[:4] == (dataset_seed, arm, ansatz_of(arm, ansatz_level), dilution)
                            and key[6] == lr_of(arm, dataset_seed, dilution, ansatz_level)
                        ]),
                        "epochs_run": arm_statistics([
                            float(test[key]["epochs_run"])
                            for key in test
                            if key[:4] == (dataset_seed, arm, ansatz_of(arm, ansatz_level), dilution)
                            and key[6] == lr_of(arm, dataset_seed, dilution, ansatz_level)
                        ]),
                        "best_epoch": arm_statistics([
                            float(test[key]["best_epoch"])
                            for key in test
                            if key[:4] == (dataset_seed, arm, ansatz_of(arm, ansatz_level), dilution)
                            and key[6] == lr_of(arm, dataset_seed, dilution, ansatz_level)
                        ]),
                    }
                    for arm, values in accuracy.items()
                }
                delta_ab = paired_statistics(accuracy["A"], accuracy["B"])
                delta_ae = paired_statistics(accuracy["A"], accuracy["E"])
                delta_be = paired_statistics(accuracy["B"], accuracy["E"])
                estimands[cell] = {
                    "delta_AB": delta_ab,
                    "delta_AE": delta_ae,
                    "delta_AF": paired_statistics(accuracy["A"], accuracy["F"]),
                    "delta_BD_matched": paired_statistics(accuracy["B"], accuracy["D_matched"]),
                    # Computed in every cell: the decomposition tells "training helps"
                    # apart from "the socket adds nothing and training only undoes the
                    # damage of a random initialisation".
                    "decomposition_delta_AE": {
                        "delta_AB": delta_ab["mean"],
                        "acc_B_minus_acc_E": delta_be["mean"],
                        "sum": delta_ab["mean"] + delta_be["mean"],
                        "delta_AE": delta_ae["mean"],
                        "residual": delta_ae["mean"] - (delta_ab["mean"] + delta_be["mean"]),
                    },
                }

    # D_best: one point per (dataset seed x width); M is selected on validation.
    d_best = {}
    for dataset_seed in dataset_seeds:
        per_width = {}
        for width in widths:
            per_width[width] = {
                "test": arm_statistics(list(by_seed(
                    dataset_seed, "D_best", "", "linear", width,
                    lr=lr_of("D_best", dataset_seed, "linear", "", width),
                ).values())),
                "val": arm_statistics(
                    list(by_seed(
                        dataset_seed, "D_best", "", "linear", width, index=val,
                        lr=lr_of("D_best", dataset_seed, "linear", "", width),
                    ).values())
                ),
                "lr_selected": lrs["d_best_lr"].get((dataset_seed, width)),
            }
        available = {w: v for w, v in per_width.items() if v["val"]["n"]}
        selected = (
            min(available, key=lambda w: (-available[w]["val"]["mean"], w)) if available else None
        )
        d_best[f"ds{dataset_seed}"] = {
            "per_width": {str(w): v for w, v in per_width.items()},
            "selected_width": selected,
            "selection_rule": "highest mean validation accuracy over the 600 val rows, ties to the smaller M",
            "delta_BD_best": {
                f"{dilution}|{ansatz_level}": paired_statistics(
                    by_seed(
                        dataset_seed, "B", ansatz_level, dilution,
                        lr=lr_of("B", dataset_seed, dilution, ansatz_level),
                    ),
                    by_seed(
                        dataset_seed, "D_best", "", "linear", selected,
                        lr=lr_of("D_best", dataset_seed, "linear", "", selected),
                    ),
                )
                for dilution in dilutions
                for ansatz_level in ansatz_levels
            } if selected is not None else {},
        }

    budget = {}
    for arm in ARMS:
        hits = [r for r in rows if r["split"] == "test" and r["arm"] == arm
                and optional_int(r["epochs_run"]) >= MAX_EPOCHS]
        total = [r for r in rows if r["split"] == "test" and r["arm"] == arm]
        budget[arm] = {
            "runs": len(total),
            "hit_budget": len(hits),
            "fraction": (len(hits) / len(total)) if total else float("nan"),
        }
    all_test = [r for r in rows if r["split"] == "test"]
    budget["ALL"] = {
        "runs": len(all_test),
        "hit_budget": sum(1 for r in all_test if optional_int(r["epochs_run"]) >= MAX_EPOCHS),
        "fraction": (
            sum(1 for r in all_test if optional_int(r["epochs_run"]) >= MAX_EPOCHS) / len(all_test)
            if all_test else float("nan")
        ),
    }

    return {
        "per_arm": per_arm,
        "estimands": estimands,
        "d_best": d_best,
        "epoch_budget": budget,
        "binomial_se_1200": binomial_se(1200),
        "gates": {f"ds{k}": v for k, v in gates.items()},
        "ridge_control": ridge_summary(rows),
    }


def ridge_summary(rows: list[dict]) -> dict:
    """The closed-form readout control, reported beside the Adam-trained head.

    The gap is a finding rather than a defect: ridge minimises squared loss on +-1 labels
    while Adam minimises BCE. Nothing here tries to close it.
    """
    out: dict[str, dict] = {}
    for arm in FROZEN_ARMS:
        pairs = [
            (float(r["accuracy"]), float(r["ridge_accuracy"]))
            for r in rows
            if r["split"] == "test" and r["arm"] == arm and r["ridge_accuracy"] not in ("", None)
        ]
        if not pairs:
            continue
        adam = [a for a, _ in pairs]
        ridge = [g for _, g in pairs]
        out[arm] = {
            "n": len(pairs),
            "adam_head": arm_statistics(adam),
            "ridge": arm_statistics(ridge),
            "gap_adam_minus_ridge": arm_statistics([a - g for a, g in pairs]),
            "alpha_selected_counts": {
                str(alpha): sum(
                    1 for r in rows
                    if r["split"] == "test" and r["arm"] == arm
                    and r["ridge_alpha_selected"] not in ("", None)
                    and float(r["ridge_alpha_selected"]) == alpha
                )
                for alpha in CONTRACT_RIDGE_ALPHA_GRID
            },
        }
    return out


def discordant_pairs(rows: list[dict], out_dir: Path, *, dataset_seeds, dilutions,
                     ansatz_levels, seeds, lrs) -> dict:
    """The contingency counts McNemar is computed from — b and c, the discordant pairs.

    Descriptive, not a test — a p-value would be a test. They matter because for Delta_AE
    sigma_Delta is smaller than the binomial SE, so the paired reading is mandatory.
    b = left arm right where the right arm was wrong, c = the other way round, per seed
    and summed.

    lr is part of the key: arms F and D_matched have no ansatz dimension while the cell lr
    does, so when L1 and L2 pick different lr those arms have TWO prediction files. Keyed
    without lr the second overwrites the first and a cell is counted against the wrong run.
    """
    vectors: dict[tuple, np.ndarray] = {}
    for row in rows:
        if row["split"] != "test":
            continue
        prediction = {
            "dataset_seed": optional_int(row["dataset_seed"]),
            "arm": row["arm"],
            "ansatz_level": row["ansatz_level"],
            "dilution": row["dilution"],
            "seed": optional_int(row["seed"]),
            "width": optional_int(row["rff_width"]),
            "lr": float(row["lr_selected"]),
        }
        path = prediction_path(Path(out_dir), prediction)
        if not path.exists():
            continue
        vectors[(prediction["dataset_seed"], prediction["arm"], prediction["ansatz_level"],
                 prediction["dilution"], prediction["width"], prediction["seed"],
                 _lr_key(prediction["lr"]))] = read_prediction(path)

    def counts(left_key, right_key):
        left, right = vectors.get(left_key), vectors.get(right_key)
        if left is None or right is None or left.size != right.size:
            return None
        return {
            "b_left_right_only": int(np.sum(left & ~right)),
            "c_right_left_only": int(np.sum(right & ~left)),
            "n_test": int(left.size),
        }

    out: dict[str, dict] = {}
    for dataset_seed in dataset_seeds:
        for dilution in dilutions:
            for ansatz_level in ansatz_levels:
                cell = f"ds{dataset_seed}|{dilution}|{ansatz_level}"
                pairs = {
                    "delta_AB": ("A", ansatz_level, dilution, None, "B", ansatz_level, dilution, None),
                    "delta_AE": ("A", ansatz_level, dilution, None, "E", "", dilution, None),
                    "delta_AF": ("A", ansatz_level, dilution, None, "F", PRODUCT_ANSATZ, dilution, None),
                    "delta_BD_matched": ("B", ansatz_level, dilution, None, "D_matched", "", dilution, None),
                }
                block: dict[str, dict] = {}
                for name, (la, laa, ld, lw, ra, raa, rd, rw) in pairs.items():
                    # ansatz_level, not the RECORDED ansatz: F carries "product" in its
                    # rows but is paired at the cell lr of the level it sits inside.
                    left_lr = paired_lr(la, lrs=lrs, dataset_seed=dataset_seed,
                                        dilution=ld, ansatz_level=ansatz_level, width=lw)
                    right_lr = paired_lr(ra, lrs=lrs, dataset_seed=dataset_seed,
                                         dilution=rd, ansatz_level=ansatz_level, width=rw)
                    per_seed = {}
                    for seed in seeds:
                        got = counts(
                            (dataset_seed, la, laa, ld, lw, seed, left_lr),
                            (dataset_seed, ra, raa, rd, rw, seed, right_lr),
                        )
                        if got is not None:
                            per_seed[seed] = got
                    if not per_seed:
                        continue
                    block[name] = {
                        "per_seed": per_seed,
                        "b_total": sum(v["b_left_right_only"] for v in per_seed.values()),
                        "c_total": sum(v["c_right_left_only"] for v in per_seed.values()),
                        "seeds": sorted(per_seed),
                    }
                if block:
                    out[cell] = block
    return out


def determinism_check(result_rows: list[dict], lr_rows: list[dict]) -> dict:
    """Determinism at the pipeline level, for free.

    Stage 1 measured arms A and B at every lr of the grid, including the selected one;
    stage 3 re-measured the same cells in a different process and task order. Every
    overlapping cell must agree bit for bit, checked on what actually ran rather than on a
    re-derivation of the recipe.
    """
    stage1 = {}
    for row in lr_rows:
        key = (optional_int(row["dataset_seed"]), row["arm"], row["ansatz_level"],
               row["dilution"], optional_int(row["seed"]), f"{float(row['lr']):g}")
        stage1[key] = float(row["val_accuracy"])

    compared, mismatches = 0, []
    for row in result_rows:
        if row["split"] != "val" or row["arm"] not in ("A", "B"):
            continue
        key = (optional_int(row["dataset_seed"]), row["arm"], row["ansatz_level"],
               row["dilution"], optional_int(row["seed"]), f"{float(row['lr_selected']):g}")
        if key not in stage1:
            continue
        compared += 1
        if float(row["accuracy"]) != stage1[key]:
            mismatches.append(
                {"cell": key, "stage1": stage1[key], "stage3": float(row["accuracy"])}
            )
    return {
        "overlapping_cells_compared": compared,
        "mismatches": mismatches,
        "passed": compared > 0 and not mismatches,
        "note": (
            "arms A and B at the selected lr and seeds 1-3 are measured twice: once in the "
            "lr stage and once in the main grid, in different processes. Bit-for-bit "
            "equality is test T-I at the pipeline level."
        ),
    }


def annotate(rows: list[dict]) -> list[dict]:
    """In-memory copies carrying dataset_seed. NEVER written: append_result_row rejects
    unknown columns, and the CSV schema is closed."""
    return [{**row, "dataset_seed": dataset_seed_of(row["dataset"])} for row in rows]


# --- the verdict table, declared before the run --------------------------------------

# "theta did not move". Declared, not fitted: the diagnostic separates "training is
# unnecessary" from "the optimiser never moved", which is a different claim.
# Share of runs hitting the epoch budget above which the run is "about the budget".


def verdicts(summary: dict) -> list[dict]:
    """Mechanical evaluation of the declared verdict table; it does not interpret.

    Every threshold is either declared above or recomputed from the sigma measured in this
    series. The pilot MDE is deliberately not used as a decision threshold.
    """
    out = []
    for cell, estimand in summary["estimands"].items():
        delta_ab = estimand["delta_AB"]
        theta_a = summary["per_arm"][cell]["A"]["theta_displacement"]["mean"]
        cell_mde = delta_ab["mde_from_this_series"]
        # Each row is read against the MDE of its own contrast: sigma_Delta depends on
        # the contrast (an order of magnitude larger for A-B than for A-E), so judging
        # acc(A) ~ acc(E) against the A-B threshold would fire that stop row whenever A-B
        # is noisy.
        ae_mde = estimand["delta_AE"]["mde_from_this_series"]

        if np.isfinite(theta_a) and theta_a < THETA_STILL and abs(delta_ab["mean"]) < cell_mde:
            out.append({
                "cell": cell, "row": "Delta_AB ~ 0 with theta_displacement ~ 0",
                "verdict": "STOP",
                "why": (
                    "this is an OPTIMISER FAILURE, not 'training is unnecessary' "
                    "(SPEC 7.7). Different thesis, different paper — it may not be "
                    "written in after the fact."
                ),
                "numbers": {"delta_AB": delta_ab["mean"], "theta_displacement": theta_a,
                            "mde_this_series": cell_mde},
            })

        acc = summary["per_arm"][cell]
        a, b, e = acc["A"]["test"]["mean"], acc["B"]["test"]["mean"], acc["E"]["test"]["mean"]
        if np.isfinite(ae_mde) and abs(a - e) < ae_mde and b < e:
            out.append({
                "cell": cell, "row": "acc(A) ~ acc(E) with acc(B) < acc(E)",
                "verdict": "STOP",
                "why": (
                    "'the socket adds nothing; training only undoes the damage of a "
                    "random initialisation' (SPEC section 9) — a result about "
                    "ARCHITECTURE, not about trainability."
                ),
                "numbers": {"acc_A": a, "acc_B": b, "acc_E": e,
                            "mde_AE_this_series": ae_mde},
            })

        if np.isfinite(cell_mde) and abs(delta_ab["mean"]) < cell_mde:
            out.append({
                "cell": cell, "row": "|Delta| below the MDE of this series",
                "verdict": "UNDECIDABLE at n=10",
                "why": "report the CI; do NOT write 'there is no effect' (raport.tex 3.7)",
                "numbers": {"delta_AB": delta_ab["mean"], "mde_this_series": cell_mde},
            })
        elif delta_ab["mean"] > 0:
            out.append({
                "cell": cell, "row": "Delta_AB > 0, theta moves",
                "verdict": "baseline scenario",
                "why": "analyse per A6",
                "numbers": {"delta_AB": delta_ab["mean"], "theta_displacement": theta_a,
                            "mde_this_series": cell_mde},
            })

    for key, gate in summary["gates"].items():
        if gate.get("passed") is False:
            out.append({
                "cell": key, "row": "a generator seed fails G1/G2 at the final lr",
                "verdict": "STOP",
                "why": "Z8: exactly three seeds pass and there is nothing to replace one with",
                "numbers": {"g1_margin": gate["g1_margin"],
                            "g1_failures": gate["g1"]["failures"],
                            "g2_failures": gate["g2"]["failures"]},
            })

    budget = summary["epoch_budget"]["ALL"]
    if np.isfinite(budget["fraction"]) and budget["fraction"] > BUDGET_HIT_NOTE_FRACTION:
        out.append({
            "cell": "ALL", "row": "> 20 % of runs hit the 300-epoch budget",
            "verdict": "NOTE the number",
            "why": "CONTRACTS 7.1: 'budget too small -> the result is about the budget'",
            "numbers": budget,
        })
    return out


# --- console report ------------------------------------------------------------------


def report(summary: dict, *, lrs: dict, determinism: dict, wall: float, out_dir: Path) -> None:
    line = "=" * 100
    print(line)
    print("A7 — main series: descriptive statistics. Tests, CI, TOST and figures are A8.")
    print(line)

    print("\nSELECTED lr (cell = dataset seed x dilution x ansatz; arms A and B, seeds 1-3)")
    for key in sorted(lrs["cell_lr"]):
        dataset_seed, dilution, ansatz_level = key
        selection = lrs["cell_selection"][key]
        table = " ".join(f"{lr:g}:{selection.mean_by_lr[lr]:.4f}" for lr in selection.grid)
        print(f"  ds{dataset_seed:<3} {dilution:<7} {ansatz_level:<3} -> {lrs['cell_lr'][key]:<6g} | {table}")
    print("\nSELECTED lr, arm E (own grid, one point wider — D-30)")
    for key in sorted(lrs["arm_e_lr"]):
        print(f"  ds{key[0]:<3} {key[1]:<7} -> {lrs['arm_e_lr'][key]:g}")
    print("\nSELECTED lr, arm D_best (own grid, per width)")
    for key in sorted(lrs["d_best_lr"]):
        print(f"  ds{key[0]:<3} M={key[1]:<4} -> {lrs['d_best_lr'][key]:g}")

    print("\nGATES at the final lr")
    for key, gate in sorted(summary["gates"].items()):
        if not gate.get("evaluated", True):
            print(f"  {key}: NOT EVALUATED — {gate['reason']}")
            continue
        print(f"  {key}: G1 {'PASS' if gate['g1']['passed'] else 'FAIL'} "
              f"margin {gate['g1_margin']:+.6f} strong {gate['g1']['strong']['accuracy']:.4f} "
              f"floor {gate['g1']['floor']['accuracy']:.4f} | "
              f"G2 {'PASS' if gate['g2']['passed'] else 'FAIL'} evr1 {gate['g2']['top_share']:.4f}")

    print("\nPER-ARM means and sigma on the 1200 test rows — EVERY ARM SEPARATELY")
    print(f"  {'cell':<22} {'arm':<10} {'n':>3} {'mean':>8} {'sd':>8} {'theta_disp':>11} {'epochs':>7}")
    for cell in sorted(summary["per_arm"]):
        for arm, stats in summary["per_arm"][cell].items():
            print(f"  {cell:<22} {arm:<10} {stats['test']['n']:>3} "
                  f"{stats['test']['mean']:>8.4f} {stats['test']['sd']:>8.4f} "
                  f"{stats['theta_displacement']['mean']:>11.2e} "
                  f"{stats['epochs_run']['mean']:>7.1f}")

    print("\nESTIMANDS, paired by seed. MDE recomputed from the sigma of THIS series.")
    print(f"  {'cell':<22} {'estimand':<17} {'n':>3} {'mean':>9} {'sd':>8} {'MDE':>8} {'>=MDE':>6}")
    for cell in sorted(summary["estimands"]):
        for name, stats in summary["estimands"][cell].items():
            if name.startswith("decomposition"):
                continue
            flag = "yes" if abs(stats["mean"]) >= stats["mde_from_this_series"] else "no"
            print(f"  {cell:<22} {name:<17} {stats['n']:>3} {stats['mean']:>+9.4f} "
                  f"{stats['sd']:>8.4f} {stats['mde_from_this_series']:>8.4f} {flag:>6}")

    print("\nDECOMPOSITION Delta_AE = Delta_AB + (acc(B) - acc(E)) — mandatory in every cell")
    for cell in sorted(summary["estimands"]):
        d = summary["estimands"][cell]["decomposition_delta_AE"]
        print(f"  {cell:<22} {d['delta_AB']:+.4f} + {d['acc_B_minus_acc_E']:+.4f} = "
              f"{d['sum']:+.4f}  (Delta_AE {d['delta_AE']:+.4f}, residual {d['residual']:+.2e})")

    print("\nARM D_best — off the dilution axis, one point per (dataset seed x M)")
    for key, block in sorted(summary["d_best"].items()):
        print(f"  {key}: selected M = {block['selected_width']}")
        for width, stats in block["per_width"].items():
            print(f"    M={width:<4} val {stats['val']['mean']:.4f} test {stats['test']['mean']:.4f} "
                  f"(sd {stats['test']['sd']:.4f}, lr {stats['lr_selected']})")

    print("\nRIDGE readout control — REPORTED BESIDE the Adam head, not a consistency condition (D-19)")
    for arm, block in sorted(summary["ridge_control"].items()):
        print(f"  {arm:<10} n {block['n']:>4} adam {block['adam_head']['mean']:.4f} "
              f"ridge {block['ridge']['mean']:.4f} gap {block['gap_adam_minus_ridge']['mean']:+.4f} "
              f"alpha {block['alpha_selected_counts']}")

    print("\nEPOCH BUDGET — how many runs hit 300 epochs")
    for arm, block in summary["epoch_budget"].items():
        print(f"  {arm:<10} {block['hit_budget']:>4} / {block['runs']:<4} "
              f"= {100 * block['fraction']:.1f} %" if block["runs"] else f"  {arm:<10} no runs")

    print("\nTHREE UNCERTAINTY ACCOUNTS side by side, never combined (raport.tex 3.7)")
    print(f"  (2) binomial SE on 1200 test rows: {summary['binomial_se_1200']:.4f}")
    print("  (1) sigma_Delta per cell: the 'sd' column of the estimand table above")
    print("  (3) discordant pairs b/c — the input of McNemar; the test itself is A8")
    for cell, block in sorted(summary.get("discordant_pairs", {}).items()):
        parts = " ".join(f"{name}: b {v['b_total']} c {v['c_total']}" for name, v in block.items())
        print(f"      {cell:<22} {parts}")

    print(f"\nDETERMINISM (T-I, pipeline): {determinism['overlapping_cells_compared']} overlapping "
          f"cells compared, {len(determinism['mismatches'])} mismatches -> "
          f"{'PASS' if determinism['passed'] else 'FAIL'}")

    print("\nVERDICTS from the table declared before the run")
    rows = verdicts(summary)
    if not rows:
        print("  (no row of the verdict table fired)")
    for row in rows:
        print(f"  [{row['verdict']}] {row['cell']}: {row['row']}")
        print(f"      {row['why']}")
        print(f"      {row['numbers']}")

    print(f"\nwall {wall:.1f} s   outputs in {out_dir}")
    print(line)


# --- validation: usage errors fail before any training ------------------------------


def validate(*, dataset_seeds, dilutions, ansatz_levels, seeds, widths, allow_generate: bool,
             f_dilutions=ARM_F_DILUTIONS):
    """Everything that can be wrong about the request, checked before the first run."""
    unknown = [d for d in dilutions if d not in HEAD_PARAM_COUNTS]
    if unknown:
        raise ValueError(f"unknown dilution(s) {unknown}; the axis is {list(DILUTION_AXIS)}")
    unknown = [a for a in ansatz_levels if a not in ANSATZ_LEVELS]
    if unknown:
        raise ValueError(f"unknown ansatz level(s) {unknown}; expected {list(ANSATZ_LEVELS)}")
    unknown = [s for s in dataset_seeds if s not in DATASET_SEEDS]
    if unknown:
        raise ValueError(
            f"generator seed(s) {unknown} are not among {list(DATASET_SEEDS)}. Z8: exactly "
            "these three pass the binding gates and there is nothing to replace them with."
        )
    unknown = [w for w in widths if w not in D_BEST_WIDTHS]
    if unknown:
        raise ValueError(f"D_best width(s) {unknown} outside the declared {list(D_BEST_WIDTHS)}")
    if not seeds:
        raise ValueError("no training seeds")
    unknown = [d for d in f_dilutions if d not in dilutions]
    if unknown:
        raise ValueError(
            f"arm F was asked for dilution(s) {unknown}, which are not in the run's "
            f"dilutions {list(dilutions)}: Delta_AF would have no arm A to pair with"
        )

    # h42 and the historical name mlp42 must be one head; asserted in the run, not only
    # in the test suite.
    assert canonical_head_name("h42") == "mlp42"
    assert HEAD_PARAM_COUNTS["h42"] == HEAD_PARAM_COUNTS["mlp42"]
    for seed in list(seeds)[:3]:
        for a, b in zip(make_head("h42", seed=seed).parameters(),
                        make_head("mlp42", seed=seed).parameters()):
            assert torch.equal(a, b), f"h42 and mlp42 differ at seed {seed}"

    # 7h+1 and 5+1, on the objects that will actually run.
    for dilution in dilutions:
        built = sum(p.numel() for p in make_head(dilution, seed=1).parameters())
        assert built == HEAD_PARAM_COUNTS[dilution], f"{dilution}: {built} parameters"

    manifests = {}
    for dataset_seed in dataset_seeds:
        manifests[dataset_seed] = ensure_dataset(dataset_seed, allow_generate=allow_generate)

    for ansatz_level in ansatz_levels:
        for seed in seeds:
            assert_theta_pairing(seed, ansatz_level)

    return manifests


def jacobian_ranks(ansatz_levels) -> dict:
    """socket_params_effective: the numerical rank of d<Z_i>/d(theta).

    Not the Fisher-information effective dimension of Abbas et al. Every row carries both
    the nominal parameter count and this rank.
    """
    ranks = {}
    for name in tuple(ansatz_levels) + (PRODUCT_ANSATZ,):
        ranks[name] = int(
            effective_dimension(
                lambda n_qubits, R, _name=name: build_socket_circuit(_name, n_qubits, R),
                R_CONTRACT,
            )
        )
    return ranks


# --- the run -------------------------------------------------------------------------


def run(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    dataset_seeds=DATASET_SEEDS,
    dilutions=DILUTIONS,
    ansatz_levels=ANSATZ_LEVELS,
    seeds=SEEDS,
    widths=D_BEST_WIDTHS,
    lr_seeds=LR_SELECTION_SEEDS,
    f_dilutions=ARM_F_DILUTIONS,
    workers: int | None = None,
    allow_generate: bool = True,
    stop_after: str | None = None,
) -> dict:
    started = time.perf_counter()
    workers = default_workers() if workers is None else workers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_CSV
    lr_path = out_dir / LR_TABLE_CSV

    run_id = f"a7_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    commit, environment = git_commit(), env_hash()
    context = {"run_id": run_id, "commit": commit, "environment": environment}

    print(f"# A7 main series  run_id {run_id}  commit {commit}  env {environment}")
    print(f"#   generator seeds {list(dataset_seeds)}  dilutions {list(dilutions)}  "
          f"ansatze {list(ansatz_levels)}  seeds {list(seeds)}  widths {list(widths)}")
    print(f"#   workers {workers}  patience {PATIENCE}  budget {MAX_EPOCHS}  "
          f"backend {DEFAULT_BACKEND}")

    manifests = validate(
        dataset_seeds=dataset_seeds, dilutions=dilutions, ansatz_levels=ansatz_levels,
        seeds=seeds, widths=widths, allow_generate=allow_generate, f_dilutions=f_dilutions,
    )
    for dataset_seed, manifest in manifests.items():
        print(f"#   ds{dataset_seed}: {manifest['frozen_name']} "
              f"dataset_hash {manifest['dataset_hash'][:16]} pca_hash {manifest['pca_hash'][:16]}")

    ranks = jacobian_ranks(ansatz_levels)
    print(f"#   Jacobian rank of the socket per ansatz: {ranks} "
          f"(nominal {socket_param_count(DEFAULT_N_QUBITS, R_CONTRACT)})")

    # --- stage 1: lr selection -------------------------------------------------------
    lr_rows: list[dict] = []
    if lr_path.exists():
        import pandas as pd

        lr_rows = pd.read_csv(
            lr_path, keep_default_na=False, float_precision="round_trip"
        ).to_dict("records")
        for row in lr_rows:
            row["width"] = optional_int(row["rff_width"])
        print(f"#   resuming: {len(lr_rows)} lr rows already on disk")

    lr_tasks = plan_lr_stage(
        dataset_seeds=dataset_seeds, dilutions=dilutions, ansatz_levels=ansatz_levels,
        seeds=lr_seeds, widths=widths, ranks=ranks, context=context,
        done=existing_lr_keys(lr_path),
    )
    print(f"\n## stage 1 — lr selection: {len(lr_tasks)} tasks")
    completed = [0]

    def on_lr_result(result: dict) -> None:
        rows = lr_table_rows(result)
        for row in rows:
            row["width"] = optional_int(row["rff_width"])
        append_rows(lr_path, LR_TABLE_COLUMNS, rows)  # on disk BEFORE anything else
        lr_rows.extend(rows)
        completed[0] += 1
        task = result["task"]
        print(f"  [lr {completed[0]:>4}/{len(lr_tasks)}] ds{task['dataset_seed']} "
              f"{task['arm']:<10} {task['ansatz_level'] or '-':<7} seed {task['seed']} "
              f"{len(rows)} cells", flush=True)

    execute(lr_tasks, workers=workers, on_result=on_lr_result)

    lrs = select_all_lrs(
        lr_rows, dataset_seeds=dataset_seeds, dilutions=dilutions,
        ansatz_levels=ansatz_levels, widths=widths, seeds=lr_seeds,
    )
    selection_summary = lr_selection_summary(
        lrs, dataset_seeds=dataset_seeds, dilutions=dilutions,
        ansatz_levels=ansatz_levels, widths=widths,
    )
    (out_dir / LR_SELECTION_JSON).write_text(
        json.dumps(selection_summary, indent=2, default=str), encoding="utf-8"
    )
    if stop_after == "lr":
        # Stopping here exists to inspect the sweep before committing the main series, so
        # the selection is printed and written, not just returned.
        report_lr_selection(selection_summary)
        print(f"\nlr table   {lr_path}")
        print(f"selection  {out_dir / LR_SELECTION_JSON}")
        print("\nThe main series runs from these numbers: re-run the same command with the")
        print("same --out-dir and without --stop-after; stage 1 will plan 0 tasks and the")
        print("selection is recomputed from the table on disk.")
        return {"lrs": lrs, "lr_rows": lr_rows, "lr_selection": selection_summary}

    # --- stage 2: gates at the final lr ----------------------------------------------
    # G1's floor is arm E at the linear head, so the gate can only be evaluated by a run
    # that has the linear dilution; a probe on a single non-linear dilution would
    # otherwise die here after paying for stage 1.
    #
    # The gates are a property of the dataset, and frozen datasets are hash-verified at
    # load, so a probe may cite the gate verdict of the run that computed it but not claim
    # to have established it. It is therefore skipped, recorded as skipped, and
    # `g1_margin` stays empty in every row rather than carrying an unmeasured number.
    if "linear" in dilutions:
        print("\n## stage 2 — G1/G2 per generator seed at the final lr")
        gates = {}
        for dataset_seed in dataset_seeds:
            gate = run_gates(dataset_seed,
                             arm_e_lr=lrs["arm_e_lr"][(dataset_seed, "linear")],
                             seeds=lr_seeds)
            gates[dataset_seed] = gate
            print(f"  ds{dataset_seed}: G1 {'PASS' if gate['g1']['passed'] else 'FAIL'} "
                  f"margin {gate['g1_margin']:+.6f} | "
                  f"G2 {'PASS' if gate['g2']['passed'] else 'FAIL'} "
                  f"evr1 {gate['g2']['top_share']:.4f}", flush=True)
    else:
        print("\n## stage 2 — NOT EVALUATED (no linear dilution in this run)")
        print("   G1's floor is arm E at the LINEAR head, so this run cannot evaluate it.")
        print("   The verdict of the run that did stands; this one does NOT re-establish")
        print("   it, and g1_margin stays empty in every row below.")
        gates = {
            dataset_seed: {
                "dataset_seed": dataset_seed,
                "evaluated": False,
                "passed": None,
                "g1_margin": None,
                "reason": "no linear dilution in this run; G1's floor is arm E at the "
                          "linear head. NOT evaluated, and NOT to be read as passed.",
            }
            for dataset_seed in dataset_seeds
        }
    (out_dir / GATES_JSON).write_text(json.dumps(gates, indent=2, default=str), encoding="utf-8")

    # `is False`, not `not ...`: an unevaluated gate is None and must not read as a failure.
    failed = [seed for seed, gate in gates.items() if gate.get("passed") is False]
    if failed:
        # Stop row of the verdict table, not a warning: there is nothing to substitute.
        raise SystemExit(
            f"STOP: generator seed(s) {failed} fail G1/G2 at the final "
            f"lr. Exactly three seeds pass the binding gates (Z8) and there is nothing to "
            f"replace them with. Gate detail in {out_dir / GATES_JSON}. The lr table is on "
            f"disk; nothing of the main grid was run."
        )
    if stop_after == "gates":
        # Same output as --stop-after lr, plus the gate verdicts. The recommended end of
        # phase one: the gates cost minutes, and a failing generator seed is better
        # learned here than nine hours later.
        report_lr_selection(selection_summary)
        print(f"\nlr table   {lr_path}")
        print(f"selection  {out_dir / LR_SELECTION_JSON}")
        print(f"gates      {out_dir / GATES_JSON}")
        print("\nThe main series runs from these numbers: re-run the same command with the")
        print("same --out-dir and without --stop-after; stage 1 will plan 0 tasks and the")
        print("selection is recomputed from the table on disk.")
        return {"lrs": lrs, "gates": gates, "lr_rows": lr_rows,
                "lr_selection": selection_summary}

    # --- stage 3: the main grid ------------------------------------------------------
    result_rows: list[dict] = []
    if results_path.exists():
        import pandas as pd

        result_rows = pd.read_csv(
            results_path, keep_default_na=False, float_precision="round_trip"
        ).to_dict("records")
        print(f"#   resuming: {len(result_rows)} result rows already on disk")

    margins = {seed: gate.get("g1_margin") for seed, gate in gates.items()}
    main_tasks = plan_main_stage(
        dataset_seeds=dataset_seeds, dilutions=dilutions, ansatz_levels=ansatz_levels,
        seeds=seeds, widths=widths, lrs=lrs, ranks=ranks, margins=margins,
        context=context, done=existing_row_keys(results_path), f_dilutions=f_dilutions,
    )
    skipped = [d for d in dilutions if d not in f_dilutions]
    if skipped:
        # No silent caps: a narrowed arm F must be visible in the log and in the summary,
        # or the missing rows read as a bug rather than as a decision.
        print(f"#   arm F runs on {list(f_dilutions)} only; NOT on {skipped} "
              f"(-{len(skipped) * len(seeds) * len(dataset_seeds)} expensive runs)")
    print(f"\n## stage 3 — main grid: {len(main_tasks)} tasks")
    completed = [0]

    def on_main_result(result: dict) -> None:
        # Raw rows to disk first, one at a time, through the schema-validating writer.
        for row in result["rows"]:
            append_result_row(results_path, row)
        for prediction in result["predictions"]:
            write_prediction(out_dir, prediction)
        for record in result.get("weights", []):
            write_weights(out_dir, record["identity"], arrays=record["arrays"])
        result_rows.extend(result["rows"])
        completed[0] += 1
        task = result["task"]
        seconds = float(result["rows"][0]["wall_seconds"]) if result["rows"] else 0.0
        print(f"  [main {completed[0]:>4}/{len(main_tasks)}] ds{task['dataset_seed']} "
              f"{task['arm']:<10} {task['ansatz_level'] or '-':<7} seed {task['seed']} "
              f"{len(result['rows'])} rows  {seconds:.1f} s/cell", flush=True)

    execute(main_tasks, workers=workers, on_result=on_main_result)

    # --- stage 4: descriptive statistics --------------------------------------------
    annotated = annotate(result_rows)
    summary = summarise(
        annotated, dataset_seeds=dataset_seeds, dilutions=dilutions,
        ansatz_levels=ansatz_levels, widths=widths, seeds=seeds, gates=gates, lrs=lrs,
    )
    determinism = determinism_check(annotated, lr_rows)
    summary["determinism_T_I"] = determinism
    # Three uncertainty accounts side by side, never combined into one number:
    # sigma_Delta over the paired differences, the binomial SE on the test rows, and the
    # discordant-pair counts McNemar needs. The test itself lives in the analysis script.
    summary["discordant_pairs"] = discordant_pairs(
        annotated, out_dir, dataset_seeds=dataset_seeds, dilutions=dilutions,
        ansatz_levels=ansatz_levels, seeds=seeds, lrs=lrs,
    )
    summary["lr_selection"] = selection_summary
    summary["selected_lr"] = {
        "cell": {f"ds{k[0]}|{k[1]}|{k[2]}": v for k, v in lrs["cell_lr"].items()},
        "arm_E": {f"ds{k[0]}|{k[1]}": v for k, v in lrs["arm_e_lr"].items()},
        "arm_D_best": {f"ds{k[0]}|M{k[1]}": v for k, v in lrs["d_best_lr"].items()},
        "contract_grid": list(CONTRACT_LR_GRID),
        "arm_E_grid": list(ARM_E_LR_GRID),
        "rule": "mean validation accuracy over arms A and B at seeds 1-3, ties to the lower lr",
    }
    summary["configuration"] = {
        "run_id": run_id, "git_commit": commit, "env_hash": environment,
        "dataset_seeds": list(dataset_seeds), "dilutions": list(dilutions),
        "ansatz_levels": list(ansatz_levels), "seeds": list(seeds),
        "d_best_widths": list(widths), "arm_F_dilutions": list(f_dilutions),
        "R": R_CONTRACT, "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
        "backend": DEFAULT_BACKEND, "jacobian_rank": ranks,
        "ridge_alpha_grid": list(CONTRACT_RIDGE_ALPHA_GRID),
        "workers": workers,
    }
    summary["verdicts"] = verdicts(summary)
    wall = time.perf_counter() - started
    summary["wall_seconds"] = wall
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    report(summary, lrs=lrs, determinism=determinism, wall=wall, out_dir=out_dir)
    return {"summary": summary, "rows": result_rows, "lr_rows": lr_rows, "lrs": lrs,
            "gates": gates}


# --- CLI -----------------------------------------------------------------------------

# 2 seeds x 1 dilution x 2 ansatze x every arm, ~20 min: catches column, cache and
# seeding mistakes before the full run.
DRY_RUN = {
    "dataset_seeds": (11,),
    "dilutions": ("linear",),
    "ansatz_levels": ANSATZ_LEVELS,
    "seeds": (1, 2),
    "widths": D_BEST_WIDTHS,
    "lr_seeds": (1,),
    "f_dilutions": ("linear",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="2 seeds x 1 dilution x 2 ansatze x every arm, 1 lr seed")
    parser.add_argument("--dataset-seeds", type=int, nargs="+", default=list(DATASET_SEEDS))
    parser.add_argument("--dilutions", nargs="+", default=list(DILUTIONS))
    parser.add_argument("--ansatz-levels", nargs="+", default=list(ANSATZ_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(D_BEST_WIDTHS))
    parser.add_argument("--f-dilutions", nargs="+", default=list(ARM_F_DILUTIONS),
                        help="dilutions arm F runs on (default: all four). Narrowing this "
                             "narrows Delta_AF; the two ENDS of the axis keep the control "
                             "control argument at half the cost")
    parser.add_argument("--lr-seeds", type=int, nargs="+", default=list(LR_SELECTION_SEEDS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stop-after", choices=("lr", "gates"), default=None)
    parser.add_argument("--no-generate", action="store_true",
                        help="refuse to generate a missing generator-seed dataset")
    parser.add_argument("--prepare-datasets", action="store_true",
                        help="generate and hash-check the generator-seed datasets, then "
                             "exit; run this once before starting cluster jobs")
    args = parser.parse_args()

    options = dict(
        dataset_seeds=tuple(args.dataset_seeds), dilutions=tuple(args.dilutions),
        ansatz_levels=tuple(args.ansatz_levels), seeds=tuple(args.seeds),
        widths=tuple(args.widths), lr_seeds=tuple(args.lr_seeds),
        f_dilutions=tuple(args.f_dilutions),
    )
    out_dir = args.out_dir
    if args.dry_run:
        options = dict(DRY_RUN)
        out_dir = out_dir if args.out_dir != DEFAULT_OUT_DIR else DEFAULT_OUT_DIR / "dry_run"

    torch.set_num_threads(1)
    if args.prepare_datasets:
        for dataset_seed in options["dataset_seeds"]:
            manifest = ensure_dataset(dataset_seed, allow_generate=not args.no_generate)
            print(f"ds{dataset_seed}  {manifest['frozen_name']}")
            print(f"    dataset_hash {manifest['dataset_hash']}")
            print(f"    pca_hash     {manifest['pca_hash']}")
            print(f"    file_sha256  {manifest['file_sha256']}")
            print(f"    evr1         {manifest['pca']['explained_variance_ratio_'][0]:.6f}")
        print("datasets ready and hash-checked; start the jobs with --no-generate")
        return
    run(out_dir=out_dir, workers=args.workers, allow_generate=not args.no_generate,
        stop_after=args.stop_after, **options)


if __name__ == "__main__":
    main()
