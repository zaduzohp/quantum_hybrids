"""Correlator-readout probe: what does freezing the socket cost under a RICHER readout?

The main series reads five <Z_i> and measures Delta_AB = +0.0883 (L1, linear head). The
objection is that the QELM/QRC convention rests on a rich readout, so a cost measured at
the narrowest possible readout is measured where the convention cannot work by
construction. Estimand: Delta_{A'B'} at 15 observables, placed NEXT TO Delta_AB at 5.

Rule fixed BEFORE the run and not changed after the numbers were seen: Delta_{A'B'} <= MDE
means the cost is a function of readout width rather than freezing; >= Delta_AB - MDE
means widening does not compensate freezing; in between is partial compensation, reported
as a fraction of Delta_AB.

STATUS: exploratory, outside the confirmatory family, exactly like arm D.

Scope (D-34): order=2 (5 <Z_i> + 10 <Z_i Z_j>); arms A_corr / B_corr / D_corr; linear head
only, NOT a point of the dilution axis; L1, L2 behind --ansatz; ds11/22/33 x 10 seeds; its
own lr selection family. order=5 is out of scope but supported, so adding it is a flag.

A separate script because no row of the main series may change. It borrows run_cell through
that function's injection points, so the feature cache and ridge control stay covered by
tests/test_feature_cache.py.

WEIGHTS are saved in BOTH stages, unlike the main series (whose runs predate write_weights,
leaving 970 sets unrecoverable). A cell measured in both stages writes one filename twice;
the second write verifies rather than overwrites, making the repeat a determinism witness.

SCHEMA is RESULT_COLUMNS + readout_order, in its OWN directory. The column is deliberately
NOT added to RESULT_COLUMNS: the resume reader and run_a8_analysis compare schemas for
equality, so a 47th column would make the pipeline refuse the 46-column main-series results
raport.tex was computed from.

Cluster: scripts/wcss/README_probe.md. Locally: --dry-run, then --workers 10.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Before numpy and torch: the BLAS pools read these at import and never again.
from qsocket.core import (
    FEATURE_RANGE,
    derive,
    pin_blas_threads,
    readout_size,
)

pin_blas_threads()

import numpy as np
import run_main_series as a7
import torch

from qsocket.ansatzes import build_socket_circuit, socket_param_count
from qsocket.head import make_linear_readout
from qsocket.rank import effective_dimension
from qsocket.results import RESULT_COLUMNS
from qsocket.socket import DEFAULT_N_QUBITS, make_socket
from qsocket.training import lr_selection_from_measurements

# --- contract ------------------------------------------------------------------------

PROBE_ID = "probe_corr"
# Predictions of the selection runs live beside the grid's, not among them: they are
# measurements of four lr values per cell, not of the estimand.
LR_STAGE_DIR = "lr_stage"
DEFAULT_OUT_DIR = ROOT / "outputs" / PROBE_ID

# order=2 is the scope of this run. The code takes any order the register admits.
DEFAULT_READOUT_ORDER = 2

# One point, and it is not a point of the dilution axis: make_head hardcodes
# SOCKET_WIDTH = 5, the MLP heads are not to be touched, and the "quantum share" column
# of the dilution table is computed over 5 outputs. The probe stands in the cell where
# the confirmatory question stands, and nowhere else.
PROBE_DILUTION = "linear"

TRAINED_ARMS: tuple[str, ...] = ("A_corr",)
FROZEN_ARMS: tuple[str, ...] = ("B_corr", "D_corr")
ARMS: tuple[str, ...] = ("A_corr", "B_corr", "D_corr")
# Mirror of a7.LR_SELECTION_ARMS = ("A", "B"): widening the readout changes the scale of
# the head gradient, so an lr chosen at 5 outputs is not the same lr. D_corr is selected
# on itself, as D_best is, because a baseline denied its own lr is a straw baseline.
LR_SELECTION_ARMS: tuple[str, ...] = ("A_corr", "B_corr")
ANSATZ_FREE_ARMS: tuple[str, ...] = ("D_corr",)

# The main-series arm each probe arm is the widened twin of. Used for two things: the
# initialisation keys (which MUST coincide, see assert_theta_pairing_across_readouts) and
# the generic provenance of a results row (see _probe_row).
TWIN: dict[str, str] = {"A_corr": "A", "B_corr": "B", "D_corr": "D_best"}

PROBE_RESULT_COLUMNS: tuple[str, ...] = RESULT_COLUMNS + ("readout_order",)

LR_TABLE_COLUMNS: tuple[str, ...] = a7.LR_TABLE_COLUMNS + ("readout_order",)

# Mean wall time of one arm-A run of the main series, measured 2026-08-18.
MAIN_SERIES_RUN_MINUTES = 7.65

# Per-step slowdown at a wider readout, MEASURED, not assumed. The obvious guess is linear
# in the observables (adjoint differentiates once per observable, so 15 "should" be 3x); it
# is not, because the forward pass is shared and lightning amortises the backward sweep --
# order=2 costs 1.24-1.28x at batch 32/64. Guessing 3x overstated the bill by 2.4x. This is
# a per-step ratio, not a measured full run. An order with no entry falls back to the
# linear guess, pessimistic and therefore safe.
READOUT_SLOWDOWN: dict[int, float] = {1: 1.0, 2: 1.29, 5: 1.73}

# Calibration from the per-step ratio to a full run. Measured at order=2: the per-step
# benchmark said 1.29x while a real run cost 12.23/7.65 = 1.60x, so the step ratio
# understates a run by 1.24x. It misses the evaluation passes and the contention of ten
# workers on twelve cores. Applied to order=5 this turns the 1.73x step ratio into a
# 2.14x run ratio, i.e. ~16.4 min -- against the 38.2 min the bare linear fallback
# (float(order) = 5.00x) would have claimed. Scaling is strongly SUBLINEAR in the number
# of observables (6.2x observables -> 2.1x time): the forward pass is shared and the
# adjoint backward sweep amortises across observables.
STEP_TO_RUN_CALIBRATION = 1.24

# MEASURED wall time of one A_corr run (ds11 lr stage, 2026-08-29): 15 runs, 10 workers on
# 12 cores, mean 12.23 min, max 26.53. Supersedes MAIN_SERIES_RUN_MINUTES x
# READOUT_SLOWDOWN, which gave 9.6 and was 1.27x too low -- the per-step ratio misses the
# evaluation passes and the contention of 10-way parallelism.
MEASURED_RUN_MINUTES: dict[int, float] = {2: 12.23}
MEASURED_RUN_MAX_MINUTES: dict[int, float] = {2: 26.53}

# Orders with a measured PER-STEP ratio but no full run yet: estimate through the
# calibration above rather than through the linear fallback, and say so.
CALIBRATED_RUN_MINUTES: dict[int, float] = {
    order: MAIN_SERIES_RUN_MINUTES * ratio * STEP_TO_RUN_CALIBRATION
    for order, ratio in READOUT_SLOWDOWN.items()
    if order not in MEASURED_RUN_MINUTES
}

# --- the lr-grid extension rule, DECLARED BEFORE THE RUN --------------------------------
#
# The contract grid ends at 3e-2 and the selection can land on that end. The main series
# met exactly this (12 of 24 cells on the upper edge), measured one point above on all 24
# cells of arm A, and its pre-declared rule did not fire: argmax on 0.1 in 1 of 24 cells,
# no cell gaining above the validation noise, mean gain -0.0433. The grid stayed
# (raport.tex, "Sprawdzenie siatki lr"; D-18/D-21).
#
# That verdict does NOT carry over here: 15 outputs instead of 5 changes the scale of the
# head gradient, so an lr chosen at five observables is not the same lr. The probe
# therefore measures the same extra point and applies the SAME rule, restated
# proportionally because it has fewer cells. Declared here, before any run, because
# widening a grid after seeing where the selection landed is a forking path, not a fix.
LR_EXTENSION_POINT = 1e-1
# "at least half the cells" — the main series' 12 of 24.
LR_EXTENSION_MIN_SHARE = 0.5
# Validation noise, the main series' number.
LR_EXTENSION_MIN_GAIN = 0.020
EXTENDED_LR_GRID: tuple[float, ...] = tuple(a7.CONTRACT_LR_GRID) + (LR_EXTENSION_POINT,)


def lr_extension_verdict(lr_rows, *, dataset_seeds, ansatz_levels, seeds, cell_lr) -> dict:
    """Does the contract grid have to be widened by LR_EXTENSION_POINT?

    Both conditions must hold, exactly as in the main series:
      1. the argmax of arm A_corr over grid + {0.1} lands on 0.1 in at least half the
         cells, AND
      2. the median gain of 0.1 over the selected contract lr exceeds the validation
         noise, 0.020.

    Gain is read on arm A_corr and on VALIDATION, per cell as the median over seeds, and
    the rule's number is the median of those per-cell medians. The main series reports
    both a per-cell median and a mean over cells, which leaves the aggregate ambiguous;
    the median of per-cell medians is the reading taken here, stated so it cannot be
    chosen after the fact.
    """
    by_cell: dict[tuple, dict] = defaultdict(dict)
    for row in lr_rows:
        if row["arm"] != "A_corr":
            continue
        key = (row["dataset_seed"], row["ansatz_level"])
        by_cell[key][(float(row["lr"]), a7.optional_int(row["seed"]))] = float(
            row["val_accuracy"]
        )

    full_grid = tuple(a7.CONTRACT_LR_GRID) + (LR_EXTENSION_POINT,)
    cells, on_edge, gains = [], 0, []
    for dataset_seed in dataset_seeds:
        for ansatz_level in ansatz_levels:
            table = by_cell[(dataset_seed, ansatz_level)]
            if any((lr, seed) not in table for lr in full_grid for seed in seeds):
                # The extension point was not measured for this cell: the rule cannot be
                # evaluated, and an unevaluable rule may not silently pass as "no".
                return {"measured": False, "fired": False,
                        "reason": f"lr={LR_EXTENSION_POINT:g} missing for "
                                  f"ds{dataset_seed} {ansatz_level or '-'}"}
            means = {lr: float(np.mean([table[(lr, s)] for s in seeds])) for lr in full_grid}
            argmax = max(full_grid, key=lambda lr: (means[lr], -lr))
            contract_best = float(cell_lr[(dataset_seed, ansatz_level)])
            per_seed_gain = [
                table[(LR_EXTENSION_POINT, s)] - table[(contract_best, s)] for s in seeds
            ]
            gain = float(np.median(per_seed_gain))
            gains.append(gain)
            on_edge += int(argmax == LR_EXTENSION_POINT)
            cells.append({
                "cell": f"ds{dataset_seed}|{ansatz_level or '-'}",
                "contract_best": contract_best,
                "argmax_over_extended_grid": float(argmax),
                "median_gain_of_extension": gain,
            })

    share = on_edge / len(cells) if cells else 0.0
    median_gain = float(np.median(gains)) if gains else 0.0
    fired = share >= LR_EXTENSION_MIN_SHARE and median_gain > LR_EXTENSION_MIN_GAIN
    return {
        "measured": True,
        "rule": (f"argmax of A_corr lands on {LR_EXTENSION_POINT:g} in >= "
                 f"{LR_EXTENSION_MIN_SHARE:.0%} of cells AND median gain > "
                 f"{LR_EXTENSION_MIN_GAIN}"),
        "declared": "before the run, scripts/probe_readout_order.py",
        "cells_on_extension": on_edge,
        "cells_total": len(cells),
        "share": share,
        "median_gain": median_gain,
        "fired": fired,
        "per_cell": cells,
    }


def probe_width(readout_order: int) -> int:
    """Socket output width, hence head input width, hence D_corr's number of features.

    Derived, never written as 15: at another order every one of those three numbers moves
    together, and a literal would silently decouple them.
    """
    return readout_size(DEFAULT_N_QUBITS, order=readout_order)


# --- arms ----------------------------------------------------------------------------


def build_socket_for(arm: str, *, ansatz: str | None, seed: int, width: int | None = None):
    """The socket of one probe arm. This is the ONLY thing that differs between arms.

    A_corr / B_corr are the SAME circuit as A / B -- same ansatz, same R, same theta_init
    key -- read out at a higher Pauli weight. Only requires_grad differs between them.

    D_corr is frozen random Fourier features of the probe width: the control for "is the
    gain the quantum state, or is it the expansion to 15 dimensions". Its width matches
    the quantum readout; its frequency support matches R, as arm D's always has.
    """
    order = _order()
    if arm == "A_corr":
        return make_socket(
            "quantum", R=a7.R_CONTRACT, ansatz=ansatz, trainable=True, seed=seed,
            readout_order=order,
        )
    if arm == "B_corr":
        return make_socket(
            "quantum", R=a7.R_CONTRACT, ansatz=ansatz, trainable=False, seed=seed,
            readout_order=order,
        )
    if arm == "D_corr":
        if width is not None and int(width) != probe_width(order):
            raise ValueError(
                f"D_corr width {width} does not match the quantum readout width "
                f"{probe_width(order)} at order {order}; the classical control exists to "
                "match that width, so a mismatch is a planning error, not a variant"
            )
        # readout_order is not passed: the classical control matches the quantum readout
        # by WIDTH, not by Pauli weight, and Socket refuses a Pauli weight it cannot have.
        return make_socket(
            "random", R=a7.R_CONTRACT, ansatz=None, trainable=False, seed=seed,
            rff_width=probe_width(order),
        )
    raise ValueError(f"unknown probe arm {arm!r}; expected one of {ARMS}")


def head_for(arm: str, dilution: str, *, seed: int, width: int | None = None):
    """Linear readout on probe_width inputs, identical for all three arms.

    Same key for all three -- derive(seed, "linear_readout", width) carries no arm -- so
    A_corr, B_corr and D_corr start from the same head, the same invariant that makes
    theta_init identical across A and B.
    """
    if dilution != PROBE_DILUTION:
        raise ValueError(
            f"the probe runs at dilution {PROBE_DILUTION!r} only, got {dilution!r}: the "
            "MLP heads hardcode SOCKET_WIDTH = 5 and are not part of this scope"
        )
    return make_linear_readout(probe_width(_order()), seed=seed)


def head_params(arm: str, dilution: str, *, width: int | None = None) -> int:
    return probe_width(_order()) + 1


def socket_params_nominal(arm: str, *, width: int | None = None) -> int:
    if arm in ("A_corr", "B_corr"):
        return socket_param_count(DEFAULT_N_QUBITS, a7.R_CONTRACT)
    if arm == "D_corr":
        # Omega (width x n_inputs) + b (width), the convention arm D already uses.
        return int(probe_width(_order()) * (DEFAULT_N_QUBITS + 1))
    raise ValueError(f"unknown probe arm {arm!r}")


def socket_init_seed(arm: str, *, ansatz: str | None, seed: int, width: int | None = None) -> str:
    """The key of the socket draw, recorded so a row can be replayed.

    A_corr and B_corr use derive(seed, ansatz, R) -- byte for byte the key of A and B, so
    all four arms start from ONE theta and the probe is anchored to the main series
    rather than being a second, unrelated experiment.
    """
    if arm in ("A_corr", "B_corr"):
        value = derive(seed, ansatz, a7.R_CONTRACT)
    elif arm == "D_corr":
        value = derive(seed, "RFF", a7.R_CONTRACT, probe_width(_order()))
    else:
        raise ValueError(f"unknown probe arm {arm!r}")
    return f"0x{value:016x}"


# --- pairing invariants, asserted per run --------------------------------------------


def assert_theta_pairing_across_readouts(seed: int, ansatz: str) -> None:
    """A, B, A_corr and B_corr all start from a bit-for-bit identical theta.

    Checked for THIS seed in THIS run rather than cited from the seeding contract. If it
    ever stopped holding, Delta_{A'B'} would stop being comparable with Delta_AB and
    nothing else in the pipeline would notice.
    """
    reference = a7.build_socket_for("A", ansatz=ansatz, seed=seed).theta_init
    for arm in ("B",):
        other = a7.build_socket_for(arm, ansatz=ansatz, seed=seed).theta_init
        assert torch.equal(reference, other), f"theta_init(A) != theta_init({arm}) at seed {seed}"
    for arm in ("A_corr", "B_corr"):
        other = build_socket_for(arm, ansatz=ansatz, seed=seed).theta_init
        assert torch.equal(reference, other), (
            f"theta_init(A) != theta_init({arm}) at seed {seed}, ansatz {ansatz}: the "
            "probe is no longer anchored to the main series"
        )


def assert_head_pairing(seed: int) -> None:
    """A_corr, B_corr and D_corr start from an identical head."""
    reference = head_for("A_corr", PROBE_DILUTION, seed=seed).state_dict()
    for arm in ("B_corr", "D_corr"):
        other = head_for(arm, PROBE_DILUTION, seed=seed).state_dict()
        for key, value in reference.items():
            assert torch.equal(value, other[key]), f"head({arm}) differs from head(A_corr) at {key}"


def assert_readout_prefix(seed: int, ansatz: str) -> None:
    """The first five columns of the probe readout ARE the main-series readout.

    The strongest of the cheap checks: it fails on a wrong observable order, a wrong
    endianness, and on a circuit that is not the circuit of the main series. Compared at
    the socket, in the dtype the training path actually uses.
    """
    x = torch.as_tensor(
        np.random.default_rng(seed).uniform(*FEATURE_RANGE, size=(8, DEFAULT_N_QUBITS)),
        dtype=torch.float32,
    )
    with torch.no_grad():
        narrow = a7.build_socket_for("B", ansatz=ansatz, seed=seed)(x)
        wide = build_socket_for("B_corr", ansatz=ansatz, seed=seed)(x)
    assert wide.shape[1] == probe_width(_order()), (
        f"probe socket returned {wide.shape[1]} columns, expected "
        f"{probe_width(_order())}"
    )
    assert torch.equal(narrow, wide[:, : narrow.shape[1]]), (
        "the first columns of the widened readout are not the main-series readout; the "
        "observable order or the endianness of one backend has drifted"
    )


# --- results row ---------------------------------------------------------------------


def _probe_row(*, arm: str, dilution: str, width: int | None, **kwargs) -> dict:
    """One probe row: a7._cell_row for the generic provenance, overridden where it is
    arm-specific, plus readout_order.

    Built through the twin rather than re-implemented, so the ~40 generic columns have one
    definition in the project. a7._cell_row dispatches on the arm name in six places, all
    handled here: R (twin is never "E", so R_CONTRACT), socket_params_nominal and
    head_params (both overridden below), init_seed and rff_omega_seed (the twin's keys are
    exactly the probe's), init_spec_id (correct as is).

    An arm whose twin does not carry the right provenance means revisiting this list, not
    adding a branch elsewhere.
    """
    twin = TWIN[arm]
    # The twin of D_corr is D_best, whose provenance is computed FROM the width, so the
    # width has to reach it. The quantum arms have no width and must not acquire one.
    twin_width = probe_width(_order()) if arm == "D_corr" else None
    if arm == "D_corr" and width is not None and int(width) != twin_width:
        raise ValueError(f"D_corr row width {width} != readout width {twin_width}")
    row = a7._cell_row(arm=twin, dilution=dilution, width=twin_width, **kwargs)
    row["arm"] = arm
    row["socket_params_nominal"] = socket_params_nominal(arm, width=width)
    row["head_params"] = head_params(arm, dilution, width=width)
    row["readout_order"] = _order()
    # rff_width belongs to arm D only; for the quantum arms the readout width is carried
    # by readout_order, and putting 15 in both columns would double-count it.
    row["rff_width"] = "" if arm != "D_corr" else int(probe_width(_order()))
    assert set(row) == set(PROBE_RESULT_COLUMNS), (
        f"probe row schema drifted: unexpected {sorted(set(row) - set(PROBE_RESULT_COLUMNS))}, "
        f"missing {sorted(set(PROBE_RESULT_COLUMNS) - set(row))}"
    )
    return row


# --- execution -----------------------------------------------------------------------

# Set once per process by _worker_init / the serial path. A module-level value rather than
# a threaded argument because the arm factories are handed to a7.run_cell by reference and
# their signature is fixed by that contract.
#
# None until set, and read through _order(): a module-level default of 2 would turn
# "somebody imported build_socket_for and forgot to set the order" into a silently wrong
# readout, which is the one class of bug this whole probe exists to rule out.
_ORDER_IN_PROCESS: int | None = None


def _order() -> int:
    if _ORDER_IN_PROCESS is None:
        raise RuntimeError(
            "the readout order of this process was never set; call _worker_init(order) "
            "(execute() and main() do) before building a probe socket or head"
        )
    return _ORDER_IN_PROCESS


def _worker_init(order: int) -> None:
    global _ORDER_IN_PROCESS
    _ORDER_IN_PROCESS = int(order)
    # One torch thread per worker: lightning.qubit at 5 qubits does not scale with
    # threads, so multi-threaded workers would only oversubscribe the machine.
    torch.set_num_threads(1)


def _worker_run(task: dict) -> dict:
    """Execute one task: one expensive run, or one cache group of cheap runs."""
    _worker_init(task["readout_order"])
    dataset, splits, manifest = a7._worker_splits(task["dataset_seed"])
    arm, seed, width = task["arm"], task["seed"], task["width"]

    cached = None
    cache_seconds = 0.0
    if arm in FROZEN_ARMS and len(task["cells"]) > 1:
        socket = build_socket_for(arm, ansatz=task["ansatz_level"], seed=seed, width=width)
        started = time.perf_counter()
        cached = {
            split: a7.frozen_socket_features(socket, splits[split][0])
            for split in ("train", "val", "test")
        }
        cache_seconds = time.perf_counter() - started

    rows, predictions, weights = [], [], []
    for cell in task["cells"]:
        cell_rows, correctness, trained_state = a7.run_cell(
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
            socket_factory=build_socket_for,
            head_factory=head_for,
            row_builder=_probe_row,
            trained_arms=TRAINED_ARMS,
        )
        rows.extend(cell_rows)
        identity = {
            "dataset_seed": task["dataset_seed"],
            "arm": arm,
            "ansatz_level": task["ansatz_level"],
            "dilution": cell["dilution"],
            "seed": seed,
            "width": width,
            "lr": cell["lr"],
        }
        predictions.append({**identity, "correct": np.asarray(correctness, dtype=bool)})

        # The trained parameters, saved for the same reasons the main series saves them
        # (a7.write_weights): A_corr's final theta is otherwise recoverable only by
        # re-running the training in the same environment, and B_corr's HEAD is not
        # recoverable at all -- its socket theta follows from the seed, but the head was
        # trained on the cached features and for acc(A') - acc(B') it matters as much.
        # run_supplementary.displacement_rows reads these to compute K(tau), how many
        # gates moved by more than tau, which is the diagnostic raport.tex names in
        # Ograniczenia as the thing theta_displacement is NOT. Under 2 kB per run.
        #
        # Arrays, not live modules: a torch Module does not belong in a multiprocessing
        # payload.
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


def execute(tasks: list[dict], *, workers: int, order: int, on_result) -> None:
    """Run tasks and hand every result to on_result as it arrives.

    Rows reach disk the moment they exist, so a failure late in the grid cannot cost the
    runs that already finished.
    """
    if not tasks:
        return
    if workers <= 1:
        _worker_init(order)
        for task in tasks:
            on_result(_worker_run(task))
        return
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers, initializer=_worker_init, initargs=(order,)
    ) as pool:
        for result in pool.imap_unordered(_worker_run, tasks):
            on_result(result)


# --- resume --------------------------------------------------------------------------


def row_key(row: dict) -> tuple:
    """Identity of a probe row. readout_order is IN the key.

    Without it, the same (dataset, arm, dilution, seed, split, lr) at two readout orders
    is one key, and a resume of an order=5 run would skip every cell an order=2 run had
    already written. This is the failure of decision D-32 in a new costume.
    """
    return a7._key(
        (
            row["dataset"], row["arm"], row["ansatz_level"], row["dilution"],
            a7.optional_int(row["seed"]), row["split"], a7.optional_int(row["rff_width"]),
            "" if row["lr_selected"] == "" else f"{float(row['lr_selected']):g}",
            a7.optional_int(row["readout_order"]),
        )
    )


def existing_row_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    if frame.columns.tolist() != list(PROBE_RESULT_COLUMNS):
        raise ValueError(
            f"{path} has a different column schema than PROBE_RESULT_COLUMNS; a resumed "
            "run would mix two schemas in one file. Move the old file aside deliberately."
        )
    return {row_key(row) for row in frame.to_dict("records")}


def existing_lr_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    return {
        a7._key((r["dataset"], r["arm"], r["ansatz_level"], r["dilution"],
                 a7.optional_int(r["seed"]), a7.optional_int(r["rff_width"]),
                 f"{float(r['lr']):g}", a7.optional_int(r["readout_order"])))
        for r in frame.to_dict("records")
    }


# --- planning ------------------------------------------------------------------------


def make_task(*, stage, dataset_seed, arm, ansatz_level, seed, width, cells, lr_grid,
              effective_rank, g1_margin, run_id, commit, environment, readout_order) -> dict:
    task = a7.make_task(
        stage=stage, dataset_seed=dataset_seed, arm=arm, ansatz_level=ansatz_level,
        seed=seed, width=width, cells=cells, lr_grid=lr_grid,
        effective_rank=effective_rank, g1_margin=g1_margin, run_id=run_id,
        commit=commit, environment=environment,
    )
    task["readout_order"] = int(readout_order)
    return task


def ansatz_of(arm: str, ansatz_level: str) -> str:
    """D_corr has no ansatz dimension: its frequency support depends on R only."""
    return "" if arm in ANSATZ_FREE_ARMS else ansatz_level


def plan_lr_stage(*, dataset_seeds, ansatz_levels, seeds, ranks, context, done, order):
    """Stage 1: the lr selection, and nothing else.

    A_corr is one task per (cell, lr, seed) -- the expensive units to spread over
    workers. B_corr and D_corr are one task per cache group, so their features are
    computed once for every lr in the group.
    """
    tasks: list[dict] = []

    def add(arm, dataset_seed, ansatz_level, seed, width, cells, grid):
        recorded = ansatz_of(arm, ansatz_level)
        cells = [
            c for c in cells
            if a7._key((
                a7.dataset_location(dataset_seed)[0], arm, recorded, c["dilution"], seed,
                "" if width is None else width, f"{float(c['lr']):g}", order,
            )) not in done
        ]
        if not cells:
            return
        tasks.append(make_task(
            stage="lr", dataset_seed=dataset_seed, arm=arm, ansatz_level=recorded,
            seed=seed, width=width, cells=cells, lr_grid=grid,
            effective_rank=ranks.get(recorded), g1_margin=None, readout_order=order,
            **context,
        ))

    for dataset_seed in dataset_seeds:
        for seed in seeds:
            for ansatz_level in ansatz_levels:
                # The contract grid PLUS the extension point. The extra point is measured
                # unconditionally so the pre-declared rule can be evaluated at all; it
                # does not enter the selection unless the rule fires.
                for lr in EXTENDED_LR_GRID:
                    add("A_corr", dataset_seed, ansatz_level, seed, None,
                        [{"dilution": PROBE_DILUTION, "lr": lr}], EXTENDED_LR_GRID)
                add("B_corr", dataset_seed, ansatz_level, seed, None,
                    [{"dilution": PROBE_DILUTION, "lr": lr} for lr in EXTENDED_LR_GRID],
                    EXTENDED_LR_GRID)
            # D_corr carries its width, as D_best does: it belongs in the task, not in
            # the row builder, because it also enters the resume key, the prediction
            # filename and the cache group.
            add("D_corr", dataset_seed, "", seed, probe_width(order),
                [{"dilution": PROBE_DILUTION, "lr": lr} for lr in a7.CONTRACT_LR_GRID],
                a7.CONTRACT_LR_GRID)
    return tasks


def select_all_lrs(lr_rows: list[dict], *, dataset_seeds, ansatz_levels, seeds,
                   grid=None) -> dict:
    """Two declared selection families, mirroring the main series:

        (dataset x ansatz)   arms A_corr and B_corr jointly, contract grid
        (dataset)            arm D_corr alone, contract grid
    """
    measured: dict[tuple, dict] = defaultdict(dict)
    for row in lr_rows:
        key = (row["dataset_seed"], row["arm"], row["ansatz_level"])
        measured[key][(float(row["lr"]), row["arm"], a7.optional_int(row["seed"]))] = float(
            row["val_accuracy"]
        )

    def merge(keys):
        joined: dict = {}
        for key in keys:
            joined.update(measured[key])
        return joined

    grid = tuple(a7.CONTRACT_LR_GRID) if grid is None else tuple(grid)
    cell_lr, cell_selection = {}, {}
    for dataset_seed in dataset_seeds:
        for ansatz_level in ansatz_levels:
            selection = lr_selection_from_measurements(
                merge([(dataset_seed, "A_corr", ansatz_level),
                       (dataset_seed, "B_corr", ansatz_level)]),
                grid=grid, seeds=seeds,
                arms=LR_SELECTION_ARMS, selection_arms=LR_SELECTION_ARMS,
            )
            cell_lr[(dataset_seed, ansatz_level)] = selection.best
            cell_selection[(dataset_seed, ansatz_level)] = selection

    d_corr_lr, d_corr_selection = {}, {}
    for dataset_seed in dataset_seeds:
        selection = lr_selection_from_measurements(
            merge([(dataset_seed, "D_corr", "")]),
            grid=a7.CONTRACT_LR_GRID, seeds=seeds,
            arms=("D_corr",), selection_arms=("D_corr",),
        )
        d_corr_lr[dataset_seed] = selection.best
        d_corr_selection[dataset_seed] = selection

    return {
        "cell_lr": cell_lr, "cell_selection": cell_selection,
        "d_corr_lr": d_corr_lr, "d_corr_selection": d_corr_selection,
    }


def plan_main_stage(*, dataset_seeds, ansatz_levels, seeds, lrs, ranks, context, done, order):
    """Stage 3: the probe grid.

    A_corr and B_corr run at the cell lr of their own (dataset x ansatz); a paired
    difference taken at two different lr values is not one paired difference. D_corr runs
    at its own lr, and having no ansatz dimension it runs once per distinct cell lr --
    once when L1 and L2 agree, twice otherwise, which is why lr is part of its key.
    """
    tasks: list[dict] = []
    cell_lr, d_corr_lr = lrs["cell_lr"], lrs["d_corr_lr"]

    def add(arm, dataset_seed, ansatz_level, seed, width, cells, grid):
        recorded = ansatz_of(arm, ansatz_level)
        cells = [
            c for c in cells
            if any(
                a7._key((
                    a7.dataset_location(dataset_seed)[0], arm, recorded, c["dilution"],
                    seed, split, "" if width is None else width,
                    f"{float(c['lr']):g}", order,
                )) not in done
                for split in a7.SPLITS_REPORTED
            )
        ]
        if not cells:
            return
        tasks.append(make_task(
            stage="main", dataset_seed=dataset_seed, arm=arm, ansatz_level=recorded,
            seed=seed, width=width, cells=cells, lr_grid=grid,
            effective_rank=ranks.get(recorded), g1_margin=None, readout_order=order,
            **context,
        ))

    for dataset_seed in dataset_seeds:
        for seed in seeds:
            for ansatz_level in ansatz_levels:
                lr = cell_lr[(dataset_seed, ansatz_level)]
                for arm in ("A_corr", "B_corr"):
                    add(arm, dataset_seed, ansatz_level, seed, None,
                        [{"dilution": PROBE_DILUTION, "lr": lr}], a7.CONTRACT_LR_GRID)
            add("D_corr", dataset_seed, "", seed, probe_width(order),
                [{"dilution": PROBE_DILUTION, "lr": d_corr_lr[dataset_seed]}],
                a7.CONTRACT_LR_GRID)
    return tasks


# --- writing -------------------------------------------------------------------------


def lr_table_rows(result: dict) -> list[dict]:
    rows = a7.lr_table_rows(result)
    for row in rows:
        row["in_selection"] = row["arm"] in LR_SELECTION_ARMS
        row["readout_order"] = int(result["task"]["readout_order"])
    return rows


def selection_to_json(selection) -> dict:
    """An LrSelection as plain JSON: the winner AND the whole table it won on.

    Written to disk rather than printed, because "does the selected lr sit on the EDGE of
    the contract grid" (D-18/D-21) is a decision a human takes between the two phases, and
    a decision input that exists only in a job log is a decision input that can be lost.
    Keeping the per-(lr, arm, seed) numbers also means the selection can be recomputed
    over a different set of arms later without repeating a single run.
    """
    return {
        "best": float(selection.best),
        "grid": [float(v) for v in selection.grid],
        "seeds": [int(s) for s in selection.seeds],
        "arms": list(selection.arms),
        "selection_arms": list(selection.selection_arms),
        "on_grid_edge": float(selection.best) in (
            float(selection.grid[0]), float(selection.grid[-1])
        ),
        "mean_by_lr": {f"{float(k):g}": float(v) for k, v in selection.mean_by_lr.items()},
        "by_lr_arm": {
            f"{float(lr):g}|{arm}": float(v) for (lr, arm), v in selection.by_lr_arm.items()
        },
        "by_lr_arm_seed": {
            f"{float(lr):g}|{arm}|{seed}": float(v)
            for (lr, arm, seed), v in selection.by_lr_arm_seed.items()
        },
    }


def append_probe_rows(path: Path, rows: list[dict]) -> None:
    a7.append_rows(path, PROBE_RESULT_COLUMNS, rows)


def write_weights_checked(out_dir: Path, identity: dict, *, arrays: dict) -> Path:
    """Save one run's trained parameters; if the file exists, VERIFY instead of clobbering.

    The lr stage and the grid measure the same cell at the same lr whenever the grid runs
    at a value the selection already measured, and the weight filename is keyed by
    (arm, ansatz, dilution, seed, lr, M) -- so the second write targets the first one's
    file. The two runs are the same run: same seed, same lr, same data, same batch order,
    so the trajectory is identical and the overwrite is a no-op.

    Rather than rely on that, this checks it. Every repeated cell becomes a free
    per-run determinism witness, and if the two stages ever DID diverge -- someone changes
    how lr reaches a cell, say -- one silently replacing the other is exactly the failure
    that would otherwise never surface.
    """
    path = a7.weights_path(out_dir, identity)
    if path.exists():
        existing = a7.read_weights(path)
        stored = {k: v for k, v in existing.items() if k != "meta"}
        if set(stored) != set(arrays):
            raise ValueError(
                f"{path} already holds arrays {sorted(stored)}, this run produced "
                f"{sorted(arrays)}; two different runs are claiming one filename"
            )
        for key, value in arrays.items():
            if not np.array_equal(np.asarray(value, dtype=np.float64), stored[key]):
                raise ValueError(
                    f"{path} already exists and its {key!r} differs from this run's. The "
                    "lr stage and the grid measure the same cell at the same lr and must "
                    "produce an identical trajectory; they did not, so one of them is "
                    "training something else. Nothing was overwritten."
                )
        return path
    return a7.write_weights(out_dir, identity, arrays=arrays)


# --- main ----------------------------------------------------------------------------


def measure_ranks(ansatz_levels, *, order: int) -> dict:
    """Numerical rank of d(outputs)/d(theta) per ansatz, at THIS readout order.

    Recorded in socket_params_effective of every row. Widening the readout adds rows to
    the Jacobian and never columns, so if this equals the main-series rank the probe is a
    pure change of readout -- which is the sentence the report needs, and it has to come
    from a measurement.
    """
    ranks = {}
    for level in ansatz_levels:
        ranks[level] = effective_dimension(
            lambda n, R, _level=level: build_socket_circuit(_level, n, R),
            a7.R_CONTRACT,
            readout_order=order,
        )
    ranks[""] = None  # D_corr has no theta
    return ranks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan every stage and print the cost; run nothing")
    parser.add_argument("--stop-after", choices=("lr", "main"), default=None,
                        help="'lr' stops AFTER the lr selection and before the grid — the "
                             "phase boundary where a human decides whether the selected "
                             "lr sits on the edge of the contract grid (D-18/D-21). "
                             "'main' is the last stage, so it means 'run everything' and "
                             "is the same as omitting the flag; it does NOT stop before "
                             "the grid")
    parser.add_argument("--readout-order", type=int, default=DEFAULT_READOUT_ORDER,
                        help="Pauli weight of the readout. 2 is this run's scope; higher "
                             "orders work but are not part of it")
    parser.add_argument("--ansatz", action="append", choices=("L1", "L2"), default=None,
                        help="repeatable; default L1 only")
    parser.add_argument("--dataset-seeds", type=int, nargs="+", default=list(a7.DATASET_SEEDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(a7.SEEDS),
                        help="seeds of the MAIN grid (default 1-10)")
    parser.add_argument("--lr-seeds", type=int, nargs="+", default=list(a7.LR_SELECTION_SEEDS),
                        help="seeds the lr selection is measured on (default 1-3, the "
                             "main series' own LR_SELECTION_SEEDS). Selecting on all ten "
                             "would cost 4x and make the choice incomparable with the "
                             "main series, which is the only reason the probe sits next "
                             "to Delta_AB at all")
    parser.add_argument("--no-generate", action="store_true",
                        help="refuse to generate a missing dataset instead of building it")
    args = parser.parse_args(argv)

    order = int(args.readout_order)
    ansatz_levels = tuple(args.ansatz or ("L1",))
    dataset_seeds = tuple(args.dataset_seeds)
    seeds = tuple(args.seeds)
    lr_seeds = tuple(args.lr_seeds)
    unknown = sorted(set(lr_seeds) - set(seeds))
    if unknown:
        parser.error(f"--lr-seeds {unknown} are not in --seeds; the lr is selected on a "
                     "subset of the grid, never on cells the grid does not contain")
    workers = a7.default_workers() if args.workers is None else args.workers
    out_dir = args.out_dir
    results_path = out_dir / "probe_results.csv"
    lr_path = out_dir / "probe_lr_table.csv"
    lr_results_path = out_dir / "probe_lr_results.csv"
    manifest_path = out_dir / "probe_manifest.json"
    selection_path = out_dir / "probe_lr_selection.json"

    _worker_init(order)

    print(f"probe: readout_order={order}, width={probe_width(order)}, "
          f"head_params={probe_width(order) + 1}")
    print(f"arms={list(ARMS)} ansatz={list(ansatz_levels)} dilution={PROBE_DILUTION!r} "
          f"datasets={list(dataset_seeds)} seeds={len(seeds)} "
          f"(lr selected on {list(lr_seeds)})")

    for dataset_seed in dataset_seeds:
        a7.ensure_dataset(dataset_seed, allow_generate=not args.no_generate)

    # Invariants, before anything expensive runs.
    for ansatz_level in ansatz_levels:
        for seed in seeds:
            assert_theta_pairing_across_readouts(seed, ansatz_level)
        assert_readout_prefix(seeds[0], ansatz_level)
    for seed in seeds:
        assert_head_pairing(seed)
    print(f"invariants OK: theta shared by A/B/A_corr/B_corr on {len(seeds)} seeds; "
          f"head shared by all three probe arms; first {DEFAULT_N_QUBITS} readout columns "
          f"identical to the main series")

    ranks = measure_ranks(ansatz_levels, order=order)
    print("jacobian rank per ansatz at this order: "
          + ", ".join(f"{k or 'D_corr'}={v}" for k, v in ranks.items()))

    context = {
        "run_id": f"{PROBE_ID}-{uuid.uuid4().hex[:8]}",
        "commit": a7.git_commit(),
        "environment": a7.env_hash(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "what": "correlator-readout probe (exploratory, outside the confirmatory family)",
            "probe_id": PROBE_ID,
            "readout_order": order,
            "readout_width": probe_width(order),
            "head_params": probe_width(order) + 1,
            "arms": list(ARMS),
            "trained_arms": list(TRAINED_ARMS),
            "dilution": PROBE_DILUTION,
            "ansatz_levels": list(ansatz_levels),
            "dataset_seeds": list(dataset_seeds),
            "seeds": list(seeds),
            "lr_selection_seeds": list(lr_seeds),
            "lr_selection_arms": list(LR_SELECTION_ARMS),
            "lr_grid": [float(v) for v in a7.CONTRACT_LR_GRID],
            "R": a7.R_CONTRACT,
            "jacobian_rank_per_ansatz": {k or "D_corr": v for k, v in ranks.items()},
            "batch_size": a7.BATCH_SIZE,
            "max_epochs": a7.MAX_EPOCHS,
            "patience": a7.PATIENCE,
            "backend": a7.DEFAULT_BACKEND,
            "run_id": context["run_id"],
            "git_commit": context["commit"],
            "env_hash": context["environment"],
            "timestamp_utc": a7.utc_now(),
        },
        manifest_path.open("w"),
        indent=2,
    )
    print(f"wrote {manifest_path}")

    lr_done = existing_lr_keys(lr_path)
    lr_tasks = plan_lr_stage(
        dataset_seeds=dataset_seeds, ansatz_levels=ansatz_levels, seeds=lr_seeds,
        ranks=ranks, context=context, done=lr_done, order=order,
    )

    if args.dry_run:
        expensive = sum(len(t["cells"]) for t in lr_tasks if t["arm"] in TRAINED_ARMS)
        cheap = sum(len(t["cells"]) for t in lr_tasks if t["arm"] in FROZEN_ARMS)
        main_cells_expensive = len(dataset_seeds) * len(seeds) * len(ansatz_levels)
        main_cells_cheap = main_cells_expensive + len(dataset_seeds) * len(seeds)
        print(f"\nstage 1 (lr): {len(lr_tasks)} tasks, "
              f"{expensive} trained cells + {cheap} frozen cells")
        print(f"stage 3 (main): {main_cells_expensive} trained cells + "
              f"{main_cells_cheap} frozen cells")
        measured = MEASURED_RUN_MINUTES.get(order)
        calibrated = CALIBRATED_RUN_MINUTES.get(order)
        per_run_minutes = measured or calibrated or (
            MAIN_SERIES_RUN_MINUTES * READOUT_SLOWDOWN.get(order, float(order))
        )
        total = (expensive + main_cells_expensive) * per_run_minutes
        if measured:
            print(f"\ntrained-run cost: {per_run_minutes:.1f} min each, MEASURED on the "
                  f"ds11 lr stage (15 runs, 10 workers on 12 cores; median 10.9, "
                  f"max {MEASURED_RUN_MAX_MINUTES[order]:.1f})")
        elif calibrated:
            print(f"\ntrained-run CALIBRATED ESTIMATE: {per_run_minutes:.1f} min each "
                  f"= {MAIN_SERIES_RUN_MINUTES} min x {READOUT_SLOWDOWN[order]:.2f} "
                  f"(measured per-step at this order) x {STEP_TO_RUN_CALIBRATION} "
                  f"(step->run, calibrated on order=2). No full run measured at this order.")
        else:
            print(f"\ntrained-run ESTIMATE: {per_run_minutes:.1f} min each "
                  f"= {MAIN_SERIES_RUN_MINUTES} min (main series at 5 observables) x "
                  f"{READOUT_SLOWDOWN.get(order, float(order)):.2f}. Not measured at this "
                  "order; the order=2 estimate built this way was 1.27x too low.")
        print(f"wall-clock estimate at {workers} workers: {total / 60 / workers:.1f} h "
              f"({total / 60:.1f} core-hours); frozen arms are cache-cheap and excluded")
        print(f"\nresume state: {len(lr_done)} lr rows and "
              f"{len(existing_row_keys(results_path))} result rows already on disk")
        print("nothing was run")
        return 0

    def on_lr_result(result: dict) -> None:
        rows = lr_table_rows(result)
        # Full 47-column rows of the SELECTION runs plus their correctness vectors. Not
        # needed to pick an lr and thrown away by the main series, kept here because they
        # are already computed. Their own file: probe_results.csv is the estimand, and
        # mixing in runs at four different lr values would sit next to the thirty that
        # carry the measurement.
        #
        # probe_lr_results.csv is INFORMATIONAL and may hold duplicates: a kill between it
        # and the lr table leaves rows the resume key cannot see, so the cell re-runs and
        # appends again -- identical bar the timestamp. The authoritative artefacts do not
        # have this property: the lr table is the commit marker and weights verify.
        for prediction in result["predictions"]:
            a7.write_prediction(out_dir / LR_STAGE_DIR, prediction)
        # ARTEFACTS FIRST, ROW LAST. The row is what resume reads, so it is the commit
        # marker: a job killed between the two must lose the cell and re-run it, never
        # keep a row whose weights were never written. The other order is how the main
        # series ended up with 310 prediction files and an empty weights directory.
        #
        # Weights from the lr stage too, not only from the grid, at under 2 kB each; a
        # cell measured at four lr values gives four files, which the filename already
        # distinguishes by lr.
        for record in result["weights"]:
            write_weights_checked(out_dir, record["identity"], arrays=record["arrays"])
        append_probe_rows(lr_results_path, result["rows"])
        # The lr table LAST: it is what existing_lr_keys reads, so it is this stage's
        # commit marker.
        a7.append_rows(lr_path, LR_TABLE_COLUMNS, rows)
        print(f"  lr {result['task']['arm']:<7} seed {result['task']['seed']:>2} "
              f"ds{result['task']['dataset_seed']} {len(rows)} rows, "
              f"{len(result['weights'])} weight files")

    print(f"\nstage 1 (lr): {len(lr_tasks)} tasks on {workers} workers")
    execute(lr_tasks, workers=workers, order=order, on_result=on_lr_result)

    # Read the whole table back: a resumed run must select on every measurement, not only
    # on the ones this process produced.
    import pandas as pd

    lr_frame = pd.read_csv(lr_path, keep_default_na=False, float_precision="round_trip")
    lr_rows = [
        r for r in lr_frame.to_dict("records") if a7.optional_int(r["readout_order"]) == order
    ]
    lrs = select_all_lrs(
        lr_rows, dataset_seeds=dataset_seeds, ansatz_levels=ansatz_levels, seeds=lr_seeds
    )

    # The pre-declared extension rule, applied to the contract-grid selection. Only if it
    # fires does the selection get redone over the wider grid; otherwise the extension
    # point stays on disk as a measurement and changes nothing.
    verdict = lr_extension_verdict(
        lr_rows, dataset_seeds=dataset_seeds, ansatz_levels=ansatz_levels,
        seeds=lr_seeds, cell_lr=lrs["cell_lr"],
    )
    if verdict["measured"]:
        print(f"\n  lr-grid extension rule ({verdict['rule']}):")
        print(f"    argmax on {LR_EXTENSION_POINT:g} in {verdict['cells_on_extension']} of "
              f"{verdict['cells_total']} cells (share {verdict['share']:.2f}), "
              f"median gain {verdict['median_gain']:+.4f}")
        print(f"    -> {'FIRED: the grid is extended' if verdict['fired'] else 'did not fire: the contract grid stands'}")
    else:
        print(f"\n  lr-grid extension rule NOT EVALUATED: {verdict['reason']}")

    if verdict["fired"]:
        lrs = select_all_lrs(
            lr_rows, dataset_seeds=dataset_seeds, ansatz_levels=ansatz_levels,
            seeds=lr_seeds, grid=EXTENDED_LR_GRID,
        )
    selection_json = {
        "readout_order": order,
        "run_id": context["run_id"],
        "cell": {
            f"ds{ds}|{ansatz}": selection_to_json(sel)
            for (ds, ansatz), sel in lrs["cell_selection"].items()
        },
        "d_corr": {f"ds{ds}": selection_to_json(sel) for ds, sel in lrs["d_corr_selection"].items()},
        "extension_rule": verdict,
    }
    json.dump(selection_json, selection_path.open("w"), indent=2)
    print(f"wrote {selection_path}")

    edge = []
    for key, value in sorted(lrs["cell_lr"].items(), key=lambda kv: str(kv[0])):
        on_edge = selection_json["cell"][f"ds{key[0]}|{key[1]}"]["on_grid_edge"]
        edge.append(on_edge)
        print(f"  selected lr {value:g} for ds{key[0]} {key[1]} (A_corr+B_corr)"
              + ("   <-- ON THE EDGE OF THE GRID" if on_edge else ""))
    for key, value in sorted(lrs["d_corr_lr"].items()):
        on_edge = selection_json["d_corr"][f"ds{key}"]["on_grid_edge"]
        edge.append(on_edge)
        print(f"  selected lr {value:g} for ds{key} D_corr"
              + ("   <-- ON THE EDGE OF THE GRID" if on_edge else ""))
    if any(edge):
        print(f"\n  {sum(edge)} of {len(edge)} selections sit on an END of the contract "
              "grid. Whether that is acceptable is decision D-18/D-21 and is NOT this "
              "script's to take: read probe_lr_selection.json before the grid.")

    if args.stop_after == "lr":
        print("\n--stop-after lr: stopping before the main grid. Read the selected lr "
              "above; if it sits on the edge of the contract grid that is a decision "
              "(D-18/D-21), not something this script may take.")
        return 0

    done = existing_row_keys(results_path)
    main_tasks = plan_main_stage(
        dataset_seeds=dataset_seeds, ansatz_levels=ansatz_levels, seeds=seeds,
        lrs=lrs, ranks=ranks, context=context, done=done, order=order,
    )

    def on_main_result(result: dict) -> None:
        # ARTEFACTS FIRST, ROW LAST -- see on_lr_result. The results row is the resume
        # key, so writing it before the weights would let a walltime kill in between
        # leave a cell that resume considers done and whose parameters do not exist.
        for prediction in result["predictions"]:
            a7.write_prediction(out_dir, prediction)
        for record in result["weights"]:
            write_weights_checked(out_dir, record["identity"], arrays=record["arrays"])
        append_probe_rows(results_path, result["rows"])
        test_rows = [r for r in result["rows"] if r["split"] == "test"]
        for row in test_rows:
            print(f"  {row['arm']:<7} ds{result['task']['dataset_seed']} "
                  f"seed {row['seed']:>2} acc {float(row['accuracy']):.4f}")

    print(f"\nstage 3 (main): {len(main_tasks)} tasks on {workers} workers")
    execute(main_tasks, workers=workers, order=order, on_result=on_main_result)

    print(f"\nwrote {results_path}")
    print(f"wrote {lr_path}")
    print("\nAnalysis is NOT part of this script: Delta_{A'B'} and Delta_{B'D_corr} are")
    print("exploratory estimands and belong in run_a8_analysis.py, reading this CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
