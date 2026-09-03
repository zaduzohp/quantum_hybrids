"""Pilot sigma: does Delta_AB exist at all, and what is sigma_Delta(A-B)?

Arm A has not been trained to completion anywhere else in this project — every earlier
measurement ran arm E, and the gradient-SNR probe skips the optimiser step — so the main
estimand had no measurements while the power analysis and the schedule were built around
it. This script is that first measurement, and it settles sigma_Delta(A-B), on which the
feasibility of delta = 0.02 depends.

Configuration: L1, R = 2, linear head, frozen production dataset. Linear because the MLP
heads are saturated by the head alone, so it is the only dilution where Delta_AB can be
non-zero.

  1. lr curve, arms A and B, grid wider than the contract grid, seeds 1-3 (18 arm-A runs)
  2. pilot series at the selected lr, seeds 1-10, arms A and B (10 arm-A runs)
  3. sigma_seed of arms B and E on seeds 1-25, both free (cache / no socket)

Asserted rather than assumed: the frozen dataset by hash before anything trains;
theta_init(A) == theta_init(B) per seed; every threshold recomputed from scipy at the
measured sigma; and independence from --workers, checked by re-running one parallel cell
sequentially in the timing probe.

    python scripts/run_pilot_sigma.py --timing-probe     # safety valve, run first
    python scripts/run_pilot_sigma.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import time
from pathlib import Path

# Before numpy and torch: the BLAS pools read these at import and never again.
from qsocket.core import pin_blas_threads

pin_blas_threads()

import numpy as np
import torch
from scipy.stats import t

from qsocket import stats
from qsocket.datasets import (
    DEFAULT_DATA_DIR,
    PRODUCTION_DATASET,
    load_splits,
    verify_frozen_identity,
)
from qsocket.head import HEAD_PARAM_COUNTS, make_head
from qsocket.socket import frozen_socket_features, make_socket
from qsocket.training import TrainConfig, train_model
from qsocket.vendored.metrics_cls import accuracy_from_z

# --- fixed configuration -------------------------------------------------------------

# The production dataset. --dataset picks the name; its hashes are asserted against the
# frozen-dataset registry in qsocket.datasets rather than trusted.
DATASET = PRODUCTION_DATASET

ANSATZ = "L1"
R_CONTRACT = 2
DILUTION = "linear"

# The contract lr is chosen from these four points and nothing else.
CONTRACT_LR_GRID: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
# The contract four plus two above them, so a curve rising to the edge is detectable.
# The two extra points are diagnostic only.
PROBE_LR_GRID: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)

CURVE_SEEDS: tuple[int, ...] = (1, 2, 3)
PILOT_SEEDS: tuple[int, ...] = tuple(range(1, 11))
SIGMA_SEEDS: tuple[int, ...] = tuple(range(1, 26))

# The cell the safety valve times, and the one used for the workers-independence
# assertion.
PROBE_LR = 1e-2
PROBE_SEED = 1
# Verdict thresholds of the timing probe, in seconds.
SAFETY_VALVE_GREEN_S = 25 * 60
SAFETY_VALVE_RED_S = 45 * 60
# Per-run wall time measured for the backend, kept as a standing assumption.
CLAIMED_RUN_SECONDS = 21 * 60

# The validation noise arm E's lr scan was read against. Used only as the declared
# yardstick for "arm B is flat", never as a fitted quantity.
VALIDATION_NOISE = 0.020
# The TOST decision. Not changed here; the power at this value is what the pilot reports.
DELTA_TOST = 0.02

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "a4_pilot"
CURVE_CSV = "a4_lr_curve.csv"
PILOT_CSV = "a4_pilot_per_run.csv"
SIGMA_CSV = "a4_sigma_seed.csv"
SUMMARY_CSV = "a4_summary.csv"
# Crash insurance: every measured row is dumped the moment the grid finishes and before
# any statistic is computed, so an exception in the analysis cannot destroy hours of
# arm-A runs.
RAW_CSV = "a4_raw_rows.csv"

# --- a threshold the verdict table leaves undefined ----------------------------------
#
# The table separates "|d| < MDE with theta_displacement clearly > 0" — training the socket
# does not buy accuracy — from "|d| < MDE with theta_displacement ~ 0", an optimisation
# failure that stops the run. It does not say where the boundary is.
# theta_displacement is ||dtheta|| / sqrt(P), in the same units as theta, whose U[0, 2pi)
# draw has sd 1.814, so the threshold below is ~0.06 % of that scale: deliberately
# permissive, because the stop row must not fire on a merely small movement.
#
# Anything between THETA_MOVED_TOL and AMBIGUITY_FACTOR x THETA_MOVED_TOL is reported as
# ambiguous rather than resolved here.
THETA_MOVED_TOL = 1e-3
THETA_AMBIGUITY_FACTOR = 10.0

# The declared verdict rows read "sigma_Delta <= ~0.02" and "sigma_Delta > ~0.03"; the
# band between them is covered by no row and is reported as such.
SIGMA_SMALL = 0.02
SIGMA_LARGE = 0.03

ARMS = ("A", "B", "E")


# --- thresholds, recomputed -----------------------------------------------------------


# One implementation of each, shared with the series driver and the analysis script.
mde = stats.mde
sigma_ci = stats.sigma_confidence_interval
binomial_se = stats.binomial_se


def tost_bound(sigma: float, n: int) -> float:
    """The delta below which TOST cannot conclude equivalence at any outcome."""
    df = n - 1
    return float((t.ppf(0.95, df) + t.ppf(0.80, df)) / np.sqrt(n) * sigma)


def ci_half_width_90(sigma: float, n: int) -> float:
    """Half width of the 90 % CI for Delta -- the quantity TOST compares against delta."""
    return float(t.ppf(0.95, n - 1) * sigma / np.sqrt(n))


def tost_power_at_zero(delta: float, sigma: float, n: int) -> float:
    """TOST power at true Delta = 0 — qsocket.stats.tost_power in this file's argument
    order, kept because the pilot report reads it positionally."""
    return stats.tost_power(sigma=sigma, n=n, delta=delta)


def paired_t_p_value(differences: np.ndarray) -> float:
    """Two-sided p of the paired t test, H0: mean difference = 0."""
    n = len(differences)
    sd = float(np.std(differences, ddof=1))
    if sd == 0.0:
        return 0.0 if float(np.mean(differences)) != 0.0 else 1.0
    statistic = float(np.mean(differences)) / (sd / np.sqrt(n))
    return float(2 * t.sf(abs(statistic), n - 1))


# --- dataset --------------------------------------------------------------------------


def verify_dataset(dataset: str = DATASET, data_dir=DEFAULT_DATA_DIR) -> dict:
    """Assert that `dataset` is a registered frozen artefact before anything is trained."""
    return verify_frozen_identity(dataset, out_dir=data_dir)


# --- sockets and the pairing proof ----------------------------------------------------


def build_socket(arm: str, seed: int):
    if arm == "A":
        return make_socket(
            "quantum", R=R_CONTRACT, ansatz=ANSATZ, trainable=True, seed=seed
        )
    if arm == "B":
        return make_socket(
            "quantum", R=R_CONTRACT, ansatz=ANSATZ, trainable=False, seed=seed
        )
    if arm == "E":
        return make_socket("identity", R=None, ansatz=ANSATZ, trainable=False, seed=seed)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def assert_theta_pairing(seed: int) -> None:
    """Acceptance criterion 4: A and B start from bit-for-bit identical theta.

    Proved in THIS run for THIS seed, not cited from the seeding contract. Without it the
    paired difference stops meaning "what training did from this starting point".
    """
    a, b = build_socket("A", seed), build_socket("B", seed)
    assert torch.equal(a.theta_init, b.theta_init), (
        f"theta_init(A) != theta_init(B) at seed {seed}: the A/B pairing is broken and "
        f"Delta_AB would not be a paired difference"
    )
    assert torch.equal(a.theta(), b.theta()), f"theta(A) != theta(B) at seed {seed}"


# --- one training run -----------------------------------------------------------------


def _accuracy(logits, y) -> float:
    return float(accuracy_from_z(np.asarray(logits).reshape(-1), np.asarray(y).reshape(-1)))


def _blank_row(arm: str, seed: int, lr: float) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "lr": float(lr),
        "ansatz": ANSATZ,
        "R": R_CONTRACT if arm != "E" else "",
        "dilution": DILUTION,
        "head_params": HEAD_PARAM_COUNTS[DILUTION],
    }


def run_arm_A(splits, *, seed: int, lr: float) -> dict:
    """One arm-A run: trainable quantum socket + linear head, joint training."""
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    socket = build_socket("A", seed)
    head = make_head(DILUTION, seed=seed)
    started = time.perf_counter()
    result = train_model(
        socket, head, X_tr, y_tr, X_val, y_val, cfg=TrainConfig(lr=lr), seed=seed
    )
    train_wall = time.perf_counter() - started

    with torch.no_grad():
        logits = head(socket(torch.as_tensor(np.asarray(X_te), dtype=torch.float32)))
    row = _blank_row("A", seed, lr)
    row.update(
        {
            "val_accuracy": float(result.val_accuracy),
            "test_accuracy": _accuracy(logits, y_te),
            "train_accuracy": float(result.train_accuracy),
            "val_macro_f1": float(result.val_macro_f1),
            "best_epoch": int(result.best_epoch),
            "epochs_run": int(result.epochs_run),
            "theta_displacement": float(result.theta_displacement),
            "grad_rms_start": float(result.grad_rms_start),
            "grad_rms_end": float(result.grad_rms_end),
            "socket_convergence_epoch": (
                "" if result.socket_convergence_epoch is None else int(result.socket_convergence_epoch)
            ),
            "cache_seconds": 0.0,
            "train_seconds": float(train_wall),
            "wall_seconds": float(train_wall),
            "used_feature_cache": False,
        }
    )
    return row


def run_arm_cached(splits, *, arm: str, seed: int, lrs) -> list[dict]:
    """Arms B and E: frozen socket, so the socket output is constant across the run.

    The features are computed once per seed and every lr trains on them through an identity
    socket, which is ~100x cheaper per arm-B run.
    """
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    socket = build_socket(arm, seed)
    started = time.perf_counter()
    F_tr = frozen_socket_features(socket, X_tr)
    F_val = frozen_socket_features(socket, X_val)
    F_te = frozen_socket_features(socket, X_te)
    cache_wall = time.perf_counter() - started

    rows = []
    for lr in lrs:
        identity = build_socket("E", seed)
        head = make_head(DILUTION, seed=seed)
        started = time.perf_counter()
        result = train_model(
            identity, head, F_tr, y_tr, F_val, y_val, cfg=TrainConfig(lr=lr), seed=seed
        )
        train_wall = time.perf_counter() - started
        with torch.no_grad():
            logits = head(torch.as_tensor(F_te, dtype=torch.float32))
        row = _blank_row(arm, seed, lr)
        row.update(
            {
                "val_accuracy": float(result.val_accuracy),
                "test_accuracy": _accuracy(logits, y_te),
                "train_accuracy": float(result.train_accuracy),
                "val_macro_f1": float(result.val_macro_f1),
                "best_epoch": int(result.best_epoch),
                "epochs_run": int(result.epochs_run),
                # Frozen socket: displacement is 0.0 by construction, recorded so a future
                # wiring mistake shows up in the CSV rather than in a conclusion.
                "theta_displacement": float(result.theta_displacement),
                "grad_rms_start": float(result.grad_rms_start),
                "grad_rms_end": float(result.grad_rms_end),
                "socket_convergence_epoch": "",
                "cache_seconds": float(cache_wall),
                "train_seconds": float(train_wall),
                "wall_seconds": float(cache_wall + train_wall),
                "used_feature_cache": True,
            }
        )
        rows.append(row)
    return rows


# --- parallel execution ---------------------------------------------------------------

_SPLITS = None


def _worker_init(dataset: str, data_dir: str) -> None:
    """ONE torch thread per worker.

    A run of arm A occupies effectively one core (measured 99.6 % of one core; lightning
    .qubit at 5 qubits does not scale with threads), so 10 workers x 8 threads would put
    80 threads on 12 cores and slow everything down.
    """
    global _SPLITS
    torch.set_num_threads(1)
    _SPLITS = load_splits(dataset, out_dir=Path(data_dir))


def _worker_run(task):
    arm, seed, lrs = task
    if arm == "A":
        return [run_arm_A(_SPLITS, seed=seed, lr=lr) for lr in lrs]
    return run_arm_cached(_SPLITS, arm=arm, seed=seed, lrs=lrs)


def make_tasks(*, arm_a_cells, cached_cells) -> list[tuple]:
    """Arm A: one task per (seed, lr), because those are the expensive units to spread.
    Arms B/E: one task per seed carrying every lr, so the feature cache is built once.
    """
    tasks = [("A", seed, (lr,)) for seed, lr in arm_a_cells]
    tasks += [(arm, seed, tuple(lrs)) for arm, seed, lrs in cached_cells]
    return tasks


def execute(tasks, *, workers: int, dataset: str, data_dir) -> list[dict]:
    if workers <= 1:
        _worker_init(dataset, str(data_dir))
        rows: list[dict] = []
        for task in tasks:
            rows.extend(_worker_run(task))
        return rows
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers, initializer=_worker_init, initargs=(dataset, str(data_dir))
    ) as pool:
        return [row for chunk in pool.imap_unordered(_worker_run, tasks) for row in chunk]


def default_workers() -> int:
    return max(1, min(10, (os.cpu_count() or 3) - 2))


# --- lr selection ---------------------------------------------------------------------


def lr_curve_table(rows: list[dict], *, arm: str | None = None) -> dict[float, float]:
    """Mean validation accuracy per lr, over the given arm (or over arms A and B)."""
    table: dict[float, float] = {}
    for lr in sorted({r["lr"] for r in rows}):
        picked = [
            r["val_accuracy"]
            for r in rows
            if r["lr"] == lr and (arm is None or r["arm"] == arm)
        ]
        if picked:
            table[lr] = float(np.mean(picked))
    return table


def select_contract_lr(curve_rows: list[dict]) -> tuple[float, dict[float, float]]:
    """Best mean validation accuracy over arms A and B, restricted to the four contract
    points. Ties go to the lower lr.

    The two probe points are excluded by construction, not by convention: extending the
    contract grid is a separate decision.
    """
    eligible = [r for r in curve_rows if r["arm"] in ("A", "B") and r["lr"] in CONTRACT_LR_GRID]
    if not eligible:
        raise ValueError("no arm A/B rows on the contract lr grid")
    table = lr_curve_table(eligible)
    best = max(table, key=lambda lr: (table[lr], -lr))
    return float(best), table


def probe_optimum_lr(curve_rows: list[dict], arm: str) -> tuple[float, dict[float, float]]:
    """Best lr for one arm over the full six-point probe grid. Diagnostic only."""
    table = lr_curve_table([r for r in curve_rows if r["arm"] == arm], arm=arm)
    best = max(table, key=lambda lr: (table[lr], -lr))
    return float(best), table


def flatness_verdict(curve_rows: list[dict]) -> dict:
    """Does arm B react to lr at all?

    Arm B has a frozen socket and a linear head, i.e. logistic regression on fixed features
    — a convex problem, like arm E, whose validation accuracy spread across three decades
    of lr stayed close to the validation noise.
    """
    spreads = {}
    for arm in ("A", "B"):
        table = lr_curve_table([r for r in curve_rows if r["arm"] == arm], arm=arm)
        values = np.array(list(table.values()))
        spreads[arm] = {
            "table": table,
            "spread": float(values.max() - values.min()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    b_flat = spreads["B"]["spread"] < VALIDATION_NOISE
    a_moves = spreads["A"]["spread"] > spreads["B"]["spread"]
    if b_flat and a_moves:
        verdict = "B is flat, A is not"
        meaning = (
            "The 'shared vs per-arm lr' dispute is moot: arm B has nothing to tune, so a "
            "shared lr is arm A's lr and harms nobody. the lr rule stands at no cost."
        )
    elif not b_flat:
        verdict = "both arms react to lr"
        meaning = (
            "A shared lr is a real compromise -> B9 ('lr per arm') stops being an optional "
            "resilience cell. REPORT for decision; do not change the contract here."
        )
    else:
        verdict = "B is flat and A is no wider than B"
        meaning = (
            "Not covered by the two declared rows: neither arm reacts to lr. Report and ask "
            "the owner."
        )
    return {
        "arm_A": spreads["A"],
        "arm_B": spreads["B"],
        "validation_noise": VALIDATION_NOISE,
        "b_below_noise": bool(b_flat),
        "verdict": verdict,
        "meaning": meaning,
    }


# --- statistics -----------------------------------------------------------------------


def _by_seed(rows: list[dict], arm: str, lr: float) -> dict[int, dict]:
    return {r["seed"]: r for r in rows if r["arm"] == arm and r["lr"] == lr}


def arm_statistics(rows: list[dict], arm: str, lr: float, seeds) -> dict:
    index = _by_seed(rows, arm, lr)
    test = np.array([index[s]["test_accuracy"] for s in seeds])
    val = np.array([index[s]["val_accuracy"] for s in seeds])
    displacement = np.array([index[s]["theta_displacement"] for s in seeds])
    return {
        "arm": arm,
        "lr": lr,
        "n_seeds": len(seeds),
        "test_mean": float(test.mean()),
        "sigma_seed": float(test.std(ddof=1)),
        "test_min": float(test.min()),
        "test_max": float(test.max()),
        "test_range": float(test.max() - test.min()),
        "val_mean": float(val.mean()),
        "theta_displacement_mean": float(displacement.mean()),
        "theta_displacement_min": float(displacement.min()),
        "theta_displacement_max": float(displacement.max()),
        "best_epoch_median": float(np.median([index[s]["best_epoch"] for s in seeds])),
        "epochs_run_max": int(max(index[s]["epochs_run"] for s in seeds)),
        "grad_rms_start_median": float(np.median([index[s]["grad_rms_start"] for s in seeds])),
        "grad_rms_end_median": float(np.median([index[s]["grad_rms_end"] for s in seeds])),
        "wall_seconds_median": float(np.median([index[s]["wall_seconds"] for s in seeds])),
        "per_seed_test": [float(index[s]["test_accuracy"]) for s in seeds],
    }


def paired_statistics(rows: list[dict], lr: float, seeds, n_test: int) -> dict:
    a = _by_seed(rows, "A", lr)
    b = _by_seed(rows, "B", lr)
    differences = np.array([a[s]["test_accuracy"] - b[s]["test_accuracy"] for s in seeds])
    n = len(seeds)
    mean = float(differences.mean())
    sigma = float(differences.std(ddof=1))
    lo, hi = sigma_ci(sigma, n)
    half = float(t.ppf(0.975, n - 1) * sigma / np.sqrt(n))
    return {
        "lr": lr,
        "n_seeds": n,
        "per_seed": [float(d) for d in differences],
        "mean": mean,
        "sigma_delta": sigma,
        "sigma_ci95": (lo, hi),
        "ci95_mean": (mean - half, mean + half),
        "ci90_half_width": ci_half_width_90(sigma, n),
        "p_value": paired_t_p_value(differences),
        "mde": mde(sigma, n),
        "tost_bound": tost_bound(sigma, n),
        "tost_power_at_delta": tost_power_at_zero(DELTA_TOST, sigma, n),
        "binomial_se": binomial_se(n_test),
    }


def verdict(paired: dict, arms: dict[str, dict]) -> tuple[str, str, str]:
    """The verdict table, DECLARED BEFORE THE MEASUREMENT. Returns (row, name, action)."""
    d = paired["mean"]
    sigma = paired["sigma_delta"]
    threshold = paired["mde"]
    displacement = arms["A"]["theta_displacement_mean"]
    moved = displacement >= THETA_MOVED_TOL * THETA_AMBIGUITY_FACTOR
    still = displacement < THETA_MOVED_TOL

    # The stop row is checked first: it can co-occur with |d| < MDE.
    if d < 0 and paired["p_value"] < 0.05:
        return (
            "row 5",
            "STOP -- training HURTS",
            "Unforeseen. STOP and ask the owner. Check lr and patience as confounds first.",
        )

    if d >= threshold:
        if sigma <= SIGMA_SMALL:
            return (
                "row 1",
                "BEST CASE",
                "Delta detectable, delta = 0.02 feasible, the TOST decision closes unchanged. Plan unchanged.",
            )
        if sigma > SIGMA_LARGE:
            return (
                "row 2",
                "effect present, power weak",
                "Delta reportable but TOST fails (the TOST decision exit 3: the CI alone). Consider more seeds.",
            )
        return (
            "between rows 1 and 2",
            f"effect present, sigma_delta = {sigma:.4f} in the uncovered band "
            f"({SIGMA_SMALL} , {SIGMA_LARGE}]",
            "Not covered by the declared verdict table. Report and ask the owner.",
        )

    if abs(d) < threshold:
        if moved:
            return (
                "row 3",
                "Delta ~ 0 but the optimiser moved theta",
                "This IS the result: 'training the socket does not buy accuracy'. SPEC "
                "section 9, row 'Delta ~ 0 everywhere'. The thesis stands.",
            )
        if still:
            return (
                "row 4",
                "STOP -- a different paper",
                "SPEC section 9: this is OPTIMISATION FAILURE, not the redundancy of "
                "training. REPORT IMMEDIATELY, do not continue the plan.",
            )
        return (
            "between rows 3 and 4",
            f"Delta ~ 0, theta_displacement = {displacement:.3e} in the ambiguous band "
            f"[{THETA_MOVED_TOL:g}, {THETA_MOVED_TOL * THETA_AMBIGUITY_FACTOR:g})",
            "'significantly > 0' was never defined. Report and ask the owner.",
        )

    return (
        "none",
        f"unexpected: mean = {d:+.4f}, MDE = {threshold:.4f}",
        "Not covered by the declared verdict table. STOP and ask the owner.",
    )


def ceiling_row_check(arms: dict[str, dict], binomial: float) -> dict:
    """Last row of the declared verdict table: acc(A) ~ acc(E) while acc(B) < acc(E). Reported as a
    separate flag because it can co-occur with any of the rows above."""
    a, b, e = arms["A"]["test_mean"], arms["B"]["test_mean"], arms["E"]["test_mean"]
    fired = abs(a - e) < binomial and (e - b) > binomial
    return {
        "fired": bool(fired),
        "acc_A": a,
        "acc_B": b,
        "acc_E": e,
        "action": (
            "STOP: SPEC section 9, new row -- the socket adds nothing and training merely "
            "undoes the harm of randomness. A DIFFERENT thesis."
            if fired
            else "not fired"
        ),
    }


# --- csv ------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the timing safety valve ---------------------------------------------------------


def safety_valve(seconds: float) -> tuple[str, str]:
    if seconds <= SAFETY_VALVE_GREEN_S:
        return "GREEN", "<= 25 min: proceed with the full grid."
    if seconds <= SAFETY_VALVE_RED_S:
        return (
            "AMBER",
            f"25-45 min: proceed, but REPORT the discrepancy against the claimed "
            f"{CLAIMED_RUN_SECONDS / 60:.0f} min per run.",
        )
    return (
        "RED",
        "> 45 min: STOP. Do not launch the 28 runs. Report the time and wait for a "
        "decision -- it invalidates the schedule of the main series.",
    )


def timing_probe(dataset: str, data_dir) -> dict:
    """ONE arm-A run at lr = 1e-2, seed 1, sequential, timed — the safety valve."""
    torch.set_num_threads(1)
    verify_dataset(dataset, data_dir)
    assert_theta_pairing(PROBE_SEED)
    splits = load_splits(dataset, out_dir=data_dir)
    started = time.perf_counter()
    row = run_arm_A(splits, seed=PROBE_SEED, lr=PROBE_LR)
    wall = time.perf_counter() - started
    light, action = safety_valve(wall)
    return {"row": row, "wall_seconds": wall, "light": light, "action": action}


def report_timing_probe(probe: dict) -> None:
    row, wall = probe["row"], probe["wall_seconds"]
    print("=" * 100)
    print("the pilot section 3a SAFETY VALVE -- one arm-A run, timed, BEFORE the full grid")
    print("=" * 100)
    print(f"cell            arm A, lr = {PROBE_LR:g}, seed = {PROBE_SEED}, L1, R = {R_CONTRACT}, linear head")
    print(f"WALL TIME       {wall:.1f} s = {wall / 60:.2f} min")
    print(f"  claimed       {CLAIMED_RUN_SECONDS:.0f} s = {CLAIMED_RUN_SECONDS / 60:.0f} min"
          f"   ratio measured/claimed = {wall / CLAIMED_RUN_SECONDS:.2f}x")
    print(f"epochs_run      {row['epochs_run']}   best_epoch {row['best_epoch']}")
    print(f"val_accuracy    {row['val_accuracy']:.6f}   test_accuracy {row['test_accuracy']:.6f}")
    print(f"theta_displacement {row['theta_displacement']:.6e}")
    print(f"grad_rms        start {row['grad_rms_start']:.6e}  end {row['grad_rms_end']:.6e}")
    print()
    print(f"SAFETY VALVE    {probe['light']}")
    print(f"  {probe['action']}")
    estimate = wall * 28 / default_workers()
    print()
    print(f"projected full grid (28 arm-A runs / {default_workers()} workers, perfect "
          f"packing): {estimate / 60:.0f} min")


# --- the measurement ------------------------------------------------------------------


def run(
    *,
    curve_seeds=CURVE_SEEDS,
    pilot_seeds=PILOT_SEEDS,
    sigma_seeds=SIGMA_SEEDS,
    probe_lr_grid=PROBE_LR_GRID,
    dataset: str = DATASET,
    data_dir=DEFAULT_DATA_DIR,
    out_dir=DEFAULT_OUT_DIR,
    workers: int | None = None,
    probe: dict | None = None,
) -> dict:
    workers = default_workers() if workers is None else workers

    # Fail on a usage error before spending arm-A time: sigma of the paired difference
    # needs ddof=1, so a single pilot seed gives df = 0 and every downstream statistic
    # divides by zero after the grid has already run.
    if len(set(pilot_seeds)) < 2:
        raise ValueError(
            f"pilot_seeds must contain at least 2 distinct seeds to estimate "
            f"sigma_delta(A-B); got {sorted(set(pilot_seeds))}. This is the whole point "
            f"of the pilot (a standing assumption), so it is refused up front rather than after the grid."
        )
    if not set(curve_seeds):
        raise ValueError("curve_seeds must not be empty: step 1 selects the contract lr on them")

    manifest = verify_dataset(dataset, data_dir)
    splits = load_splits(dataset, out_dir=data_dir)
    n_test = len(splits["test"][0])

    # The theta pairing, asserted for every seed this run touches.
    for seed in sorted(set(curve_seeds) | set(pilot_seeds)):
        assert_theta_pairing(seed)

    started_all = time.perf_counter()

    # --- step 1: the lr curve on the six-point probe grid ----------------------------
    curve_tasks = make_tasks(
        arm_a_cells=[(s, lr) for s in curve_seeds for lr in probe_lr_grid],
        cached_cells=[("B", s, probe_lr_grid) for s in curve_seeds],
    )
    curve_rows = execute(curve_tasks, workers=workers, dataset=dataset, data_dir=data_dir)
    step1_seconds = time.perf_counter() - started_all

    contract_lr, contract_table = select_contract_lr(curve_rows)
    probe_optimum_A, curve_A = probe_optimum_lr(curve_rows, "A")
    probe_optimum_B, curve_B = probe_optimum_lr(curve_rows, "B")
    flatness = flatness_verdict(curve_rows)

    # Retreat rule, declared before the measurement: if arm A's optimum falls outside the
    # contract grid, the contract is not changed — step 2 runs twice and both sigmas are
    # reported.
    retreat_fired = probe_optimum_A not in CONTRACT_LR_GRID
    measurement_lrs = [contract_lr] + ([probe_optimum_A] if retreat_fired else [])

    # --- step 2: the pilot series, and step 3: sigma_seed of B and E -----------------
    started_step2 = time.perf_counter()
    pilot_tasks = make_tasks(
        arm_a_cells=[(s, lr) for lr in measurement_lrs for s in pilot_seeds],
        cached_cells=(
            [("B", s, tuple(measurement_lrs)) for s in pilot_seeds]
            # Step 3: arms B and E on seeds 1-25 at the contract lr. Seeds already covered
            # for B above are skipped; the run is deterministic, so nothing is lost.
            + [("B", s, (contract_lr,)) for s in sigma_seeds if s not in pilot_seeds]
            + [("E", s, tuple(measurement_lrs)) for s in sigma_seeds]
        ),
    )
    measurement_rows = execute(
        pilot_tasks, workers=workers, dataset=dataset, data_dir=data_dir
    )
    step23_seconds = time.perf_counter() - started_step2

    results: dict = {
        "dataset": dataset,
        "manifest": manifest,
        "n_test": n_test,
        "workers": workers,
        "curve_rows": curve_rows,
        "measurement_rows": measurement_rows,
        "contract_lr": contract_lr,
        "contract_table": contract_table,
        "curve_A": curve_A,
        "curve_B": curve_B,
        "probe_optimum_A": probe_optimum_A,
        "probe_optimum_B": probe_optimum_B,
        "retreat_fired": retreat_fired,
        "measurement_lrs": measurement_lrs,
        "flatness": flatness,
        "step1_seconds": step1_seconds,
        "step23_seconds": step23_seconds,
        "probe": probe,
        "sigma_seeds": list(sigma_seeds),
        "pilot_seeds": list(pilot_seeds),
        "curve_seeds": list(curve_seeds),
    }

    # --- crash insurance: dump the raw rows before any analysis ----------------------
    # The measurement is the expensive part; the statistics are cheap and re-runnable from
    # this file, so writing at the end would risk every arm-A run of the grid.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_CSV
    write_csv(raw_path, sorted(measurement_rows, key=lambda r: (r["arm"], r["lr"], r["seed"])))
    print(f"raw rows written before analysis: {raw_path}  ({len(measurement_rows)} rows)")

    # --- per-lr analysis --------------------------------------------------------------
    analyses = {}
    for lr in measurement_lrs:
        arms = {
            "A": arm_statistics(measurement_rows, "A", lr, pilot_seeds),
            "B": arm_statistics(measurement_rows, "B", lr, pilot_seeds),
            "E": arm_statistics(measurement_rows, "E", lr, pilot_seeds),
        }
        paired = paired_statistics(measurement_rows, lr, pilot_seeds, n_test)
        delta_AE = arms["A"]["test_mean"] - arms["E"]["test_mean"]
        row, name, action = verdict(paired, arms)
        analyses[lr] = {
            "arms": arms,
            "paired": paired,
            "verdict_row": row,
            "verdict_name": name,
            "verdict_action": action,
            "ceiling_row": ceiling_row_check(arms, paired["binomial_se"]),
            "delta_AE": float(delta_AE),
            "delta_AE_decomposed": {
                "delta_AB": paired["mean"],
                "acc_B_minus_acc_E": float(arms["B"]["test_mean"] - arms["E"]["test_mean"]),
                "sum": float(paired["mean"] + arms["B"]["test_mean"] - arms["E"]["test_mean"]),
            },
            "sigma_ratio_to_A": (
                float(paired["sigma_delta"] / arms["A"]["sigma_seed"])
                if arms["A"]["sigma_seed"] > 0
                else float("inf")
            ),
            "sigma_ratio_to_B": (
                float(paired["sigma_delta"] / arms["B"]["sigma_seed"])
                if arms["B"]["sigma_seed"] > 0
                else float("inf")
            ),
        }
    results["analyses"] = analyses

    # --- step 3 proper: sigma_seed of B and E on 25 seeds -----------------------------
    results["sigma25"] = {
        arm: arm_statistics(measurement_rows, arm, contract_lr, sigma_seeds)
        for arm in ("B", "E")
    }

    # --- the workers-independence assertion ---------------
    results["workers_check"] = check_workers_independence(results, probe)

    # --- output -----------------------------------------------------------------------
    # out_dir already created above, when the raw rows were dumped.
    paths = {RAW_CSV: (raw_path, sha256(raw_path))}
    sigma_rows = [
        r
        for r in measurement_rows
        if r["arm"] in ("B", "E") and r["lr"] == contract_lr
    ]
    for name, rows in (
        (CURVE_CSV, sorted(curve_rows, key=lambda r: (r["arm"], r["lr"], r["seed"]))),
        (
            PILOT_CSV,
            sorted(
                [r for r in measurement_rows if r["seed"] in pilot_seeds],
                key=lambda r: (r["arm"], r["lr"], r["seed"]),
            ),
        ),
        (SIGMA_CSV, sorted(sigma_rows, key=lambda r: (r["arm"], r["seed"]))),
    ):
        path = out_dir / name
        write_csv(path, rows)
        paths[name] = (path, sha256(path))

    summary_rows = []
    for lr, analysis in analyses.items():
        for arm in ARMS:
            stats = dict(analysis["arms"][arm])
            stats["per_seed_test"] = " ".join(f"{v:.6f}" for v in stats["per_seed_test"])
            stats["role"] = "contract" if lr == contract_lr else "retreat_rule"
            summary_rows.append(stats)
    summary_path = out_dir / SUMMARY_CSV
    write_csv(summary_path, summary_rows)
    paths[SUMMARY_CSV] = (summary_path, sha256(summary_path))
    results["paths"] = paths
    results["total_seconds"] = time.perf_counter() - started_all
    return results


def check_workers_independence(results: dict, probe: dict | None) -> dict:
    """The result must not depend on --workers.

    The sequential timing probe computes one cell; if the parallel grid computed the same
    cell, the two are compared bit for bit. Batch order comes from batch_order_rng(seed),
    so parallelism cannot change anything — checked rather than assumed.
    """
    if probe is None:
        return {"checked": False, "reason": "no sequential timing probe in this run"}
    reference = probe["row"]
    match = [
        r
        for r in results["curve_rows"]
        if r["arm"] == "A" and r["seed"] == reference["seed"] and r["lr"] == reference["lr"]
    ]
    if not match:
        return {"checked": False, "reason": "the probe cell is not in the parallel grid"}
    parallel = match[0]
    fields = ("val_accuracy", "test_accuracy", "train_accuracy", "best_epoch", "epochs_run",
              "theta_displacement", "grad_rms_start", "grad_rms_end")
    mismatches = {f: (reference[f], parallel[f]) for f in fields if reference[f] != parallel[f]}
    assert not mismatches, (
        f"the result depends on --workers: sequential (workers=1) and parallel "
        f"(workers={results['workers']}) disagree on {mismatches}"
    )
    return {
        "checked": True,
        "cell": f"arm A, lr={reference['lr']:g}, seed={reference['seed']}",
        "workers_sequential": 1,
        "workers_parallel": results["workers"],
        "fields": list(fields),
        "identical": True,
    }


# --- reporting ------------------------------------------------------------------------


def report(results: dict) -> None:
    print("=" * 100)
    print("the pilot — pilot sigma: does Delta_AB exist, and what is sigma_Delta(A-B)")
    print("=" * 100)
    print(f"dataset      {results['dataset']}")
    print(f"  dataset_hash {results['manifest']['dataset_hash']}")
    print(f"  pca_hash     {results['manifest']['pca_hash']}")
    print(f"config       L1, R = {R_CONTRACT}, {DILUTION} head "
          f"({HEAD_PARAM_COUNTS[DILUTION]} params), backend pennylane")
    print(f"n_test       {results['n_test']}    workers {results['workers']}")
    print()

    if results.get("probe"):
        probe = results["probe"]
        print("-- section 3a safety valve ------------------------------------------------")
        print(f"  one arm-A run (lr={PROBE_LR:g}, seed={PROBE_SEED}): "
              f"{probe['wall_seconds']:.1f} s = {probe['wall_seconds'] / 60:.2f} min "
              f"[{probe['light']}]")
        print(f"  vs the claimed {CLAIMED_RUN_SECONDS / 60:.0f} min per run: "
              f"{probe['wall_seconds'] / CLAIMED_RUN_SECONDS:.2f}x")
        print(f"  {probe['action']}")
        print()

    print("-- step 1: lr curve on the SIX-point probe grid, seeds "
          f"{results['curve_seeds']} ----------")
    header = f"{'lr':>8} {'val acc A':>10} {'val acc B':>10} {'mean A,B':>10} {'contract?':>10}"
    print(header)
    print("-" * len(header))
    for lr in sorted(results["curve_A"]):
        a = results["curve_A"][lr]
        b = results["curve_B"].get(lr, float("nan"))
        in_contract = lr in CONTRACT_LR_GRID
        mean = f"{results['contract_table'][lr]:.6f}" if in_contract else "  (probe)"
        print(f"{lr:>8g} {a:>10.6f} {b:>10.6f} {mean:>10} {in_contract!s:>10}")
    print()
    print(f"  CONTRACT lr  = {results['contract_lr']:g}   "
          f"(best mean val accuracy over arms A AND B, from the four contract points only)")
    print(f"  arm A optimum over the full probe grid = {results['probe_optimum_A']:g}"
          f"{'   <-- OUTSIDE the contract grid, retreat rule FIRED' if results['retreat_fired'] else ''}")
    print(f"  arm B optimum over the full probe grid = {results['probe_optimum_B']:g}")
    print()
    flat = results["flatness"]
    print("-- flatness test of arm B (settles the lr rule / B9 empirically) -------------------")
    print(f"  spread of val acc(A) over the six lr: {flat['arm_A']['spread']:.4f} "
          f"[{flat['arm_A']['min']:.4f}, {flat['arm_A']['max']:.4f}]")
    print(f"  spread of val acc(B) over the six lr: {flat['arm_B']['spread']:.4f} "
          f"[{flat['arm_B']['min']:.4f}, {flat['arm_B']['max']:.4f}]")
    print(f"  validation noise yardstick (FINDINGS 1.5c): {flat['validation_noise']:.3f}")
    print(f"  -> {flat['verdict']}")
    print(f"     {flat['meaning']}")
    print()

    for lr, analysis in results["analyses"].items():
        role = "CONTRACT lr" if lr == results["contract_lr"] else "RETREAT-RULE lr"
        print("=" * 100)
        print(f"-- steps 2 and 3 at {role} = {lr:g}, seeds {results['pilot_seeds']} ----")
        print("=" * 100)
        head = (f"{'arm':>4} {'test mean':>10} {'sigma_seed':>11} {'[min, max]':>18} "
                f"{'val mean':>9} {'best ep':>8} {'ep max':>7} {'wall s':>9}")
        print(head)
        print("-" * len(head))
        for arm in ARMS:
            s = analysis["arms"][arm]
            print(f"{arm:>4} {s['test_mean']:>10.6f} {s['sigma_seed']:>11.6f} "
                  f"[{s['test_min']:.4f}, {s['test_max']:.4f}] {s['val_mean']:>9.4f} "
                  f"{s['best_epoch_median']:>8.1f} {s['epochs_run_max']:>7} "
                  f"{s['wall_seconds_median']:>9.1f}")
        print()
        for arm in ARMS:
            s = analysis["arms"][arm]
            print(f"  acc({arm}) per seed: " + " ".join(f"{v:.4f}" for v in s["per_seed_test"]))
        print()
        p = analysis["paired"]
        print("  Delta_AB per seed: " + " ".join(f"{v:+.4f}" for v in p["per_seed"]))
        print(f"  mean Delta_AB          {p['mean']:+.6f}   "
              f"95 % CI [{p['ci95_mean'][0]:+.4f}, {p['ci95_mean'][1]:+.4f}]   "
              f"paired t p = {p['p_value']:.4g}")
        print(f"  sigma_Delta(A-B)       {p['sigma_delta']:.6f}   "
              f"95 % CI [{p['sigma_ci95'][0]:.4f}, {p['sigma_ci95'][1]:.4f}]")
        print(f"  sigma_Delta / sigma_seed(A) = {analysis['sigma_ratio_to_A']:.3f}    "
              f"/ sigma_seed(B) = {analysis['sigma_ratio_to_B']:.3f}")
        print(f"  theta_displacement (arm A): mean {analysis['arms']['A']['theta_displacement_mean']:.6e} "
              f"[{analysis['arms']['A']['theta_displacement_min']:.3e}, "
              f"{analysis['arms']['A']['theta_displacement_max']:.3e}]")
        print(f"  grad_rms (arm A): start {analysis['arms']['A']['grad_rms_start_median']:.4e} "
              f"end {analysis['arms']['A']['grad_rms_end_median']:.4e}")
        print()
        d = analysis["delta_AE_decomposed"]
        print(f"  Delta_AE = Delta_AB + (acc(B) - acc(E)) = {d['delta_AB']:+.6f} "
              f"{d['acc_B_minus_acc_E']:+.6f} = {d['sum']:+.6f}   "
              f"(direct: {analysis['delta_AE']:+.6f})")
        print()
        print("  thresholds RECOMPUTED at the MEASURED sigma_Delta (SPEC section 7.6):")
        print(f"    MDE            = 0.995 x {p['sigma_delta']:.4f} = {p['mde']:.4f}")
        print(f"    TOST bound     = 0.859 x {p['sigma_delta']:.4f} = {p['tost_bound']:.4f}")
        print(f"    90 % CI half width = {p['ci90_half_width']:.4f}  "
              f"(TOST can conclude only for delta above this)")
        print(f"    binomial SE on {results['n_test']} = {p['binomial_se']:.4f}")
        print(f"    TOST POWER at delta = {DELTA_TOST} and the measured sigma_Delta: "
              f"{p['tost_power_at_delta']:.4f}")
        print()
        print(f"  VERDICT (declared BEFORE the measurement) -- {analysis['verdict_row']}")
        print(f"    {analysis['verdict_name']}")
        print(f"    {analysis['verdict_action']}")
        ceiling = analysis["ceiling_row"]
        print(f"  ceiling row (acc(A) ~ acc(E) while acc(B) < acc(E)): "
              f"{'FIRED' if ceiling['fired'] else 'not fired'}")
        if ceiling["fired"]:
            print(f"    {ceiling['action']}")
        print()

    print("=" * 100)
    print(f"-- step 3: sigma_seed on {len(results['sigma_seeds'])} seeds, "
          f"lr = {results['contract_lr']:g} --------------------")
    head = f"{'arm':>4} {'n':>4} {'test mean':>10} {'sigma_seed':>11} {'[min, max]':>18} {'median wall s':>14}"
    print(head)
    print("-" * len(head))
    for arm in ("B", "E"):
        s = results["sigma25"][arm]
        print(f"{arm:>4} {s['n_seeds']:>4} {s['test_mean']:>10.6f} {s['sigma_seed']:>11.6f} "
              f"[{s['test_min']:.4f}, {s['test_max']:.4f}] {s['wall_seconds_median']:>14.2f}")
    print()
    b_wall = results["sigma25"]["B"]["wall_seconds_median"]
    print(f"  arm-B run through the FEATURE CACHE: median {b_wall:.2f} s "
          f"(without the cache one arm-B run is ~596 s -> "
          f"{596 / b_wall:.0f}x). Acceptance criterion 3.")
    print()

    check = results["workers_check"]
    print("-- workers independence ------------------------------------------------------")
    if check.get("checked"):
        print(f"  cell {check['cell']}: workers={check['workers_sequential']} vs "
              f"workers={check['workers_parallel']} -> IDENTICAL on {check['fields']}")
    else:
        print(f"  NOT CHECKED: {check['reason']}")
    print()
    print(f"wall time: step 1 {results['step1_seconds'] / 60:.1f} min, "
          f"steps 2+3 {results['step23_seconds'] / 60:.1f} min, "
          f"total {results['total_seconds'] / 60:.1f} min")
    print()
    for name, (path, digest) in results["paths"].items():
        print(f"{path}  sha256 {digest}")


# --- cli ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timing-probe", action="store_true",
                        help="section 3a safety valve only: ONE timed arm-A run, then exit")
    parser.add_argument("--skip-timing-probe", action="store_true",
                        help="skip the safety valve (only if it was already run separately)")
    parser.add_argument("--curve-seeds", type=int, nargs="+", default=list(CURVE_SEEDS))
    parser.add_argument("--pilot-seeds", type=int, nargs="+", default=list(PILOT_SEEDS))
    parser.add_argument("--sigma-seeds", type=int, nargs="+", default=list(SIGMA_SEEDS))
    parser.add_argument("--probe-lr", type=float, nargs="+", default=list(PROBE_LR_GRID))
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    torch.set_num_threads(1)

    probe = None
    if not args.skip_timing_probe:
        probe = timing_probe(args.dataset, args.data_dir)
        report_timing_probe(probe)
        if args.timing_probe:
            return
        if probe["light"] == "RED":
            print()
            print("STOPPING: the safety valve is RED. The full grid was NOT launched.")
            return
        print()

    result = run(
        curve_seeds=tuple(args.curve_seeds),
        pilot_seeds=tuple(args.pilot_seeds),
        sigma_seeds=tuple(args.sigma_seeds),
        probe_lr_grid=tuple(args.probe_lr),
        dataset=args.dataset,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        workers=args.workers,
        probe=probe,
    )
    report(result)


if __name__ == "__main__":
    main()
