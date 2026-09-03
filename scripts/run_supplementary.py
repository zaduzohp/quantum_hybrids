"""Supplementary measurements — report numbers the main analysis does not produce.

A number here disagreeing with the main analysis is a defect in this file.

One CSV per section, into <out-dir>/tables/:

  degeneracy.csv        runs that collapsed to a constant classifier, per axis point x arm
  diagnostics.csv       Adam - ridge, theta_displacement, best_epoch, budget, per dilution
  ridge_contrast.csv    Delta_AB with arm B read out in closed form, paired per seed
  probe_estimands.csv   estimands of an off-axis probe, same blocked estimator
  head_init.csv         how often a hidden ReLU unit is born dead, by width
  fourier_support.csv   exact spectrum of <Z_i>(x); what arm D shares with arms A/B
  data_rank.csv         singular values of the raw 20-dimensional features
  shared_lr_bias.csv    what the one-lr-for-A-and-B rule does to Delta_AB
  displacement.csv      how many gates actually moved, from the saved weights

    python scripts/run_supplementary.py --main <combined.csv> --probe <probe.csv> \
                                        --out-dir outputs/supplementary
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_a8_analysis as a8
import run_main_series as a7

from qsocket.ansatzes import build_socket_circuit, socket_param_count
from qsocket.contract import MAX_EPOCHS, dataset_location
from qsocket.core import derive
from qsocket.datasets import load_frozen
from qsocket.head import _init_linear
from qsocket.rank import z_expectation_batch
from qsocket.socket import initial_theta

AXIS_ORDER = ("linear", "h2", "h3", "h4", "h42")
TRAINED_ARMS = ("A", "F")


# =====================================================================================
# shared helpers
# =====================================================================================

def class_shares(dataset_seeds) -> dict:
    """{dataset_seed: {share of each class in the test rows}}.

    A run predicting one class for every row scores exactly the share of that class, the
    signature the degeneracy check looks for. Read off the frozen data rather than
    hard-coded, so it cannot drift away from the datasets.
    """
    out = {}
    for dataset_seed in dataset_seeds:
        name, out_dir = dataset_location(dataset_seed)
        _, y = load_frozen(name, "test", out_dir=out_dir)
        y = np.asarray(y).reshape(-1)
        out[dataset_seed] = {float(np.mean(y > 0)), float(np.mean(y <= 0))}
    return out


def _row_identity(row: dict) -> tuple:
    """What makes two rows the SAME measurement. Deliberately the cell, not the run:
    run_id and env_hash differ between a run and its re-export, the numbers do not."""
    return (
        row["dataset_seed"], row["arm"], row["ansatz_level"], row["dilution"],
        row["width_int"], row["seed_int"], row["split"], f"{row['lr_float']:g}",
    )


def _deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    """Rows from several CSVs, each cell kept once. Returns (rows, how many were dropped).

    The inputs are allowed to overlap — a probe directory may re-run an arm the main
    series already covered — but a cell measured twice with two DIFFERENT accuracies is
    not a duplicate, it is two answers to one question, and this file does not choose
    between them.
    """
    seen: dict[tuple, dict] = {}
    dropped = 0
    for row in rows:
        key = _row_identity(row)
        first = seen.get(key)
        if first is None:
            seen[key] = row
            continue
        if first["accuracy_float"] != row["accuracy_float"]:
            raise SystemExit(
                f"STOP: cell {key} appears in two input CSVs with different accuracies "
                f"({first['accuracy_float']} vs {row['accuracy_float']}). That is two "
                "measurements, not a duplicate; this script does not choose between them."
            )
        dropped += 1
    return list(seen.values()), dropped


def frame_of(rows: list[dict]) -> pd.DataFrame:
    """Result rows as a frame with the columns the tables below need."""
    frame = pd.DataFrame(rows)
    frame["accuracy"] = frame["accuracy_float"]
    frame["ridge"] = [float(v) if v not in ("", None) else np.nan for v in frame["ridge_accuracy"]]
    frame["theta"] = [float(v) if v not in ("", None) else np.nan
                      for v in frame["theta_displacement"]]
    return frame


def mark_degenerate(frame: pd.DataFrame, shares: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["degenerate"] = [
        any(abs(acc - s) < 1e-9 for s in shares[ds])
        for acc, ds in zip(frame["accuracy"], frame["dataset_seed"])
    ]
    return frame


def dilution_key(dilution: str) -> int:
    return AXIS_ORDER.index(dilution) if dilution in AXIS_ORDER else len(AXIS_ORDER)


# =====================================================================================
# 1. degeneracy
# =====================================================================================

def degeneracy_rows(test: pd.DataFrame) -> list[dict]:
    """Runs whose accuracy equals a class share exactly, i.e. a constant classifier.

    The rule is post hoc — written after the series, and the report says so. It stays
    usable because it is measurable without looking at any Delta, and it fires in arm E,
    which has no socket at all.
    """
    rows = []
    for dilution, per_dilution in test.groupby("dilution"):
        for arm, group in per_dilution.groupby("arm"):
            rows.append({
                "scope": "axis point x arm", "dilution": dilution, "arm": arm,
                "degenerate": int(group["degenerate"].sum()), "runs": len(group),
                "seeds_affected": "|".join(
                    str(s) for s in sorted(set(group.loc[group["degenerate"], "seed_int"]))),
            })
    for dilution, group in test.groupby("dilution"):
        rows.append({
            "scope": "axis point, all arms", "dilution": dilution, "arm": "ALL",
            "degenerate": int(group["degenerate"].sum()), "runs": len(group),
            "seeds_affected": "|".join(
                str(s) for s in sorted(set(group.loc[group["degenerate"], "seed_int"]))),
        })
    rows.sort(key=lambda r: (r["scope"], dilution_key(r["dilution"]), r["arm"]))
    return rows


# =====================================================================================
# 2. per-dilution diagnostics
# =====================================================================================

def diagnostic_rows(test: pd.DataFrame) -> list[dict]:
    """Adam - ridge, theta_displacement, best_epoch and the epoch budget, per dilution.

    Per dilution rather than pooled: for the epoch budget pooling is harmless, but the
    ridge gap changes sign along the axis, so a pooled figure describes no point of it —
    least of all the linear head, where the confirmatory question stands.
    """
    rows = []

    def add(quantity, dilution, arm, value, n, note=""):
        rows.append({"quantity": quantity, "dilution": dilution, "arm": arm,
                     "value": value, "n": n, "note": note})

    for (dilution, arm), g in test.groupby(["dilution", "arm"]):
        if g["ridge"].notna().any():
            gap = g["accuracy"] - g["ridge"]
            add("adam_minus_ridge", dilution, arm, float(gap.mean()), int(gap.notna().sum()),
                "frozen sockets only; arms A and F have no closed-form readout")
        if arm in TRAINED_ARMS:
            add("theta_displacement_mean", dilution, arm, float(g["theta"].mean()), len(g))
            add("theta_displacement_min", dilution, arm, float(g["theta"].min()), len(g),
                "0 means the optimiser never moved the socket — verdict row (i)")
        add("best_epoch_mean", dilution, arm, float(np.mean(g["best_epoch_int"])), len(g))
        add("epoch_budget_hits", dilution, arm,
            int((np.asarray(g["epochs_run_int"]) >= MAX_EPOCHS).sum()), len(g))
    rows.sort(key=lambda r: (r["quantity"], dilution_key(r["dilution"]), r["arm"]))
    return rows


# =====================================================================================
# 3. the contrast against a closed-form readout
# =====================================================================================

def ridge_contrast_rows(rows_a8: list[dict], dataset_seeds, seeds) -> list[dict]:
    """Delta_AB with arm B read out in CLOSED FORM instead of by Adam.

    The QELM literature trains a linear readout in closed form, while arm B trains its
    head with Adam on BCE, so identifying arm B with the QELM convention is an
    approximation. `ridge_accuracy` is on every frozen row, so the literal convention is
    measurable: acc(A) - ridge_acc(B), paired per (generator seed, training seed),
    through the same blocked estimator as everything else.
    """
    test = a8.accuracy_index(rows_a8, "test")
    lrs = a8.selected_lrs(rows_a8)

    # accuracy_index carries the whole row, so the ridge column travels with it.
    def ridge_series(arm, ansatz_level, dilution, lr_of):
        out = {}
        for dataset_seed in dataset_seeds:
            lr = lr_of(dataset_seed)
            if lr is None:
                continue
            for seed in seeds:
                key = (dataset_seed, arm, ansatz_level, dilution, None, seed, f"{float(lr):g}")
                row = test.get(key)
                if row is not None and row["ridge_accuracy"] not in ("", None):
                    out[a8.PairKey(dataset_seed, seed)] = float(row["ridge_accuracy"])
        return out

    out = []
    dilutions = sorted({r["dilution"] for r in rows_a8 if r["arm"] in ("A", "B")},
                       key=dilution_key)
    ansatz_levels = sorted({r["ansatz_level"] for r in rows_a8 if r["arm"] == "A"})
    for dilution in dilutions:
        for ansatz_level in ansatz_levels:
            def lr_of(dataset_seed, _d=dilution, _a=ansatz_level):
                return lrs["cell_lr"].get((dataset_seed, _d, _a))
            arm_a = a8.series(test, dataset_seeds=dataset_seeds, arm="A",
                              ansatz_level=ansatz_level, dilution=dilution,
                              lr_of=lr_of, seeds=seeds)
            arm_b_adam = a8.series(test, dataset_seeds=dataset_seeds, arm="B",
                                   ansatz_level=ansatz_level, dilution=dilution,
                                   lr_of=lr_of, seeds=seeds)
            arm_b_ridge = ridge_series("B", ansatz_level, dilution, lr_of)
            for label, right in (("adam", arm_b_adam), ("ridge", arm_b_ridge)):
                if not right:
                    continue
                pair = a8.paired_differences(arm_a, right, label=f"delta_AB_{label}")
                point = a8.estimate_blocked(pair)
                out.append({
                    "estimand": f"delta_AB_B_readout_{label}",
                    "dilution": dilution, "ansatz": ansatz_level, "n": point["n"],
                    "mean": point["mean"], "ci95_low": point.get("ci95_low"),
                    "ci95_high": point.get("ci95_high"), "sigma_delta": point["sd"],
                    "mde": point.get("mde"),
                    "p_sign_exact": point["p_sign_exact"],
                    "p_wilcoxon_exact": point["p_wilcoxon_exact"],
                })
    out.sort(key=lambda r: (dilution_key(r["dilution"]), r["ansatz"], r["estimand"]))
    return out


# =====================================================================================
# 4. the estimands of an off-axis probe
# =====================================================================================

def probe_estimand_rows(rows_a8: list[dict], dataset_seeds, seeds, dilution: str) -> list[dict]:
    """Every estimand of one off-axis dilution, through the main analysis' own blocked
    estimator.

    An off-axis probe is not a point of the declared axis and is never folded into the
    contract grid, which would make the finished main series read as incomplete. It is
    analysed here and reported beside the axis.

    delta_BE is included because acc(B) - acc(E) asks whether the frozen quantum socket
    beats no socket at all — an intermediate of the delta_AE decomposition that the main
    analysis never emits.
    """
    test = a8.accuracy_index(rows_a8, "test")
    lrs = a8.selected_lrs(rows_a8)
    ansatz_levels = sorted({r["ansatz_level"] for r in rows_a8 if r["arm"] == "A"})
    out = []
    for ansatz_level in ansatz_levels:
        def cell_lr(dataset_seed, _a=ansatz_level):
            return lrs["cell_lr"].get((dataset_seed, dilution, _a))

        def arm_e_lr(dataset_seed):
            return lrs["arm_e_lr"].get((dataset_seed, dilution))

        arms = {
            "A": a8.series(test, dataset_seeds=dataset_seeds, arm="A",
                           ansatz_level=ansatz_level, dilution=dilution,
                           lr_of=cell_lr, seeds=seeds),
            "B": a8.series(test, dataset_seeds=dataset_seeds, arm="B",
                           ansatz_level=ansatz_level, dilution=dilution,
                           lr_of=cell_lr, seeds=seeds),
            "E": a8.series(test, dataset_seeds=dataset_seeds, arm="E", ansatz_level="",
                           dilution=dilution, lr_of=arm_e_lr, seeds=seeds),
            "F": a8.series(test, dataset_seeds=dataset_seeds, arm="F",
                           ansatz_level=a8.PRODUCT_ANSATZ, dilution=dilution,
                           lr_of=cell_lr, seeds=seeds),
            "D_matched": a8.series(test, dataset_seeds=dataset_seeds, arm="D_matched",
                                   ansatz_level="", dilution=dilution,
                                   lr_of=cell_lr, seeds=seeds),
        }
        for name, (left, right) in {
            "delta_AB": ("A", "B"), "delta_AE": ("A", "E"), "delta_BE": ("B", "E"),
            "delta_AF": ("A", "F"), "delta_BD_matched": ("B", "D_matched"),
        }.items():
            if not arms[left] or not arms[right]:
                continue
            pair = a8.paired_differences(arms[left], arms[right], label=name)
            if not pair["differences"]:
                continue
            point = a8.estimate_blocked(pair)
            out.append({
                "estimand": name, "dilution": dilution, "ansatz": ansatz_level,
                "n": point["n"], "mean": point["mean"],
                "ci95_low": point.get("ci95_low"), "ci95_high": point.get("ci95_high"),
                "sigma_delta": point["sd"], "mde": point.get("mde"),
                "above_mde": "yes" if point["above_mde"] else "no",
                "ci95_excludes_zero": "yes" if point["ci95_excludes_zero"] else "no",
                "p_sign_exact": point["p_sign_exact"],
                "p_wilcoxon_exact": point["p_wilcoxon_exact"],
                "unpaired": len(pair["unpaired_left"]) + len(pair["unpaired_right"]),
            })
    return out


# =====================================================================================
# 5. how often a hidden ReLU unit is born dead
# =====================================================================================

def head_init_rows(dataset_seed: int, *, draws: int = 2000, widths=(2, 3, 4)) -> list[dict]:
    """The mechanism behind the h2 collapse.

    torch's default nn.Linear initialisation draws weight and bias from
    U(-1/sqrt(fan_in), +1/sqrt(fan_in)). At fan_in = 5 the bias reaches +-0.447 while the
    weighted sum has a standard deviation of about 0.13 on these features, so one unlucky
    bias draw pushes a whole unit below zero for nearly every input.

    Reported: the share of units born nearly dead, which does not depend on width, and the
    share of draws in which every unit is.
    """
    import torch
    from torch import nn

    name, out_dir = dataset_location(dataset_seed)
    X, _ = load_frozen(name, "train", out_dir=out_dir)
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    fan_in = X.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    weight_sd = bound / math.sqrt(3.0)
    sum_sd = math.sqrt(fan_in) * weight_sd * float(np.asarray(X).std())

    rows = [{
        "quantity": "bias_bound_over_weighted_sum_sd", "width": "", "value": bound / sum_sd,
        "note": f"bias reaches +-{bound:.4f}; std(w.x) = {sum_sd:.4f} on the frozen features",
    }, {
        "quantity": "he_variance_over_default_variance", "width": "",
        "value": (2.0 / fan_in) / weight_sd**2,
        "note": "He for ReLU against torch's legacy nn.Linear default",
    }]
    for width in widths:
        generator = torch.Generator()
        generator.manual_seed(derive("head_init_probe", width) % (2**63))
        dead_units = 0
        all_dead = 0
        for _ in range(draws):
            layer = nn.Linear(fan_in, width)
            _init_linear(layer, generator)
            with torch.no_grad():
                alive = (layer(X_t) > 0).float().mean(0).numpy()
            dead_units += int((alive < 0.05).sum())
            all_dead += int((alive < 0.05).all())
        rows.append({"quantity": "share_of_units_born_nearly_dead", "width": width,
                     "value": dead_units / (draws * width),
                     "note": f"{draws} draws; 'nearly dead' = positive on < 5 % of inputs"})
        rows.append({"quantity": "share_of_draws_with_every_unit_dead", "width": width,
                     "value": all_dead / draws,
                     "note": "this is what collapses the head to a constant classifier"})
    return rows


# =====================================================================================
# 6. the Fourier support arm D does and does not share
# =====================================================================================

def fourier_rows(R: int, n_qubits: int, levels=("L1", "L2", "product"), seeds=(1, 2, 3)) -> list[dict]:
    """Exact multivariate spectrum of <Z_i>(x), by DFT on a (2R+1)^n grid.

    The circuit output is a trigonometric polynomial of degree <= R per coordinate, so a
    (2R+1)-point DFT per axis is exact — no aliasing, no sampling error.

    Arm D draws Omega uniformly from the integer lattice |omega| <= R, so it shares the
    frequency support of arms A and B as a set, which is measured here. The distribution
    of spectral mass differs substantially, so delta_BD may only be read as "same support,
    same size", never as "same frequency distribution".
    """
    grid = 2.0 * np.pi * np.arange(2 * R + 1) / (2 * R + 1)
    X = np.array(list(itertools.product(*[grid] * n_qubits)))
    freqs = np.fft.fftfreq(2 * R + 1, d=1.0 / (2 * R + 1)).astype(int)
    rows = []
    for level in levels:
        circuit = build_socket_circuit(level, n_qubits, R)
        n_params = socket_param_count(n_qubits, R)
        mass = np.zeros((2 * R + 1,) * n_qubits)
        for seed in seeds:
            theta = initial_theta(ansatz=level, R=R, seed=seed, n_params=n_params)
            Z = z_expectation_batch(circuit, theta, X, n_qubits=n_qubits)
            for i in range(n_qubits):
                spectrum = np.fft.fftn(Z[:, i].reshape((2 * R + 1,) * n_qubits)) / X.shape[0]
                mass += np.abs(spectrum) ** 2
        live = mass > 1e-12 * mass.max()
        present = np.argwhere(live)
        omega = np.array([[freqs[p[j]] for j in range(n_qubits)] for p in present])
        active = (omega != 0).sum(axis=1)
        total = mass.sum()
        rows.append({"ansatz": level, "quantity": "lattice_points_with_mass",
                     "value": int(live.sum()), "of": int(mass.size), "note": ""})
        for k in range(n_qubits + 1):
            sel = active == k
            if not sel.any():
                continue
            rows.append({"ansatz": level, "quantity": f"mass_share_active_coords_{k}",
                         "value": float(mass[live][sel].sum() / total),
                         "of": int(sel.sum()),
                         "note": "share of spectral mass on terms with k active coordinates"})
    # What a uniform draw over the same lattice puts on k active coordinates, for contrast.
    for k in range(n_qubits + 1):
        p = math.comb(n_qubits, k) * (2 * R / (2 * R + 1)) ** k * (1 / (2 * R + 1)) ** (n_qubits - k)
        rows.append({"ansatz": "arm D (uniform on the same lattice)",
                     "quantity": f"mass_share_active_coords_{k}", "value": p, "of": "",
                     "note": "probability that a drawn Omega row has k active coordinates"})
    return rows


# =====================================================================================
# 7. numerical rank of the data
# =====================================================================================

def data_rank_rows(dataset_seeds) -> list[dict]:
    """Singular values of the centred RAW training features."""
    rows = []
    for dataset_seed in dataset_seeds:
        name, out_dir = dataset_location(dataset_seed)
        X, _ = load_frozen(name, "train", out_dir=out_dir, raw=True)
        X = np.asarray(X, dtype=np.float64)
        s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
        rows.append({
            "dataset_seed": dataset_seed, "n_rows": X.shape[0], "n_features": X.shape[1],
            "singular_values_top6": " ".join(f"{v:.4e}" for v in s[:6]),
            "s5_over_s0": float(s[5] / s[0]),
            "numerical_rank_tol_1e-8": int((s > 1e-8 * s[0]).sum()),
        })
    return rows


# =====================================================================================
# 8. what the shared-lr rule costs
# =====================================================================================

def shared_lr_rows(lr_tables) -> list[dict]:
    """The one lr is chosen on the mean of arms A and B, while the estimand is A - B.

    That makes the hyperparameter selected on the same quantity the estimand is built
    from. The lr tables carry validation and test accuracy at every grid point for the
    selection seeds, so the effect is measurable: Delta_AB at the shared lr against
    Delta_AB with each arm at its own best validation lr.
    """
    rows = []
    for path in lr_tables:
        table = pd.read_csv(path, float_precision="round_trip")
        table = table[table["in_selection"].astype(str).str.lower().isin(("true", "1"))]
        for (dataset_seed, dilution, ansatz), g in table.groupby(
                ["dataset_seed", "dilution", "ansatz_level"]):
            val = g.pivot_table(index="lr", columns="arm", values="val_accuracy", aggfunc="mean")
            test = g.pivot_table(index="lr", columns="arm", values="test_accuracy", aggfunc="mean")
            if not {"A", "B"} <= set(val.columns):
                continue
            def argmax(column):  # ties to the lower lr, as everywhere else
                return column.index[np.lexsort((column.index, -column.values))[0]]
            shared = argmax(val.mean(axis=1))
            rows.append({
                "dataset_seed": int(dataset_seed), "dilution": dilution, "ansatz": ansatz,
                "lr_shared": float(shared),
                "lr_best_A": float(argmax(val["A"])), "lr_best_B": float(argmax(val["B"])),
                "delta_AB_at_shared_lr": float(test.loc[shared, "A"] - test.loc[shared, "B"]),
                "delta_AB_each_own_lr": float(
                    test.loc[argmax(val["A"]), "A"] - test.loc[argmax(val["B"]), "B"]),
            })
    for row in rows:
        row["bias_of_the_shared_rule"] = (
            row["delta_AB_at_shared_lr"] - row["delta_AB_each_own_lr"])
    return rows


# =====================================================================================
# 9. K(tau) — how many gates actually moved
# =====================================================================================

def displacement_rows(weights_dirs, taus=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0)) -> list[dict]:
    """The number of gates that moved by more than tau, from the saved weights.

    theta_displacement is a norm divided by sqrt(P) and only answers whether the optimiser
    moved. Reads whatever weight files exist; cells run before weights were saved cannot appear.
    """
    rows = []
    for weights_dir in weights_dirs:
        for path in sorted(Path(weights_dir).rglob("*.npz")):
            record = a7.read_weights(path)
            if "socket_theta" not in record or "socket_theta_init" not in record:
                continue
            moved = np.abs(np.asarray(record["socket_theta"])
                           - np.asarray(record["socket_theta_init"]))
            if not moved.any():
                continue  # a frozen arm: nothing moved, by construction
            meta = record["meta"]
            row = {
                "source": str(Path(weights_dir).parent.name),
                "dataset_seed": meta.get("dataset_seed"), "arm": meta.get("arm"),
                "ansatz": meta.get("ansatz_level"), "dilution": meta.get("dilution"),
                "seed": meta.get("seed"), "n_params": int(moved.size),
                "norm_over_sqrt_P": float(np.linalg.norm(moved) / math.sqrt(moved.size)),
                "mean_abs": float(moved.mean()), "median_abs": float(np.median(moved)),
                "max_abs": float(moved.max()),
            }
            for tau in taus:
                row[f"K_gt_{tau}"] = int((moved > tau).sum())
            rows.append(row)
    return rows


# =====================================================================================
# driver
# =====================================================================================

def dump(out_dir: Path, name: str, rows: list[dict]) -> Path:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / name
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  {name:24s} {len(rows):4d} rows")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, required=True,
                        help="the combined A7 results CSV of the main series")
    parser.add_argument("--probe", type=Path, action="append", default=[],
                        help="an off-axis probe results CSV; repeatable")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/supplementary"))
    parser.add_argument("--lr-table", type=Path, action="append", default=[],
                        help="A7 lr tables for the shared-lr bias; repeatable")
    parser.add_argument("--weights-dir", type=Path, action="append", default=[],
                        help="a weights/ directory for K(tau); repeatable")
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip the Fourier spectrum and the head-init Monte Carlo")
    args = parser.parse_args()

    print("=" * 86)
    print("Supplementary measurements. Estimators are IMPORTED from run_a8_analysis.")
    print("=" * 86)

    main_rows = a8.load_rows(args.main)
    all_rows = list(main_rows)
    for probe in args.probe:
        all_rows.extend(a8.load_rows(probe))
    all_rows, duplicated = _deduplicate(all_rows)
    if duplicated:
        print(f"\n{duplicated} row(s) appear in more than one input CSV and were counted")
        print("ONCE. Identical accuracies, so this is a re-export, not two measurements;")
        print("a disagreement would have stopped the run in _deduplicate.")

    dataset_seeds = sorted({r["dataset_seed"] for r in all_rows})
    seeds = sorted({r["seed_int"] for r in all_rows if r["seed_int"] is not None})
    shares = class_shares(dataset_seeds)
    test = mark_degenerate(frame_of([r for r in all_rows if r["split"] == "test"]), shares)
    print(f"\ninput      {len(all_rows)} rows, {len(test)} of them test")
    print(f"datasets   {dataset_seeds}   seeds {seeds}")
    print(f"dilutions  {sorted(set(test['dilution']), key=dilution_key)}\n")

    dump(args.out_dir, "degeneracy.csv", degeneracy_rows(test))
    dump(args.out_dir, "diagnostics.csv", diagnostic_rows(test))
    dump(args.out_dir, "ridge_contrast.csv",
         ridge_contrast_rows(main_rows, dataset_seeds, seeds))

    probe_rows = []
    for probe in args.probe:
        rows = a8.load_rows(probe)
        for dilution in sorted({r["dilution"] for r in rows if r["arm"] == "A"},
                               key=dilution_key):
            probe_rows.extend(probe_estimand_rows(
                rows, sorted({r["dataset_seed"] for r in rows}),
                sorted({r["seed_int"] for r in rows if r["seed_int"] is not None}), dilution))
    if probe_rows:
        dump(args.out_dir, "probe_estimands.csv", probe_rows)

    if args.lr_table:
        dump(args.out_dir, "shared_lr_bias.csv", shared_lr_rows(args.lr_table))
    if args.weights_dir:
        dump(args.out_dir, "displacement.csv", displacement_rows(args.weights_dir))

    dump(args.out_dir, "data_rank.csv", data_rank_rows(dataset_seeds))
    if not args.skip_slow:
        dump(args.out_dir, "head_init.csv", head_init_rows(dataset_seeds[0]))
        dump(args.out_dir, "fourier_support.csv",
             fourier_rows(a7.R_CONTRACT, a7.DEFAULT_N_QUBITS))

    print(f"\ntables in {args.out_dir / 'tables'}")


if __name__ == "__main__":
    main()
