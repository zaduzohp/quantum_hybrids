"""The dataset sweep that closes the choice of production cell and seed.

Two stages in one pass, because stage 2 needs the frozen files stage 1 produced.

  stage 1, cheap screen, no training loop. Per cell x generator seed, on the frozen
      PCA-k representation: evr1 (-> G2), acc(strong) = SVC(rbf) over G1_SVC_GRID
      selected on val and scored on test, acc(logreg), their difference as the ceiling,
      BW = logreg(PCA-10) - logreg(PCA-5), plus ceilings at k=8 and k=10 as context.

  stage 2, binding G1, only where G2 passed AND acc(strong) is in [0.65, 0.90]. The
      floor is the real arm E — identity socket, linear head, the full train_model loop,
      lr from G1_LR_GRID on val. Both readings recorded: the binding one on PCA-5, and
      "strong model on the full pre-PCA features", reported and never gating.

Nothing here generates data or runs PCA on its own: generate_and_freeze is the only path
to a frozen dataset and check_g1_headroom / check_g2_effective_dim the only path to a
verdict. Generator arguments always go by name — `offset` sits before `noise` in
generate_two_curves.

One CSV row per cell x generator seed, plus the JSON of every gate record. The frozen
.npz files are scratch and deleted once a cell is done (manifests kept); they deliberately
do NOT go to data/.

  python scripts/run_dataset_sweep.py --out-dir outputs/a3_sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

# Before numpy: stage 2 runs the real training loop, and a multi-threaded BLAS makes the
# arm-E floor of G1 depend on the machine rather than on the dataset.
from qsocket.core import pin_blas_threads

pin_blas_threads()


from qsocket.datasets import N_SAMPLES_TOTAL, generate_and_freeze, load_splits
from qsocket.gates import (
    G1_LR_GRID,
    G1_STRONG_ACCURACY_BAND,
    G1_SVC_GRID,
    check_g1_headroom,
    check_g2_effective_dim,
    make_arm_e_linear_floor_model,
    make_svc_strong_model,
)

# --- the grid ------------------------------------------------------------------------

# Generator seeds; the first three match the earlier replication run, so the
# overlapping cells can be read against it.
DATASET_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)

# PCA widths. k=5 is the frozen project width and the only one any gate is stated on;
# k=8 and k=10 are context, and k=10 is the second half of the bandwidth predictor.
K_VALUES: tuple[int, ...] = (5, 8, 10)
K_GATING = 5

# two_curves: `noise` is its G2 knob — the two curves are intrinsically one-dimensional,
# so evr1 stays high until noise lifts it — while `degree` and `offset` move difficulty.
# hidden_manifold has manifold_dimension and nothing else.
GRIDS: dict[str, dict[str, tuple]] = {
    "two_curves": {
        "n_features": (10, 20),
        "degree": (3, 5, 10),
        "noise": (0.01, 0.05, 0.1, 0.2),
        "offset": (0.05, 0.1),
    },
    "hidden_manifold": {
        "n_features": (10, 20),
        "manifold_dimension": (3, 6, 10),
    },
}

# Column order of the cell id and of the frozen name, so a cell always prints and hashes
# the same way.
CELL_KEYS: dict[str, tuple[str, ...]] = {
    "two_curves": ("n_features", "degree", "offset", "noise"),
    "hidden_manifold": ("n_features", "manifold_dimension"),
}

# Training seeds arm E is averaged over inside G1. Not a contract value: one seed leaves
# the floor with an uncertainty comparable to the threshold itself, and three cut that
# spread by ~sqrt(3) at three times the (small) cost of arm E.
ARM_E_SEEDS: tuple[int, ...] = (1, 2, 3)

CSV_COLUMNS: tuple[str, ...] = (
    # cell identity
    "dataset", "n_features", "degree", "offset", "noise", "manifold_dimension",
    "dataset_seed", "cell_id", "frozen_name_k5", "dataset_hash_k5", "pca_hash_k5",
    # stage 1 — G2
    "evr1_k5", "g2_passed", "shares_of_retained_k5",
    # stage 1 — the three scales at the gating width
    "acc_strong_k5", "strong_C_k5", "strong_gamma_k5", "acc_logreg_k5", "ceiling_k5",
    # stage 1 — context widths and the bandwidth predictor
    "acc_strong_k8", "acc_logreg_k8", "ceiling_k8",
    "acc_strong_k10", "acc_logreg_k10", "ceiling_k10", "bw",
    # stage 1 verdict
    "strong_in_band", "stage1_pass",
    # stage 2 — binding G1 on PCA-5
    "stage2_run", "acc_arm_e", "arm_e_lr_selected", "arm_e_val_accuracy",
    "g1_headroom", "g1_passed", "g1_binding",
    # stage 2 — reported, never gating
    "acc_strong_full_features", "g1_headroom_full_features",
    # cost
    "wall_seconds",
)


def cells_of(dataset: str) -> list[dict]:
    """Every combination of the knobs of `dataset`, in declared key order."""
    keys = CELL_KEYS[dataset]
    return [dict(zip(keys, values)) for values in itertools.product(*(GRIDS[dataset][k] for k in keys))]


def cell_id(dataset: str, kwargs: dict) -> str:
    parts = [f"{key}{_short(kwargs[key])}" for key in CELL_KEYS[dataset]]
    return f"{dataset}_" + "_".join(parts)


def _short(value) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    return str(value)


# --- the cheap models ---------------------------------------------------------------


def logreg_accuracy(splits: dict) -> float:
    """LogisticRegression(max_iter=5000) on the frozen representation, scored on test.

    No hyperparameter selection, hence no selection split: this is the linear reference
    the ceiling is measured against, the gap a nonlinear socket could at most close. Same
    estimator as the earlier probes, so the numbers stay comparable.
    """
    from sklearn.linear_model import LogisticRegression

    X_tr, y_tr = splits["train"]
    X_te, y_te = splits["test"]
    return float(LogisticRegression(max_iter=5000).fit(X_tr, y_tr).score(X_te, y_te))


def cached(record: dict):
    """Wrap an already-computed model record so check_g1_headroom can take it.

    SVC(rbf) on fixed data with fixed hyperparameters is deterministic, so re-fitting the
    nine-cell grid inside the gate would return exactly the record stage 1 produced.
    Passing the cached one keeps the gate as the single place the verdict is formed
    without paying for the grid twice.
    """
    return lambda dataset: record


# --- one cell x one generator seed --------------------------------------------------


def run_cell(
    dataset: str,
    kwargs: dict,
    *,
    dataset_seed: int,
    freeze_dir: Path,
    stage1_only: bool,
    arm_e_seeds: tuple[int, ...],
    keep_frozen: bool,
) -> dict:
    started = time.perf_counter()
    identifier = cell_id(dataset, kwargs)
    row: dict = {column: "" for column in CSV_COLUMNS}
    row.update({"dataset": dataset, "dataset_seed": dataset_seed, "cell_id": identifier})
    row.update({key: kwargs[key] for key in CELL_KEYS[dataset]})
    detail: dict = {"cell_id": identifier, "dataset_seed": dataset_seed,
                    "generator_kwargs": dict(kwargs)}

    # Freeze once per PCA width rather than truncating a single k=10 fit: scikit-learn's
    # svd_solver="auto" can pick the randomized solver, whose components need not agree
    # between n_components=5 and 10.
    manifests: dict[int, dict] = {}
    splits: dict[int, dict] = {}
    frozen_names: dict[int, str] = {}
    for k in K_VALUES:
        frozen_names[k] = f"{identifier}_seed{dataset_seed}_k{k}"
        manifests[k] = generate_and_freeze(
            dataset,
            n_samples=N_SAMPLES_TOTAL,
            generator_kwargs=kwargs,
            dataset_seed=dataset_seed,
            out_dir=freeze_dir,
            frozen_name=frozen_names[k],
            n_components=k,
        )
        splits[k] = load_splits(frozen_names[k], out_dir=freeze_dir)

    row["frozen_name_k5"] = frozen_names[K_GATING]
    row["dataset_hash_k5"] = manifests[K_GATING]["dataset_hash"]
    row["pca_hash_k5"] = manifests[K_GATING]["pca_hash"]

    # G2 — binding, computed from the k=5 manifest and renormalised over the retained
    # components by check_g2_effective_dim itself.
    g2 = check_g2_effective_dim(manifests[K_GATING])
    detail["G2"] = g2
    row["evr1_k5"] = round(g2["top_share"], 6)
    row["g2_passed"] = g2["passed"]
    row["shares_of_retained_k5"] = json.dumps([round(v, 5) for v in g2["share_of_retained"]])

    # The three scales, at every width.
    strong_records: dict[int, dict] = {}
    for k in K_VALUES:
        strong_records[k] = make_svc_strong_model()(splits[k])
        logreg = logreg_accuracy(splits[k])
        row[f"acc_strong_k{k}"] = round(strong_records[k]["accuracy"], 6)
        row[f"acc_logreg_k{k}"] = round(logreg, 6)
        row[f"ceiling_k{k}"] = round(strong_records[k]["accuracy"] - logreg, 6)
    row["strong_C_k5"] = strong_records[K_GATING]["selected"]["C"]
    row["strong_gamma_k5"] = strong_records[K_GATING]["selected"]["gamma"]
    row["bw"] = round(float(row["acc_logreg_k10"]) - float(row["acc_logreg_k5"]), 6)
    detail["strong"] = {str(k): strong_records[k] for k in K_VALUES}

    low, high = G1_STRONG_ACCURACY_BAND
    in_band = low <= strong_records[K_GATING]["accuracy"] <= high
    row["strong_in_band"] = in_band
    row["stage1_pass"] = bool(g2["passed"] and in_band)

    # Stage 2 — the binding G1, with the real arm E.
    row["stage2_run"] = bool(row["stage1_pass"] and not stage1_only)
    if row["stage2_run"]:
        g1 = check_g1_headroom(
            splits[K_GATING],
            strong_model=cached(strong_records[K_GATING]),
            floor_model=make_arm_e_linear_floor_model(seeds=arm_e_seeds),
        )
        detail["G1"] = g1
        row["acc_arm_e"] = round(g1["floor"]["accuracy"], 6)
        row["arm_e_lr_selected"] = g1["floor"]["lr_selected"]
        row["arm_e_val_accuracy"] = round(g1["floor"]["val_accuracy"], 6)
        row["g1_headroom"] = round(g1["headroom"], 6)
        row["g1_passed"] = g1["passed"]
        row["g1_binding"] = g1["binding"]

        # Reported, never gating: the same strong model on the full pre-PCA features,
        # against the same arm-E floor, which stays on PCA-5 as the contract
        # representation.
        raw = load_splits(frozen_names[K_GATING], out_dir=freeze_dir, raw=True)
        full = make_svc_strong_model()(raw)
        detail["strong_on_full_features"] = full
        row["acc_strong_full_features"] = round(full["accuracy"], 6)
        row["g1_headroom_full_features"] = round(
            full["accuracy"] - g1["floor"]["accuracy"], 6
        )

    if not keep_frozen:
        # The .npz files are scratch; the manifests stay as the provenance of the hashes.
        for k in K_VALUES:
            (freeze_dir / f"{frozen_names[k]}.npz").unlink(missing_ok=True)

    row["wall_seconds"] = round(time.perf_counter() - started, 2)
    detail["row"] = dict(row)
    return {"row": row, "detail": detail}


# --- driver -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=sorted(GRIDS), choices=sorted(GRIDS))
    parser.add_argument("--dataset-seeds", nargs="+", type=int, default=list(DATASET_SEEDS))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/a3_sweep"))
    parser.add_argument("--freeze-dir", type=Path, default=None,
                        help="where frozen datasets are written (default: <out-dir>/frozen)")
    parser.add_argument("--keep-frozen", action="store_true",
                        help="keep the .npz files instead of deleting them per cell")
    parser.add_argument("--stage1-only", action="store_true",
                        help="skip the binding G1 (no training loop runs)")
    parser.add_argument("--arm-e-seeds", nargs="+", type=int, default=list(ARM_E_SEEDS))
    parser.add_argument("--limit", type=int, default=None, help="first N cells, for a smoke run")
    parser.add_argument("--skip", type=int, default=0,
                        help="skip the first N cell x seed runs, to resume an interrupted sweep. "
                             "The job order is deterministic (datasets in the given order, then "
                             "cells in CELL_KEYS order, then seeds), so the resumed CSV continues "
                             "the interrupted one and the two concatenate")
    args = parser.parse_args()

    freeze_dir = args.freeze_dir or (args.out_dir / "frozen")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    freeze_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    jobs = [
        (dataset, kwargs, seed)
        for dataset in args.datasets
        for kwargs in cells_of(dataset)
        for seed in args.dataset_seeds
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    total_jobs = len(jobs)
    if args.skip:
        jobs = jobs[args.skip :]

    print(f"# the dataset sweep dataset sweep, {len(jobs)} cell x seed runs"
          + (f" (resumed: skipped the first {args.skip} of {total_jobs})" if args.skip else ""))
    print(f"# split 4200/600/1200 out of {N_SAMPLES_TOTAL}, PCA widths {list(K_VALUES)}, "
          f"gating width k={K_GATING}")
    print(f"# stage 1: G2 + SVC(rbf) grid {G1_SVC_GRID} (selected on val, scored on test) "
          f"+ LogisticRegression(max_iter=5000)")
    print(f"# stage 1 filter into stage 2: G2 passed AND acc(strong) in "
          f"{list(G1_STRONG_ACCURACY_BAND)}")
    print(f"# stage 2: binding G1, arm E = identity socket + linear head via "
          f"training.train_model, lr from {list(G1_LR_GRID)} on val, seeds {args.arm_e_seeds}")
    print("# G1_SVC_GRID is 3x3; the earlier replication used 4x4 (the 3x3 rule) — numbers are "
          "NOT directly comparable to FINDINGS section 4\n")

    csv_path = args.out_dir / f"a3_sweep_{stamp}.csv"
    json_path = args.out_dir / f"a3_sweep_{stamp}.json"
    details = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for index, (dataset, kwargs, seed) in enumerate(jobs, start=args.skip + 1):
            result = run_cell(
                dataset,
                kwargs,
                dataset_seed=seed,
                freeze_dir=freeze_dir,
                stage1_only=args.stage1_only,
                arm_e_seeds=tuple(args.arm_e_seeds),
                keep_frozen=args.keep_frozen,
            )
            writer.writerow(result["row"])
            handle.flush()
            details.append(result["detail"])
            _print_row(index, total_jobs, result["row"])

    json_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "dataset_seeds": list(args.dataset_seeds),
                "k_values": list(K_VALUES),
                "gating_k": K_GATING,
                "svc_grid": {k: list(v) for k, v in G1_SVC_GRID.items()},
                "lr_grid": list(G1_LR_GRID),
                "arm_e_seeds": list(args.arm_e_seeds),
                "stage1_only": bool(args.stage1_only),
                "cells": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {csv_path}\nwritten: {json_path}")


def _print_row(index: int, total: int, row: dict) -> None:
    stage2 = (
        f" E {row['acc_arm_e']:.3f} head {float(row['g1_headroom']):+.3f} "
        f"G1 {'PASS' if row['g1_passed'] else 'FAIL'} full {row['acc_strong_full_features']:.3f}"
        if row["stage2_run"]
        else ""
    )
    print(
        f"[{index:>3}/{total}] {row['cell_id']} S={row['dataset_seed']:<3} "
        f"evr1 {row['evr1_k5']:.3f} G2 {'PASS' if row['g2_passed'] else 'FAIL'} "
        f"strong {row['acc_strong_k5']:.3f} logreg {row['acc_logreg_k5']:.3f} "
        f"ceil {row['ceiling_k5']:+.3f} BW {row['bw']:+.3f} "
        f"s1 {'PASS' if row['stage1_pass'] else 'FAIL'}{stage2} "
        f"[{row['wall_seconds']:.1f}s]"
    )


if __name__ == "__main__":
    main()
