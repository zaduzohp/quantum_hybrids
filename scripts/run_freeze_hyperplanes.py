"""Scan five generator seeds of the `hyperplanes` cell and freeze the production dataset.

Declared above the measurement, so the outcome cannot bias it: n_features = 20 for
continuity with the frozen two_curves; cell hyperplanes(n_features=20, n_hyperplanes=3,
dim_hyperplanes=5); seed rule = the smallest passing seed of {11, 22, 33, 44, 55}, the
same rule used for two_curves.

Per seed, through the contract chain (generate_and_freeze -> load_splits -> gates):

    evr1                  share of the top retained component     G2:  <= 0.80
    acc(strong)           SVC(rbf) over G1_SVC_GRID, on val       G1a: in [0.65, 0.90]
    acc(E, linear head)   contract arm E, seeds 1-3               G1b: headroom >= 0.05
    gain(PC1) vs gain(5)  SVC - logreg at k=1 and k=5             reported, not gating
    class balance         > 5 pp off 50/50 and the cell is out
    acc(strong MLP)       best of {mlp42, mlp4285}                diagnostic only

Only the SVM ceiling is the contract verdict. An MLP head beats the tuned RBF-SVM by ~11
points on this parity task, so acc(strong) understates the ceiling; which reading to use
is still open. No gate constant is written here.

The picked seed is frozen into data/ and reproduced from its own manifest into a temporary
directory; all three hashes must come back identical. The frozen two_curves is never
touched.

    caffeinate -dimsu .venv/bin/python scripts/run_freeze_hyperplanes.py [--scan-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Has to happen before numpy/torch import their BLAS. One definition of the list, shared
# with every other driver — a local copy is how one of them ends up missing a pool.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsocket.core import pin_blas_threads

pin_blas_threads()

import numpy as np
import torch

torch.set_num_threads(1)

from qsocket.datasets import (
    DEFAULT_DATA_DIR,
    N_COMPONENTS,
    N_SAMPLES_TOTAL,
    SPLIT_NAMES,
    dataset_paths,
    generate_and_freeze,
    load_manifest,
    load_splits,
)
from qsocket.gates import (
    G1_LR_GRID,
    G1_MIN_HEADROOM,
    G1_STRONG_ACCURACY_BAND,
    G1_SVC_GRID,
    G2_MAX_COMPONENT_SHARE,
    check_g1_headroom,
    check_g2_effective_dim,
    make_arm_e_linear_floor_model,
    make_svc_strong_model,
)
from qsocket.head import HEAD_PARAM_COUNTS, make_head
from qsocket.socket import make_socket
from qsocket.training import TrainConfig, train_model
from qsocket.vendored.metrics_cls import accuracy_from_z

# --- candidate configuration, declared before the measurement -----------------------

DATASET = "hyperplanes"

# Arguments by name: generate_hyperplanes_parity takes n_hyperplanes before
# dim_hyperplanes, both small ints, so a positional call could silently swap them.
PRODUCTION_GENERATOR_KWARGS: dict = {
    "n_features": 20,
    "n_hyperplanes": 3,
    "dim_hyperplanes": 5,
}

# n_features = 20, for continuity with the frozen two_curves. Declared before the
# measurement and independent of it: nf=10 and nf=20 both pass everything.
CELL_RULE = "n_features = 20, for continuity with the frozen two_curves (also n_features = 20)"
SCREEN_ALTERNATIVE_CELL: dict = {"n_features": 10, "n_hyperplanes": 3, "dim_hyperplanes": 5}

# The smallest seed of the grid, the same rule used for two_curves.
DATASET_SEED_GRID: tuple[int, ...] = (11, 22, 33, 44, 55)
PRODUCTION_DATASET_SEED = min(DATASET_SEED_GRID)
SEED_RULE = "the smallest seed of the grid"

# On-disk name in data/: generator, every knob, seed. PCA width is not in the name
# because it is frozen at 5.
PRODUCTION_FROZEN_NAME = (
    "hyperplanes_n_features20_n_hyperplanes3_dim_hyperplanes5_seed11"
)

# The arm-E training seeds the sweep and the earlier freeze used, not the ten seeds of
# the main series.
ARM_E_SEEDS: tuple[int, ...] = (1, 2, 3)

# Shape-of-the-ceiling comparison. Declared here, never gating.
G2_PRIME_FACTOR = 0.7

# Dilution levels used here as the second ceiling reading only.
MLP_DILUTIONS: tuple[str, ...] = ("mlp42", "mlp4285")

# More than 5 pp off 50/50 on any split and the cell is out.
BALANCE_TOLERANCE_PP = 5.0

CSV_COLUMNS: tuple[str, ...] = (
    "dataset_seed", "n_features", "n_hyperplanes", "dim_hyperplanes",
    "dataset_hash", "pca_hash", "file_sha256",
    # G2
    "evr1", "g2_passed", "shares_of_retained", "total_variance_explained",
    # G1 (binding)
    "acc_strong_svm", "strong_C", "strong_gamma", "strong_in_band",
    "acc_e_linear", "e_linear_lr_selected", "e_linear_val_accuracy",
    "g1_headroom", "g1_margin", "g1_passed", "g1_binding",
    # second ceiling, diagnostic for the ceiling decision
    "acc_e_mlp42", "mlp42_lr_selected", "acc_e_mlp4285", "mlp4285_lr_selected",
    "acc_strong_mlp", "strong_mlp_which", "headroom_vs_mlp", "strong_mlp_in_band",
    # G2' (NOT a gate)
    "acc_svc_pc1", "acc_logreg_pc1", "gain_pc1",
    "acc_svc_all5", "acc_logreg_all5", "gain_all5",
    "g2prime_threshold", "g2prime_passed",
    # class balance per split
    "frac_pos_train", "frac_pos_val", "frac_pos_test",
    "dev_pp_train", "dev_pp_val", "dev_pp_test", "balance_ok",
    # overall
    "gates_passed", "wall_seconds",
)


# --- reference models the scan needs beyond the gate helpers ------------------------


def logreg_accuracy(splits: dict, *, n_components: int) -> float:
    """LogisticRegression(max_iter=5000) on the leading `n_components`, scored on test.

    Same estimator and same absence of hyperparameter selection as run_dataset_sweep and
    probe_hyperplanes, so gain(PC1) and gain(all5) are comparable with the cell screen numbers.
    """
    from sklearn.linear_model import LogisticRegression

    X_tr, y_tr = splits["train"]
    X_te, y_te = splits["test"]
    model = LogisticRegression(max_iter=5000).fit(X_tr[:, :n_components], y_tr)
    return float(model.score(X_te[:, :n_components], y_te))


def svc_accuracy(splits: dict, *, n_components: int) -> dict:
    """The gate's own strong model restricted to a column slice — same selection rule at
    every width, so the two gains differ only in how many components are visible."""
    sliced = {split: (X[:, :n_components], y) for split, (X, y) in splits.items()}
    return make_svc_strong_model(grid=G1_SVC_GRID)(sliced)


def arm_e_with_head(splits: dict, *, dilution: str, lr_grid=G1_LR_GRID, seeds=ARM_E_SEEDS) -> dict:
    """Arm E (identity socket + `dilution` head) through the contract training loop.

    Same construction as gates.make_arm_e_linear_floor_model — same identity socket, same
    training.train_model, same per-cell lr selection with ties to the lower lr — with the
    head taken as an argument. It lives here rather than in gates.py because with an MLP
    head this is not contract arm E: it is the second ceiling reading, and
    `is_contract_arm_e` is deliberately absent so it can never enter a binding verdict.

    tests/test_dataset_freeze_hyperplanes.py asserts that at dilution="linear" this
    reproduces the gate helper's number exactly, so the MLP readings are known to come off
    the same path as the binding one.
    """
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]
    X_te_t = torch.as_tensor(np.asarray(X_te), dtype=torch.float32)

    started = time.perf_counter()
    table = []
    for lr in lr_grid:
        for seed in seeds:
            socket = make_socket("identity", R=None, ansatz="L1", trainable=False, seed=seed)
            head = make_head(dilution, seed=seed)
            result = train_model(
                socket, head, X_tr, y_tr, X_val, y_val, cfg=TrainConfig(lr=lr), seed=seed
            )
            with torch.no_grad():
                logits = head(socket(X_te_t)).reshape(-1).numpy()
            table.append(
                {
                    "lr": float(lr),
                    "seed": int(seed),
                    "val_accuracy": float(result.val_accuracy),
                    "test_accuracy": accuracy_from_z(
                        np.asarray(logits).reshape(-1), np.asarray(y_te).reshape(-1)
                    ),
                    "train_accuracy": float(result.train_accuracy),
                    "best_epoch": int(result.best_epoch),
                    "epochs_run": int(result.epochs_run),
                }
            )

    per_lr = {lr: [r for r in table if r["lr"] == lr] for lr in (float(v) for v in lr_grid)}
    mean_val = {lr: float(np.mean([r["val_accuracy"] for r in rows])) for lr, rows in per_lr.items()}
    best_lr = max(mean_val, key=lambda lr: (mean_val[lr], -lr))  # ties to the LOWER lr

    return {
        "accuracy": float(np.mean([r["test_accuracy"] for r in per_lr[best_lr]])),
        "label": f"arm E: identity socket + {dilution} head, training.train_model",
        "dilution": dilution,
        "head_params": HEAD_PARAM_COUNTS[dilution],
        "is_contract_arm_e": dilution == "linear",
        "lr_selected": float(best_lr),
        "lr_grid": [float(v) for v in lr_grid],
        "val_accuracy": mean_val[best_lr],
        "mean_val_accuracy_per_lr": {str(lr): v for lr, v in mean_val.items()},
        "seeds": [int(s) for s in seeds],
        "runs": table,
        "wall_seconds": time.perf_counter() - started,
    }


# --- one generator seed --------------------------------------------------------------


def frozen_name_for(seed: int, kwargs: dict) -> str:
    knobs = "_".join(f"{key}{kwargs[key]}" for key in ("n_features", "n_hyperplanes", "dim_hyperplanes"))
    return f"{DATASET}_{knobs}_seed{seed}"


def scan_seed(
    seed: int, *, kwargs: dict, freeze_dir: Path, keep_frozen: bool = False
) -> tuple[dict, dict]:
    started = time.perf_counter()
    name = frozen_name_for(seed, kwargs)

    manifest = generate_and_freeze(
        DATASET,
        n_samples=N_SAMPLES_TOTAL,
        generator_kwargs=dict(kwargs),
        dataset_seed=seed,
        out_dir=freeze_dir,
        frozen_name=name,
        n_components=N_COMPONENTS,
    )
    splits = load_splits(name, out_dir=freeze_dir)

    # G2 through the gate function, not recomputed here.
    g2 = check_g2_effective_dim(manifest)

    # G1, binding: SVC strong model vs contract arm E with a linear head, both on PCA-5.
    strong_all5 = svc_accuracy(splits, n_components=5)
    e_linear = make_arm_e_linear_floor_model(lr_grid=G1_LR_GRID, seeds=ARM_E_SEEDS)(splits)
    g1 = check_g1_headroom(
        splits,
        strong_model=lambda _d, record=strong_all5: record,
        floor_model=lambda _d, record=e_linear: record,
    )

    # Second ceiling reading, diagnostic only: the same features, an MLP head.
    mlp = {d: arm_e_with_head(splits, dilution=d) for d in MLP_DILUTIONS}
    best_mlp_name = max(MLP_DILUTIONS, key=lambda d: mlp[d]["accuracy"])
    acc_strong_mlp = mlp[best_mlp_name]["accuracy"]
    low, high = G1_STRONG_ACCURACY_BAND

    # The shape of the ceiling. Not a gate.
    logreg_all5 = logreg_accuracy(splits, n_components=5)
    strong_pc1 = svc_accuracy(splits, n_components=1)
    logreg_pc1 = logreg_accuracy(splits, n_components=1)
    gain_all5 = strong_all5["accuracy"] - logreg_all5
    gain_pc1 = strong_pc1["accuracy"] - logreg_pc1
    g2prime_threshold = G2_PRIME_FACTOR * gain_all5
    # A non-positive gain(all5) leaves no ceiling to be multi-dimensional about, so
    # "passed" there would be a false positive.
    g2prime_passed = bool(gain_all5 > 0 and gain_pc1 < g2prime_threshold)

    # Class balance measured on the labels of every split, not only on the manifest.
    balance = {}
    for split in SPLIT_NAMES:
        _, y = splits[split]
        balance[split] = float(np.mean(np.asarray(y) > 0))
        assert manifest["class_balance"][split]["fraction_positive"] == balance[split]
    deviation_pp = {s: abs(v - 0.5) * 100.0 for s, v in balance.items()}
    balance_ok = all(v <= BALANCE_TOLERANCE_PP for v in deviation_pp.values())

    row = {
        "dataset_seed": seed,
        **{key: kwargs[key] for key in ("n_features", "n_hyperplanes", "dim_hyperplanes")},
        "dataset_hash": manifest["dataset_hash"][:16],
        "pca_hash": manifest["pca_hash"][:16],
        "file_sha256": manifest["file_sha256"][:16],
        "evr1": round(g2["top_share"], 6),
        "g2_passed": g2["passed"],
        "shares_of_retained": " ".join(f"{v:.4f}" for v in g2["share_of_retained"]),
        "total_variance_explained": round(g2["total_variance_explained"], 6),
        "acc_strong_svm": round(strong_all5["accuracy"], 6),
        "strong_C": strong_all5["selected"]["C"],
        "strong_gamma": strong_all5["selected"]["gamma"],
        "strong_in_band": g1["strong_in_band"],
        "acc_e_linear": round(e_linear["accuracy"], 6),
        "e_linear_lr_selected": e_linear["lr_selected"],
        "e_linear_val_accuracy": round(e_linear["val_accuracy"], 6),
        "g1_headroom": round(g1["headroom"], 6),
        "g1_margin": round(g1["g1_margin"], 6),
        "g1_passed": g1["passed"],
        "g1_binding": g1["binding"],
        "acc_e_mlp42": round(mlp["mlp42"]["accuracy"], 6),
        "mlp42_lr_selected": mlp["mlp42"]["lr_selected"],
        "acc_e_mlp4285": round(mlp["mlp4285"]["accuracy"], 6),
        "mlp4285_lr_selected": mlp["mlp4285"]["lr_selected"],
        "acc_strong_mlp": round(acc_strong_mlp, 6),
        "strong_mlp_which": best_mlp_name,
        "headroom_vs_mlp": round(acc_strong_mlp - e_linear["accuracy"], 6),
        "strong_mlp_in_band": bool(low <= acc_strong_mlp <= high),
        "acc_svc_pc1": round(strong_pc1["accuracy"], 6),
        "acc_logreg_pc1": round(logreg_pc1, 6),
        "gain_pc1": round(gain_pc1, 6),
        "acc_svc_all5": round(strong_all5["accuracy"], 6),
        "acc_logreg_all5": round(logreg_all5, 6),
        "gain_all5": round(gain_all5, 6),
        "g2prime_threshold": round(g2prime_threshold, 6),
        "g2prime_passed": g2prime_passed,
        **{f"frac_pos_{s}": round(balance[s], 6) for s in SPLIT_NAMES},
        **{f"dev_pp_{s}": round(deviation_pp[s], 4) for s in SPLIT_NAMES},
        "balance_ok": balance_ok,
        # The binding gates plus the balance rule. The PC1 comparison and the MLP
        # ceiling are not part of this verdict.
        "gates_passed": bool(g1["passed"] and g2["passed"] and balance_ok),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }

    detail = {
        "dataset_seed": seed,
        "generator_kwargs": dict(kwargs),
        "frozen_name": name,
        "manifest": manifest,
        "g2": g2,
        "g1_binding": g1,
        "second_ceiling_diagnostic": {
            "note": "DIAGNOSTIC for open the ceiling decision, not gating: best of {mlp42, mlp4285} "
                    "on the same PCA-5 features, arm E construction with an MLP head",
            "which": best_mlp_name,
            "accuracy": acc_strong_mlp,
            "headroom_vs_e_linear": acc_strong_mlp - e_linear["accuracy"],
            "per_dilution": mlp,
        },
        "ceiling_shape_G2prime": {
            "note": "G2' is PROPOSED, NOT a gate",
            "factor": G2_PRIME_FACTOR,
            "gain_pc1": gain_pc1,
            "gain_all5": gain_all5,
            "threshold": g2prime_threshold,
            "passed": g2prime_passed,
            "svc_pc1": strong_pc1,
            "logreg_pc1": logreg_pc1,
            "logreg_all5": logreg_all5,
        },
        "class_balance": {
            "fraction_positive": balance,
            "deviation_pp": deviation_pp,
            "tolerance_pp": BALANCE_TOLERANCE_PP,
            "ok": balance_ok,
        },
    }

    if not keep_frozen:
        for path in dataset_paths(name, out_dir=freeze_dir):
            path.unlink(missing_ok=True)

    return row, detail


# --- freezing the chosen seed --------------------------------------------------------


def freeze_and_verify(*, out_dir: Path, staging: Path, replay: Path) -> tuple[dict, dict]:
    """Freeze into `staging`, publish into `out_dir`, then replay from the manifest.

    Staging first makes "nothing was written" true rather than merely intended. The replay
    reads every argument back out of the written manifest, which catches a knob that
    silently fell back to its default during freezing.
    """
    manifest = generate_and_freeze(
        DATASET,
        n_samples=N_SAMPLES_TOTAL,
        generator_kwargs=PRODUCTION_GENERATOR_KWARGS,
        dataset_seed=PRODUCTION_DATASET_SEED,
        out_dir=staging,
        frozen_name=PRODUCTION_FROZEN_NAME,
        n_components=N_COMPONENTS,
    )

    data_path, manifest_path = dataset_paths(PRODUCTION_FROZEN_NAME, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in dataset_paths(PRODUCTION_FROZEN_NAME, out_dir=staging):
        shutil.copy2(source, out_dir / source.name)
    print(f"# written: {data_path}")
    print(f"# written: {manifest_path}")

    # Reload through the loader, which re-verifies file_sha256 and both content digests.
    load_splits(PRODUCTION_FROZEN_NAME, out_dir=out_dir)
    published = load_manifest(PRODUCTION_FROZEN_NAME, out_dir=out_dir)

    # Reproduction: every argument comes from the written manifest, not the constants
    # above.
    print(f"\n# T2 — reproduce from the manifest into {replay}")
    print("  recorded manifest is the source of every argument below")
    print(f"    generator        {published['generator']}")
    print(f"    generator_kwargs {published['generator_kwargs']}")
    print(f"    n_samples        {published['n_samples']}")
    print(f"    dataset_seed     {published['dataset_seed']}")
    print(f"    generator_commit {published['generator_commit']}")
    print(f"    pca.n_components {published['pca']['n_components']}")
    replayed = generate_and_freeze(
        published["generator"],
        n_samples=published["n_samples"],
        generator_kwargs=dict(published["generator_kwargs"]),
        dataset_seed=published["dataset_seed"],
        out_dir=replay,
        frozen_name=published["frozen_name"],
        n_components=published["pca"]["n_components"],
    )

    identical = True
    for key in ("dataset_hash", "pca_hash", "file_sha256"):
        same = replayed[key] == published[key]
        identical &= same
        print(f"  {key}:")
        print(f"      manifest    {published[key]}")
        print(f"      regenerated {replayed[key]}")
        print(f"      -> {'IDENTICAL' if same else 'DIFFERENT'}")
    for key in ("shuffle_seed", "generator_commit"):
        same = replayed[key] == published[key]
        identical &= same
        print(f"  {key}: {published[key]} vs {replayed[key]} -> "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    if not identical:
        raise SystemExit(
            "ABORTED: the frozen dataset does not reproduce from its own manifest. "
            "Every downstream result would be unreconstructable."
        )
    print("  VERDICT T2: all three hashes identical")
    return published, replayed


# --- printing -----------------------------------------------------------------------


def print_scan_header() -> None:
    header = (f"{'seed':>5} | {'evr1':>6} G2 | {'strong':>6} band | {'E lin':>6} {'hdrm':>7} G1 |"
              f" {'MLP':>6} {'hdrmMLP':>8} {'which':>8} | {'g(PC1)':>7} {'g(all5)':>7} G2' |"
              f" {'dev pp tr/val/te':>18} bal | GATES")
    print(header)
    print("-" * len(header))


def print_scan_line(r: dict) -> None:
    print(
        f"{r['dataset_seed']:5d} | "
        f"{r['evr1']:6.3f} {'ok' if r['g2_passed'] else 'NO'} | "
        f"{r['acc_strong_svm']:6.3f} {'ok ' if r['strong_in_band'] else 'NO '}| "
        f"{r['acc_e_linear']:6.3f} {r['g1_headroom']:+7.3f} "
        f"{'ok' if r['g1_passed'] else 'NO'} | "
        f"{r['acc_strong_mlp']:6.3f} {r['headroom_vs_mlp']:+8.3f} {r['strong_mlp_which']:>8} | "
        f"{r['gain_pc1']:7.3f} {r['gain_all5']:7.3f} "
        f"{'ok ' if r['g2prime_passed'] else 'NO '}| "
        f"{r['dev_pp_train']:5.2f}/{r['dev_pp_val']:5.2f}/{r['dev_pp_test']:5.2f} "
        f"{'ok ' if r['balance_ok'] else 'NO '}| "
        f"{'PASS' if r['gates_passed'] else 'FAIL'}",
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="where the frozen production dataset lands (default: data/)")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/ds2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DATASET_SEED_GRID))
    parser.add_argument("--scan-only", action="store_true",
                        help="run the five-seed scan and stop; nothing is written to data/")
    parser.add_argument("--skip-scan", action="store_true",
                        help="freeze the chosen seed without re-running the scan")
    args = parser.parse_args(argv)

    print("# this freeze — scan five generator seeds of the hyperplanes cell, then freeze")
    print(f"# cell rule (declared BEFORE the measurement): {CELL_RULE}")
    print(f"#   the cell screen found two passing cells; the other one is {SCREEN_ALTERNATIVE_CELL}")
    print(f"# cell: {DATASET} {PRODUCTION_GENERATOR_KWARGS}, PCA k={N_COMPONENTS}")
    print(f"# seed rule (declared BEFORE the measurement): \"{SEED_RULE}\" over "
          f"{list(DATASET_SEED_GRID)} -> {PRODUCTION_DATASET_SEED}")
    print(f"# split {N_SAMPLES_TOTAL} -> 4200/600/1200")
    print(f"# G2 evr1 <= {G2_MAX_COMPONENT_SHARE}; G1 band {list(G1_STRONG_ACCURACY_BAND)}, "
          f"headroom >= {G1_MIN_HEADROOM}")
    print(f"# G1 strong: SVC(rbf) grid {G1_SVC_GRID} (3x3, the 3x3 rule), selected on val, scored on test")
    print(f"# G1 floor: arm E = identity socket + LINEAR head via training.train_model, "
          f"lr from {list(G1_LR_GRID)}, seeds {list(ARM_E_SEEDS)}")
    print(f"# second ceiling (DIAGNOSTIC, the ceiling decision open, NOT gating): best of "
          f"{list(MLP_DILUTIONS)} on the same features")
    print(f"# G2' (REPORTED, NOT GATING): gain(PC1) < {G2_PRIME_FACTOR} * gain(all5)")
    print(f"# class balance: more than {BALANCE_TOLERANCE_PP:.0f} pp off 50/50 on any split "
          f"and the cell is OUT")
    print()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []
    details: list[dict] = []

    if not args.skip_scan:
        print_scan_header()
        with tempfile.TemporaryDirectory(prefix="ds2_scan_") as scratch:
            for seed in args.seeds:
                row, detail = scan_seed(
                    seed, kwargs=PRODUCTION_GENERATOR_KWARGS, freeze_dir=Path(scratch)
                )
                print_scan_line(row)
                rows.append(row)
                details.append(detail)

        print()
        n = len(rows)
        print(f"  G2  (evr1 <= {G2_MAX_COMPONENT_SHARE})            passed by "
              f"{sum(r['g2_passed'] for r in rows)} / {n} seeds")
        print(f"  G1a band {list(G1_STRONG_ACCURACY_BAND)}              passed by "
              f"{sum(r['strong_in_band'] for r in rows)} / {n} seeds")
        print(f"  G1  (binding, headroom >= {G1_MIN_HEADROOM})     passed by "
              f"{sum(r['g1_passed'] for r in rows)} / {n} seeds")
        print(f"  class balance within {BALANCE_TOLERANCE_PP:.0f} pp        passed by "
              f"{sum(r['balance_ok'] for r in rows)} / {n} seeds")
        print(f"  G2' (NOT GATING)                    satisfied by "
              f"{sum(r['g2prime_passed'] for r in rows)} / {n} seeds")
        print(f"  binding gates + balance             passed by "
              f"{sum(r['gates_passed'] for r in rows)} / {n} seeds")

        csv_path = outputs_dir / f"ds2_seed_scan_{stamp}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        json_path = outputs_dir / f"ds2_seed_scan_{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    "task": "this freeze",
                    "rules_fixed": "cell rule and seed rule below were fixed before this scan ran",
                    "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "generator": DATASET,
                    "generator_kwargs": PRODUCTION_GENERATOR_KWARGS,
                    "cell_rule": CELL_RULE,
                    "ds1_alternative_cell": SCREEN_ALTERNATIVE_CELL,
                    "seed_grid": list(DATASET_SEED_GRID),
                    "seed_rule": SEED_RULE,
                    "chosen_seed": PRODUCTION_DATASET_SEED,
                    "arm_e_seeds": list(ARM_E_SEEDS),
                    "g2_prime_factor": G2_PRIME_FACTOR,
                    "g2_prime_is_gating": False,
                    "mlp_ceiling_is_gating": False,
                    "balance_tolerance_pp": BALANCE_TOLERANCE_PP,
                    "seeds": details,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {csv_path}\nwrote {json_path}")

        chosen = next((r for r in rows if r["dataset_seed"] == PRODUCTION_DATASET_SEED), None)
        if chosen is None:
            print(f"\n! seed {PRODUCTION_DATASET_SEED} was not in the scan; not freezing")
            return 0
        if not chosen["gates_passed"]:
            raise SystemExit(
                f"ABORTED: seed {PRODUCTION_DATASET_SEED}, chosen by the declared rule, does "
                f"not pass the binding gates (G1 {chosen['g1_passed']}, G2 {chosen['g2_passed']}, "
                f"balance {chosen['balance_ok']}). SPEC section 6: a dataset that fails a gate "
                "is reported as EXCLUDED, not rescued. Nothing was written to data/."
            )

    if args.scan_only:
        print("\n# --scan-only: nothing written to data/")
        return 0

    print(f"\n# freezing seed {PRODUCTION_DATASET_SEED} into {args.out_dir}")
    with tempfile.TemporaryDirectory(prefix="ds2_staging_") as staging, \
            tempfile.TemporaryDirectory(prefix="ds2_replay_") as replay:
        published, _ = freeze_and_verify(
            out_dir=args.out_dir, staging=Path(staging), replay=Path(replay)
        )

    print("\n# manifest summary")
    print(f"  frozen_name  {published['frozen_name']}")
    print(f"  dataset_hash {published['dataset_hash']}")
    print(f"  pca_hash     {published['pca_hash']}")
    print(f"  file_sha256  {published['file_sha256']}")
    print(f"  explained_variance_ratio_of_retained "
          f"{[round(v, 6) for v in published['pca']['explained_variance_ratio_of_retained']]}")
    print(f"  total_variance_explained {published['pca']['total_variance_explained']:.6f}")
    for split, b in published["class_balance"].items():
        print(f"  class balance {split:<5} n={b['n']:>4} fraction_positive "
              f"{b['fraction_positive']:.6f} ({abs(b['fraction_positive'] - 0.5) * 100:.2f} pp off) "
              f"counts {b['counts']}")
    print(f"  clipped_fraction {published['scaling']['clipped_fraction']}")
    print(f"  max_overshoot_before_clipping {published['scaling']['max_overshoot_before_clipping']}")

    summary = outputs_dir / f"ds2_freeze_{stamp}.json"
    summary.write_text(
        json.dumps(
            {
                "task": "this freeze",
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "dataset": DATASET,
                "generator_kwargs": PRODUCTION_GENERATOR_KWARGS,
                "cell_rule": CELL_RULE,
                "seed_rule": SEED_RULE,
                "seed_grid": list(DATASET_SEED_GRID),
                "dataset_seed": PRODUCTION_DATASET_SEED,
                "frozen_name": PRODUCTION_FROZEN_NAME,
                "out_dir": str(args.out_dir),
                "manifest": published,
                "scan_rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
