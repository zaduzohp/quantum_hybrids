"""The analysis pipeline: estimands, exact tests, figures, and the audit of the report.

What it computes:

  * the paired contrast on the differences per training seed, per
    (generator seed x dilution x ansatz) cell: Delta_AB (the research question),
    Delta_AE (with its mandatory decomposition), Delta_AF, Delta_BD_matched and
    Delta_BD_best (both exploratory),
  * replication across the three generator seeds side by side, then pooled estimation
    with the generator seed as a FIXED effect — 30 blocked paired differences, which is
    a bigger sample with a blocking factor, NOT a variance component,
  * the exact sign test and exact Wilcoxon plus t for
    comparison, MixedLM as a CHECK, TOST for Delta_AE only,
  * the three uncertainty accounts side by side and never summed: the CI over 10 paired
    differences, the binomial SE on 1200 test rows, and McNemar on discordant pairs,

Usage:
    python scripts/run_a8_analysis.py --results outputs/a7/dry_run/a7_results.csv
    python scripts/run_a8_analysis.py --results outputs/a7/main/a7_results.csv \
        --out-dir outputs/a8/main
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_main_series as a7

from qsocket import stats
from qsocket.ansatzes import build_socket_circuit, socket_param_count
from qsocket.core import derive
from qsocket.datasets import (
    N_COMPONENTS,
    PCA_WHITEN,
    SPLIT_SIZES,
    load_frozen,
    load_manifest,
    load_splits,
)
from qsocket.estimators import (  # noqa: F401
    MIXEDLM_EQUIVALENCE_TOL,
    POOLING_DIVERGENCE_RULE,
    TOST_ALPHA,
    TOST_DELTA,
    TOST_POWER_FLOOR,
    PairKey,
    arm_summary,
    divergence_check,
    estimate,
    estimate_blocked,
    in_reference_units,
    mcnemar_from_vectors,
    mixedlm_check,
    ols_block_crosscheck,
    paired_differences,
    pooled_blocked_estimate,
    seeds_needed_for_mde,
    seeds_needed_for_tost,
    tost,
    tost_power,
)
from qsocket.gates import (
    G1_LR_GRID,
    G1_MIN_HEADROOM,
    ceiling,
    check_g1_headroom,
    make_arm_e_linear_floor_model,
    make_svc_strong_model,
)
from qsocket.head import DILUTION_AXIS, make_head
from qsocket.rank import effective_dimension

# --- declared before the analysis ----------------------------------------------------

CONTRACT_DATASET_SEEDS: tuple[int, ...] = a7.DATASET_SEEDS
CONTRACT_DILUTIONS: tuple[str, ...] = DILUTION_AXIS
CONTRACT_ANSATZ_LEVELS: tuple[str, ...] = a7.ANSATZ_LEVELS
CONTRACT_SEEDS: tuple[int, ...] = a7.SEEDS
CONTRACT_N_TEST: int = SPLIT_SIZES["test"]

CONFIRMATORY_FAMILY: tuple[str, ...] = ("delta_AB",)
HOLM_NOTE = (
    "the confirmatory family has ONE element (H1 = Delta_AB at the linear head), so the "
    "Holm correction is not applied. P1 and P2 are descriptive and do not enter the "
    "family. Stated explicitly so it does not read as an omission."
)

TOST_REPORTED_AS_TEST: tuple[str, ...] = ("delta_AE",)
TOST_COMPUTED_NOT_REPORTED: tuple[str, ...] = ("delta_AB",)
TOST_POWER_D4_FLOOR = 0.80

# "theta did not move"
THETA_STILL = a7.THETA_STILL

PRODUCT_ANSATZ = a7.PRODUCT_ANSATZ
MAX_EPOCHS = a7.MAX_EPOCHS

# Arm E is tuned on the contract grid plus one point
ARM_E_LR_GRID = a7.ARM_E_LR_GRID
GATE_ARM_E_SEEDS: tuple[int, ...] = (1, 2, 3)

# outputs/stats/ is where the committed analysis lives and what REPRODUCE.md passes
# explicitly; the default used to be outputs/a8/, which silently wrote a second,
# competing analysis directory whenever --out-dir was omitted.
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "stats"

STATUS_PROVISIONAL = "PROVISIONAL — NOT A RESULT"
STATUS_COMPLETE = "complete contract grid"


# =====================================================================================
# 1. Input: rows, provenance, and the refusal to call an incomplete grid a result
# =====================================================================================


def load_rows(results_csv: Path) -> list[dict]:
    """The result rows as dicts, with dataset_seed attached and integer columns coerced.

    Read with dtype=str and coerced here: pandas turns an int column holding one empty
    value into float, so rff_width would come back as "32.0" and stop matching the 32 in
    a prediction filename.
    """
    results_csv = Path(results_csv)
    frame = pd.read_csv(results_csv, dtype=str, keep_default_na=False)
    expected = list(a7.RESULT_COLUMNS)
    if frame.columns.tolist() != expected:
        raise SystemExit(
            f"STOP: {results_csv} has columns {frame.columns.tolist()}, which are not "
            f"RESULT_COLUMNS. A8 reads the A7 schema and nothing else."
        )
    rows = []
    for record in frame.to_dict("records"):
        row = dict(record)
        row["dataset_seed"] = a7.dataset_seed_of(row["dataset"])
        row["seed_int"] = a7.optional_int(row["seed"])
        row["width_int"] = a7.optional_int(row["rff_width"])
        row["epochs_run_int"] = a7.optional_int(row["epochs_run"])
        row["best_epoch_int"] = a7.optional_int(row["best_epoch"])
        row["n_eval_int"] = a7.optional_int(row["n_eval"])
        row["accuracy_float"] = float(row["accuracy"])
        row["lr_float"] = float(row["lr_selected"])
        rows.append(row)
    if not rows:
        raise SystemExit(f"STOP: {results_csv} has no rows.")
    return rows


def provenance(rows: list[dict], *, results_csv: Path) -> dict:
    """What the input covers versus what the contract asks for.

    Decides whether any number downstream may be called a result. It never fills a gap or
    lowers the bar — it reports the gap.
    """
    present = {
        "dataset_seeds": sorted({r["dataset_seed"] for r in rows}),
        "dilutions": sorted({r["dilution"] for r in rows if r["arm"] != "D_best"}),
        "ansatz_levels": sorted({r["ansatz_level"] for r in rows if r["arm"] in ("A", "B")}),
        "seeds": sorted({r["seed_int"] for r in rows if r["seed_int"] is not None}),
        "arms": sorted({r["arm"] for r in rows}),
    }
    missing = {
        "dataset_seeds": [s for s in CONTRACT_DATASET_SEEDS if s not in present["dataset_seeds"]],
        "dilutions": [d for d in CONTRACT_DILUTIONS if d not in present["dilutions"]],
        "ansatz_levels": [
            a for a in CONTRACT_ANSATZ_LEVELS if a not in present["ansatz_levels"]
        ],
        "seeds": [s for s in CONTRACT_SEEDS if s not in present["seeds"]],
    }
    complete = not any(missing.values())
    n_test_rows = {r["n_eval_int"] for r in rows if r["split"] == "test"}
    return {
        "results_csv": str(results_csv),
        "n_rows": len(rows),
        "present": present,
        "missing": missing,
        "complete_contract_grid": complete,
        "status": STATUS_COMPLETE if complete else STATUS_PROVISIONAL,
        "n_training_seeds": len(present["seeds"]),
        "n_test_rows_seen": sorted(x for x in n_test_rows if x is not None),
        "arm_D_present": any(r["arm"].startswith("D") for r in rows),
        "note": (
            "every number below is a RESULT"
            if complete
            else "PROVISIONAL: the input is not the complete contract grid. The schema is "
            "real, the numbers are not results and may not be reported as such "
            "(owner's instruction, 2026-08-21). Missing rows are "
            "reported here and NOT filled in."
        ),
    }


# =====================================================================================
# 2. Pairing — the silent-failure surface of the analysis
# =====================================================================================



def accuracy_index(rows: list[dict], split: str) -> dict:
    """(dataset_seed, arm, ansatz_level, dilution, width, seed, lr) -> row, for one split.

    The lr is part of the key. Arms F and D_matched have no ansatz dimension while the
    cell lr does, so when L1 and L2 select different lr those arms are measured twice,
    once per lr. Keyed without lr, one row wins by CSV order and the paired difference for
    the other ansatz is silently taken against the wrong lr.
    """
    index: dict[tuple, dict] = {}
    for row in rows:
        if row["split"] != split:
            continue
        key = (
            row["dataset_seed"],
            row["arm"],
            row["ansatz_level"],
            row["dilution"],
            row["width_int"],
            row["seed_int"],
            f"{row['lr_float']:g}",
        )
        if key in index and row["accuracy"] != index[key]["accuracy"]:
            raise SystemExit(
                f"STOP: two rows for the same cell {key} disagree on accuracy "
                f"({index[key]['accuracy']} vs {row['accuracy']}). A8 does not choose "
                "between them."
            )
        index.setdefault(key, row)
    return index


def selected_lrs(rows: list[dict]) -> dict:
    """The lr each cell ran at, read off the rows rather than re-derived.

    Arms A and B share the cell lr; arm E has its own, and arm D_best one per width.
    Re-deriving the selection here would be a second implementation of a rule that already
    ran, so the rows are the authority.
    """
    cell: dict[tuple, set] = {}
    arm_e: dict[tuple, set] = {}
    d_best: dict[tuple, set] = {}
    for row in rows:
        if row["arm"] in ("A", "B"):
            cell.setdefault(
                (row["dataset_seed"], row["dilution"], row["ansatz_level"]), set()
            ).add(row["lr_float"])
        elif row["arm"] == "E":
            arm_e.setdefault((row["dataset_seed"], row["dilution"]), set()).add(row["lr_float"])
        elif row["arm"] == "D_best":
            d_best.setdefault((row["dataset_seed"], row["width_int"]), set()).add(row["lr_float"])

    def one(mapping, label):
        out = {}
        for key, values in mapping.items():
            if len(values) != 1:
                raise SystemExit(
                    f"STOP: {label} {key} carries more than one lr {sorted(values)}. "
                    "Arms A and B must share the cell lr, or the difference between them "
                    "is not a paired difference."
                )
            out[key] = next(iter(values))
        return out

    return {
        "cell_lr": one(cell, "cell"),
        "arm_e_lr": one(arm_e, "arm E cell"),
        "d_best_lr": one(d_best, "arm D_best (dataset seed, width)"),
    }


def series(
    index: dict,
    *,
    dataset_seeds,
    arm: str,
    ansatz_level: str,
    dilution: str,
    lr_of,
    width=None,
    seeds=CONTRACT_SEEDS,
) -> dict:
    """{PairKey(dataset_seed, seed) -> accuracy} for one arm at the lr it must be read at.

    `lr_of` is a callable (dataset_seed) -> lr, so the caller states which lr this arm is
    paired at instead of letting the index pick. Arms F and D_matched are paired at the lr
    of the cell they are compared inside; arms E and D_best at their own.
    """
    out: dict[PairKey, float] = {}
    for dataset_seed in dataset_seeds:
        lr = lr_of(dataset_seed)
        if lr is None:
            continue
        for seed in seeds:
            key = (
                dataset_seed,
                arm,
                ansatz_level,
                dilution,
                width,
                seed,
                f"{float(lr):g}",
            )
            if key in index:
                out[PairKey(dataset_seed, seed)] = float(index[key]["accuracy"])
    return out




# =====================================================================================
# 3. The estimator: mean, CI, sigma with its own CI, MDE, and the three tests
# =====================================================================================


# The paired-design statistics come from qsocket.stats, re-exported here so every call
# site in this file keeps reading the same names. Three copies of MDE, the sigma CI and
# the TOST power used to live in three scripts; they agreed to the last bit, which is
# precisely why an edit to one of them would not have been noticed.
mde_constant = stats.mde_constant
sigma_confidence_interval = stats.sigma_confidence_interval








# =====================================================================================
# 4. TOST, and the power statement that replaces it for Delta_AB
# =====================================================================================










# =====================================================================================
# 5. Replication across the datasets, then pooling with the dataset as a fixed effect
# =====================================================================================














# =====================================================================================
# 6. McNemar — the third uncertainty account, mandatory for Delta_AE
# =====================================================================================


def correctness_vectors(rows: list[dict], predictions_dir) -> dict:
    """Per-test-row correctness, read from the npz files the driver wrote.

    Keyed exactly like accuracy_index, lr included, so a vector is never paired against a
    run at a different lr.

    `predictions_dir` may be one directory or several. The driver writes one run directory
    per generator seed, so a combined CSV covering three seeds has its vectors spread over
    three directories; taking only the first leaves McNemar — the account the report calls
    MANDATORY for Delta_AE — computed on a third of the data with no sign that anything is
    absent. A row counts as missing only when NO directory holds it.
    """
    directories = (
        [Path(predictions_dir)]
        if isinstance(predictions_dir, (str, Path))
        else [Path(d) for d in predictions_dir]
    )
    vectors: dict[tuple, np.ndarray] = {}
    missing = []
    for row in rows:
        if row["split"] != "test":
            continue
        prediction = {
            "dataset_seed": row["dataset_seed"],
            "arm": row["arm"],
            "ansatz_level": row["ansatz_level"],
            "dilution": row["dilution"],
            "seed": row["seed_int"],
            "width": row["width_int"],
            "lr": row["lr_float"],
        }
        # Each directory as given, not its parent with "predictions" re-appended: that
        # only resolved when --predictions-dir happened to be named "predictions", and any
        # other name reported every vector as missing.
        candidates = [a7.prediction_path_in(d, prediction) for d in directories]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            missing.append(str(candidates[0]))
            continue
        key = (
            row["dataset_seed"], row["arm"], row["ansatz_level"], row["dilution"],
            row["width_int"], row["seed_int"], f"{row['lr_float']:g}",
        )
        vectors[key] = a7.read_prediction(path)
    return {"vectors": vectors, "missing": missing}




# =====================================================================================
# 7. Recomputing every number that stands in the report
# =====================================================================================

# The values as they stand in the report, with the precision the document prints and the
# recipe for reproducing each. Category (a) must match exactly; a mismatch stops the run
# rather than being fixed, because it is unknown which side is wrong.
#
# Exposed as data rather than buried in asserts, so a test can substitute a false
# "reported" value and require the pipeline to raise.
RAPORT_DETERMINISTIC: dict[str, dict] = {
    "dataset_hash_ds11": {
        "reported": "4360508611e0e896",
        "kind": "prefix",
        "recipe": "manifest of the frozen production dataset, digest recomputed from the "
                  "arrays by datasets.load_frozen at load time",
    },
    "pca_hash_ds11": {
        "reported": "2bf856a6c49a9c38",
        "kind": "prefix",
        "recipe": "digest of (pca_components, pca_mean), recomputed at load time",
    },
    "file_sha256_ds11": {
        "reported": "5e519b749c22b488",
        "kind": "prefix",
        "recipe": "sha256 of the frozen .npz, recomputed from the bytes on disk",
    },
    "evr1_ds11": {
        "reported": 0.7585,
        "decimals": 4,
        "recipe": "PCA(5, whiten=False, random_state=derive(11,'pca')%2**31) refitted on "
                  "the 4200 RAW training rows; explained_variance_ratio_[0]",
    },
    "total_variance_explained_ds11": {
        "reported": 1.000000,
        "decimals": 6,
        "recipe": "sum of the retained shares evr/evr.sum() from the same refit",
    },
    "g1a_strong_svm_ds11": {
        "reported": 0.860,
        "decimals": 3,
        "recipe": "gates.make_svc_strong_model() over G1_SVC_GRID, selected on the 600 "
                  "val rows, evaluated on the 1200 test rows",
    },
    "g1b_headroom_ds11": {
        "reported": 0.096,
        "decimals": 3,
        "recipe": "acc(strong SVM) - acc(arm E linear) with arm E on the CONTRACT lr grid, "
                  "training seeds 1-3",
    },
    "g1_margin_ds11": {
        "reported": 0.038,
        "decimals": 3,
        "recipe": "headroom - G1_MIN_HEADROOM with arm E on the D-30 grid (contract + 0.1), "
                  "training seeds 1-3, on the arm-E grid (contract + 0.1)",
    },
    "ceiling_vs_svm_ds11": {
        "reported": 0.096,
        "decimals": 3,
        "recipe": "acc(strong SVM) - acc(arm E linear, contract grid, seeds 1-3)",
    },
    "ceiling_vs_mlp_ds11": {
        "reported": 0.213,
        "decimals": 3,
        "recipe": "acc(best of mlp42/mlp4285 on the same features) - acc(arm E linear); "
                  "gates.ceiling(), D-28",
    },
    "socket_params_R2": {
        "reported": 35,
        "kind": "int",
        "recipe": "ansatzes.socket_param_count(5, 2)",
    },
    "jacobian_rank_L1_R2": {
        "reported": 35,
        "kind": "int",
        "recipe": "rank.effective_dimension on the L1 socket circuit at R=2",
    },
    "head_params_linear": {"reported": 6, "kind": "int", "recipe": "5+1, counted on the built model"},
    "head_params_h2": {"reported": 15, "kind": "int", "recipe": "7h+1 at h=2, counted on the built model"},
    "head_params_h4": {"reported": 29, "kind": "int", "recipe": "7h+1 at h=4, counted on the built model"},
    "head_params_h42": {"reported": 295, "kind": "int", "recipe": "7h+1 at h=42, counted on the built model"},
    "quantum_share_linear": {"reported": 85.4, "decimals": 1, "recipe": "100*35/(35+6)"},
    "quantum_share_h2": {"reported": 70.0, "decimals": 1, "recipe": "100*35/(35+15)"},
    "quantum_share_h4": {"reported": 54.7, "decimals": 1, "recipe": "100*35/(35+29)"},
    "quantum_share_h42": {"reported": 10.6, "decimals": 1, "recipe": "100*35/(35+295)"},
    "binomial_se_1200": {
        "reported": 0.0144,
        "decimals": 4,
        "recipe": "sqrt(0.25/1200)",
    },
    "mde_constant_n10": {
        "reported": 0.995,
        "decimals": 3,
        "recipe": "(t.ppf(.975,9)+t.ppf(.80,9))/sqrt(10) from scipy, never hard-coded",
    },
}

# Category (b): pilot estimates from a single cell and generator seed, which the main
# series replaces. Never asserted for equality — doing so fails the pipeline on correct
# data. The constant below names the generator seed the pilot was measured on, the only
# one whose numbers compare like for like against RAPORT_PILOT.
PILOT_GENERATOR_SEED: int = 11

RAPORT_PILOT: dict[str, dict] = {
    "sigma_seed_A": {"reported": 0.0063, "source": "DS3, 10 seeds, linear, L1, ds11"},
    "sigma_seed_B": {"reported": 0.0478, "source": "DS3"},
    "sigma_seed_E": {"reported": 0.0031, "source": "DS3"},
    "sigma_delta_AB": {"reported": 0.0489, "source": "DS3 5.2"},
    "sigma_delta_AE": {"reported": 0.0075, "source": "DS3 5.2"},
    "sigma_delta_BE": {"reported": 0.0471, "source": "DS3 5.2"},
    "mde_AB": {"reported": 0.0486, "source": "0.995 * sigma_delta_AB (pilot)"},
    "mde_AE": {"reported": 0.0075, "source": "0.995 * sigma_delta_AE (pilot)"},
    "tost_power_AE_delta002": {"reported": 1.000, "source": "D-4, closed on this number"},
    "tost_power_AB_delta002": {"reported": 0.000, "source": "D-4"},
    "delta_AB": {"reported": 0.0919, "source": "DS3 — a RESULT of the pilot, not a reference"},
    "delta_AE": {"reported": 0.0600, "source": "DS3 — a RESULT of the pilot"},
}


# The published TOST power figures, with the sigma each was taken at. Recomputing a power
# at its own sigma reproduces a reported number rather than replacing a pilot estimate, so
# it belongs in the audit.
PILOT_POWER_CLAIMS: tuple[dict, ...] = (
    {"label": "TOST power A<->E, delta=0.02", "sigma": 0.0075, "n": 10, "reported": 1.000,
     "source": "raport.tex 3.7 / D-4"},
    {"label": "TOST power A<->B, delta=0.02", "sigma": 0.0489, "n": 10, "reported": 0.000,
     "source": "raport.tex 3.7 / D-4"},
    {"label": "TOST power, allocation-contrast sigma_Delta", "sigma": 0.0612, "n": 10, "reported": 0.003,
     "source": "SPEC 7.5 table"},
    {"label": "TOST power, sigma_Delta(A-B) smoke n=3", "sigma": 0.0298, "n": 10,
     "reported": 0.044, "source": "SPEC 7.5 / 7.6 remark"},
)


def pilot_power_audit() -> list[dict]:
    """Every published TOST power figure, recomputed at the sigma it was published with.

    Two of these do not reproduce. They are pilot figures, so a disagreement is reported
    rather than fatal, and neither changes a decision: both values sit far below the power
    floor either way.
    """
    rows = []
    for claim in PILOT_POWER_CLAIMS:
        recomputed = tost_power(sigma=claim["sigma"], n=claim["n"], delta=TOST_DELTA)
        rows.append(
            {
                **claim,
                "recomputed": recomputed,
                "agrees_at_3dp": bool(
                    round(recomputed, 3) == round(float(claim["reported"]), 3)),
                "changes_a_decision": bool(
                    (recomputed >= TOST_POWER_FLOOR) != (float(claim["reported"]) >= TOST_POWER_FLOOR)
                ),
            }
        )
    return rows


def recompute_dataset_numbers(dataset_seed: int) -> dict:
    """evr1 and the retained shares, refitted on the RAW training rows.

    Recomputed from source, never from the manifest's own PCA block, which would only
    check a copy of the number.
    """
    from sklearn.decomposition import PCA

    name, out_dir = a7.dataset_location(dataset_seed)
    # load_frozen re-verifies the file, content and PCA digests.
    X_raw, _ = load_frozen(name, "train", out_dir=out_dir, raw=True)
    manifest = load_manifest(name, out_dir=out_dir)
    random_state = derive(dataset_seed, "pca") % (2**31)
    pca = PCA(n_components=N_COMPONENTS, whiten=PCA_WHITEN, random_state=random_state)
    pca.fit(X_raw)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    retained = evr / evr.sum()
    return {
        "dataset": name,
        "dataset_seed": dataset_seed,
        "n_train_rows_used": int(X_raw.shape[0]),
        "evr": [float(v) for v in evr],
        "evr1": float(evr[0]),
        "retained_shares": [float(v) for v in retained],
        "total_variance_explained": float(retained.sum()),
        "dataset_hash": manifest["dataset_hash"],
        "pca_hash": manifest["pca_hash"],
        "file_sha256": manifest["file_sha256"],
    }


def recompute_gate_numbers(dataset_seed: int) -> dict:
    """G1 on both arm-E lr grids, and the ceiling. The expensive part of the audit.

    Two floor readings, because the report quotes two numbers taken on two different lr
    grids. Conflating them would compare quantities that are not the same.
    """
    name, out_dir = a7.dataset_location(dataset_seed)
    splits = load_splits(name, out_dir=out_dir)
    strong = make_svc_strong_model()

    contract = check_g1_headroom(
        splits,
        strong_model=strong,
        floor_model=make_arm_e_linear_floor_model(
            lr_grid=tuple(G1_LR_GRID), seeds=GATE_ARM_E_SEEDS
        ),
    )
    wide = check_g1_headroom(
        splits,
        strong_model=strong,
        floor_model=make_arm_e_linear_floor_model(
            lr_grid=tuple(ARM_E_LR_GRID), seeds=GATE_ARM_E_SEEDS
        ),
    )
    ceiling_record = ceiling(
        splits,
        svc_model=strong,
        floor_model=make_arm_e_linear_floor_model(
            lr_grid=tuple(G1_LR_GRID), seeds=GATE_ARM_E_SEEDS
        ),
    )
    return {
        "dataset_seed": dataset_seed,
        "dataset": name,
        "strong_svm_accuracy": float(contract["strong"]["accuracy"]),
        "arm_e_linear_contract_grid": float(contract["floor"]["accuracy"]),
        "arm_e_linear_d30_grid": float(wide["floor"]["accuracy"]),
        "headroom_contract_grid": float(contract["headroom"]),
        "g1_margin_contract_grid": float(contract["g1_margin"]),
        "headroom_d30_grid": float(wide["headroom"]),
        "g1_margin_d30_grid": float(wide["g1_margin"]),
        "g1_min_headroom": float(G1_MIN_HEADROOM),
        "mlp_accuracy": float(ceiling_record["mlp"]["accuracy"]),
        "ceiling": float(ceiling_record["ceiling"]),
        "ceiling_strong_accuracy": float(ceiling_record["strong_accuracy"]),
        "ceiling_which": ceiling_record["strong_which"],
        "ceiling_vs_svm_only": float(ceiling_record["ceiling_vs_svm_only"]),
        "ceiling_vs_mlp_only": float(ceiling_record["ceiling_vs_mlp_only"]),
        "gates_passed": bool(contract["passed"]),
    }


def recompute_structural_numbers() -> dict:
    """Parameter counts and Jacobian ranks — counted on the built objects, not on tables."""
    head_params = {}
    for dilution in CONTRACT_DILUTIONS:
        head = make_head(dilution, seed=1)
        head_params[dilution] = int(sum(p.numel() for p in head.parameters()))
    socket_params = int(socket_param_count(a7.DEFAULT_N_QUBITS, a7.R_CONTRACT))
    ranks = {
        level: int(
            effective_dimension(
                lambda n_qubits, R, _level=level: build_socket_circuit(_level, n_qubits, R),
                a7.R_CONTRACT,
            )
        )
        for level in tuple(CONTRACT_ANSATZ_LEVELS) + (PRODUCT_ANSATZ,)
    }
    return {
        "socket_params_nominal_R2": socket_params,
        "jacobian_rank": ranks,
        "head_params": head_params,
        "quantum_share_percent": {
            dilution: 100.0 * socket_params / (socket_params + params)
            for dilution, params in head_params.items()
        },
    }


def verification_context(dataset_seeds, *, cache_path: Path | None, skip_slow: bool) -> dict:
    """Everything the audit needs, computed once and optionally cached by dataset hash.

    The cache is keyed by the dataset hashes, so an entry cannot survive a change to the
    data it describes. Only measurements are cached, never a verdict.
    """
    cached: dict = {}
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())

    context: dict = {"datasets": {}, "gates": {}, "structural": recompute_structural_numbers()}
    for dataset_seed in dataset_seeds:
        numbers = recompute_dataset_numbers(dataset_seed)
        context["datasets"][str(dataset_seed)] = numbers
        cache_key = f"gates:{dataset_seed}:{numbers['dataset_hash'][:16]}"
        if cache_key in cached:
            context["gates"][str(dataset_seed)] = cached[cache_key]
            context["gates"][str(dataset_seed)]["from_cache"] = True
        elif skip_slow:
            context["gates"][str(dataset_seed)] = {
                "dataset_seed": dataset_seed, "skipped": True,
                "reason": "--skip-slow-verification: the G1 and ceiling readings were NOT "
                          "recomputed. They are reported as not verified, never as passing.",
            }
        else:
            gate_numbers = recompute_gate_numbers(dataset_seed)
            gate_numbers["from_cache"] = False
            context["gates"][str(dataset_seed)] = gate_numbers
            cached[cache_key] = gate_numbers
    if cache_path is not None and not skip_slow:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=1, sort_keys=True))
    return context


def _computed_deterministic(context: dict) -> dict:
    """quantity -> recomputed value, in the same names as RAPORT_DETERMINISTIC."""
    ds11 = context["datasets"].get("11")
    gates11 = context["gates"].get("11", {})
    structural = context["structural"]
    socket = structural["socket_params_nominal_R2"]
    computed: dict = {
        "socket_params_R2": socket,
        "jacobian_rank_L1_R2": structural["jacobian_rank"].get("L1"),
        "binomial_se_1200": float(math.sqrt(0.25 / CONTRACT_N_TEST)),
        "mde_constant_n10": mde_constant(10),
    }
    for dilution, key in (
        ("linear", "head_params_linear"), ("h2", "head_params_h2"),
        ("h4", "head_params_h4"), ("h42", "head_params_h42"),
    ):
        computed[key] = structural["head_params"].get(dilution)
    for dilution, key in (
        ("linear", "quantum_share_linear"), ("h2", "quantum_share_h2"),
        ("h4", "quantum_share_h4"), ("h42", "quantum_share_h42"),
    ):
        computed[key] = structural["quantum_share_percent"].get(dilution)
    if ds11 is not None:
        computed.update(
            {
                "dataset_hash_ds11": ds11["dataset_hash"],
                "pca_hash_ds11": ds11["pca_hash"],
                "file_sha256_ds11": ds11["file_sha256"],
                "evr1_ds11": ds11["evr1"],
                "total_variance_explained_ds11": ds11["total_variance_explained"],
            }
        )
    if gates11 and not gates11.get("skipped"):
        computed.update(
            {
                "g1a_strong_svm_ds11": gates11["strong_svm_accuracy"],
                "g1b_headroom_ds11": gates11["headroom_contract_grid"],
                "g1_margin_ds11": gates11["g1_margin_d30_grid"],
                "ceiling_vs_svm_ds11": gates11["ceiling_vs_svm_only"],
                "ceiling_vs_mlp_ds11": gates11["ceiling_vs_mlp_only"],
            }
        )
    return computed


def verification_table(context: dict, *, series_numbers: dict,
                       reported: dict | None = None) -> tuple[list[dict], list[dict]]:
    """The audit: (rows, mismatches). A mismatch in category (a) stops the run.

    `reported` defaults to RAPORT_DETERMINISTIC and is a parameter so a test can inject a
    false reported value and require the mismatch to surface.
    """
    reported = RAPORT_DETERMINISTIC if reported is None else reported
    computed = _computed_deterministic(context)
    rows: list[dict] = []
    mismatches: list[dict] = []

    for quantity, spec in reported.items():
        value = computed.get(quantity)
        kind = spec.get("kind", "float")
        if value is None:
            rows.append(
                {
                    "quantity": quantity,
                    "in_raport": spec["reported"],
                    "recomputed": "NOT RECOMPUTED",
                    "category": "a (deterministic)",
                    "agrees": "not verified",
                    "recipe": spec["recipe"],
                }
            )
            continue
        if kind == "prefix":
            agrees = str(value).startswith(str(spec["reported"]))
            shown = str(value)[: len(str(spec["reported"]))]
        elif kind == "int":
            agrees = int(value) == int(spec["reported"])
            shown = int(value)
        else:
            decimals = spec["decimals"]
            agrees = round(float(value), decimals) == round(float(spec["reported"]), decimals)
            shown = f"{float(value):.{decimals + 3}f}"
        rows.append(
            {
                "quantity": quantity,
                "in_raport": spec["reported"],
                "recomputed": shown,
                "category": "a (deterministic)",
                "agrees": "YES" if agrees else "NO — STOP",
                "recipe": spec["recipe"],
            }
        )
        if not agrees:
            mismatches.append(
                {"quantity": quantity, "reported": spec["reported"], "computed": value,
                 "recipe": spec["recipe"]}
            )

    for quantity, spec in RAPORT_PILOT.items():
        value = series_numbers.get(quantity)
        rows.append(
            {
                "quantity": quantity,
                "in_raport": spec["reported"],
                "recomputed": "" if value is None else f"{float(value):.4f}",
                "category": "b (pilot — REPLACED, not confirmed)",
                "agrees": "n/a — a pilot estimate is REPLACED by the main series, never confirmed against it",
                "recipe": spec["source"],
            }
        )
    return rows, mismatches


# =====================================================================================
# 8. The analysis proper
# =====================================================================================


def analyse(rows: list[dict], *, predictions_dir, context: dict, prov: dict) -> dict:
    """Every estimand, per cell, plus replication, pooling and the checks."""
    test = accuracy_index(rows, "test")
    val = accuracy_index(rows, "val")
    lrs = selected_lrs(rows)
    dataset_seeds = prov["present"]["dataset_seeds"]
    dilutions = [d for d in CONTRACT_DILUTIONS if d in prov["present"]["dilutions"]]
    ansatz_levels = [a for a in CONTRACT_ANSATZ_LEVELS if a in prov["present"]["ansatz_levels"]]
    seeds = prov["present"]["seeds"]
    arm_d_matched = any(r["arm"] == "D_matched" for r in rows)
    arm_d_best = any(r["arm"] == "D_best" for r in rows)

    def cell_lr_of(dilution, ansatz_level):
        return lambda dataset_seed: lrs["cell_lr"].get((dataset_seed, dilution, ansatz_level))

    def arm_e_lr_of(dilution):
        return lambda dataset_seed: lrs["arm_e_lr"].get((dataset_seed, dilution))

    def d_best_lr_of(width):
        return lambda dataset_seed: lrs["d_best_lr"].get((dataset_seed, width))

    def arm_series(arm, ansatz_level, dilution, lr_of, width=None, index=None):
        return series(
            test if index is None else index,
            dataset_seeds=dataset_seeds, arm=arm, ansatz_level=ansatz_level,
            dilution=dilution, lr_of=lr_of, width=width, seeds=seeds,
        )

    # The ceiling denominator, per generator seed. gates.ceiling() returns
    # max{SVM, MLP} - acc(E linear) for one dataset, so this is the map and never an
    # average, which would describe no dataset. Empty for a seed whose gates were skipped
    # and reported as missing, never replaced by another seed's ceiling.
    ceilings = {
        str(dataset_seed): context["gates"].get(str(dataset_seed), {}).get("ceiling")
        for dataset_seed in dataset_seeds
    }
    ceilings = {k: v for k, v in ceilings.items() if v is not None}

    # --- arm D_best: one point per (dataset seed x M), M selected on validation -------
    d_best: dict = {}
    if arm_d_best:
        widths = sorted({r["width_int"] for r in rows if r["arm"] == "D_best"
                         and r["width_int"] is not None})
        # One M per generator seed: `arm_series` spans every dataset seed, so the series
        # must be filtered to this block before the argmax. Otherwise the per-seed
        # selection is a pooled one wearing a per-seed dict and every seed gets the same M.
        def _for_seed(mapping, dataset_seed):
            return [v for k, v in mapping.items() if k.dataset_seed == dataset_seed]

        for dataset_seed in dataset_seeds:
            per_width = {}
            for width in widths:
                per_width[width] = {
                    "test": arm_summary(_for_seed(arm_series(
                        "D_best", "", "linear", d_best_lr_of(width), width), dataset_seed)),
                    "val": arm_summary(_for_seed(arm_series(
                        "D_best", "", "linear", d_best_lr_of(width), width, index=val),
                        dataset_seed)),
                    "lr_selected": lrs["d_best_lr"].get((dataset_seed, width)),
                }
            available = {w: v for w, v in per_width.items() if v["val"]["n"]}
            selected = (
                min(available, key=lambda w: (-available[w]["val"]["mean"], w))
                if available else None
            )
            d_best[dataset_seed] = {
                "per_width": per_width,
                "selected_width": selected,
                "selection_rule": "highest mean validation accuracy on THIS generator "
                                  "seed's 10 validation runs, ties to the smaller M "
                                  "(the rule A7 used; not re-decided here)",
            }

    # --- per cell ---------------------------------------------------------------------
    cells: dict = {}
    for dilution in dilutions:
        for ansatz_level in ansatz_levels:
            lr_of = cell_lr_of(dilution, ansatz_level)
            arms = {
                "A": arm_series("A", ansatz_level, dilution, lr_of),
                "B": arm_series("B", ansatz_level, dilution, lr_of),
                "E": arm_series("E", "", dilution, arm_e_lr_of(dilution)),
                # F and D_matched are paired at the cell lr: they have no ansatz
                # dimension but the lr does.
                "F": arm_series("F", PRODUCT_ANSATZ, dilution, lr_of),
            }
            if arm_d_matched:
                arms["D_matched"] = arm_series("D_matched", "", dilution, lr_of)

            per_arm = {}
            for arm, values in arms.items():
                ansatz_for_arm = a7.ansatz_of(arm, ansatz_level)
                keys = [
                    key for key in test
                    if key[1] == arm and key[2] == ansatz_for_arm and key[3] == dilution
                    and key[6] == f"{float(lr_of(key[0]) or float('nan')):g}"
                ]
                per_arm[arm] = {
                    "test": arm_summary(list(values.values())),
                    # Per generator seed as well, so sigma_seed can be reported within a
                    # dataset instead of only pooled across datasets of differing
                    # difficulty. See sigma_decomposition.
                    "test_by_dataset": {
                        str(ds): arm_summary(
                            [v for k, v in values.items() if k.dataset_seed == ds]
                        )
                        for ds in dataset_seeds
                    },
                    "val": arm_summary(list(arm_series(
                        arm, ansatz_for_arm, dilution,
                        arm_e_lr_of(dilution) if arm == "E" else lr_of, index=val).values())),
                    # Per generator seed, not just the first one. The seeds do NOT always
                    # agree — at h2 the two ansatz levels selected 0.03 on ds11 and 0.01 on
                    # ds22/ds33 — and reporting seed[0]'s value as "the lr" put a number in
                    # the table that two thirds of the runs were never trained at.
                    "lr_selected": {
                        str(ds): (arm_e_lr_of(dilution)(ds) if arm == "E" else lr_of(ds))
                        for ds in dataset_seeds
                    },
                    "theta_displacement": arm_summary(
                        [float(test[k]["theta_displacement"]) for k in keys]),
                    "best_epoch": arm_summary([float(test[k]["best_epoch_int"]) for k in keys]),
                    "epochs_run": arm_summary([float(test[k]["epochs_run_int"]) for k in keys]),
                }

            pairs = {
                "delta_AB": paired_differences(arms["A"], arms["B"], label="delta_AB"),
                "delta_AE": paired_differences(arms["A"], arms["E"], label="delta_AE"),
                "delta_AF": paired_differences(arms["A"], arms["F"], label="delta_AF"),
                # acc(B) - acc(E) exists only for the decomposition of Delta_AE.
                "_delta_BE": paired_differences(arms["B"], arms["E"], label="delta_BE"),
            }
            if arm_d_matched:
                pairs["delta_BD_matched"] = paired_differences(
                    arms["B"], arms["D_matched"], label="delta_BD_matched")
            if arm_d_best:
                d_best_series = {}
                for dataset_seed in dataset_seeds:
                    selected = d_best.get(dataset_seed, {}).get("selected_width")
                    if selected is None:
                        continue
                    d_best_series.update({
                        k: v for k, v in arm_series(
                            "D_best", "", "linear", d_best_lr_of(selected), selected).items()
                        if k.dataset_seed == dataset_seed
                    })
                if d_best_series:
                    pairs["delta_BD_best"] = paired_differences(
                        arms["B"], d_best_series, label="delta_BD_best")

            estimands = {}
            for name, pair in pairs.items():
                if name.startswith("_"):
                    continue
                # Blocked, not iid: the generator seed is a fixed effect. The point
                # estimate is unchanged; the interval and sigma_Delta come from the
                # within-dataset residual.
                point = estimate_blocked(pair)
                left_arm, right_arm = {
                    "delta_AB": ("A", "B"), "delta_AE": ("A", "E"), "delta_AF": ("A", "F"),
                    "delta_BD_matched": ("B", "D_matched"), "delta_BD_best": ("B", "D_best"),
                }[name]
                point.update(
                    in_reference_units(
                        point,
                        sigma_seed_left=per_arm.get(left_arm, {}).get("test", {}).get("sd"),
                        sigma_seed_right=(
                            per_arm.get(right_arm, {}).get("test", {}).get("sd")
                            if right_arm in per_arm else float("nan")
                        ),
                        ceilings=ceilings,
                    )
                )
                point["unpaired_left"] = [tuple(k) for k in pair["unpaired_left"]]
                point["unpaired_right"] = [tuple(k) for k in pair["unpaired_right"]]
                point["differences"] = pair["differences"]
                point["keys"] = [tuple(k) for k in pair["keys"]]
                point["exploratory"] = name in ("delta_BD_matched", "delta_BD_best")
                estimands[name] = point

            delta_ab_mean = estimands["delta_AB"]["mean"]
            delta_ae_mean = estimands["delta_AE"]["mean"]
            delta_be_mean = estimate(pairs["_delta_BE"]["differences"])["mean"]
            cells[(dilution, ansatz_level)] = {
                "dilution": dilution,
                "ansatz_level": ansatz_level,
                "cell_lr": {str(s): lrs["cell_lr"].get((s, dilution, ansatz_level))
                            for s in dataset_seeds},
                "per_arm": per_arm,
                "estimands": estimands,
                # Computed in every cell.
                "decomposition_delta_AE": {
                    "delta_AB": delta_ab_mean,
                    "acc_B_minus_acc_E": delta_be_mean,
                    "sum": delta_ab_mean + delta_be_mean,
                    "delta_AE": delta_ae_mean,
                    "residual": delta_ae_mean - (delta_ab_mean + delta_be_mean),
                },
                "pairs": pairs,
            }

    # --- replication, then pooling with the dataset as a fixed effect -----------------
    replication: dict = {}
    for name in ("delta_AB", "delta_AE"):
        for ansatz_level in ansatz_levels:
            for dilution in dilutions:
                cell = cells.get((dilution, ansatz_level))
                if cell is None or name not in cell["estimands"]:
                    continue
                pair = cell["pairs"][name]
                per_dataset = {}
                for dataset_seed in dataset_seeds:
                    subset = [
                        float(v) for k, v in zip(pair["keys"], pair["differences"])
                        if k.dataset_seed == dataset_seed
                    ]
                    if subset:
                        per_dataset[dataset_seed] = estimate(subset)
                pooled = pooled_blocked_estimate(pair)
                # The ceiling denominator lives here as well as in the estimand: each
                # dataset is divided by its own ceiling.
                for dataset_seed, point in per_dataset.items():
                    ceiling = ceilings.get(str(dataset_seed))
                    point["ceiling"] = ceiling
                    point["in_ceiling"] = (
                        float(point["mean"] / ceiling)
                        if ceiling not in (None, 0) and np.isfinite(ceiling)
                        else float("nan")
                    )
                replication[(name, dilution, ansatz_level)] = {
                    "per_dataset": per_dataset,
                    "pooled": pooled,
                    "ols_crosscheck": ols_block_crosscheck(pair),
                    "divergence": divergence_check(per_dataset, pooled),
                    "ceilings": dict(ceilings),
                }

    # --- MixedLM, as a check ----------------------------------------------------------
    mixedlm: dict = {}
    for ansatz_level in ansatz_levels:
        by_dilution = {
            dilution: cells[(dilution, ansatz_level)]["pairs"]["delta_AB"]
            for dilution in dilutions if (dilution, ansatz_level) in cells
        }
        if by_dilution:
            mixedlm[ansatz_level] = mixedlm_check(by_dilution)

    # --- TOST -------------------------------------------------------------------------
    tost_results: dict = {}
    for (dilution, ansatz_level), cell in cells.items():
        for name in ("delta_AE", "delta_AB"):
            if name not in cell["estimands"]:
                continue
            differences = cell["pairs"][name]["differences"]
            point = cell["estimands"][name]
            # Run the test on the same residual the headline interval uses: a blocked CI
            # beside an iid TOST is not one analysis.
            if point.get("blocked_computable"):
                record = tost(differences, se=point["se"], df=point["df_residual"],
                              sd=point["sd"])
                record["residual"] = "blocked (within-dataset), SPEC 7.3"
            else:
                record = tost(differences)
                record["residual"] = "iid"
            sigma = point["sd"]
            n = point["n"]
            # The power of the test that was actually run: same df, same residual. Left at
            # n - 1 it reported the power of an iid test beside a blocked one.
            record["power_at_delta_true_zero"] = tost_power(
                sigma=sigma, n=n, df=record["df"]
            )
            record["reported_as_test"] = name in TOST_REPORTED_AS_TEST
            record["seeds_needed_for_80pct_power"] = seeds_needed_for_tost(
                sigma=sigma, blocks=point.get("n_blocks", 1)
            )
            if name in TOST_COMPUTED_NOT_REPORTED:
                record["how_to_report"] = (
                    f"⛔ NOT a verdict. TOST at delta = {TOST_DELTA} for {name} has power "
                    f"{record['power_at_delta_true_zero']:.3f} and is not reported as a "
                    f"test; the 90% CI alone is reported. Declaring equivalence at this "
                    f"delta would need n ~ {record['seeds_needed_for_80pct_power']} seeds. "
                    "This is a methodological contribution to the discussion (SPEC 7.5), "
                    "not a missing result."
                )
            else:
                record["how_to_report"] = (
                    "reported as a test (D-4). A verdict of NON-equivalence at power ~1 is "
                    "a POSITIVE result: the test COULD have declared equivalence and did not."
                )
            tost_results[(dilution, ansatz_level, name)] = record

    # --- McNemar, the third account ---------------------------------------------------
    predictions = correctness_vectors(rows, predictions_dir)
    mcnemar_results: dict = {}
    for (dilution, ansatz_level), cell in cells.items():
        for name, (left, right) in {
            "delta_AB": (("A", ansatz_level, dilution, None), ("B", ansatz_level, dilution, None)),
            "delta_AE": (("A", ansatz_level, dilution, None), ("E", "", dilution, None)),
            "delta_AF": (("A", ansatz_level, dilution, None),
                         ("F", PRODUCT_ANSATZ, dilution, None)),
            "delta_BD_matched": (("B", ansatz_level, dilution, None),
                                 ("D_matched", "", dilution, None)),
        }.items():
            if name not in cell["estimands"]:
                continue
            lr_of = cell_lr_of(dilution, ansatz_level)
            per_seed = {}
            for dataset_seed in dataset_seeds:
                for seed in seeds:
                    def vector(spec, is_arm_e):
                        arm, ansatz, dil, width = spec
                        lr = arm_e_lr_of(dil)(dataset_seed) if is_arm_e else lr_of(dataset_seed)
                        if lr is None:
                            return None
                        return predictions["vectors"].get(
                            (dataset_seed, arm, ansatz, dil, width, seed, f"{float(lr):g}")
                        )

                    left_vector = vector(left, False)
                    right_vector = vector(right, right[0] == "E")
                    if left_vector is None or right_vector is None:
                        continue
                    if left_vector.size != right_vector.size:
                        continue
                    per_seed[(dataset_seed, seed)] = mcnemar_from_vectors(
                        left_vector, right_vector)
            if per_seed:
                b_total = sum(v["b_left_only"] for v in per_seed.values())
                c_total = sum(v["c_right_only"] for v in per_seed.values())
                mcnemar_results[(dilution, ansatz_level, name)] = {
                    "per_seed": per_seed,
                    "b_total": b_total,
                    "c_total": c_total,
                    "mandatory": name == "delta_AE",
                    "mandatory_note": (
                        "MANDATORY for Delta_AE: sigma_Delta(A-E) is SMALLER than the "
                        "binomial SE, so the CI over seeds narrows the uncertainty by "
                        "itself (raport.tex 3.7)." if name == "delta_AE" else
                        "supplementary: the paired reading of the same population as the "
                        "binomial SE"
                    ),
                }

    return {
        "cells": cells,
        "d_best": d_best,
        "replication": replication,
        "mixedlm": mixedlm,
        "tost": tost_results,
        "mcnemar": mcnemar_results,
        "predictions_missing": predictions["missing"],
        "lrs": lrs,
        "axis": {"dilutions": dilutions, "ansatz_levels": ansatz_levels,
                 "dataset_seeds": dataset_seeds, "seeds": seeds},
        "uncertainty_accounts": {
            "1_ci_over_paired_differences": "MAIN. Generalises over initialisations at a fixed split.",
            "2_binomial_se": float(math.sqrt(0.25 / CONTRACT_N_TEST)),
            "3_mcnemar": "same population as (2) but paired, therefore smaller",
            "rule": "⛔ THREE numbers, three labels, one marked main. Never summed, no "
                    "'total uncertainty' is computed.",
        },
        "holm": HOLM_NOTE,
        "confirmatory_family": list(CONFIRMATORY_FAMILY),
    }


def diagnostics(rows: list[dict], analysis: dict, context: dict) -> dict:
    """Section 1.5: theta, the epoch budget, the ridge control, headroom along the axis."""
    test_rows = [r for r in rows if r["split"] == "test"]

    budget = {}
    for arm in sorted({r["arm"] for r in test_rows}):
        arm_rows = [r for r in test_rows if r["arm"] == arm]
        hits = [r for r in arm_rows if (r["epochs_run_int"] or 0) >= MAX_EPOCHS]
        budget[arm] = {
            "runs": len(arm_rows),
            "hit_budget": len(hits),
            "fraction": len(hits) / len(arm_rows) if arm_rows else float("nan"),
            "best_epoch": arm_summary([float(r["best_epoch_int"]) for r in arm_rows
                                       if r["best_epoch_int"] is not None]),
        }

    # Ridge is reported beside the Adam head for B, E and D only: arms A and F have a
    # trained socket, so the closed-form readout has no defined argument.
    ridge = {}
    for arm in ("B", "E", "D_matched", "D_best"):
        pairs = [
            (float(r["accuracy"]), float(r["ridge_accuracy"]))
            for r in test_rows
            if r["arm"] == arm and r["ridge_accuracy"] not in ("", None)
        ]
        if not pairs:
            continue
        ridge[arm] = {
            "n": len(pairs),
            "adam": arm_summary([a for a, _ in pairs]),
            "ridge": arm_summary([g for _, g in pairs]),
            "gap_adam_minus_ridge": arm_summary([a - g for a, g in pairs]),
            "note": "D-19 closed: reported BESIDE, not a consistency condition. The gap is "
                    "the measured price of the readout convention.",
        }
    forbidden = [
        r["arm"] for r in test_rows
        if r["arm"] in a7.TRAINED_ARMS and r["ridge_accuracy"] not in ("", None)
    ]

    # Headroom of every axis point against the ceiling max{SVM, MLP}.
    axis_headroom = {}
    for dataset_seed in analysis["axis"]["dataset_seeds"]:
        gates = context["gates"].get(str(dataset_seed), {})
        strong = gates.get("ceiling_strong_accuracy")
        for dilution in analysis["axis"]["dilutions"]:
            e_rows = [r for r in test_rows if r["arm"] == "E" and r["dilution"] == dilution
                      and r["dataset_seed"] == dataset_seed]
            if not e_rows or strong is None:
                continue
            e_mean = float(np.mean([r["accuracy_float"] for r in e_rows]))
            cell = analysis["cells"].get((dilution, analysis["axis"]["ansatz_levels"][0]))
            mde_here = (
                cell["estimands"]["delta_AB"]["mde"] if cell and "delta_AB" in cell["estimands"]
                else float("nan")
            )
            axis_headroom[(dataset_seed, dilution)] = {
                "acc_E": e_mean,
                "ceiling_strong": float(strong),
                "headroom": float(strong) - e_mean,
                "mde_delta_AB_this_series": mde_here,
                "headroom_exceeds_mde": bool(
                    np.isfinite(mde_here) and (float(strong) - e_mean) >= mde_here),
            }

    theta = {}
    for arm in sorted({r["arm"] for r in test_rows}):
        values = [float(r["theta_displacement"]) for r in test_rows if r["arm"] == arm]
        theta[arm] = arm_summary(values)
        theta[arm]["all_still"] = bool(values and max(values) < THETA_STILL)
        theta[arm]["expected_still"] = arm not in a7.TRAINED_ARMS
    return {
        "epoch_budget": budget,
        "ridge_control": ridge,
        "ridge_forbidden_arms_present": sorted(set(forbidden)),
        "axis_headroom": axis_headroom,
        "theta_displacement": theta,
    }


# =====================================================================================
# 9. The verdict table, declared before the analysis
# =====================================================================================


def verdicts(analysis: dict, diag: dict, mismatches: list[dict], prov: dict) -> list[dict]:
    """Section 6, row by row. Produces verdicts, never an opinion on whether they are good."""
    out: list[dict] = []

    def add(row, verdict, detail, stop=False):
        out.append({"row": row, "verdict": verdict, "detail": detail, "stop": stop})

    if mismatches:
        add(
            "a deterministic number of section 1.6(a) disagrees",
            "⛔ STOP",
            "; ".join(
                f"{m['quantity']}: raport {m['reported']} vs recomputed {m['computed']} "
                f"[{m['recipe']}]" for m in mismatches
            ),
            stop=True,
        )
    else:
        add("section 1.6(a) deterministic numbers", "✅ all reproduce", "no mismatch")

    for (dilution, ansatz_level), cell in sorted(analysis["cells"].items()):
        label = f"{dilution}|{ansatz_level}"
        delta_ab = cell["estimands"].get("delta_AB")
        if delta_ab is None:
            continue
        theta_a = cell["per_arm"].get("A", {}).get("theta_displacement", {})
        theta_still = np.isfinite(theta_a.get("max", float("nan"))) and \
            theta_a["max"] < THETA_STILL

        if theta_still and abs(delta_ab["mean"]) < delta_ab["mde"]:
            add(
                f"{label}: Delta_AB ~ 0 with theta_displacement ~ 0",
                "⛔ STOP",
                "optimiser failure, not 'training is unnecessary' (SPEC 7.7). A different "
                "thesis, and it may not be written in after the fact.",
                stop=True,
            )
        if delta_ab["ci95_excludes_zero"] and delta_ab["mean"] > 0 and dilution == "linear":
            add(
                f"{label}: Delta_AB > 0 at the linear head, CI excludes 0",
                "✅ H1 estimated",
                f"Delta_AB = {delta_ab['mean']:+.4f}, 95% CI "
                f"[{delta_ab['ci95_low']:+.4f}; {delta_ab['ci95_high']:+.4f}], "
                f"= {delta_ab['in_sigma_seed_left']:.2f} sigma_seed(A); "
                f"of the ceiling PER GENERATOR SEED: "
                + ", ".join(f"ds{seed} {value:.3f}"
                            for seed, value in delta_ab["in_ceiling_by_seed"].items())
                + ". Report WITH both denominators.",
            )
        if not delta_ab["above_mde"]:
            n_needed = seeds_needed_for_mde(
                sigma=delta_ab["sd"], effect=abs(delta_ab["mean"]))
            add(
                f"{label}: |Delta_AB| below MDE",
                "🟡 undecidable at this n",
                f"|{delta_ab['mean']:+.4f}| < MDE {delta_ab['mde']:.4f} (n = {delta_ab['n']}). "
                f"⛔ Do NOT write 'there is no effect'. Seeds needed to decide: ~{n_needed}.",
            )

        arm_a = cell["per_arm"].get("A", {}).get("test", {})
        arm_b = cell["per_arm"].get("B", {}).get("test", {})
        arm_e = cell["per_arm"].get("E", {}).get("test", {})
        delta_ae = cell["estimands"].get("delta_AE")
        if delta_ae is not None and arm_b.get("n") and arm_e.get("n"):
            # Judged against the MDE of its own contrast.
            if abs(delta_ae["mean"]) < delta_ae["mde"] and arm_b["mean"] < arm_e["mean"]:
                add(
                    f"{label}: acc(A) ~ acc(E) while acc(B) < acc(E)",
                    "⛔ STOP",
                    "'the socket adds nothing, training only undoes the damage of a random "
                    "initialisation' (SPEC 9) — a result about ARCHITECTURE. Judged against "
                    f"MDE(A-E) = {delta_ae['mde']:.4f}, not MDE(A-B).",
                    stop=True,
                )

        d_matched = cell["estimands"].get("delta_BD_matched")
        if d_matched is not None and abs(d_matched["mean"]) < d_matched["mde"]:
            add(
                f"{label}: Delta_BD_matched ~ 0",
                "🟡 exploratory",
                "permitted sentence: 'this quantum socket is a trigonometric feature map "
                "with support |omega| <= R, and a classical map of the same support and the "
                "same size does the same thing'. ⛔ NOT 'quantumness is useless'.",
            )
        d_best = cell["estimands"].get("delta_BD_best")
        if d_best is not None and d_best["mean"] < 0:
            add(
                f"{label}: Delta_BD_best < 0",
                "🟡 exploratory, expected",
                "permitted: 'a correctly sized classical random-feature model beats the "
                "frozen quantum socket'. A strong result about BASELINES, not about "
                "quantumness — D_best has a different feature count, so the comparison is "
                "not matched and that must be written.",
            )

    # The dilution curve is descriptive, never a test of H1.
    for ansatz_level in analysis["axis"]["ansatz_levels"]:
        curve = [
            (dilution, analysis["cells"][(dilution, ansatz_level)]["estimands"]["delta_AB"]["mean"])
            for dilution in analysis["axis"]["dilutions"]
            if (dilution, ansatz_level) in analysis["cells"]
            and "delta_AB" in analysis["cells"][(dilution, ansatz_level)]["estimands"]
        ]
        if len(curve) < 2:
            continue
        values = [v for _, v in curve]
        monotone = all(b <= a for a, b in zip(values, values[1:]))
        add(
            f"P1 curve, ansatz {ansatz_level}",
            "🟡 descriptive" + (" (monotone decreasing)" if monotone else " (not decreasing)"),
            "⛔ does NOT 'confirm H1'. At h42 arm E beats the strong model, so Delta -> 0 "
            "happens BECAUSE OF THE HEAD (SPEC 7.9). "
            + ("Not decreasing is an honest, publishable result: 'freezing costs this "
               "much, independently of dilution'." if not monotone else ""),
        )

    for ansatz_level, record in sorted(analysis["mixedlm"].items()):
        if not record.get("computable"):
            continue
        numbers = (
            f"intercept {record['intercept']:+.6f} vs reference-cell "
            f"({record['reference_level']}) mean "
            f"{record['reference_level_mean_for_comparison']:+.6f}, difference "
            f"{record['intercept_minus_reference_mean']:+.2e} against a tolerance of "
            f"{record['equivalence_tolerance']:g}"
        )
        if record["equivalent_to_paired_contrast"]:
            add(f"MixedLM vs paired contrast, ansatz {ansatz_level}",
                "✅ equivalent, as predicted", numbers)
        else:
            add(
                f"MixedLM vs paired contrast, ansatz {ansatz_level}",
                "🟡 report, do not choose",
                f"{numbers}. The report predicts equivalence; its absence is a finding "
                "about the model.",
            )

    for (dilution, ansatz_level, name), record in sorted(analysis["tost"].items()):
        if name != "delta_AE" or not record.get("computable"):
            continue
        power = record["power_at_delta_true_zero"]
        if np.isfinite(power) and power < TOST_POWER_D4_FLOOR:
            add(
                f"{dilution}|{ansatz_level}: TOST power for A<->E dropped to {power:.3f}",
                "⛔ STOP and report",
                "D-4 ('TOST stays for Delta_AE') stands on power 1.000. Below ~0.8 that "
                "decision has to be reopened — and that is NOT A8's decision.",
                stop=True,
            )

    if diag["ridge_forbidden_arms_present"]:
        add(
            "ridge for a trained-socket arm",
            "⛔ STOP",
            f"arms {diag['ridge_forbidden_arms_present']} carry a ridge accuracy. The "
            "closed-form readout has no defined argument for a trained socket.",
            stop=True,
        )

    if not prov["complete_contract_grid"]:
        add(
            "input coverage",
            "⛔ PROVISIONAL — NOT A RESULT",
            f"missing {prov['missing']}. The schema is real, the numbers are not results. "
            "Missing rows are reported, never filled in.",
        )
    if not prov["arm_D_present"]:
        add(
            "arm D absent from the CSV",
            "🟡 Delta_BD skipped, with a reason",
            "a run without arm D is allowed — it is an exploratory passenger. "
            "⛔ Nothing is substituted for D and no Delta_BD is computed from another arm.",
        )
    return out


# =====================================================================================
# 10. Tables — on disk before any figure is drawn
# =====================================================================================

ESTIMAND_COLUMNS: tuple[str, ...] = (
    "status", "estimand", "exploratory", "generator_seed", "dilution", "ansatz",
    "cell_lr", "n", "mean", "ci95_low", "ci95_high", "ci90_low", "ci90_high",
    "n_blocks", "df_residual",
    "sigma_delta", "sigma_ci95_low", "sigma_ci95_high", "mde_this_series", "mde_constant",
    "above_mde", "ci95_excludes_zero", "p_sign_exact", "p_wilcoxon_exact", "p_t",
    "n_positive", "n_negative", "in_sigma_seed_left", "in_sigma_seed_right",
    "in_sigma_delta", "in_ceiling", "ceiling_used", "in_ceiling_by_seed",
    "ceiling_by_seed", "unpaired_left", "unpaired_right",
)

ARM_COLUMNS: tuple[str, ...] = (
    "status", "generator_seed", "dilution", "ansatz", "arm_carries_ansatz", "arm", "lr",
    "split",
    "n", "mean", "sigma_seed", "sigma_ci95_low", "sigma_ci95_high", "min", "max",
    "theta_displacement_mean", "theta_displacement_max", "best_epoch_mean",
    "epochs_run_mean",
)

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "status", "quantity", "scope", "value", "note",
)

REPLICATION_COLUMNS: tuple[str, ...] = (
    "status", "estimand", "dilution", "ansatz", "level", "generator_seed", "n",
    "mean", "ci95_low", "ci95_high", "p_sign_exact", "sigma", "df",
    "ceiling", "in_ceiling",
    "diverges_from_pooled", "divergence_rule",
)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def estimand_rows(analysis: dict, status: str) -> list[dict]:
    rows = []
    for (dilution, ansatz_level), cell in sorted(analysis["cells"].items()):
        for name, point in sorted(cell["estimands"].items()):
            rows.append(
                {
                    "status": status,
                    "estimand": name,
                    "exploratory": "yes" if point["exploratory"] else "no",
                    "generator_seed": "all present in the CSV",
                    "dilution": dilution,
                    "ansatz": ansatz_level,
                    "cell_lr": json.dumps(cell["cell_lr"], sort_keys=True),
                    "n": point["n"],
                    # The blocked design spends one degree of freedom per generator seed.
                    # Both travel with the row so a reader recomputing the power or the
                    # sample size uses the df the test was actually run on.
                    "n_blocks": point.get("n_blocks", 1),
                    "df_residual": point.get("df_residual", point["n"] - 1),
                    "mean": _fmt(point["mean"]),
                    "ci95_low": _fmt(point.get("ci95_low")),
                    "ci95_high": _fmt(point.get("ci95_high")),
                    "ci90_low": _fmt(point.get("ci90_low")),
                    "ci90_high": _fmt(point.get("ci90_high")),
                    "sigma_delta": _fmt(point["sd"]),
                    "sigma_ci95_low": _fmt(point.get("sigma_ci95_low")),
                    "sigma_ci95_high": _fmt(point.get("sigma_ci95_high")),
                    "mde_this_series": _fmt(point.get("mde")),
                    "mde_constant": _fmt(point.get("mde_constant")),
                    "above_mde": "yes" if point["above_mde"] else "no",
                    "ci95_excludes_zero": "yes" if point["ci95_excludes_zero"] else "no",
                    "p_sign_exact": _fmt(point["p_sign_exact"]),
                    "p_wilcoxon_exact": _fmt(point["p_wilcoxon_exact"]),
                    "p_t": _fmt(point["p_t"]),
                    "n_positive": point["sign_positive"],
                    "n_negative": point["sign_negative"],
                    "in_sigma_seed_left": _fmt(point["in_sigma_seed_left"]),
                    "in_sigma_seed_right": _fmt(point["in_sigma_seed_right"]),
                    "in_sigma_delta": _fmt(point["in_sigma_delta"]),
                    "in_ceiling": _fmt(point["in_ceiling"]),
                    "ceiling_used": _fmt(point["ceiling_used"]),
                    "in_ceiling_by_seed": json.dumps(
                        {k: round(v, 6) for k, v in point["in_ceiling_by_seed"].items()},
                        sort_keys=True),
                    "ceiling_by_seed": json.dumps(
                        {k: round(v, 6) for k, v in point["ceiling_by_seed"].items()},
                        sort_keys=True),
                    "unpaired_left": len(point["unpaired_left"]),
                    "unpaired_right": len(point["unpaired_right"]),
                }
            )
    return rows


def _lr_column(lr_by_seed) -> str:
    """One lr when every generator seed agrees, otherwise all of them, seed by seed.

    Collapsing a disagreement to a single value is how a cell whose seeds trained at
    0.03 and 0.01 came to be labelled "0.03".
    """
    if not isinstance(lr_by_seed, dict):
        return _fmt(lr_by_seed)
    values = {v for v in lr_by_seed.values() if v is not None}
    if len(values) == 1:
        return _fmt(next(iter(values)))
    return json.dumps({k: v for k, v in sorted(lr_by_seed.items())}, sort_keys=True)


def arm_rows(analysis: dict, status: str) -> list[dict]:
    rows = []
    for (dilution, ansatz_level), cell in sorted(analysis["cells"].items()):
        for arm, record in sorted(cell["per_arm"].items()):
            for split in ("test", "val"):
                stats = record[split]
                rows.append(
                    {
                        "status": status,
                        "generator_seed": "all present in the CSV",
                        "dilution": dilution,
                        # The ansatz level of the CELL, always. The ansatz-free arms are
                        # read once per level — at the cell lr of that level, which is a
                        # different reading when the levels chose different lr — so blanking
                        # this column emitted two rows that carried different numbers and
                        # could not be told apart.
                        "ansatz": ansatz_level,
                        "arm_carries_ansatz": "yes" if arm in ("A", "B") else "no",
                        "arm": arm,
                        "lr": _lr_column(record["lr_selected"]),
                        "split": split,
                        "n": stats["n"],
                        "mean": _fmt(stats["mean"]),
                        "sigma_seed": _fmt(stats["sd"]),
                        "sigma_ci95_low": _fmt(stats["sigma_ci95_low"]),
                        "sigma_ci95_high": _fmt(stats["sigma_ci95_high"]),
                        "min": _fmt(stats["min"]),
                        "max": _fmt(stats["max"]),
                        "theta_displacement_mean": _fmt(record["theta_displacement"]["mean"]),
                        "theta_displacement_max": _fmt(record["theta_displacement"]["max"]),
                        "best_epoch_mean": _fmt(record["best_epoch"]["mean"]),
                        "epochs_run_mean": _fmt(record["epochs_run"]["mean"]),
                    }
                )
    return rows


def replication_rows(analysis: dict, status: str) -> list[dict]:
    rows = []
    for (name, dilution, ansatz_level), record in sorted(analysis["replication"].items()):
        divergence = record["divergence"]
        for dataset_seed, point in sorted(record["per_dataset"].items()):
            rows.append(
                {
                    "status": status,
                    "estimand": name,
                    "dilution": dilution,
                    "ansatz": ansatz_level,
                    "level": "per dataset",
                    "generator_seed": dataset_seed,
                    "n": point["n"],
                    "mean": _fmt(point["mean"]),
                    "ci95_low": _fmt(point.get("ci95_low")),
                    "ci95_high": _fmt(point.get("ci95_high")),
                    "p_sign_exact": _fmt(point["p_sign_exact"]),
                    "sigma": _fmt(point["sd"]),
                    "df": point["n"] - 1 if point["n"] > 1 else "",
                    "ceiling": _fmt(point.get("ceiling")),
                    "in_ceiling": _fmt(point.get("in_ceiling")),
                    "diverges_from_pooled": (
                        "YES" if dataset_seed in divergence.get("diverging_dataset_seeds", [])
                        else "no"
                    ),
                    "divergence_rule": divergence["rule"],
                }
            )
        pooled = record["pooled"]
        if pooled.get("computable"):
            rows.append(
                {
                    "status": status,
                    "estimand": name,
                    "dilution": dilution,
                    "ansatz": ansatz_level,
                    "level": "pooled (generator seed = FIXED effect, NOT a variance component)",
                    "generator_seed": "|".join(str(b) for b in pooled["blocks"]),
                    "n": pooled["n"],
                    "mean": _fmt(pooled["mean"]),
                    "ci95_low": _fmt(pooled.get("ci95_low")),
                    "ci95_high": _fmt(pooled.get("ci95_high")),
                    "p_sign_exact": _fmt(pooled.get("p_sign_exact")),
                    "sigma": _fmt(pooled.get("sigma_within")),
                    "df": pooled.get("df_residual", ""),
                    # Deliberately blank: a pooled ceiling denominator describes no
                    # dataset. Read it off the per-dataset rows instead.
                    "ceiling": "per generator seed — see the rows above",
                    "in_ceiling": "per generator seed — see the rows above",
                    "diverges_from_pooled": (
                        "DIVERGENCE: " + str(divergence.get("diverging_dataset_seeds"))
                        if divergence.get("diverged") else "no"
                    ),
                    "divergence_rule": divergence["rule"],
                }
            )
    return rows


def diagnostic_rows(analysis: dict, diag: dict, status: str) -> list[dict]:
    rows = []

    def add(quantity, scope, value, note=""):
        rows.append({"status": status, "quantity": quantity, "scope": scope,
                     "value": _fmt(value), "note": note})

    for arm, record in sorted(diag["theta_displacement"].items()):
        add("theta_displacement mean", f"arm {arm}", record["mean"],
            "socket frozen — 0 is correct here" if record["expected_still"]
            else "socket trained — 0 would be an optimiser failure (SPEC 7.7)")
        add("theta_displacement CI95 on sigma", f"arm {arm}",
            f"[{_fmt(record['sigma_ci95_low'])}; {_fmt(record['sigma_ci95_high'])}]", "")
    for arm, record in sorted(diag["epoch_budget"].items()):
        add(f"runs hitting the {MAX_EPOCHS}-epoch budget", f"arm {arm}",
            f"{record['hit_budget']}/{record['runs']}",
            f"fraction {record['fraction']:.3f}")
        add("best_epoch mean", f"arm {arm}", record["best_epoch"]["mean"], "")
    for arm, record in sorted(diag["ridge_control"].items()):
        add("ridge control: adam - ridge", f"arm {arm}", record["gap_adam_minus_ridge"]["mean"],
            record["note"])
    for (dataset_seed, dilution), record in sorted(diag["axis_headroom"].items()):
        add("headroom against the ceiling max{SVM,MLP}", f"ds{dataset_seed}|{dilution}",
            record["headroom"],
            f"acc(E) {record['acc_E']:.4f}, ceiling {record['ceiling_strong']:.4f}, "
            f"exceeds MDE: {'yes' if record['headroom_exceeds_mde'] else 'no'}")
    for (dilution, ansatz_level, name), record in sorted(analysis["tost"].items()):
        if not record.get("computable"):
            continue
        add(f"TOST delta={TOST_DELTA} — p", f"{dilution}|{ansatz_level}|{name}",
            record["pvalue"],
            ("REPORTED as a test (D-4)" if record["reported_as_test"]
             else "⛔ NOT reported as a test — see the power row below"))
        add(f"TOST delta={TOST_DELTA} — 90% CI", f"{dilution}|{ansatz_level}|{name}",
            f"[{record['ci90_low']:+.4f}; {record['ci90_high']:+.4f}]",
            f"half width {record['ci90_half_width']:.4f}"
            + (" > delta — NO result could declare equivalence"
               if record["half_width_exceeds_delta"] else ""))
        add(f"TOST delta={TOST_DELTA} — moc przy Δ=0", f"{dilution}|{ansatz_level}|{name}",
            record["power_at_delta_true_zero"], record["how_to_report"])
        add("seeds needed for 80% TOST power", f"{dilution}|{ansatz_level}|{name}",
            record["seeds_needed_for_80pct_power"], "")
    for (dilution, ansatz_level, name), record in sorted(analysis["mcnemar"].items()):
        add("McNemar b/c (summed over seeds)", f"{dilution}|{ansatz_level}|{name}",
            f"b={record['b_total']} c={record['c_total']}", record["mandatory_note"])
        for (dataset_seed, seed), per_seed in sorted(record["per_seed"].items()):
            add("McNemar p (exact, per seed)",
                f"{dilution}|{ansatz_level}|{name}|ds{dataset_seed}|seed{seed}",
                per_seed["pvalue"],
                f"b={per_seed['b_left_only']} c={per_seed['c_right_only']} "
                f"discordant {per_seed['discordant']}")
    for (dilution, ansatz_level), cell in sorted(analysis["cells"].items()):
        decomposition = cell["decomposition_delta_AE"]
        add("decomposition Δ_AE = Δ_AB + (acc(B)-acc(E))", f"{dilution}|{ansatz_level}",
            f"{decomposition['delta_AB']:+.6f} + {decomposition['acc_B_minus_acc_E']:+.6f} "
            f"= {decomposition['sum']:+.6f}",
            f"Δ_AE {decomposition['delta_AE']:+.6f}, residual {decomposition['residual']:+.2e}")
    for claim in pilot_power_audit():
        add("TOST power recomputed at the ORIGINAL sigma", claim["label"],
            f"{claim['recomputed']:.4f}",
            f"documents say {claim['reported']:.3f} ({claim['source']}); "
            + ("agrees" if claim["agrees_at_3dp"] else
               "⚠️ DIVERGENCE — category 1.6(b), reported, NOT a STOP; "
               + ("changes a decision" if claim["changes_a_decision"]
                  else "changes no decision: both values sit on the same side of the "
                       f"{TOST_POWER_FLOOR} floor")))
    accounts = analysis["uncertainty_accounts"]
    add("account 1 — CI over the paired differences", "MAIN", accounts["1_ci_over_paired_differences"], "")
    add("account 2 — binomial SE", "1200 test rows", accounts["2_binomial_se"], "")
    add("account 3 — McNemar", "discordant pairs", accounts["3_mcnemar"], accounts["rule"])
    add("Holm correction", "confirmatory family", "NOT APPLIED", analysis["holm"])
    for ansatz_level, record in sorted(analysis["mixedlm"].items()):
        add("MixedLM (check)", f"ansatz {ansatz_level}",
            _fmt(record.get("intercept")), record.get("note", "") + " " + record.get("reason", ""))
    return rows


def write_tables(out_dir: Path, analysis: dict, diag: dict, verification: list[dict],
                 verdict_rows: list[dict], prov: dict, status: str) -> dict:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written = {}

    def dump(name, rows, columns=None):
        frame = pd.DataFrame(rows)
        if columns is not None:
            frame = frame.reindex(columns=list(columns))
        path = tables / name
        frame.to_csv(path, index=False)
        written[name] = len(frame)
        return path

    dump("00_provenance.csv", [
        {"field": key, "value": json.dumps(value, sort_keys=True, ensure_ascii=False)
         if isinstance(value, (dict, list)) else value}
        for key, value in prov.items()
    ])
    dump("estimands.csv", estimand_rows(analysis, status), ESTIMAND_COLUMNS)
    dump("arms.csv", arm_rows(analysis, status), ARM_COLUMNS)
    dump("replication.csv", replication_rows(analysis, status), REPLICATION_COLUMNS)
    dump("diagnostics.csv", diagnostic_rows(analysis, diag, status), DIAGNOSTIC_COLUMNS)
    dump("raport_verification.csv", verification,
         ("quantity", "in_raport", "recomputed", "category", "agrees", "recipe"))
    dump("verdicts.csv", [
        {"status": status, "row": v["row"], "verdict": v["verdict"],
         "detail": v["detail"], "stop": "YES" if v["stop"] else "no"}
        for v in verdict_rows
    ])
    return written


# =====================================================================================
# 11. Figures — drawn from the tables, after they are on disk
# =====================================================================================

# Distinct marker and linestyle per series: colour is never the only carrier.
MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", "--", ":", "-.")


def _stamp(figure, status: str) -> None:
    if status == STATUS_COMPLETE:
        return
    figure.text(
        0.5, 0.5, "PROVISIONAL\nNOT A RESULT", fontsize=34, color="0.82",
        ha="center", va="center", rotation=28, zorder=0, alpha=0.9,
    )


def draw_figures(out_dir: Path, analysis: dict, diag: dict, rows: list[dict],
                 status: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    dilutions = analysis["axis"]["dilutions"]
    ansatz_levels = analysis["axis"]["ansatz_levels"]
    x = np.arange(len(dilutions), dtype=float)

    def save(figure, name):
        figure.savefig(figures_dir / name, format="pdf", bbox_inches="tight")
        plt.close(figure)
        written.append(name)

    # 1 — one figure per estimand, one panel per generator seed.
    # Everything on one pair of axes was unreadable: two estimands x two ansatzes x three
    # datasets, plus an MDE line and three ceilings. Split twice — Delta_AB and Delta_AE
    # get their own figure, and each figure gets one panel per generator seed — so a panel
    # carries two series and its own ceiling and nothing else.
    dataset_seeds = sorted(analysis["axis"]["dataset_seeds"])

    def draw_axis_figure(name: str, label: str, tag: str, number: str) -> None:
        figure, axes = plt.subplots(
            1, max(len(dataset_seeds), 1), figsize=(3.1 * max(len(dataset_seeds), 1), 3.7),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        # The MDE is not one number along the axis: sigma_Delta grows by a factor of
        # about three from the linear head to h2, so a single flat line taken from one
        # cell would understate what h2 could not have detected. The band follows the
        # per-dilution MDE, taking the worse ansatz, and is drawn as steps so it reads as
        # one value per cell rather than an interpolation between cells.
        mde_per_dilution = []
        for dilution in dilutions:
            values = [
                analysis["cells"][(dilution, level)]["estimands"][name]["mde"]
                for level in ansatz_levels
                if (dilution, level) in analysis["cells"]
                and name in analysis["cells"][(dilution, level)]["estimands"]
            ]
            values = [v for v in values if np.isfinite(v)]
            mde_per_dilution.append(max(values) if values else float("nan"))
        band_drawn = any(np.isfinite(v) for v in mde_per_dilution)

        for panel, dataset_seed in zip(axes, dataset_seeds):
            if band_drawn:
                # Padded to the panel edges so the band does not stop half a cell short
                # of the frame and read as missing at the ends.
                xb = np.concatenate(([x[0] - 0.5], x, [x[-1] + 0.5]))
                yb = np.array([mde_per_dilution[0], *mde_per_dilution,
                               mde_per_dilution[-1]], dtype=float)
                panel.fill_between(
                    xb, -yb, yb, step="mid", color="0.55", alpha=0.16, linewidth=0,
                    zorder=0, label="±MDE of this cell (worse ansatz)",
                )
            # Zero is the line every point is read against, so it is the darkest thing
            # on the panel.
            panel.axhline(0.0, color="black", linewidth=1.4, zorder=1)
            for j, ansatz_level in enumerate(ansatz_levels):
                means, low, high, xs = [], [], [], []
                for k, dilution in enumerate(dilutions):
                    record = analysis["replication"].get((name, dilution, ansatz_level))
                    point = (record or {}).get("per_dataset", {}).get(dataset_seed)
                    if point is None:
                        continue
                    xs.append(x[k] + 0.07 * (j - (len(ansatz_levels) - 1) / 2))
                    means.append(point["mean"])
                    low.append(point["mean"] - point.get("ci95_low", point["mean"]))
                    high.append(point.get("ci95_high", point["mean"]) - point["mean"])
                if not xs:
                    continue
                panel.errorbar(
                    xs, means, yerr=[low, high], marker=MARKERS[j % len(MARKERS)],
                    linestyle=LINESTYLES[j % len(LINESTYLES)], capsize=3, markersize=4.5,
                    linewidth=1.2, zorder=3, label=f"{label}, {ansatz_level}",
                )
            # The ceiling of THIS dataset seed: max{SVM, MLP} - acc(E linear), read off
            # the linear head, so "small" can be read against what is available here.
            record = diag["axis_headroom"].get((dataset_seed, "linear"))
            if record is not None:
                ceiling = record["ceiling_strong"] - record["acc_E"]
                panel.axhline(ceiling, linestyle=(0, (1, 1.5)), color="0.15",
                              linewidth=1.0, zorder=2,
                              label="ceiling of this generator seed")
                # The value is annotated in the panel: it differs per seed, so a shared
                # legend cannot carry it.
                panel.annotate(
                    f"ceiling {ceiling:.3f}", xy=(0.02, ceiling),
                    xycoords=panel.get_yaxis_transform(), xytext=(0, 2),
                    textcoords="offset points", fontsize=6, color="0.15", va="bottom")
            panel.set_xticks(x)
            panel.set_xticklabels(dilutions, fontsize=7)
            panel.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            panel.set_title(f"generator seed {dataset_seed}", fontsize=9)
            panel.set_xlabel("dilution", fontsize=8)
            panel.tick_params(labelsize=7)
        axes[0].set_ylabel(f"{label} (accuracy units, not normalised)", fontsize=8)
        # One legend for the whole row: the series are the same in every panel.
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, fontsize=7, loc="lower center",
                      ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, -0.11))
        figure.suptitle(
            f"{number}. {label} along the dilution axis, per generator seed, 95 % CI",
            fontsize=10,
        )
        _stamp(figure, status)
        save(figure, f"fig1{tag}_{name.lower()}_along_axis.pdf")

    draw_axis_figure("delta_AB", "Δ_AB", "a", "1a")
    draw_axis_figure("delta_AE", "Δ_AE", "b", "1b")

    # 2 — accuracy of every arm per axis point, one point per training seed.
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    test_rows = [r for r in rows if r["split"] == "test"]
    arms = sorted({r["arm"] for r in test_rows})
    for i, arm in enumerate(arms):
        xs, ys = [], []
        for k, dilution in enumerate(dilutions):
            for row in test_rows:
                if row["arm"] != arm or row["dilution"] != dilution:
                    continue
                xs.append(x[k] + 0.07 * (i - len(arms) / 2))
                ys.append(row["accuracy_float"])
        if xs:
            axis.scatter(xs, ys, marker=MARKERS[i % len(MARKERS)], s=26, label=arm,
                         edgecolors="black", linewidths=0.4)
    axis.set_xticks(x)
    axis.set_xticklabels(dilutions)
    axis.set_xlabel("dilution")
    axis.set_ylabel("accuracy on the 1200 test rows")
    axis.set_title("2. Every arm, one point per seed — the scatter a mean does not show")
    axis.legend(fontsize=7, ncol=2)
    _stamp(figure, status)
    save(figure, "fig2_arms_per_seed.pdf")

    # 5 — best_epoch per arm with the epoch-budget line.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    data, names = [], []
    for arm in arms:
        values = [float(r["best_epoch_int"]) for r in test_rows
                  if r["arm"] == arm and r["best_epoch_int"] is not None]
        if values:
            data.append(values)
            names.append(arm)
    if data:
        axis.boxplot(data, tick_labels=names, showmeans=True)
        for i, values in enumerate(data, start=1):
            axis.scatter([i] * len(values), values, s=16, color="0.35", zorder=3,
                         edgecolors="black", linewidths=0.3)
    axis.axhline(MAX_EPOCHS, linestyle="--", color="black",
                 label=f"{MAX_EPOCHS}-epoch budget")
    axis.set_ylabel("best_epoch")
    axis.set_title("5. best_epoch distribution per arm, with the budget line")
    axis.legend(fontsize=7)
    _stamp(figure, status)
    save(figure, "fig5_best_epoch.pdf")
    return written


# =====================================================================================
# 12. Console report and entry point
# =====================================================================================


def report(prov: dict, analysis: dict, diag: dict, verification: list[dict],
           mismatches: list[dict], verdict_rows: list[dict], status: str,
           written: dict, figures: list[str], wall: float) -> None:
    line = "=" * 86
    print(line)
    print("A8 — analysis pipeline. Methodology: docs/raport.tex 3.7. A8 decides nothing.")
    print(line)
    if status != STATUS_COMPLETE:
        print()
        print("#" * 86)
        print(f"#  {STATUS_PROVISIONAL}")
        print("#  The input is not the complete contract grid. The schema is real, the")
        print("#  numbers are NOT results and may not be reported as such.")
        print(f"#  missing: {json.dumps(prov['missing'], sort_keys=True)}")
        print(f"#  training seeds present: {prov['present']['seeds']}")
        print("#" * 86)
    print()
    print(f"input      {prov['results_csv']}  ({prov['n_rows']} rows)")
    print(f"arms       {prov['present']['arms']}")
    print(f"dilutions  {prov['present']['dilutions']}   ansatzes {prov['present']['ansatz_levels']}")
    print(f"datasets   {prov['present']['dataset_seeds']}   seeds {prov['present']['seeds']}")

    print()
    print("-- ESTIMANDS (paired contrast on the differences per seed) " + "-" * 27)
    for (dilution, ansatz_level), cell in sorted(analysis["cells"].items()):
        print(f"  cell {dilution}|{ansatz_level}   lr {cell['cell_lr']}")
        for name, point in sorted(cell["estimands"].items()):
            tag = " [EXPLORATORY]" if point["exploratory"] else ""
            print(
                f"    {name:<18} n {point['n']:>2}  mean {point['mean']:+.6f}  "
                f"95% CI [{point.get('ci95_low', float('nan')):+.6f}; "
                f"{point.get('ci95_high', float('nan')):+.6f}]  "
                f"sigma_d {point['sd']:.6f}  MDE {point.get('mde', float('nan')):.6f}  "
                f"p_sign {point['p_sign_exact']:.4f}  p_wilcoxon {point['p_wilcoxon_exact']:.4f}"
                f"{tag}"
            )
        decomposition = cell["decomposition_delta_AE"]
        print(
            f"    decomposition  {decomposition['delta_AB']:+.6f} "
            f"+ {decomposition['acc_B_minus_acc_E']:+.6f} = {decomposition['sum']:+.6f}  "
            f"(Δ_AE {decomposition['delta_AE']:+.6f}, residual "
            f"{decomposition['residual']:+.2e})"
        )

    print()
    print("-- REPLICATION and POOLED (dataset seed = FIXED effect) " + "-" * 30)
    for (name, dilution, ansatz_level), record in sorted(analysis["replication"].items()):
        print(f"  {name} {dilution}|{ansatz_level}")
        for dataset_seed, point in sorted(record["per_dataset"].items()):
            print(
                f"    ds{dataset_seed}  n {point['n']:>2}  {point['mean']:+.6f}  "
                f"95% CI [{point.get('ci95_low', float('nan')):+.6f}; "
                f"{point.get('ci95_high', float('nan')):+.6f}]  p_sign {point['p_sign_exact']:.4f}"
            )
        pooled = record["pooled"]
        if pooled.get("computable"):
            print(
                f"    pooled n {pooled['n']} over {pooled['n_blocks']} block(s)  "
                f"{pooled['mean']:+.6f}  95% CI "
                f"[{pooled.get('ci95_low', float('nan')):+.6f}; "
                f"{pooled.get('ci95_high', float('nan')):+.6f}]  df {pooled['df_residual']}"
            )
        print(f"    divergence: {record['divergence']['verdict']}")

    print()
    print("-- TOST " + "-" * 78)
    for (dilution, ansatz_level, name), record in sorted(analysis["tost"].items()):
        if not record.get("computable"):
            continue
        print(f"  {dilution}|{ansatz_level} {name}: p {record['pvalue']:.4f}  90% CI "
              f"[{record['ci90_low']:+.6f}; {record['ci90_high']:+.6f}]  "
              f"power@Δ=0 {record['power_at_delta_true_zero']:.3f}")
        print(f"    {record['how_to_report']}")

    print()
    print("-- McNEMAR (third uncertainty account; never summed with the others) " + "-" * 18)
    for (dilution, ansatz_level, name), record in sorted(analysis["mcnemar"].items()):
        mark = "MANDATORY" if record["mandatory"] else "supplementary"
        print(f"  {dilution}|{ansatz_level} {name:<18} b {record['b_total']:>5} "
              f"c {record['c_total']:>5}   [{mark}]")
    if analysis["predictions_missing"]:
        print(f"  ⚠️ prediction files missing: {len(analysis['predictions_missing'])}")

    print()
    print("-- SECTION 1.6 AUDIT OF raport.tex " + "-" * 51)
    for row in verification:
        if row["category"].startswith("a"):
            print(f"  [{row['agrees']:<12}] {row['quantity']:<32} raport {row['in_raport']!s:<20} "
                  f"recomputed {row['recomputed']}")
    print("  published TOST power figures, recomputed AT THEIR OWN sigma:")
    for claim in pilot_power_audit():
        mark = "YES" if claim["agrees_at_3dp"] else "⚠️ DIVERGENCE (category b, not a STOP)"
        print(f"    [{mark:<28}] {claim['label']:<42} docs {claim['reported']:.3f}  "
              f"recomputed {claim['recomputed']:.4f}   [{claim['source']}]")
    print("  category (b), pilot — REPLACED, never confirmed:")
    for row in verification:
        if row["category"].startswith("b"):
            print(f"    {row['quantity']:<28} raport {row['in_raport']!s:<10} "
                  f"this series {row['recomputed'] or '—'}")

    decomposition = (analysis.get("series_numbers") or {}).get("_decomposition") or {}
    if decomposition:
        seed = decomposition.get("pilot_generator_seed")
        print(f"  sigma DECOMPOSED — the pilot measured ds{seed} alone, so only the "
              f"'ds{seed}' column compares like for like:")
        print(f"    {'quantity':<18} {'ds'+str(seed):>9} {'within':>9} {'between':>9} "
              f"{'pooled':>9} {'headline':>9}")
        for key, block in sorted(decomposition.items()):
            if not isinstance(block, dict):
                continue
            def fmt(value):
                return f"{value:9.4f}" if isinstance(value, (int, float)) else f"{'—':>9}"
            print(f"    {key:<18} {fmt(block.get('like_for_like'))} "
                  f"{fmt(block.get('within_rms'))} {fmt(block.get('between_spread'))} "
                  f"{fmt(block.get('pooled_iid'))} {fmt(block.get('blocked_headline'))}")
        print("    ⚠️ 'pooled' contains the difference in difficulty BETWEEN the three "
              "datasets and is therefore NOT comparable to the pilot.")

    print()
    print("-- VERDICTS (table declared BEFORE the analysis) " + "-" * 37)
    for verdict in verdict_rows:
        print(f"  {verdict['verdict']:<28} {verdict['row']}")
        print(f"      {verdict['detail']}")

    print()
    print(f"tables  {json.dumps(written, sort_keys=True)}")
    print(f"figures {figures}")
    print(f"wall    {wall:.1f} s")
    if mismatches:
        print()
        print("⛔ STOP: a deterministic number of section 1.6(a) does not reproduce. "
              "Both values and the recipe are above. Do NOT 'fix' either side — it is "
              "unknown which one is wrong.")


def run(*, results_csv: Path, out_dir: Path, predictions_dir, skip_slow: bool,
        no_figures: bool) -> int:
    started = time.perf_counter()
    results_csv = Path(results_csv)
    out_dir = Path(out_dir)
    # A list, because the driver writes one run directory per generator seed: a combined
    # CSV over three seeds has its prediction files in three places, and taking only one
    # of them computes McNemar on a third of the data without saying so.
    if not predictions_dir:
        predictions_dirs = [results_csv.parent / a7.PREDICTIONS_DIR]
    elif isinstance(predictions_dir, (str, Path)):
        predictions_dirs = [Path(predictions_dir)]
    else:
        predictions_dirs = [Path(d) for d in predictions_dir]

    rows = load_rows(results_csv)
    prov = provenance(rows, results_csv=results_csv)
    status = prov["status"]

    context = verification_context(
        prov["present"]["dataset_seeds"],
        cache_path=out_dir / "verification_cache.json",
        skip_slow=skip_slow,
    )
    analysis = analyse(rows, predictions_dir=predictions_dirs, context=context, prov=prov)
    diag = diagnostics(rows, analysis, context)

    # This series' own sigma and MDE, beside the pilot numbers.
    series_numbers = series_side_by_side(analysis)
    # The report prints the sigma decomposition from here, so it has to reach it.
    analysis["series_numbers"] = series_numbers
    verification, mismatches = verification_table(context, series_numbers=series_numbers)
    verdict_rows = verdicts(analysis, diag, mismatches, prov)

    # Results on disk before drawing: the figure is reproducible from the table, the
    # table from the CSV.
    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_tables(out_dir, analysis, diag, verification, verdict_rows, prov, status)
    (out_dir / "a8_summary.json").write_text(
        json.dumps(
            {
                "provenance": prov,
                "verification": verification,
                "verification_mismatches": mismatches,
                "verdicts": verdict_rows,
                "context": context,
                "series_numbers_vs_pilot": series_numbers,
                "pilot_power_audit": pilot_power_audit(),
                "uncertainty_accounts": analysis["uncertainty_accounts"],
                "holm": analysis["holm"],
                "mixedlm": {k: v for k, v in analysis["mixedlm"].items()},
            },
            indent=1, sort_keys=True, default=str,
        )
    )
    if status != STATUS_COMPLETE:
        (out_dir / "PROVISIONAL.txt").write_text(
            f"{STATUS_PROVISIONAL}\n\n{prov['note']}\n\nmissing: "
            f"{json.dumps(prov['missing'], sort_keys=True)}\n"
            f"training seeds present: {prov['present']['seeds']}\n"
        )
    else:
        marker = out_dir / "PROVISIONAL.txt"
        if marker.exists():
            marker.unlink()

    figures = [] if no_figures else draw_figures(out_dir, analysis, diag, rows, status)
    report(prov, analysis, diag, verification, mismatches, verdict_rows, status,
           written, figures, time.perf_counter() - started)
    # A stop is an exit code, so a failed run cannot be mistaken for a clean one.
    return 3 if mismatches else 0


def series_side_by_side(analysis: dict) -> dict:
    """This series' numbers under the names of the pilot quantities.

    Taken at the linear head and the first ansatz level present, the cell the pilot
    measured. Never asserted against the pilot: the main series covers four axis points
    and two ansatzes, so these are expected to differ.
    """
    levels = analysis["axis"]["ansatz_levels"]
    if not levels or ("linear", levels[0]) not in analysis["cells"]:
        return {}
    cell = analysis["cells"][("linear", levels[0])]
    out: dict = {}
    for arm, key in (("A", "sigma_seed_A"), ("B", "sigma_seed_B"), ("E", "sigma_seed_E")):
        stats = cell["per_arm"].get(arm, {}).get("test")
        if stats:
            out[key] = stats["sd"]
    for name, key in (("delta_AB", "sigma_delta_AB"), ("delta_AE", "sigma_delta_AE")):
        point = cell["estimands"].get(name)
        if point:
            out[key] = point["sd"]
            out[key.replace("sigma_delta", "mde").replace("_AB", "_AB").replace("_AE", "_AE")] = \
                point.get("mde")
            out[name] = point["mean"]
    for name, key in (("delta_AE", "tost_power_AE_delta002"), ("delta_AB", "tost_power_AB_delta002")):
        record = analysis["tost"].get(("linear", levels[0], name))
        if record and record.get("computable"):
            out[key] = record["power_at_delta_true_zero"]

    # The pilot measured one generator seed, so a pooled-across-datasets sigma beside it
    # would read as "the pilot was far too optimistic" when most of the pooled variance is
    # between-dataset difficulty the pilot could not have seen. Four numbers, not one:
    #   like_for_like   the generator seed the pilot used — the only direct comparison
    #   within          the within-dataset residual, RMS over the blocks
    #   between         the spread of the block means — a finding in its own right
    #   pooled          the iid spread, kept but not comparable to the pilot
    out["_decomposition"] = sigma_decomposition(analysis, levels[0])
    return out


def sigma_decomposition(analysis: dict, ansatz_level: str) -> dict:
    """Within / between / pooled / like-for-like, for every sigma the report quotes.

    `like_for_like` is the pilot's own generator seed. Without it the audit compares a
    within-dataset number against a pooled one and reads the difference as a change in
    precision, which it is not.
    """
    pilot_seed = PILOT_GENERATOR_SEED
    cell = analysis["cells"].get(("linear", ansatz_level))
    replication = analysis.get("replication", {})
    out: dict = {"pilot_generator_seed": pilot_seed, "ansatz_level": ansatz_level}
    if cell is None:
        return out

    for arm, key in (("A", "sigma_seed_A"), ("B", "sigma_seed_B"), ("E", "sigma_seed_E")):
        per_seed = cell["per_arm"].get(arm, {}).get("test_by_dataset") or {}
        sds = [v["sd"] for v in per_seed.values() if np.isfinite(v.get("sd", float("nan")))]
        means = [v["mean"] for v in per_seed.values() if np.isfinite(v.get("mean", float("nan")))]
        stats = cell["per_arm"].get(arm, {}).get("test") or {}
        out[key] = {
            "like_for_like": per_seed.get(str(pilot_seed), {}).get("sd"),
            "within_rms": float(np.sqrt(np.mean(np.square(sds)))) if sds else None,
            "between_spread": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "pooled_iid": stats.get("sd"),
        }

    for name, key in (("delta_AB", "sigma_delta_AB"), ("delta_AE", "sigma_delta_AE")):
        point = cell["estimands"].get(name) or {}
        rep = replication.get((name, "linear", ansatz_level), {})
        per_dataset = rep.get("per_dataset", {}) or {}
        sds = [
            p["sd"] for p in per_dataset.values()
            if np.isfinite(p.get("sd", float("nan")))
        ]
        out[key] = {
            "like_for_like": (per_dataset.get(pilot_seed) or {}).get("sd"),
            "within_rms": float(np.sqrt(np.mean(np.square(sds)))) if sds else None,
            "between_spread": point.get("between_block_spread"),
            "pooled_iid": point.get("sd_pooled_iid", point.get("sd")),
            "blocked_headline": point.get("sd"),
        }
    return out


# =====================================================================================
# 14. The correlator-readout probe (EXPLORATORY, never the confirmatory family)
# =====================================================================================
#
# A separate entry point rather than a branch in analyse(): that function is built around
# the main-series grid and produced the numbers in raport.tex, while the probe has three
# arms, one dilution and a 47-column schema. Only the ESTIMATOR is shared.
#
# Estimand: Delta_{A'B'} at 15 observables, placed NEXT TO Delta_AB at 5. Rule fixed
# before the run: <= MDE means the cost is a function of readout width rather than
# freezing; >= Delta_AB - MDE means widening does not compensate freezing; in between is
# partial compensation, reported as a fraction of Delta_AB.


PROBE_ARMS: tuple[str, ...] = ("A_corr", "B_corr", "D_corr")
PROBE_DILUTION = "linear"
# The main-series contrast this is placed next to: L1, linear head, five observables.
# Read from the main-series analysis when it is available, and stated here so the rule can
# be applied even when only the probe CSV is at hand.
MAIN_SERIES_DELTA_AB = 0.0883
MAIN_SERIES_MDE = 0.0241


def probe_module():
    """The probe driver, imported lazily.

    Its constants (which arms carry an ansatz, the schema) are the authority; restating
    them here is how the analysis and the driver drift apart.
    """
    import probe_readout_order as probe

    return probe


def load_probe_rows(results_csv: Path) -> list[dict]:
    """Probe rows, validated against the PROBE schema rather than the A7 one.

    The probe writes RESULT_COLUMNS + readout_order. That column is deliberately absent
    from RESULT_COLUMNS: append_result_row, the resume reader and load_rows above all
    compare the schema for equality, so widening it would make this pipeline refuse to
    read the 46-column main-series results raport.tex was computed from.
    """
    probe = probe_module()

    results_csv = Path(results_csv)
    frame = pd.read_csv(results_csv, dtype=str, keep_default_na=False)
    expected = list(probe.PROBE_RESULT_COLUMNS)
    if frame.columns.tolist() != expected:
        raise SystemExit(
            f"STOP: {results_csv} has columns {frame.columns.tolist()}, which are not "
            f"PROBE_RESULT_COLUMNS. Use --results without --probe for the main series."
        )
    rows = []
    for record in frame.to_dict("records"):
        row = dict(record)
        row["dataset_seed"] = a7.dataset_seed_of(row["dataset"])
        row["seed_int"] = a7.optional_int(row["seed"])
        row["width_int"] = a7.optional_int(row["rff_width"])
        row["best_epoch_int"] = a7.optional_int(row["best_epoch"])
        row["n_eval_int"] = a7.optional_int(row["n_eval"])
        row["accuracy_float"] = float(row["accuracy"])
        row["lr_float"] = float(row["lr_selected"])
        row["readout_order_int"] = a7.optional_int(row["readout_order"])
        rows.append(row)
    if not rows:
        raise SystemExit(f"STOP: {results_csv} has no rows.")

    orders = sorted({r["readout_order_int"] for r in rows})
    if len(orders) != 1:
        raise SystemExit(
            f"STOP: {results_csv} mixes readout orders {orders}. Each order is a separate "
            "readout and its contrasts may not be pooled; analyse one at a time."
        )
    return rows


def probe_series(index: dict, *, dataset_seeds, arm: str, seeds, ansatz_level: str) -> dict:
    """{PairKey -> accuracy} for one probe arm, at whatever lr its rows carry.

    Unlike the main series there is nothing to choose here: the probe has one dilution and
    each arm ran at exactly one lr per dataset, so the lr is READ from the rows. Finding
    two lr values for one (dataset, arm) is an error rather than a choice — pairing across
    them would be the D-32 failure again.

    `ansatz_level` is passed in, never assumed: the probe accepts --ansatz L2, and a
    hardcoded "L1" turned an L2 probe into an empty series with no error anywhere. The
    ansatz-free arms carry an empty value, exactly as they do in the rows.
    """
    recorded = "" if arm in probe_module().ANSATZ_FREE_ARMS else ansatz_level
    by_dataset_lr: dict[tuple, set] = {}
    widths = {None}
    for key in index:
        if key[1] == arm and key[2] == recorded:
            by_dataset_lr.setdefault(key[0], set()).add(key[6])
            widths.add(key[4])

    out: dict[PairKey, float] = {}
    for dataset_seed in dataset_seeds:
        lrs = by_dataset_lr.get(dataset_seed, set())
        if not lrs:
            continue
        if len(lrs) > 1:
            raise SystemExit(
                f"STOP: arm {arm} on ds{dataset_seed} has rows at {sorted(lrs)}. A paired "
                "difference taken across two lr values is not one paired difference."
            )
        (lr,) = tuple(lrs)
        for seed in seeds:
            for width in sorted(widths, key=lambda w: (w is not None, w)):
                key = (dataset_seed, arm, recorded, PROBE_DILUTION, width, seed, lr)
                if key in index:
                    out[PairKey(dataset_seed, seed)] = float(index[key]["accuracy"])
                    break
    return out


def probe_contrast(left: dict, right: dict, *, label: str, vectors=None,
                   left_key=None, right_key=None) -> dict:
    """One exploratory contrast, through the SAME estimator the main series uses."""
    pairs = paired_differences(left, right, label=label)
    out: dict = {
        "label": label,
        "status": "EXPLORATORY",
        "n": len(pairs["differences"]),
        "blocked": estimate_blocked(pairs),
        "pooled_blocked": pooled_blocked_estimate(pairs),
        "per_dataset": {},
        "unpaired_left": [list(k) for k in pairs["unpaired_left"]],
        "unpaired_right": [list(k) for k in pairs["unpaired_right"]],
    }
    by_block: dict[int, list[float]] = {}
    for key, value in zip(pairs["keys"], pairs["differences"]):
        by_block.setdefault(key.dataset_seed, []).append(float(value))
    for block, values in sorted(by_block.items()):
        out["per_dataset"][f"ds{block}"] = estimate(values)
    out["divergence"] = divergence_check(
        {f"ds{b}": out["per_dataset"][f"ds{b}"] for b in sorted(by_block)},
        out["blocked"],
    )
    # TOST on the BLOCKED residual, not on the iid spread: the same choice the main
    # series makes, so the two are comparable. df_residual, not df — the blocked model
    # spends J degrees of freedom on the dataset means.
    blocked = out["blocked"]
    out["tost"] = tost(
        pairs["differences"],
        se=blocked["se"], df=blocked["df_residual"], sd=blocked["sd"],
    )
    # Same df as the test above, not n - 1: the blocked model spends one per dataset.
    out["tost_power"] = tost_power(
        sigma=blocked["sd"], n=out["n"] or 1, df=out["tost"].get("df")
    )
    return out


def probe_verdict(delta_ab_prime: float, *, mde: float) -> dict:
    """The pre-declared rule, applied. Never re-read after the numbers are seen."""
    below = delta_ab_prime <= mde
    uncompensated = delta_ab_prime >= MAIN_SERIES_DELTA_AB - MAIN_SERIES_MDE
    if below:
        reading = ("the cost of freezing is a function of the readout WIDTH, not of "
                   "freezing")
    elif uncompensated:
        reading = "widening the readout does NOT compensate freezing"
    else:
        reading = "PARTIAL compensation"
    return {
        "rule": ("Delta_A'B' <= MDE -> width, not freezing; >= Delta_AB - MDE -> no "
                 "compensation; in between -> partial"),
        "declared": "fixed before the probe was run; the rule is stated in full above "
                    "probe_verdict() and applied mechanically",
        "delta_a_prime_b_prime": float(delta_ab_prime),
        "probe_mde": float(mde),
        "main_series_delta_ab": MAIN_SERIES_DELTA_AB,
        "main_series_mde": MAIN_SERIES_MDE,
        "condition_below_mde": bool(below),
        "condition_uncompensated": bool(uncompensated),
        "reading": reading,
        "fraction_of_delta_ab": float(delta_ab_prime / MAIN_SERIES_DELTA_AB),
        "caveat": ("The fraction compares two readout widths and therefore two head "
                   "sizes, 16 parameters against 6. It is DESCRIPTIVE, as declared in "
                   "a decision fixed before the run; the contrast INSIDE each width is not."),
    }


def _probe_vector_pairs(vectors: dict, *, left: str, right: str, ansatz_level: str) -> dict:
    """{(dataset_seed, seed): (left key, right key)} for one contrast.

    Each arm is taken at the single (ansatz, lr) its own rows carry, and a block with more
    than one candidate on either side is dropped rather than guessed at: pairing two runs
    that were never paired produces a plausible b/c out of nothing.
    """
    probe = probe_module()

    def candidates(arm):
        recorded = "" if arm in probe.ANSATZ_FREE_ARMS else ansatz_level
        found: dict[tuple, list] = {}
        for key in vectors:
            if key[1] == arm and key[2] == recorded:
                found.setdefault((key[0], key[5]), []).append(key)
        return found

    left_keys, right_keys = candidates(left), candidates(right)
    pairs = {}
    for block, keys in left_keys.items():
        other = right_keys.get(block, [])
        if len(keys) == 1 and len(other) == 1:
            pairs[block] = (keys[0], other[0])
    return pairs


def analyse_probe(rows: list[dict], *, predictions_dir: Path, csvs=None) -> dict:
    """Every exploratory estimand of the probe, through the shared estimator."""
    seeds = sorted({r["seed_int"] for r in rows if r["seed_int"] is not None})
    dataset_seeds = sorted({r["dataset_seed"] for r in rows})
    # Read from the rows, never assumed to be L1: the probe accepts --ansatz L2, and two
    # levels in one directory are two families of contrasts that may not be pooled — the
    # same rule that already refuses two readout orders.
    ansatz_levels = sorted({r["ansatz_level"] for r in rows if r["ansatz_level"]})
    if len(ansatz_levels) != 1:
        raise SystemExit(
            f"STOP: the probe rows carry ansatz levels {ansatz_levels}. Each level is its "
            "own family of contrasts; analyse one at a time."
        )
    (ansatz_level,) = ansatz_levels
    index = accuracy_index(rows, "test")
    arms = {
        arm: probe_series(index, dataset_seeds=dataset_seeds, arm=arm, seeds=seeds,
                          ansatz_level=ansatz_level)
        for arm in PROBE_ARMS
    }

    out: dict = {
        "status": "EXPLORATORY",
        "readout_order": rows[0]["readout_order_int"],
        "ansatz_level": ansatz_level,
        "dataset_seeds": dataset_seeds,
        "seeds": seeds,
        "per_arm": {
            arm: arm_summary(list(values.values())) for arm, values in arms.items()
        },
        "per_arm_per_dataset": {
            arm: {
                f"ds{ds}": arm_summary(
                    [v for k, v in values.items() if k.dataset_seed == ds]
                )
                for ds in dataset_seeds
            }
            for arm, values in arms.items()
        },
        "estimands": {},
    }

    for label, (left, right) in {
        "Delta_A'B'": ("A_corr", "B_corr"),
        "Delta_B'D_corr": ("B_corr", "D_corr"),
        "Delta_A'D_corr": ("A_corr", "D_corr"),
    }.items():
        out["estimands"][label] = probe_contrast(arms[left], arms[right], label=label)

    primary = out["estimands"]["Delta_A'B'"]["blocked"]
    out["verdict"] = probe_verdict(primary["mean"], mde=primary["mde"])

    # Uncertainty account (3): McNemar on the per-test-row correctness, mandatory wherever
    # sigma_Delta is smaller than the binomial SE of a single accuracy.
    # Predictions live beside each dataset's own CSV, so they are gathered per directory
    # rather than from one path: a single predictions_dir would silently find none for two
    # of the three generator seeds and report a McNemar built on a third of the data.
    vectors: dict = {}
    for directory in ({Path(c).parent for c in csvs} if csvs else {predictions_dir.parent}):
        found = correctness_vectors(rows, Path(directory) / a7.PREDICTIONS_DIR)
        vectors.update(found["vectors"])
    # A row is missing only if NO directory holds its vector. Summing each directory's own
    # misses would count every ds22 row as missing while searching ds11 and report a
    # defect that is not there.
    wanted = {
        (r["dataset_seed"], r["arm"], r["ansatz_level"], r["dilution"],
         r["width_int"], r["seed_int"], f"{r['lr_float']:g}")
        for r in rows if r["split"] == "test"
    }
    out["mcnemar"] = {"missing_prediction_files": len(wanted - set(vectors))}
    for label, (left, right) in {
        "Delta_A'B'": ("A_corr", "B_corr"),
        "Delta_B'D_corr": ("B_corr", "D_corr"),
    }.items():
        b_total = c_total = 0
        # The counts have to be paired the way the accuracies are: same generator seed,
        # same training seed, and each side at the lr and ansatz its OWN series was read
        # at. Matching on (dataset, seed) alone would pair across ansatz levels and across
        # lr — the D-32 failure moved into the third uncertainty account.
        for (block, seed), (left_key, right_key) in _probe_vector_pairs(
            vectors, left=left, right=right, ansatz_level=ansatz_level
        ).items():
            vector, match = vectors[left_key], vectors[right_key]
            if vector.size != match.size:
                continue
            b_total += int(np.sum(vector & ~match))
            c_total += int(np.sum(match & ~vector))
        n_eval = rows[0]["n_eval_int"] or 1
        pairs_n = out["estimands"][label]["n"]
        out["mcnemar"][label] = {
            "b_left_better": b_total,
            "c_right_better": c_total,
            "delta_reconstructed": (b_total - c_total) / (n_eval * pairs_n) if pairs_n else float("nan"),
            "note": "(b - c) / (n_eval * pairs) must reproduce the blocked mean exactly",
        }
    return out


def probe_result_csvs(target: Path) -> list[Path]:
    """One probe CSV, or every probe CSV under a directory.

    The estimand is BLOCKED over the generator seeds and the driver writes one directory
    per seed, so the normal call is `--results outputs/probe_corr --probe`. Passing a
    single file analyses that dataset alone, which is a per-dataset reading and not the
    blocked estimand.
    """
    target = Path(target)
    if target.is_file():
        return [target]
    found = sorted(target.glob("*/probe_results.csv"))
    if not found:
        raise SystemExit(
            f"STOP: no */probe_results.csv under {target}. Point --results at a probe "
            "output directory or at one probe_results.csv."
        )
    return found


def run_probe(*, results_csv: Path, out_dir: Path, predictions_dir: Path | None) -> int:
    """Analyse one probe directory. Exploratory throughout, never the confirmatory family."""
    results_csv = Path(results_csv)
    out_dir = Path(out_dir)
    predictions_dir = (
        Path(predictions_dir) if predictions_dir is not None
        else results_csv.parent / a7.PREDICTIONS_DIR
    )
    csvs = probe_result_csvs(results_csv)
    rows = []
    for csv in csvs:
        rows.extend(load_probe_rows(csv))
    print(f"reading {len(csvs)} probe result file(s): "
          + ", ".join(str(c.parent.name) for c in csvs))
    analysis = analyse_probe(rows, predictions_dir=predictions_dir, csvs=csvs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_analysis.json").write_text(json.dumps(analysis, indent=2, default=str))

    table = []
    for label, block in analysis["estimands"].items():
        b = block["blocked"]
        table.append({
            "estimand": label,
            "status": "EXPLORATORY",
            "n": block["n"],
            "estimate": _fmt(b["mean"]),
            "ci95_low": _fmt(b["ci95_low"]),
            "ci95_high": _fmt(b["ci95_high"]),
            "sigma_delta": _fmt(b["sd"]),
            "mde": _fmt(b["mde"]),
            "above_mde": b["above_mde"],
            "ci95_excludes_zero": b["ci95_excludes_zero"],
            "n_positive": b["n_positive"],
            "p_sign": _fmt(b["p_sign_exact"]),
            "p_wilcoxon": _fmt(b["p_wilcoxon_exact"]),
            "tost_equivalent": block["tost"].get("equivalent"),
            "tost_power": _fmt(block["tost_power"]),
        })
    pd.DataFrame(table).to_csv(out_dir / "probe_estimands.csv", index=False)

    arms_table = [
        {"arm": arm, "status": "EXPLORATORY", **{k: _fmt(v) for k, v in summary.items()}}
        for arm, summary in analysis["per_arm"].items()
    ]
    pd.DataFrame(arms_table).to_csv(out_dir / "probe_arms.csv", index=False)

    print(f"probe analysis: readout_order={analysis['readout_order']}, "
          f"{len(analysis['dataset_seeds'])} datasets x {len(analysis['seeds'])} seeds")
    print(f"  {'estimand':<16} {'n':>3} {'estimate':>9}  {'95% CI':^20} "
          f"{'sigma':>7} {'MDE':>7} {'+/n':>6}")
    for row in table:
        print(f"  {row['estimand']:<16} {row['n']:>3} {row['estimate']:>+9.4f}  "
              f"[{row['ci95_low']:>+8.4f}; {row['ci95_high']:>+8.4f}] "
              f"{row['sigma_delta']:>7.4f} {row['mde']:>7.4f} "
              f"{row['n_positive']:>3}/{row['n']:<2}")
    v = analysis["verdict"]
    print(f"\n  PRE-DECLARED RULE -> {v['reading']}")
    print(f"    Delta_A'B' = {v['delta_a_prime_b_prime']:+.4f}, probe MDE {v['probe_mde']:.4f}, "
          f"main-series Delta_AB {v['main_series_delta_ab']:+.4f}")
    print(f"    fraction of Delta_AB = {v['fraction_of_delta_ab']:.3f}  ({v['caveat'][:60]}...)")
    print(f"\nwrote {out_dir/'probe_analysis.json'}, probe_estimands.csv, probe_arms.csv")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", required=True, type=Path,
        help="the A7 results CSV. There is no default on purpose: a default would make it "
             "possible to analyse the dry run by accident.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--predictions-dir", type=Path, nargs="+", default=None,
        help="repeatable. Defaults to <results CSV dir>/predictions, where A7 writes them. "
             "A combined CSV covering several generator seeds needs the predictions "
             "directory of EACH run, or McNemar is computed on part of the data",
    )
    parser.add_argument(
        "--skip-slow-verification", action="store_true",
        help="skip the G1 and ceiling recomputation of section 1.6. They are then reported "
             "as NOT VERIFIED, never as passing. For the test suite.",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--probe", action="store_true",
        help="analyse a correlator-readout PROBE directory instead of the main series. "
             "The probe schema is RESULT_COLUMNS + readout_order and its estimands are "
             "EXPLORATORY; nothing it produces joins the confirmatory family.",
    )
    args = parser.parse_args()
    if args.probe:
        raise SystemExit(
            run_probe(
                results_csv=args.results,
                out_dir=args.out_dir,
                predictions_dir=args.predictions_dir,
            )
        )
    raise SystemExit(
        run(
            results_csv=args.results,
            out_dir=args.out_dir,
            predictions_dir=args.predictions_dir,
            skip_slow=args.skip_slow_verification,
            no_figures=args.no_figures,
        )
    )


if __name__ == "__main__":
    main()
