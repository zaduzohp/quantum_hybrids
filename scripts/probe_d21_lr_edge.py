"""Is the contract lr grid clipped too low for arm A? Measures one point above it.

Phase 1 chose the top of the contract grid (lr = 0.03) in 12 of 24 cells. An earlier probe
covered one cell; this one measures the extra point across every dilution, both ansatz
levels and all three generator seeds — the 24 cells MIN_CELLS_AT_PROBE counts against.

The probe point cannot become a contract lr here: the verdict is a recommendation under a
rule fixed before the measurement, and no selected lr is written anywhere.

    WIDEN   if arm A's argmax sits at the probe point in at least half the cells AND the
            median gain over 0.03 among those cells exceeds the validation noise of 0.020.
    LEAVE   otherwise. An argmax that moves up by less than its own measurement noise is a
            plateau, not an optimum.

    caffeinate -dimsu .venv/bin/python scripts/probe_d21_lr_edge.py [--probe-lr 0.1 0.3]
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

# Before numpy and torch: the BLAS pools read these at import and never again.
from qsocket.core import pin_blas_threads

pin_blas_threads()

import numpy as np
import pandas as pd
import run_main_series as a7
import torch

# The point(s) above the contract grid. Diagnostic only.
PROBE_LR: tuple[float, ...] = (0.1,)
CONTRACT_TOP = 0.03
# The yardstick the gain is read against, not a fitted quantity.
VALIDATION_NOISE = 0.020
# The rule's thresholds, fixed in advance.
MIN_CELLS_AT_PROBE = 12  # of 24
SEEDS = a7.LR_SELECTION_SEEDS  # the same seeds select_lr uses

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "d21_lr_edge"
RAW_CSV = "d21_raw_rows.csv"
SUMMARY_JSON = "d21_summary.json"
RAW_COLUMNS = ("dataset_seed", "dataset", "arm", "ansatz_level", "dilution", "seed", "lr",
               "val_accuracy", "test_accuracy", "best_epoch", "epochs_run",
               "theta_displacement", "wall_seconds", "run_id")


def contract_curves(phase1_dir: Path) -> pd.DataFrame:
    """Arm A and B at the four contract lr values, read from the phase-1 tables."""
    frames = []
    for path in sorted(glob.glob(str(phase1_dir / "ds*" / "a7_lr_table.csv"))):
        frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
        frames.append(frame[frame.arm.isin(("A", "B"))])
    if not frames:
        raise FileNotFoundError(
            f"no phase-1 lr tables under {phase1_dir}. This probe compares AGAINST them; "
            "without them there is nothing to compare to."
        )
    return pd.concat(frames, ignore_index=True)


def cells(dataset_seeds, dilutions, ansatz_levels) -> list[tuple]:
    return [(ds, dil, ans) for ds in dataset_seeds for dil in dilutions for ans in ansatz_levels]


def _worker(task: dict) -> list[dict]:
    torch.set_num_threads(1)
    dataset, splits, manifest = a7._worker_splits(task["dataset_seed"])
    rows = []
    for arm in ("A", "B"):
        started = time.perf_counter()
        cell_rows, _, _ = a7.run_cell(
            splits, manifest=manifest, dataset=dataset, arm=arm,
            ansatz_level=task["ansatz_level"], dilution=task["dilution"],
            seed=task["seed"], lr=task["lr"], lr_grid=PROBE_LR, width=None,
            cached_features=None, effective_rank=None, g1_margin=None,
            run_id=task["run_id"], commit=task["commit"], environment=task["environment"],
        )
        by_split = {row["split"]: row for row in cell_rows}
        rows.append({
            "dataset_seed": task["dataset_seed"], "dataset": dataset, "arm": arm,
            "ansatz_level": task["ansatz_level"], "dilution": task["dilution"],
            "seed": task["seed"], "lr": task["lr"],
            "val_accuracy": by_split["val"]["accuracy"],
            "test_accuracy": by_split["test"]["accuracy"],
            "best_epoch": by_split["val"]["best_epoch"],
            "epochs_run": by_split["val"]["epochs_run"],
            "theta_displacement": by_split["val"]["theta_displacement"],
            "wall_seconds": time.perf_counter() - started,
            "run_id": task["run_id"],
        })
    return rows


def verdict(probe: pd.DataFrame, contract: pd.DataFrame, probe_lr: float) -> dict:
    """Apply the rule declared in the docstring. Arm A only; arm B reported beside."""
    per_cell = []
    for (ds, dil, ans) in sorted({(r.dataset_seed, r.dilution, r.ansatz_level)
                                  for r in probe.itertuples()}):
        def mean_at(frame, arm, lr):
            m = frame[(frame.dataset_seed == ds) & (frame.dilution == dil)
                      & (frame.ansatz_level == ans) & (frame.arm == arm)
                      & (np.isclose(frame.lr, lr))]
            return float(m.val_accuracy.mean()) if len(m) else float("nan")

        a_top = mean_at(contract, "A", CONTRACT_TOP)
        a_probe = mean_at(probe, "A", probe_lr)
        b_top = mean_at(contract, "B", CONTRACT_TOP)
        b_probe = mean_at(probe, "B", probe_lr)
        # The full contract curve, so the argmax is taken over all points.
        a_curve = {float(lr): mean_at(contract, "A", lr) for lr in a7.CONTRACT_LR_GRID}
        a_curve[probe_lr] = a_probe
        per_cell.append({
            "cell": f"ds{ds}|{dil}|{ans}",
            "A_at_contract_top": a_top, "A_at_probe": a_probe, "A_gain": a_probe - a_top,
            "B_at_contract_top": b_top, "B_at_probe": b_probe, "B_gain": b_probe - b_top,
            "A_argmax_lr": max(a_curve, key=lambda lr: a_curve[lr]),
            "A_argmax_at_probe": max(a_curve, key=lambda lr: a_curve[lr]) == probe_lr,
            "A_gain_above_noise": (a_probe - a_top) > VALIDATION_NOISE,
        })

    at_probe = [c for c in per_cell if c["A_argmax_at_probe"]]
    gains = [c["A_gain"] for c in at_probe]
    median_gain = float(np.median(gains)) if gains else float("nan")
    widen = len(at_probe) >= MIN_CELLS_AT_PROBE and median_gain > VALIDATION_NOISE
    return {
        "rule": {
            "declared_before_measurement": True,
            "widen_if": (f"arm A argmax at {probe_lr:g} in >= {MIN_CELLS_AT_PROBE} of "
                         f"{len(per_cell)} cells AND median gain > {VALIDATION_NOISE}"),
            "validation_noise": VALIDATION_NOISE,
            "arm_B_excluded": "D-18: only a measurement on arm A may move the grid",
        },
        "cells_total": len(per_cell),
        "cells_with_A_argmax_at_probe": len(at_probe),
        "median_A_gain_among_those": median_gain,
        "cells_with_gain_above_noise": sum(c["A_gain_above_noise"] for c in per_cell),
        "mean_A_gain_all_cells": float(np.mean([c["A_gain"] for c in per_cell])),
        "mean_B_gain_all_cells": float(np.mean([c["B_gain"] for c in per_cell])),
        "recommendation": "WIDEN the contract grid by one point" if widen
                          else "LEAVE the contract grid as it is",
        "owner_decision_required": True,
        "note": ("This probe may not select an lr. Widening CONTRACTS section 7.1 is D-21, "
                 "the owner's decision; the script only supplies the evidence and the rule "
                 "it was read against."),
        "per_cell": per_cell,
    }


def run(*, out_dir=DEFAULT_OUT, phase1_dir=None, probe_lr=PROBE_LR, workers=None,
        dataset_seeds=a7.DATASET_SEEDS, dilutions=a7.DILUTIONS,
        ansatz_levels=a7.ANSATZ_LEVELS, seeds=SEEDS) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = Path(phase1_dir) if phase1_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "a7_wcss")
    workers = a7.default_workers() if workers is None else workers
    raw_path = out_dir / RAW_CSV
    run_id = f"d21_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    commit, environment = a7.git_commit(), a7.env_hash()

    contract = contract_curves(phase1_dir)
    print(f"# D-21 probe  run_id {run_id}  probe lr {list(probe_lr)}  workers {workers}")
    print(f"#   contract points read from {phase1_dir} ({len(contract)} rows, NOT recomputed)")
    print(f"#   rule: widen if A argmax at probe in >= {MIN_CELLS_AT_PROBE} cells AND "
          f"median gain > {VALIDATION_NOISE}")
    for dataset_seed in dataset_seeds:
        a7.ensure_dataset(dataset_seed, allow_generate=False)

    done = set()
    if raw_path.exists():
        prior = pd.read_csv(raw_path, keep_default_na=False, float_precision="round_trip")
        done = {(int(r.dataset_seed), r.dilution, r.ansatz_level, int(r.seed),
                 f"{float(r.lr):g}") for r in prior.itertuples()}
        print(f"#   resuming: {len(prior)} rows already on disk")

    tasks = [
        {"dataset_seed": ds, "dilution": dil, "ansatz_level": ans, "seed": seed,
         "lr": lr, "run_id": run_id, "commit": commit, "environment": environment}
        for (ds, dil, ans) in cells(dataset_seeds, dilutions, ansatz_levels)
        for seed in seeds for lr in probe_lr
        if (ds, dil, ans, seed, f"{lr:g}") not in done
    ]
    print(f"\n## {len(tasks)} arm-A runs to do (arm B rides along, cached)")

    collected: list[dict] = []
    context = mp.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(processes=workers, initializer=a7._worker_init) as pool:
        for index, rows in enumerate(pool.imap_unordered(_worker, tasks), 1):
            a7.append_rows(raw_path, RAW_COLUMNS, rows)  # on disk the moment it exists
            collected.extend(rows)
            a_row = next(r for r in rows if r["arm"] == "A")
            print(f"  [{index:>3}/{len(tasks)}] ds{a_row['dataset_seed']} "
                  f"{a_row['dilution']:<7} {a_row['ansatz_level']:<3} seed {a_row['seed']} "
                  f"lr {a_row['lr']:g}  A val {a_row['val_accuracy']:.4f}  "
                  f"{a_row['wall_seconds']:.0f} s", flush=True)

    probe = pd.read_csv(raw_path, keep_default_na=False, float_precision="round_trip")
    summary = {}
    for lr in probe_lr:
        summary[f"{lr:g}"] = verdict(probe, contract, lr)
    summary["configuration"] = {
        "run_id": run_id, "git_commit": commit, "env_hash": environment,
        "probe_lr": list(probe_lr), "contract_grid": list(a7.CONTRACT_LR_GRID),
        "seeds": list(seeds), "workers": workers,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, default=str),
                                       encoding="utf-8")
    report(summary, probe_lr, out_dir)
    return summary


def report(summary: dict, probe_lr, out_dir: Path) -> None:
    line = "=" * 100
    print(f"\n{line}\nD-21 — one point above the contract grid, arm A. EVIDENCE, not a decision.\n{line}")
    for lr in probe_lr:
        v = summary[f"{lr:g}"]
        print(f"\nprobe lr = {lr:g}   rule: {v['rule']['widen_if']}")
        print(f"  {'cell':<22} {'A@0.03':>8} {'A@probe':>8} {'gain':>8} "
              f"{'argmax':>7} {'B@0.03':>8} {'B@probe':>8} {'gain':>8}")
        for c in v["per_cell"]:
            flag = "PROBE" if c["A_argmax_at_probe"] else ""
            print(f"  {c['cell']:<22} {c['A_at_contract_top']:>8.4f} {c['A_at_probe']:>8.4f} "
                  f"{c['A_gain']:>+8.4f} {flag:>7} {c['B_at_contract_top']:>8.4f} "
                  f"{c['B_at_probe']:>8.4f} {c['B_gain']:>+8.4f}")
        print(f"\n  arm A argmax at {lr:g}: {v['cells_with_A_argmax_at_probe']} of {v['cells_total']} cells"
              f"  (rule needs >= {MIN_CELLS_AT_PROBE})")
        print(f"  median gain among those: {v['median_A_gain_among_those']:+.4f}"
              f"  (rule needs > {VALIDATION_NOISE})")
        print(f"  cells whose gain clears the validation noise: {v['cells_with_gain_above_noise']} of {v['cells_total']}")
        print(f"  mean gain, all cells:  A {v['mean_A_gain_all_cells']:+.4f}   B {v['mean_B_gain_all_cells']:+.4f}")
        print(f"\n  >>> {v['recommendation']}")
        print(f"      {v['note']}")
    print(f"\nraw rows   {out_dir / RAW_CSV}\nsummary    {out_dir / SUMMARY_JSON}\n{line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase1-dir", type=Path, default=None,
                        help="where the phase-1 a7_lr_table.csv files are")
    parser.add_argument("--probe-lr", type=float, nargs="+", default=list(PROBE_LR))
    parser.add_argument("--dataset-seeds", type=int, nargs="+", default=list(a7.DATASET_SEEDS))
    parser.add_argument("--dilutions", nargs="+", default=list(a7.DILUTIONS))
    parser.add_argument("--ansatz-levels", nargs="+", default=list(a7.ANSATZ_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(out_dir=args.out_dir, phase1_dir=args.phase1_dir, probe_lr=tuple(args.probe_lr),
        workers=args.workers, dataset_seeds=tuple(args.dataset_seeds),
        dilutions=tuple(args.dilutions), ansatz_levels=tuple(args.ansatz_levels),
        seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
