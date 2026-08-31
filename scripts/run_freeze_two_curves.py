"""Freeze the two_curves production dataset into data/ and run the binding gates on it.

The cell is an input, not re-litigated here: two_curves, n_features=20, degree=3,
offset=0.05, noise=0.1, dataset_seed=11, PCA to 5. Seed rule: the smallest seed of the
sweep grid — all five pass both gates, so the rule is independent of the outcome.

  1. Freeze into a STAGING directory (arguments by name).
  2. Compare dataset_hash and pca_hash against the sweep manifest. A mismatch aborts
     before anything reaches data/ — staging first is what makes that true.
  3. Copy into data/ byte for byte, reload through load_splits, which re-verifies all
     three hashes.
  4. G2 on the frozen PCA.
  5. G1, the contract verdict: strong = SVC(rbf) over G1_SVC_GRID, floor = contract arm E
     through the full training loop, lr from G1_LR_GRID.
  6. The same G1 with extra lr values — a sensitivity reading, not the verdict; the floor
     is known to rise when the grid widens and that question is open.
  7. Reported, never gating: the same strong model on the full 20 pre-PCA features.

No constant is changed; grids are read, never written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from qsocket.datasets import (
    DEFAULT_DATA_DIR,
    N_COMPONENTS,
    N_SAMPLES_TOTAL,
    dataset_paths,
    generate_and_freeze,
    load_manifest,
    load_splits,
)
from qsocket.gates import (
    G1_LR_GRID,
    G1_SVC_GRID,
    check_g1_headroom,
    check_g2_effective_dim,
    make_arm_e_linear_floor_model,
    make_svc_strong_model,
)

# --- frozen production configuration ------------------------------------------------

DATASET = "two_curves"

# Arguments by name: `offset` precedes `noise` in generate_two_curves, unlike the
# upstream README example.
PRODUCTION_GENERATOR_KWARGS: dict = {
    "n_features": 20,
    "degree": 3,
    "offset": 0.05,
    "noise": 0.1,
}

# The dataset sweep grid; the production seed is its smallest member.
SWEEP_DATASET_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
PRODUCTION_DATASET_SEED = min(SWEEP_DATASET_SEEDS)
SEED_RULE = "the smallest seed of the dataset sweep grid"

# On-disk name in data/, following the sweep manifests minus their `_k5` suffix: PCA
# width is frozen at 5 here, so carrying it in the name would imply siblings that do not
# exist.
PRODUCTION_FROZEN_NAME = "two_curves_n_features20_degree3_offset0p05_noise0p1_seed11"

# Sweep manifest of the same cell and generator seed, used only as the provenance
# reference for the two hashes.
SWEEP_FROZEN_DIR = Path(__file__).resolve().parents[1] / "outputs" / "a3_sweep" / "frozen"
SWEEP_MANIFEST_NAME = "two_curves_n_features20_degree3_offset0p05_noise0p1_seed11_k5"

# The arm-E training seeds the dataset sweep used, kept for consistency with the numbers
# the verdict is compared against. Not the ten seeds of the main series.
ARM_E_SEEDS: tuple[int, ...] = (1, 2, 3)

# Sensitivity only, never merged into gates.G1_LR_GRID: the contract verdict has to be
# computable from the contract grid alone.
SENSITIVITY_EXTRA_LR: tuple[float, ...] = (0.1, 0.3)


# --- steps ---------------------------------------------------------------------------


def freeze_and_verify(*, out_dir: Path, a3_frozen_dir: Path, staging: Path) -> dict:
    """Freeze into `staging`, verify the hashes against the sweep, publish into `out_dir`.

    Returns the manifest of the published copy. Raises SystemExit on a hash mismatch,
    before anything is written to `out_dir`.
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

    reference = load_manifest(SWEEP_MANIFEST_NAME, out_dir=a3_frozen_dir)
    mismatches = [
        (key, manifest[key], reference[key])
        for key in ("dataset_hash", "pca_hash")
        if manifest[key] != reference[key]
    ]
    print(f"# the dataset sweep reference manifest: {a3_frozen_dir / (SWEEP_MANIFEST_NAME + '.manifest.json')}")
    for key in ("dataset_hash", "pca_hash"):
        verdict = "MATCH" if manifest[key] == reference[key] else "MISMATCH"
        print(f"#   {key:<13} {manifest[key]}  vs the dataset sweep {reference[key]}  -> {verdict}")
    if reference["generator_kwargs"] != {k: v for k, v in PRODUCTION_GENERATOR_KWARGS.items()}:
        print(f"#   ! the dataset sweep manifest kwargs {reference['generator_kwargs']} differ from "
              f"{PRODUCTION_GENERATOR_KWARGS}")
    if mismatches:
        raise SystemExit(
            "ABORTED: the production dataset does not reproduce the dataset sweep hashes "
            f"{mismatches}. Nothing was written to {out_dir}. This is a real defect, not "
            "expected drift: the same replay path reproduced the hash for degree=5."
        )

    data_path, manifest_path = dataset_paths(PRODUCTION_FROZEN_NAME, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in dataset_paths(PRODUCTION_FROZEN_NAME, out_dir=staging):
        shutil.copy2(source, out_dir / source.name)
    print(f"# written: {data_path}")
    print(f"# written: {manifest_path}")

    # Reload through the loader, which re-verifies file_sha256 and both content digests.
    load_splits(PRODUCTION_FROZEN_NAME, out_dir=out_dir)
    return load_manifest(PRODUCTION_FROZEN_NAME, out_dir=out_dir)


def _g1(splits, *, lr_grid, label: str) -> dict:
    started = time.perf_counter()
    verdict = check_g1_headroom(
        splits,
        strong_model=make_svc_strong_model(),
        floor_model=make_arm_e_linear_floor_model(lr_grid=lr_grid, seeds=ARM_E_SEEDS),
    )
    verdict["reading"] = label
    verdict["wall_seconds"] = time.perf_counter() - started
    return verdict


def _print_g1(verdict: dict) -> None:
    strong, floor = verdict["strong"], verdict["floor"]
    print(f"  {verdict['reading']}")
    print(f"    strong  SVC(rbf) C={strong['selected']['C']} gamma={strong['selected']['gamma']}"
          f"  test {strong['accuracy']:.6f}  (val {strong['selection_accuracy']:.6f})")
    print(f"    floor   arm E lr={floor['lr_selected']:g} of {floor['lr_grid']}"
          f"  test {floor['accuracy']:.6f}  (val {floor['val_accuracy']:.6f})"
          f"  seeds {floor['seeds']}")
    print(f"    mean val per lr: "
          + ", ".join(f"{lr}:{value:.6f}" for lr, value in floor["mean_val_accuracy_per_lr"].items()))
    print(f"    headroom {verdict['headroom']:+.6f}  (>= {verdict['min_headroom']:.2f}), "
          f"g1_margin {verdict['g1_margin']:+.6f}, strong_in_band {verdict['strong_in_band']}, "
          f"binding {verdict['binding']}  -> {'PASS' if verdict['passed'] else 'FAIL'}")
    for failure in verdict["failures"]:
        print(f"    ! {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="where the frozen production dataset lands (default: data/)")
    parser.add_argument("--a3-frozen-dir", type=Path, default=SWEEP_FROZEN_DIR)
    parser.add_argument("--json-out", type=Path,
                        default=Path("outputs/a5b/a5b_gates.json"))
    args = parser.parse_args()

    print(f"# the two_curves freeze — freeze the production dataset and run the binding gates")
    print(f"# cell: {DATASET} {PRODUCTION_GENERATOR_KWARGS}, dataset_seed="
          f"{PRODUCTION_DATASET_SEED}, PCA k={N_COMPONENTS}")
    print(f"# seed rule: \"{SEED_RULE}\" (the dataset sweep grid {list(SWEEP_DATASET_SEEDS)}); "
          f"outcome-independent, all five pass both gates")
    print(f"# split {N_SAMPLES_TOTAL} -> 4200/600/1200")
    print(f"# G1 strong: SVC(rbf) grid {G1_SVC_GRID} (3x3, the 3x3 rule), selected on val, scored on test")
    print(f"# G1 floor: arm E = identity socket + linear head via training.train_model, "
          f"lr from {list(G1_LR_GRID)}, seeds {list(ARM_E_SEEDS)}")
    print(f"# G1 sensitivity adds lr {list(SENSITIVITY_EXTRA_LR)} — REPORTED, NOT THE VERDICT\n")

    with tempfile.TemporaryDirectory(prefix="a5b_staging_") as staging:
        manifest = freeze_and_verify(
            out_dir=args.out_dir, a3_frozen_dir=args.a3_frozen_dir, staging=Path(staging)
        )

    print(f"\n# manifest summary")
    print(f"  dataset_hash {manifest['dataset_hash']}")
    print(f"  pca_hash     {manifest['pca_hash']}")
    print(f"  file_sha256  {manifest['file_sha256']}")
    print(f"  explained_variance_ratio_of_retained "
          f"{[round(v, 6) for v in manifest['pca']['explained_variance_ratio_of_retained']]}")
    print(f"  total_variance_explained {manifest['pca']['total_variance_explained']:.6f}")
    for split, balance in manifest["class_balance"].items():
        print(f"  class balance {split:<5} n={balance['n']:>4} "
              f"fraction_positive {balance['fraction_positive']:.6f} counts {balance['counts']}")
    print(f"  clipped_fraction {manifest['scaling']['clipped_fraction']}")
    print(f"  max_overshoot_before_clipping {manifest['scaling']['max_overshoot_before_clipping']}")

    g2 = check_g2_effective_dim(manifest)
    print(f"\n# G2 (binding)")
    print(f"  share_of_retained {[round(v, 6) for v in g2['share_of_retained']]}")
    print(f"  top component {g2['top_component']} share {g2['top_share']:.6f} "
          f"(limit {g2['max_share']:.2f}, margin {g2['max_share'] - g2['top_share']:+.6f})"
          f"  -> {'PASS' if g2['passed'] else 'FAIL'}")
    for failure in g2["failures"]:
        print(f"  ! {failure}")

    splits = load_splits(PRODUCTION_FROZEN_NAME, out_dir=args.out_dir)
    raw_splits = load_splits(PRODUCTION_FROZEN_NAME, out_dir=args.out_dir, raw=True)

    print(f"\n# G1")
    contract = _g1(splits, lr_grid=G1_LR_GRID, label="CONTRACT VERDICT — lr from G1_LR_GRID")
    _print_g1(contract)
    sensitivity = _g1(
        splits,
        lr_grid=tuple(G1_LR_GRID) + SENSITIVITY_EXTRA_LR,
        label=f"SENSITIVITY, NOT THE VERDICT — lr grid widened by {list(SENSITIVITY_EXTRA_LR)}",
    )
    _print_g1(sensitivity)
    print(f"    headroom shift vs contract "
          f"{sensitivity['headroom'] - contract['headroom']:+.6f}")

    strong_full = make_svc_strong_model()(raw_splits)
    print(f"\n# reported, never gating (SPEC section 6): strong model on the full 20 features")
    print(f"  acc {strong_full['accuracy']:.6f}  (C={strong_full['selected']['C']}, "
          f"gamma={strong_full['selected']['gamma']}), vs the same PCA-5 arm-E floor: "
          f"{strong_full['accuracy'] - contract['floor']['accuracy']:+.6f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "task": "the two_curves freeze",
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "dataset": DATASET,
                    "generator_kwargs": PRODUCTION_GENERATOR_KWARGS,
                    "dataset_seed": PRODUCTION_DATASET_SEED,
                    "seed_rule": SEED_RULE,
                    "frozen_name": PRODUCTION_FROZEN_NAME,
                    "out_dir": str(args.out_dir),
                    "arm_e_seeds": list(ARM_E_SEEDS),
                    "manifest": manifest,
                    "G2": g2,
                    "G1_contract": contract,
                    "G1_sensitivity_extended_lr": sensitivity,
                    "sensitivity_extra_lr": list(SENSITIVITY_EXTRA_LR),
                    "strong_on_full_features": strong_full,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten: {args.json_out}")


if __name__ == "__main__":
    main()
