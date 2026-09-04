"""run_supplementary.py must decide nothing: every estimator comes from the analysis
pipeline.

The file exists because the report needs numbers the main pipeline does not produce, and
the danger of such a file is a second implementation of the blocked estimator that
quietly disagrees with the first. These tests pin that it is not one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_a8_analysis as a8
import run_supplementary as sup

# The analysis output directory of the main series, as it is committed and as
# REPRODUCE.md drives it. It used to be outputs/a8/; the rename to outputs/stats/ left
# this guard behind, so every test below skipped on a complete checkout with a message
# that read like missing data instead of a stale path.
ANALYSIS_DIR = ROOT / "outputs" / "stats" / "main"
MAIN_CSV = ANALYSIS_DIR / "a7_results_combined.csv"
ESTIMANDS_CSV = ANALYSIS_DIR / "tables" / "estimands.csv"
requires_series = pytest.mark.skipif(
    not (MAIN_CSV.exists() and ESTIMANDS_CSV.exists()),
    reason=f"the main series analysis is not in this checkout ({ANALYSIS_DIR})")


# --- the degeneracy detector ---------------------------------------------------------

def test_a_constant_classifier_is_detected_and_a_normal_run_is_not():
    """The rule is 'accuracy equals a class share exactly'. It must not fire on a run that
    merely scores badly, or it would delete evidence instead of flagging failures."""
    shares = {11: {0.485, 0.515}}
    frame = pd.DataFrame({
        "accuracy": [0.485, 0.515, 0.4851, 0.8218, 0.5],
        "dataset_seed": [11] * 5,
    })
    flagged = sup.mark_degenerate(frame, shares)["degenerate"].tolist()
    assert flagged == [True, True, False, False, False]


def test_the_share_used_by_the_detector_comes_from_the_frozen_labels():
    """Hard-coding the shares would let them drift away from the datasets."""
    shares = sup.class_shares([11])
    assert len(shares[11]) == 2
    assert abs(sum(shares[11]) - 1.0) < 1e-12
    assert all(abs(s * 1200 - round(s * 1200)) < 1e-9 for s in shares[11])


# --- the estimator comes from the analysis pipeline, not a second one -----------------------------------------

@requires_series
def test_the_adam_arm_of_the_ridge_contrast_reproduces_a8_exactly():
    """`ridge_contrast_rows` computes Delta_AB twice: once against arm B trained with Adam,
    which the analysis already reports, and once against its closed-form readout, which it
    does not. The first must come out bit for bit like the analysis' delta_AB, which is
    what shows the estimator is re-used rather than re-implemented.
    """
    rows = a8.load_rows(MAIN_CSV)
    dataset_seeds = sorted({r["dataset_seed"] for r in rows})
    seeds = sorted({r["seed_int"] for r in rows if r["seed_int"] is not None})
    ours = {
        (r["dilution"], r["ansatz"]): r["mean"]
        for r in sup.ridge_contrast_rows(rows, dataset_seeds, seeds)
        if r["estimand"] == "delta_AB_B_readout_adam"
    }
    reference = pd.read_csv(ESTIMANDS_CSV, float_precision="round_trip")
    reference = reference[reference["estimand"] == "delta_AB"]
    assert ours, "no adam rows produced"
    for _, row in reference.iterrows():
        assert ours[(row["dilution"], row["ansatz"])] == row["mean"]


@requires_series
def test_the_probe_emits_delta_BE_which_a8_never_does():
    """acc(B) - acc(E) — does the frozen quantum socket beat no socket — exists in the main
    analysis only as an intermediate of the delta_AE decomposition and is never written out.
    """
    rows = a8.load_rows(MAIN_CSV)
    emitted = {r["estimand"] for r in sup.probe_estimand_rows(
        rows, sorted({r["dataset_seed"] for r in rows}),
        sorted({r["seed_int"] for r in rows if r["seed_int"] is not None}), "linear")}
    assert "delta_BE" in emitted
    a8_estimands = set(pd.read_csv(ESTIMANDS_CSV)["estimand"])
    assert "delta_BE" not in a8_estimands


@requires_series
def test_the_decomposition_closes_on_the_probe_estimands():
    """Delta_AE = Delta_AB + (acc(B) - acc(E)) must hold on whatever the probe reports, as
    it does in the main analysis. The cheapest possible check that the pairing is right.
    """
    rows = a8.load_rows(MAIN_CSV)
    got = {(r["estimand"], r["ansatz"]): r["mean"] for r in sup.probe_estimand_rows(
        rows, sorted({r["dataset_seed"] for r in rows}),
        sorted({r["seed_int"] for r in rows if r["seed_int"] is not None}), "linear")}
    for ansatz in {a for _, a in got}:
        residual = (got[("delta_AE", ansatz)]
                    - got[("delta_AB", ansatz)] - got[("delta_BE", ansatz)])
        assert abs(residual) < 1e-12, f"{ansatz}: residual {residual}"


# --- the head-initialisation measurement ---------------------------------------------

@pytest.mark.slow
def test_the_share_of_dead_units_does_not_depend_on_the_width():
    """The load-bearing claim of the h2 diagnosis: the INITIALISATION decides how often a
    unit is born dead, and the width only sets how many must die at once. If this ever
    fails, the explanation in the limitations section is wrong."""
    rows = sup.head_init_rows(11, draws=300, widths=(2, 3, 4))
    per_unit = {r["width"]: r["value"] for r in rows
                if r["quantity"] == "share_of_units_born_nearly_dead"}
    all_dead = {r["width"]: r["value"] for r in rows
                if r["quantity"] == "share_of_draws_with_every_unit_dead"}
    assert max(per_unit.values()) - min(per_unit.values()) < 0.08, per_unit
    # ... while the probability that EVERY unit dies falls off with the width.
    assert all_dead[2] > all_dead[3] > all_dead[4]
    # and it is close to (share per unit) ** width, which is the whole mechanism
    assert abs(all_dead[2] - per_unit[2] ** 2) < 0.05


# --- the Fourier support -------------------------------------------------------------

@pytest.mark.slow
def test_the_product_circuit_keeps_only_single_coordinate_frequencies():
    """Removing the CZ gates does not merely drop the Jacobian rank from 35 to 20: it
    deletes every mixed term of the spectrum. That is the sharper statement about what the
    entangling gate buys, and it is what makes arm F a control."""
    # The seeds must match the reported measurement: the support is a property of the
    # ansatz family, and at a single theta some lattice coefficients fall below the
    # numerical threshold. Summed over the reported initialisations the whole lattice
    # carries weight.
    rows = sup.fourier_rows(R=2, n_qubits=5, levels=("L1", "product"), seeds=(1, 2, 3))
    points = {r["ansatz"]: r["value"] for r in rows
              if r["quantity"] == "lattice_points_with_mass"}
    assert points["L1"] == 3125          # the whole lattice {-2..2}^5
    assert points["product"] == 21       # 1 constant + 5 coordinates x 4 nonzero frequencies
    mixed = [r for r in rows if r["ansatz"] == "product"
             and r["quantity"].startswith("mass_share_active_coords_")
             and int(r["quantity"].rsplit("_", 1)[1]) >= 2]
    assert not mixed, "the product circuit must have no mixed terms at all"
