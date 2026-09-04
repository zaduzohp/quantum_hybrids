"""Where the analysis pipeline is made to fail.

An analysis pipeline has a failure mode that a training script does not: it produces
numbers that look right. Every test here is aimed at one of those.

  * the exact tests, on inputs whose p-value is known in closed form (10 positive
    differences -> 2/1024), so a broken enumeration cannot hide behind a plausible p,
  * PAIRING: a difference must be taken within one training seed AND one generator seed.
    Mixing either is silent, so the key type itself refuses it,
  * POOLING: on synthetic data where Delta genuinely differs between the three datasets,
    the pipeline must report the divergence rather than average it away,
  * the report audit, broken on purpose: a false reported value must surface as a
    mismatch and a non-zero exit. A verification that cannot fail is not a verification,
  * determinism: two runs on one CSV, byte-identical tables.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_a8_analysis as a8
import run_main_series as a7

from qsocket.results import RESULT_COLUMNS
from qsocket.stats import (
    average_ranks,
    sign_test_exact,
    wilcoxon_signed_rank_exact,
)

REPO = Path(__file__).resolve().parents[1]
# The dry-run grid is a FIXTURE, not an output: two generator seeds x one dilution, the
# smallest CSV that still carries every column and both splits. Producing it costs a
# ~40-minute training run, so it is kept on disk rather than regenerated per session, and
# it has to travel with the repository or the tests below fail as if the pipeline were
# broken. It lives under outputs/ and is committed (see the negation in .gitignore);
# outputs_archive/ is scratch and is not. Regenerate with:
#     python scripts/run_main_series.py --dry-run --out-dir outputs/dry_run
DRY_RUN_CSV = REPO / "outputs" / "dry_run" / "a7_results.csv"

if not DRY_RUN_CSV.exists():  # pragma: no cover - a missing fixture, not a failing test
    raise RuntimeError(
        f"the dry-run fixture is missing at {DRY_RUN_CSV}. It is an input to this file, "
        "not something it produces; without it the tests below cannot tell a broken "
        "pipeline from an absent file. Regenerate it with "
        "`python scripts/run_main_series.py --dry-run --out-dir outputs/dry_run` "
        "(~40 min) and commit it."
    )

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# =====================================================================================
# The exact tests, including the assertions carried over from upstream
# =====================================================================================


def test_the_qc1_smoke_assertions_carried_over():
    """Upstream smoke assertions, moved here to guard qsocket.stats.

    qsocket.stats replaced the vendored enumeration; the numbers are unchanged,
    which is the point.
    """
    diffs = [0.1, 0.2, -0.05, 0.15]
    wilcoxon = wilcoxon_signed_rank_exact(diffs)
    sign = sign_test_exact(diffs)
    assert wilcoxon["pvalue"] is not None
    assert 0.0 <= sign["pvalue"] <= 1.0


def test_sign_test_on_ten_positive_differences_is_two_over_1024():
    """2 * (10 choose 0) / 2**10 = 2/1024."""
    result = sign_test_exact([0.01 * (i + 1) for i in range(10)])
    assert result["n_nonzero"] == 10
    assert result["positive"] == 10
    assert result["negative"] == 0
    assert result["pvalue"] == pytest.approx(2.0 / 1024.0)


def test_sign_test_known_values():
    # 9 positive, 1 negative: 2 * (C(10,0) + C(10,1)) / 1024 = 22/1024
    values = [0.1] * 9 + [-0.1]
    assert sign_test_exact(values)["pvalue"] == pytest.approx(22.0 / 1024.0)
    # exact zeros are dropped, not counted as ties on either side
    assert sign_test_exact([0.0, 0.0, 0.1])["n_nonzero"] == 1
    # a symmetric split cannot be significant
    assert sign_test_exact([0.1, -0.1])["pvalue"] == pytest.approx(1.0)


def test_wilcoxon_on_ten_positive_differences_is_two_over_1024():
    """With every difference positive the signed-rank statistic is 0, and only the
    all-positive and all-negative sign assignments reach it: 2/1024."""
    result = wilcoxon_signed_rank_exact([0.01 * (i + 1) for i in range(10)])
    assert result["statistic"] == 0.0
    assert result["pvalue"] == pytest.approx(2.0 / 1024.0)
    assert result["rank_biserial"] == pytest.approx(1.0)


def test_wilcoxon_agrees_with_scipy_exact_on_a_tie_free_sample():
    from scipy.stats import wilcoxon as scipy_wilcoxon

    values = [0.04, -0.01, 0.03, 0.07, -0.02, 0.05, 0.06, -0.03, 0.08, 0.02]
    mine = wilcoxon_signed_rank_exact(values)
    theirs = scipy_wilcoxon(values, mode="exact")
    assert mine["statistic"] == pytest.approx(float(theirs.statistic))
    assert mine["pvalue"] == pytest.approx(float(theirs.pvalue))


def test_average_ranks_handles_ties_by_averaging():
    assert average_ranks([1.0, 2.0, 2.0, 4.0]) == [1.0, 2.5, 2.5, 4.0]


# =====================================================================================
# MDE — recomputed from scipy
# =====================================================================================


def test_mde_constant_is_the_spec_formula_not_a_literal():
    df, n = 9, 10
    expected = (student_t.ppf(0.975, df) + student_t.ppf(0.80, df)) / math.sqrt(n)
    assert a8.mde_constant(10) == pytest.approx(expected)
    # The 0.995 the report prints, at the precision it prints it.
    assert round(a8.mde_constant(10), 3) == 0.995


def test_mde_constant_is_a_function_of_n():
    assert a8.mde_constant(10) > a8.mde_constant(25) > a8.mde_constant(80)


def test_no_pilot_mde_is_hard_coded_in_the_pipeline():
    """Pilot numbers may appear only in the blocks declaring what the documents say.

    The binding MDE is computed from this series' sigma, so a pilot MDE reached by the
    logic would freeze the threshold at the pilot value.
    """
    import ast

    path = REPO / "scripts" / "run_a8_analysis.py"
    source = path.read_text()
    tree = ast.parse(source)
    declared = {"RAPORT_PILOT", "RAPORT_DETERMINISTIC", "PILOT_POWER_CLAIMS"}
    declaration_lines: set[int] = set()
    for node in tree.body:
        targets = getattr(node, "targets", []) or ([node.target] if hasattr(node, "target") else [])
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & declared:
            declaration_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    assert len(declaration_lines) > 20, "the declaration blocks were not found"

    logic = [
        line for number, line in enumerate(source.splitlines(), start=1)
        if number not in declaration_lines
    ]
    for literal in ("0.0486", "0.0489", "0.0075"):
        offenders = [line.strip() for line in logic if literal in line]
        assert not offenders, (
            f"{literal} is used outside the declaration blocks — is it a threshold? "
            f"{offenders[:3]}"
        )


def test_mde_of_an_estimand_uses_this_series_sigma():
    point = a8.estimate([0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.13, 0.07, 0.12, 0.08])
    assert point["mde"] == pytest.approx(a8.mde_constant(10) * point["sd"])


def test_sigma_has_its_own_confidence_interval():
    """At n = 10 the factor on sigma is [0.69, 1.83]."""
    low, high = a8.sigma_confidence_interval(1.0, 10)
    assert low == pytest.approx(0.69, abs=0.01)
    assert high == pytest.approx(1.83, abs=0.01)
    # And at n = 3 it is the factor-of-12 interval that makes a 3-seed pilot useless.
    low3, high3 = a8.sigma_confidence_interval(1.0, 3)
    assert high3 / low3 == pytest.approx(12.0, abs=0.3)


# =====================================================================================
# TOST — on synthetic data with a known Delta and sigma
# =====================================================================================


def test_tost_declares_equivalence_at_delta_zero_with_small_sigma():
    rng = np.random.default_rng(0)
    values = list(0.0 + 0.002 * rng.standard_normal(10))
    result = a8.tost(values, delta=0.02)
    assert result["equivalent"] is True
    assert result["ci90_low"] > -0.02 and result["ci90_high"] < 0.02
    assert result["half_width_exceeds_delta"] is False


def test_tost_does_not_declare_equivalence_when_the_effect_sits_at_delta():
    rng = np.random.default_rng(1)
    values = list(0.02 + 0.002 * rng.standard_normal(10))
    assert a8.tost(values, delta=0.02)["equivalent"] is False


def test_tost_is_the_ninety_percent_ci_inside_the_margin():
    """The equivalence of the two formulations, which is why the 90% CI is reported."""
    rng = np.random.default_rng(2)
    for scale in (0.001, 0.005, 0.02, 0.05):
        values = list(0.003 + scale * rng.standard_normal(10))
        result = a8.tost(values, delta=0.02)
        inside = result["ci90_low"] > -0.02 and result["ci90_high"] < 0.02
        assert result["equivalent"] == inside


def test_tost_power_reproduces_the_whole_spec_power_table():
    """The published table was tabulated by Monte Carlo at sigma_Delta = 0.0612; the
    quadrature has to reproduce it entry for entry, which is what validates it.
    """
    for delta, expected in (
        (0.020, 0.003), (0.030, 0.058), (0.040, 0.260),
        (0.053, 0.624), (0.060, 0.774), (0.080, 0.969),
    ):
        assert a8.tost_power(sigma=0.0612, n=10, delta=delta) == pytest.approx(
            expected, abs=0.002), f"SPEC 7.5 row delta = {delta}"


def test_tost_power_for_a_to_e_is_the_one_the_decision_was_closed_on():
    """Keeping TOST for Delta_AE stands on power 1.000 at the pilot sigma_Delta = 0.0075."""
    assert a8.tost_power(sigma=0.0075, n=10, delta=0.02) == pytest.approx(1.0, abs=0.01)


def test_tost_power_for_a_to_b_is_far_below_the_floor_at_the_pilot_sigma():
    """The report prints 0.000 here while the recomputation gives 0.017.

    The recomputation reproduces the whole published power table and an independent Monte
    Carlo, so the implementation is not what is wrong. Reported, not resolved: 0.017 is as
    far below the 0.80 floor as 0.000 is, so the withdrawal of TOST for Delta_AB stands
    either way, and a pilot figure is not a fatal mismatch.
    """
    power = a8.tost_power(sigma=0.0489, n=10, delta=0.02)
    assert power < a8.TOST_POWER_FLOOR
    assert power == pytest.approx(0.0171, abs=0.001)


def test_the_three_scripts_share_one_implementation_of_each_statistic():
    """MDE, the sigma CI, the binomial SE and the TOST power were written three times.

    The copies agreed to the last bit when it was checked, which is why an edit to one of
    them would not have surfaced: nothing compared them. They are now one function, and
    the scripts must hold the SAME object rather than an equal-looking one.
    """
    import run_pilot_sigma as pilot

    from qsocket import stats

    assert a8.mde_constant is stats.mde_constant
    assert a8.sigma_confidence_interval is stats.sigma_confidence_interval
    assert a7.mde is stats.mde
    assert a7.binomial_se is stats.binomial_se
    assert pilot.mde is stats.mde
    assert pilot.sigma_ci is stats.sigma_confidence_interval
    assert pilot.binomial_se is stats.binomial_se
    # a8.tost_power and pilot.tost_power_at_zero are thin wrappers (a default delta, a
    # positional order), so identity is the wrong check — equality of the numbers is not.
    for sigma, n in ((0.05, 10), (0.02, 30), (0.1, 30)):
        expected = stats.tost_power(sigma=sigma, n=n, delta=a8.TOST_DELTA)
        assert a8.tost_power(sigma=sigma, n=n) == expected
        assert pilot.tost_power_at_zero(a8.TOST_DELTA, sigma, n) == expected


def test_the_power_is_computed_on_the_df_the_test_was_run_on():
    """A blocked TOST beside an iid power is not one analysis.

    The test spends one degree of freedom per generator seed (n - J); the power used to
    default to n - 1. The gap is small at this series' numbers but it sits directly under
    the 0.80 floor a STOP row is decided on, so it has to be the same df.
    """
    sigma, n, blocks = 0.039, 30, 3
    blocked = a8.tost_power(sigma=sigma, n=n, df=n - blocks)
    iid = a8.tost_power(sigma=sigma, n=n)
    assert blocked < iid, "fewer degrees of freedom cannot buy power"
    assert blocked == pytest.approx(iid, abs=5e-3), "and the difference is small, not a bug"

    record = a8.tost([0.01] * n, se=sigma / math.sqrt(n), df=n - blocks, sd=sigma)
    assert record["df"] == n - blocks
    assert a8.tost_power(sigma=sigma, n=n, df=record["df"]) == blocked

    # The sample size quoted alongside it answers the same question on the same df.
    needed = a8.seeds_needed_for_tost(sigma=sigma, blocks=blocks)
    assert a8.tost_power(sigma=sigma, n=needed, df=needed - blocks) >= a8.TOST_POWER_FLOOR
    assert a8.tost_power(sigma=sigma, n=needed - 1, df=needed - 1 - blocks) < a8.TOST_POWER_FLOOR


def test_tost_power_grows_with_n_and_shrinks_with_sigma():
    assert a8.tost_power(sigma=0.02, n=10) < a8.tost_power(sigma=0.02, n=80)
    assert a8.tost_power(sigma=0.06, n=10) < a8.tost_power(sigma=0.01, n=10)


def test_tost_power_is_deterministic():
    """Quadrature, not Monte Carlo, so two runs must agree exactly."""
    first = a8.tost_power(sigma=0.03, n=10, delta=0.02)
    second = a8.tost_power(sigma=0.03, n=10, delta=0.02)
    assert first == second


def test_the_number_of_seeds_equivalence_would_need_is_computed():
    """A power statement replaces the verdict, and it carries a number of seeds.

    The published "n ~ 80" reproduces at sigma_Delta = 0.0612, where the answer is 82; at
    the sigma_Delta = 0.0489 quoted alongside it the answer is 53. Reported, not resolved:
    both are far above the 10 seeds the series has.
    """
    assert a8.seeds_needed_for_tost(sigma=0.0612, delta=0.02) == 82
    assert a8.seeds_needed_for_tost(sigma=0.0489, delta=0.02) == 53
    # And the number is a function of sigma, not a constant carried over from a document.
    assert a8.seeds_needed_for_tost(sigma=0.0075, delta=0.02) < 10


def test_seeds_needed_for_mde_scales_the_right_way():
    assert a8.seeds_needed_for_mde(sigma=0.05, effect=0.10) < \
        a8.seeds_needed_for_mde(sigma=0.05, effect=0.02)
    assert a8.seeds_needed_for_mde(sigma=0.0, effect=0.1) is None


# =====================================================================================
# McNemar — by hand against statsmodels
# =====================================================================================


def test_mcnemar_matches_a_hand_built_two_by_two():
    from statsmodels.stats.contingency_tables import mcnemar

    left = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1], dtype=bool)
    right = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    result = a8.mcnemar_from_vectors(left, right)
    # Counted by hand from the two vectors above.
    assert result["both_correct"] == 3
    assert result["b_left_only"] == 3
    assert result["c_right_only"] == 2
    assert result["neither"] == 2
    assert result["discordant"] == 5
    reference = mcnemar([[3, 3], [2, 2]], exact=True)
    assert result["pvalue"] == pytest.approx(float(reference.pvalue))
    assert result["delta_from_counts"] == pytest.approx((3 - 2) / 10)


def test_mcnemar_is_symmetric_in_p_and_antisymmetric_in_delta():
    left = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    right = np.array([0, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    forward = a8.mcnemar_from_vectors(left, right)
    backward = a8.mcnemar_from_vectors(right, left)
    assert forward["pvalue"] == pytest.approx(backward["pvalue"])
    assert forward["delta_from_counts"] == pytest.approx(-backward["delta_from_counts"])
    assert forward["b_left_only"] == backward["c_right_only"]


# =====================================================================================
# PAIRING — the silent failure
# =====================================================================================


def test_a_series_keyed_by_a_bare_seed_is_refused():
    """Keying by seed alone would pair training seed 1 of dataset 11 against training
    seed 1 of dataset 22 and produce a plausible number for nothing."""
    good = {a8.PairKey(11, 1): 0.8, a8.PairKey(11, 2): 0.81}
    bad = {1: 0.7, 2: 0.71}
    with pytest.raises(ValueError, match="not PairKey"):
        a8.paired_differences(good, bad, label="delta_XY")
    with pytest.raises(ValueError, match="not PairKey"):
        a8.paired_differences(bad, good, label="delta_XY")


def test_pairing_never_crosses_a_generator_seed_or_a_training_seed():
    left = {a8.PairKey(11, 1): 0.90, a8.PairKey(22, 1): 0.70}
    right = {a8.PairKey(11, 1): 0.80, a8.PairKey(22, 1): 0.60}
    pair = a8.paired_differences(left, right, label="delta_AB")
    assert pair["keys"] == [a8.PairKey(11, 1), a8.PairKey(22, 1)]
    assert pair["differences"] == pytest.approx([0.10, 0.10])
    for key in pair["keys"]:
        assert isinstance(key, a8.PairKey)
    # A key present on one side only is reported, never folded into the mean.
    lonely = a8.paired_differences(
        {**left, a8.PairKey(33, 1): 0.5}, right, label="delta_AB")
    assert lonely["unpaired_left"] == [a8.PairKey(33, 1)]
    assert len(lonely["differences"]) == 2


def test_the_lr_is_part_of_the_accuracy_index_key():
    """Arms F and D_matched have no ansatz dimension but the cell lr does, so the same
    arm is measured once per distinct cell lr. Indexed without lr, one of those rows wins
    by CSV order and the other cell's paired difference is taken at the wrong lr."""
    rows = a8.load_rows(DRY_RUN_CSV)
    index = a8.accuracy_index(rows, "test")
    f_keys = [k for k in index if k[1] == "F"]
    lrs_per_seed: dict[int, set] = {}
    for key in f_keys:
        lrs_per_seed.setdefault(key[5], set()).add(key[6])
    # The dry run really does hold arm F at two lr values per seed.
    assert any(len(v) > 1 for v in lrs_per_seed.values()), (
        "this fixture no longer exercises the two-lr case"
    )
    a = index[(11, "F", a7.PRODUCT_ANSATZ, "linear", None, 1, "0.01")]
    b = index[(11, "F", a7.PRODUCT_ANSATZ, "linear", None, 1, "0.03")]
    assert a["accuracy"] != b["accuracy"], "two different runs, two different accuracies"


def test_arm_f_is_paired_at_the_cell_lr_of_the_ansatz_it_is_compared_inside():
    """Δ_AF for the L1 cell must use arm F at the L1 cell's lr, and likewise for L2."""
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    lrs = a8.selected_lrs(rows)
    index = a8.accuracy_index(rows, "test")
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for ansatz_level in ("L1", "L2"):
        cell_lr = lrs["cell_lr"][(11, "linear", ansatz_level)]
        pair = analysis["cells"][("linear", ansatz_level)]["pairs"]["delta_AF"]
        for key, difference in zip(pair["keys"], pair["differences"]):
            expected_f = float(index[(
                11, "F", a7.PRODUCT_ANSATZ, "linear", None, key.seed, f"{cell_lr:g}"
            )]["accuracy"])
            expected_a = float(index[(
                11, "A", ansatz_level, "linear", None, key.seed, f"{cell_lr:g}"
            )]["accuracy"])
            assert difference == pytest.approx(expected_a - expected_f)


def test_arms_a_and_b_of_one_cell_must_share_one_lr():
    rows = a8.load_rows(DRY_RUN_CSV)
    poisoned = [dict(r) for r in rows]
    for row in poisoned:
        if row["arm"] == "B" and row["ansatz_level"] == "L1":
            row["lr_float"] = 0.999
    with pytest.raises(SystemExit, match="more than one lr"):
        a8.selected_lrs(poisoned)


def test_two_rows_for_one_cell_disagreeing_on_accuracy_is_a_stop():
    rows = a8.load_rows(DRY_RUN_CSV)
    duplicate = dict(rows[0])
    duplicate["accuracy"] = str(float(duplicate["accuracy"]) + 0.01)
    with pytest.raises(SystemExit, match="disagree on accuracy"):
        a8.accuracy_index(rows + [duplicate], rows[0]["split"])


# =====================================================================================
# The mandatory decomposition, on real rows
# =====================================================================================


def test_delta_ae_decomposition_holds_to_floating_point_on_real_data():
    """Delta_AE == Delta_AB + (acc(B) - acc(E)), in every cell."""
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    assert analysis["cells"], "no cells were formed from the fixture"
    for cell in analysis["cells"].values():
        decomposition = cell["decomposition_delta_AE"]
        assert decomposition["residual"] == pytest.approx(0.0, abs=1e-12)
        assert decomposition["delta_AE"] == pytest.approx(
            decomposition["delta_AB"] + decomposition["acc_B_minus_acc_E"], abs=1e-12
        )


def test_acc_b_minus_acc_f_is_never_computed():
    """Brief 1.1 forbids it explicitly: both arms of Δ_AF are trained, and B−F is not a
    contrast anybody declared."""
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for cell in analysis["cells"].values():
        assert "delta_BF" not in cell["estimands"]
        assert not any("BF" in name for name in cell["estimands"])
        assert not any("BF" in name for name in cell["pairs"])


# =====================================================================================
# Pooling versus per dataset — the divergence must be reported, not averaged away
# =====================================================================================


def _pairs_from(mapping: dict) -> dict:
    """{(dataset_seed, seed): difference} in the shape paired_differences returns."""
    keys = [a8.PairKey(*k) for k in sorted(mapping)]
    return {
        "label": "synthetic",
        "keys": keys,
        "differences": [float(mapping[tuple(k)]) for k in keys],
        "left_values": [], "right_values": [], "unpaired_left": [], "unpaired_right": [],
    }


def test_pooling_reports_a_divergence_when_delta_differs_between_datasets():
    """Brief 1.1a: on divergence the per-dataset reading wins and the divergence is a
    finding. The pipeline must not answer with the pooled mean alone."""
    mapping = {}
    for seed in range(1, 11):
        mapping[(11, seed)] = 0.20 + 0.001 * seed
        mapping[(22, seed)] = 0.02 + 0.001 * seed
        mapping[(33, seed)] = 0.02 + 0.001 * seed
    pairs = _pairs_from(mapping)
    pooled = a8.pooled_blocked_estimate(pairs)
    per_dataset = {
        dataset_seed: a8.estimate(
            [v for k, v in mapping.items() if k[0] == dataset_seed]
        )
        for dataset_seed in (11, 22, 33)
    }
    divergence = a8.divergence_check(per_dataset, pooled)
    assert divergence["diverged"] is True
    assert 11 in divergence["diverging_dataset_seeds"]
    assert "per-dataset reading WINS" in divergence["verdict"]


def test_pooling_reports_no_divergence_when_delta_agrees():
    rng = np.random.default_rng(7)
    mapping = {
        (dataset_seed, seed): 0.09 + 0.01 * rng.standard_normal()
        for dataset_seed in (11, 22, 33)
        for seed in range(1, 11)
    }
    pairs = _pairs_from(mapping)
    pooled = a8.pooled_blocked_estimate(pairs)
    per_dataset = {
        dataset_seed: a8.estimate([v for k, v in mapping.items() if k[0] == dataset_seed])
        for dataset_seed in (11, 22, 33)
    }
    assert a8.divergence_check(per_dataset, pooled)["diverged"] is False


def test_the_pooled_estimate_is_blocked_and_not_a_variance_component():
    mapping = {
        (dataset_seed, seed): 0.10 + 0.01 * dataset_seed / 11 + 0.001 * seed
        for dataset_seed in (11, 22, 33)
        for seed in range(1, 11)
    }
    pairs = _pairs_from(mapping)
    pooled = a8.pooled_blocked_estimate(pairs)
    assert pooled["n"] == 30
    assert pooled["n_blocks"] == 3
    # 30 differences minus three block means.
    assert pooled["df_residual"] == 27
    # Balanced design: the blocked estimate is the grand mean.
    assert pooled["mean"] == pytest.approx(float(np.mean(pairs["differences"])))
    assert "FIXED" in pooled["model"]
    assert "variance component" in pooled["not_a_variance_component"]
    # And the hand computation agrees with statsmodels OLS on the same model.
    crosscheck = a8.ols_block_crosscheck(pairs)
    assert crosscheck["computable"] is True
    assert crosscheck["mse_resid"] == pytest.approx(pooled["mse_within"])
    assert crosscheck["df_resid"] == pytest.approx(pooled["df_residual"])


def test_pooling_is_honest_about_an_unbalanced_design():
    mapping = {(11, seed): 0.1 for seed in range(1, 11)}
    mapping.update({(22, seed): 0.2 for seed in range(1, 4)})
    pooled = a8.pooled_blocked_estimate(_pairs_from(mapping))
    # Unweighted mean of block means, not the row mean, so a small block is not drowned.
    assert pooled["mean"] == pytest.approx(0.15)
    assert pooled["block_counts"] == {"11": 10, "22": 3}


# =====================================================================================
# MixedLM — the check the report asks for, at a realistic n
# =====================================================================================


def _synthetic_pairs(dilutions, *, seeds=range(1, 11), dataset_seeds=(11,), effect=None):
    """Paired differences with a known per-dilution effect and a per-seed offset.

    The per-seed offset is what makes a random intercept meaningful: the same training
    seeds recur in every dilution cell.
    """
    rng = np.random.default_rng(11)
    offsets = {seed: 0.02 * rng.standard_normal() for seed in seeds}
    effect = effect or {d: 0.09 for d in dilutions}
    out = {}
    for dilution in dilutions:
        mapping = {}
        for dataset_seed in dataset_seeds:
            for seed in seeds:
                mapping[(dataset_seed, seed)] = (
                    effect[dilution] + offsets[seed] + 0.004 * rng.standard_normal()
                )
        out[dilution] = _pairs_from(mapping)
    return out


def test_mixedlm_fits_at_a_realistic_n_and_equals_the_paired_contrast():
    """The mixed model is formally redundant here, so it must come out equivalent.

    Tested at the real grid size, because at n = 2 the fit is singular — which is why the
    pipeline reports 'not computable' rather than crashing.
    """
    pairs = _synthetic_pairs(a8.CONTRACT_DILUTIONS)
    record = a8.mixedlm_check(pairs)
    assert record["computable"] is True, record.get("reason")
    assert record["formula"] == "d ~ C(dilution), groups = seed"
    assert record["n"] == 40
    # The intercept is the reference dilution's mean; the balanced design makes the fixed
    # effects reproduce the within-subject cell means.
    reference = a8.CONTRACT_DILUTIONS[0]
    assert record["reference_level"] == reference, (
        "the reference level must be the linear head, not the alphabetically first name"
    )
    cell_mean = float(np.mean(pairs[reference]["differences"]))
    assert record["intercept"] == pytest.approx(cell_mean, abs=5e-3)
    # The pipeline must compare the intercept against THIS mean — the reference cell's —
    # and not against the mean over the whole axis, which is a different quantity as soon
    # as the axis has more than one point.
    assert record["reference_level_mean_for_comparison"] == pytest.approx(cell_mean)
    # The per-seed offset is real, so the random intercept must pick up variance.
    assert record["group_variance"] > 0


def test_the_mixedlm_equivalence_check_is_decided_and_never_left_undecided():
    """The verdict row must be able to go red.

    It used to be computed only when the axis had ONE point, so on every real run the
    flag was None, the `is False` branch could not fire, and the row printed
    'equivalent, as predicted' without any comparison having been made.
    """
    for dilutions in (a8.CONTRACT_DILUTIONS, ("linear",)):
        record = a8.mixedlm_check(_synthetic_pairs(dilutions))
        assert record["computable"] is True
        assert isinstance(record["equivalent_to_paired_contrast"], bool), dilutions
        assert record["equivalent_to_paired_contrast"] is True, dilutions
        assert abs(record["intercept_minus_reference_mean"]) < record["equivalence_tolerance"]


def test_a_mixedlm_that_disagrees_with_the_paired_contrast_is_reported_as_a_finding():
    """A green verdict has to be earned: falsify the comparison and the row must turn."""
    record = a8.mixedlm_check(_synthetic_pairs(a8.CONTRACT_DILUTIONS))
    broken = dict(record)
    broken["intercept"] = record["intercept"] + 0.05
    broken["intercept_minus_reference_mean"] = 0.05
    broken["equivalent_to_paired_contrast"] = False

    rows = a8.verdicts(
        {"mixedlm": {"L1": broken}, "cells": {}, "tost": {}, "d_best": {},
         "axis": {"ansatz_levels": []}, "holm": {}},
        {"ridge_forbidden_arms_present": []}, [],
        {"complete_contract_grid": True, "arm_D_present": True, "missing": {}},
    )
    mixed = [r for r in rows if "MixedLM" in r["row"]]
    assert len(mixed) == 1
    assert "report, do not choose" in mixed[0]["verdict"]
    assert "finding about the model" in mixed[0]["detail"]


def test_mixedlm_recovers_a_declining_axis():
    """The dilution curve is descriptive, but the check still has to track the axis it is given."""
    effect = {"linear": 0.12, "h2": 0.08, "h4": 0.04, "h42": 0.00}
    record = a8.mixedlm_check(_synthetic_pairs(a8.CONTRACT_DILUTIONS, effect=effect))
    assert record["computable"] is True
    intercept = record["intercept"]
    # Reference level is `linear`; each other level's coefficient is its shift from it.
    for dilution in ("h2", "h4", "h42"):
        key = f"C(dilution)[T.{dilution}]"
        assert intercept + record["params"][key] == pytest.approx(
            effect[dilution], abs=0.01), dilution


def test_mixedlm_reports_rather_than_crashes_when_it_cannot_fit():
    """A check that dies takes the whole pipeline with it, so a failed fit is RECORDED
    and the run continues. On the dry-run fixture the fit really is singular."""
    # One training seed: nothing for a random intercept to group over.
    record = a8.mixedlm_check(_synthetic_pairs(("linear",), seeds=range(1, 2)))
    assert record["computable"] is False
    assert "reason" in record

    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for record in analysis["mixedlm"].values():
        # n = 2 makes the fit singular; the pipeline must say so rather than raise.
        assert record["computable"] is False
        assert record["reason"]


def test_mixedlm_is_labelled_a_check_and_never_the_main_analysis():
    record = a8.mixedlm_check(_synthetic_pairs(a8.CONTRACT_DILUTIONS))
    assert "CHECK" in record["note"]
    assert "formally redundant" in record["note"]


# =====================================================================================
# The estimator itself
# =====================================================================================


def test_estimate_reports_both_confidence_levels_and_the_ninety_is_narrower():
    values = [0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.13, 0.07, 0.12, 0.08]
    point = a8.estimate(values)
    assert point["n"] == 10
    assert point["ci90_half_width"] < point["ci95_half_width"]
    assert point["ci95_low"] < point["mean"] < point["ci95_high"]
    assert point["ci95_excludes_zero"] is True
    assert point["p_sign_exact"] == pytest.approx(2.0 / 1024.0)


def test_estimate_matches_the_textbook_t_interval():
    values = [0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.13, 0.07, 0.12, 0.08]
    array = np.asarray(values)
    point = a8.estimate(values)
    half = student_t.ppf(0.975, 9) * array.std(ddof=1) / math.sqrt(10)
    assert point["ci95_high"] - point["mean"] == pytest.approx(half)


def test_estimate_survives_a_single_difference_without_inventing_a_ci():
    point = a8.estimate([0.1])
    assert point["n"] == 1
    assert not np.isfinite(point["ci95_low"])
    assert point["above_mde"] is False


def test_delta_is_expressed_in_sigma_seed_and_ceiling_units():
    """H1 is phrased with those two denominators, so both are mandatory."""
    point = a8.estimate([0.09] * 5 + [0.10] * 5)
    units = a8.in_reference_units(
        point, sigma_seed_left=0.0063, sigma_seed_right=0.0478, ceilings={"11": 0.2128})
    assert units["in_sigma_seed_left"] == pytest.approx(point["mean"] / 0.0063)
    assert units["in_sigma_seed_right"] == pytest.approx(point["mean"] / 0.0478)
    # One dataset seed -> one ceiling ratio, and it is the single number too.
    assert units["in_ceiling"] == pytest.approx(point["mean"] / 0.2128)
    assert units["in_ceiling_by_seed"]["11"] == pytest.approx(point["mean"] / 0.2128)
    # A missing ceiling must be NaN, never a silent 0 or a dropped column.
    assert not np.isfinite(a8.in_reference_units(
        point, sigma_seed_left=0.01, sigma_seed_right=0.01, ceilings=None)["in_ceiling"])


def test_the_ceiling_denominator_is_per_generator_seed_and_never_pooled():
    """The ceiling differs between datasets, so the denominator is per generator seed.

    Regression test: `in_reference_units` used to take a single ceiling and compute it
    only for a one-dataset analysis, so the main series reported NaN everywhere.
    """
    point = a8.estimate_blocked({
        "differences": [0.10] * 10 + [0.06] * 10 + [0.12] * 10,
        "keys": [a8.PairKey(11, s) for s in range(10)]
              + [a8.PairKey(22, s) for s in range(10)]
              + [a8.PairKey(33, s) for s in range(10)],
    })
    ceilings = {"11": 0.20, "22": 0.15, "33": 0.30}
    units = a8.in_reference_units(
        point, sigma_seed_left=0.01, sigma_seed_right=0.05, ceilings=ceilings)

    # Each block over its own ceiling, never the pooled mean over an averaged ceiling.
    assert units["in_ceiling_by_seed"] == {
        "11": pytest.approx(0.10 / 0.20),
        "22": pytest.approx(0.06 / 0.15),
        "33": pytest.approx(0.12 / 0.30),
    }
    assert units["ceiling_by_seed"] == ceilings
    # The pooled single number stays NaN on purpose with more than one block: it is the
    # quantity the report forbids.
    assert not np.isfinite(units["in_ceiling"])
    assert not np.isfinite(units["ceiling_used"])
    # A pooled ceiling would have been mean(0.0933)/mean(0.2167) = 0.431 — a number that
    # describes none of the three blocks (0.50, 0.40, 0.40).
    assert all(np.isfinite(v) for v in units["in_ceiling_by_seed"].values())


def test_a_dataset_seed_without_a_recomputed_ceiling_is_reported_as_missing():
    """--skip-slow-verification leaves a seed without a ceiling. That seed must drop out
    of the map, never inherit another seed's ceiling."""
    point = a8.estimate_blocked({
        "differences": [0.10] * 10 + [0.06] * 10,
        "keys": [a8.PairKey(11, s) for s in range(10)]
              + [a8.PairKey(22, s) for s in range(10)],
    })
    units = a8.in_reference_units(
        point, sigma_seed_left=0.01, sigma_seed_right=0.05, ceilings={"11": 0.20})
    assert set(units["in_ceiling_by_seed"]) == {"11"}
    assert "22" not in units["ceiling_by_seed"]


# =====================================================================================
# The report audit, broken on purpose
# =====================================================================================


def test_the_deterministic_numbers_of_raport_tex_reproduce():
    """Deterministic numbers: hashes, evr1, parameter counts, shares, binomial SE, MDE
    constant. The G1 and ceiling rows need training runs and are covered end to end."""
    context = a8.verification_context([11], cache_path=None, skip_slow=True)
    rows, mismatches = a8.verification_table(context, series_numbers={})
    assert mismatches == [], f"raport.tex and the recomputation disagree: {mismatches}"
    checked = {r["quantity"] for r in rows if r["agrees"] == "YES"}
    for quantity in (
        "dataset_hash_ds11", "pca_hash_ds11", "file_sha256_ds11", "evr1_ds11",
        "total_variance_explained_ds11", "socket_params_R2", "jacobian_rank_L1_R2",
        "head_params_linear", "head_params_h2", "head_params_h4", "head_params_h42",
        "quantum_share_linear", "quantum_share_h42", "binomial_se_1200", "mde_constant_n10",
    ):
        assert quantity in checked, f"{quantity} was not verified"


def test_the_verification_fails_when_a_reported_value_is_falsified():
    """The test that breaks on purpose: a verification that can never fail is not a
    verification.
    """
    context = a8.verification_context([11], cache_path=None, skip_slow=True)
    falsified = {
        key: dict(spec) for key, spec in a8.RAPORT_DETERMINISTIC.items()
    }
    falsified["evr1_ds11"]["reported"] = 0.9999
    rows, mismatches = a8.verification_table(
        context, series_numbers={}, reported=falsified)
    assert len(mismatches) == 1
    assert mismatches[0]["quantity"] == "evr1_ds11"
    assert mismatches[0]["reported"] == 0.9999
    assert any(r["agrees"] == "NO — STOP" for r in rows)


def test_a_falsified_hash_is_also_caught():
    context = a8.verification_context([11], cache_path=None, skip_slow=True)
    falsified = {key: dict(spec) for key, spec in a8.RAPORT_DETERMINISTIC.items()}
    falsified["dataset_hash_ds11"]["reported"] = "deadbeefdeadbeef"
    _, mismatches = a8.verification_table(context, series_numbers={}, reported=falsified)
    assert [m["quantity"] for m in mismatches] == ["dataset_hash_ds11"]


def test_pilot_numbers_are_never_asserted_for_equality():
    """Pilot numbers are replaced by the main series, not confirmed. Asserting them would
    fail the pipeline on correct data.
    """
    context = a8.verification_context([11], cache_path=None, skip_slow=True)
    rows, mismatches = a8.verification_table(
        context, series_numbers={"sigma_delta_AB": 0.9, "delta_AB": -5.0})
    assert mismatches == []
    pilot = [r for r in rows if r["category"].startswith("b")]
    assert pilot, "the pilot block is missing from the audit"
    for row in pilot:
        assert row["agrees"].startswith("n/a")


def test_evr1_is_recomputed_from_the_raw_rows_not_read_from_the_manifest():
    """Brief 1.6: recompute FROM SOURCE. Reading the manifest's own PCA block would check
    a copy of the number instead of the number."""
    numbers = a8.recompute_dataset_numbers(11)
    assert numbers["n_train_rows_used"] == 4200
    assert numbers["evr1"] == pytest.approx(0.758533, abs=1e-6)
    assert numbers["total_variance_explained"] == pytest.approx(1.0, abs=1e-9)
    source = (REPO / "scripts" / "run_a8_analysis.py").read_text()
    body = source.split("def recompute_dataset_numbers")[1].split("\ndef ")[0]
    assert "explained_variance_ratio_" in body
    assert 'manifest["pca"]' not in body, "evr1 must not be read out of the manifest"


def test_the_structural_numbers_are_counted_on_built_objects():
    structural = a8.recompute_structural_numbers()
    assert structural["socket_params_nominal_R2"] == 35
    assert structural["jacobian_rank"]["L1"] == 35
    assert structural["jacobian_rank"]["L2"] == 35
    # The product circuit has 15 dead directions, so Delta_AF is an upper bound.
    assert structural["jacobian_rank"][a7.PRODUCT_ANSATZ] == 20
    assert structural["head_params"] == {"linear": 6, "h2": 15, "h4": 29, "h42": 295}
    for dilution, params in structural["head_params"].items():
        assert structural["quantum_share_percent"][dilution] == pytest.approx(
            100.0 * 35 / (35 + params))


# =====================================================================================
# Provenance — a dry run may not be reported as a result
# =====================================================================================


def test_an_incomplete_grid_is_stamped_provisional():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    assert prov["complete_contract_grid"] is False
    assert prov["status"] == a8.STATUS_PROVISIONAL
    assert prov["missing"]["dataset_seeds"] == [22, 33]
    assert prov["missing"]["seeds"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert "not results" in prov["note"]


def test_a_csv_that_is_not_the_main_series_schema_is_refused(tmp_path):
    """The pipeline reads the main-series schema and nothing else. A renamed or reordered column would make
    the analysis read NaN and look like it simply did not measure that quantity."""
    import pandas as pd

    frame = pd.read_csv(DRY_RUN_CSV, dtype=str, keep_default_na=False)
    assert frame.columns.tolist() == list(RESULT_COLUMNS)
    wrong = tmp_path / "wrong.csv"
    frame.rename(columns={"accuracy": "acc"}).to_csv(wrong, index=False)
    with pytest.raises(SystemExit, match="RESULT_COLUMNS"):
        a8.load_rows(wrong)


def test_the_contract_grid_definition_is_the_one_the_report_declares():
    assert a8.CONTRACT_DATASET_SEEDS == (11, 22, 33)
    assert a8.CONTRACT_DILUTIONS == ("linear", "h2", "h4", "h42")
    assert a8.CONTRACT_ANSATZ_LEVELS == ("L1", "L2")
    assert a8.CONTRACT_SEEDS == tuple(range(1, 11))
    assert a8.CONTRACT_N_TEST == 1200


def test_the_confirmatory_family_has_one_element_and_holm_is_not_applied():
    assert a8.CONFIRMATORY_FAMILY == ("delta_AB",)
    assert "not applied" in a8.HOLM_NOTE
    source = (REPO / "scripts" / "run_a8_analysis.py").read_text()
    assert "multipletests" not in source, "Holm must not be applied in part A"


def test_tost_is_reported_as_a_test_only_for_delta_ae():
    assert a8.TOST_REPORTED_AS_TEST == ("delta_AE",)
    assert a8.TOST_COMPUTED_NOT_REPORTED == ("delta_AB",)
    assert a8.TOST_DELTA == 0.02


def test_no_interaction_contrast_of_the_dilution_axis_is_computed():
    """H1 is an estimand and the axis is descriptive, so the axis is never called a test
    of H1.
    """
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for cell in analysis["cells"].values():
        for name in cell["estimands"]:
            assert "interaction" not in name


# =====================================================================================
# Ridge, and the arms it may not be computed for
# =====================================================================================


def test_ridge_is_never_reported_for_a_trained_socket_arm():
    """Brief 7: arms A and F have a trained socket, so the closed-form readout has no
    defined argument."""
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    diag = a8.diagnostics(rows, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    assert "A" not in diag["ridge_control"]
    assert "F" not in diag["ridge_control"]
    assert diag["ridge_forbidden_arms_present"] == []


def test_a_ridge_value_on_a_trained_arm_raises_a_stop_verdict():
    rows = a8.load_rows(DRY_RUN_CSV)
    poisoned = [dict(r) for r in rows]
    for row in poisoned:
        if row["arm"] == "A":
            row["ridge_accuracy"] = "0.5"
    prov = a8.provenance(poisoned, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        poisoned, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    diag = a8.diagnostics(poisoned, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    assert diag["ridge_forbidden_arms_present"] == ["A"]
    verdict_rows = a8.verdicts(analysis, diag, [], prov)
    stops = [v for v in verdict_rows if v["stop"] and "ridge" in v["row"]]
    assert stops, "a ridge value on a trained arm must fire a STOP"


# =====================================================================================
# Arm D_best: one width per generator seed, chosen on that seed's validation runs
# =====================================================================================


def test_d_best_width_is_chosen_per_generator_seed_not_pooled():
    """The RFF width M is selected on validation, one M per generator seed.

    Regression test: the selection loop used to iterate over the dataset seeds while
    reading a series spanning all of them, so every seed received the pooled argmax.

    The two seeds here are built to disagree — one favours M = 32, the other M = 512 — so
    a pooled selection cannot produce both.
    """
    rows = a8.load_rows(DRY_RUN_CSV)
    ds22 = "hyperplanes_n_features20_n_hyperplanes3_dim_hyperplanes5_seed22"
    # ds11: make the smallest width win on validation.
    for row in rows:
        if row["arm"] == "D_best" and row["split"] == "val":
            row["accuracy"] = {32: "0.90", 128: "0.70", 512: "0.60"}[row["width_int"]]
    # ds22: a copy of every row, with the largest width winning on validation.
    clones = []
    for row in rows:
        clone = dict(row)
        clone["dataset"], clone["dataset_seed"] = ds22, 22
        if clone["arm"] == "D_best" and clone["split"] == "val":
            clone["accuracy"] = {32: "0.60", 128: "0.70", 512: "0.90"}[clone["width_int"]]
        clones.append(clone)
    rows = rows + clones

    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    selected = {seed: record["selected_width"]
                for seed, record in analysis["d_best"].items()}
    assert selected == {11: 32, 22: 512}, selected
    # And the per-width validation means must come from one seed each, not from both.
    assert analysis["d_best"][11]["per_width"][32]["val"]["mean"] == pytest.approx(0.90)
    assert analysis["d_best"][22]["per_width"][32]["val"]["mean"] == pytest.approx(0.60)
    assert analysis["d_best"][11]["per_width"][32]["val"]["n"] == 2  # 2 seeds in the dry run


# =====================================================================================
# Arm D may be absent — skip it with a reason, do not substitute anything
# =====================================================================================


def test_a_csv_without_arm_d_still_analyses_everything_else():
    """Arm D is an exploratory passenger, so its absence skips Delta_BD and stops nothing.
    Nothing is substituted for it.
    """
    rows = [r for r in a8.load_rows(DRY_RUN_CSV) if not r["arm"].startswith("D")]
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    assert prov["arm_D_present"] is False
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for cell in analysis["cells"].values():
        assert "delta_AB" in cell["estimands"]
        assert "delta_AE" in cell["estimands"]
        assert "delta_BD_matched" not in cell["estimands"]
        assert "delta_BD_best" not in cell["estimands"]
    diag = a8.diagnostics(rows, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    verdict_rows = a8.verdicts(analysis, diag, [], prov)
    skipped = [v for v in verdict_rows if "arm D absent" in v["row"]]
    assert skipped and not skipped[0]["stop"]
    assert "Nothing is substituted" in skipped[0]["detail"]


# =====================================================================================
# End to end, on the dry-run CSV, twice — determinism
# =====================================================================================


def _run_pipeline(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "run_a8_analysis.py"),
            "--results", str(DRY_RUN_CSV),
            "--out-dir", str(out_dir),
            "--skip-slow-verification",
            "--no-figures",
        ],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def test_two_runs_on_the_same_csv_produce_identical_tables(tmp_path):
    """Brief 4: determinism, on the tables. Figures carry a creation date, so the claim
    is about the tables the figures are drawn from."""
    first, second = tmp_path / "one", tmp_path / "two"
    result_one, result_two = _run_pipeline(first), _run_pipeline(second)
    assert result_one.returncode == 0, result_one.stderr[-3000:]
    assert result_two.returncode == 0, result_two.stderr[-3000:]
    names = sorted(p.name for p in (first / "tables").glob("*.csv"))
    assert names, "no tables were written"
    for name in names:
        assert (first / "tables" / name).read_bytes() == (second / "tables" / name).read_bytes(), (
            f"{name} differs between two runs on the same CSV"
        )


def test_the_end_to_end_run_writes_every_deliverable_of_section_3(tmp_path):
    out_dir = tmp_path / "a8"
    result = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "run_a8_analysis.py"),
            "--results", str(DRY_RUN_CSV), "--out-dir", str(out_dir),
            "--skip-slow-verification",
        ],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    tables = out_dir / "tables"
    for name in (
        "estimands.csv", "arms.csv", "diagnostics.csv", "verdicts.csv",
        "raport_verification.csv", "replication.csv", "00_provenance.csv",
    ):
        assert (tables / name).exists(), f"{name} was not written"
    for name in (
        "fig1a_delta_ab_along_axis.pdf", "fig1b_delta_ae_along_axis.pdf",
        "fig2_arms_per_seed.pdf", "fig5_best_epoch.pdf",
    ):
        figure = out_dir / "figures" / name
        assert figure.exists(), f"{name} was not drawn"
        assert figure.read_bytes()[:4] == b"%PDF", f"{name} is not a vector PDF"
    # The dry run is not the contract grid, so the marker must be there.
    assert (out_dir / "PROVISIONAL.txt").exists()
    assert a8.STATUS_PROVISIONAL in (out_dir / "PROVISIONAL.txt").read_text()
    assert a8.STATUS_PROVISIONAL in result.stdout


def test_the_sigma_power_figure_draws_from_the_tables_and_guards_their_agreement(tmp_path):
    """plot_sigma_power.py is a separate script that consumes A8's tables. It must draw
    from a fresh pair of them, and must refuse a pair that disagrees."""
    out_dir = tmp_path / "a8"
    assert _run_pipeline(out_dir).returncode == 0
    tables = out_dir / "tables"
    figure = out_dir / "figures" / "fig_sigma_power.pdf"

    def draw():
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "plot_sigma_power.py"),
             "--tables", str(tables)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )

    result = draw()
    assert result.returncode == 0, result.stderr[-3000:]
    assert figure.exists(), "fig_sigma_power.pdf was not drawn"
    assert figure.read_bytes()[:4] == b"%PDF", "fig_sigma_power.pdf is not a vector PDF"

    # A sigma that no longer matches the seeds-needed column A8 wrote must stop the run:
    # the guard exists so a stale table pair is never drawn as if it were one run.
    import pandas as pd

    estimands = pd.read_csv(tables / "estimands.csv")
    estimands.loc[0, "sigma_delta"] = float(estimands.loc[0, "sigma_delta"]) * 3.0
    estimands.to_csv(tables / "estimands.csv", index=False)
    result = draw()
    assert result.returncode != 0, "a disagreeing pair of tables was drawn anyway"
    assert "not from one run" in (result.stdout + result.stderr)


def test_the_verification_columns_are_the_ones_the_brief_names(tmp_path):
    import pandas as pd

    out_dir = tmp_path / "a8"
    assert _run_pipeline(out_dir).returncode == 0
    frame = pd.read_csv(out_dir / "tables" / "raport_verification.csv")
    # Column names are English; what is pinned here is their semantics.
    assert frame.columns.tolist() == [
        "quantity", "in_raport", "recomputed", "category", "agrees", "recipe"
    ]
    frame_replication = pd.read_csv(out_dir / "tables" / "replication.csv")
    assert "pooled" in " ".join(frame_replication["level"].astype(str))
    assert set(frame_replication["estimand"]) <= {"delta_AB", "delta_AE"}


def test_every_provisional_table_says_so_in_a_status_column(tmp_path):
    import pandas as pd

    out_dir = tmp_path / "a8"
    assert _run_pipeline(out_dir).returncode == 0
    for name in ("estimands.csv", "arms.csv", "diagnostics.csv", "replication.csv"):
        frame = pd.read_csv(out_dir / "tables" / name)
        assert (frame["status"] == a8.STATUS_PROVISIONAL).all(), (
            f"{name} does not carry the provisional stamp on every row"
        )


def test_a_falsified_raport_value_makes_the_whole_run_exit_nonzero(tmp_path, monkeypatch):
    """The STOP has to be visible to whatever called the script, not only in the log."""
    monkeypatch.setitem(
        a8.RAPORT_DETERMINISTIC["binomial_se_1200"], "reported", 0.5)
    exit_code = a8.run(
        results_csv=DRY_RUN_CSV,
        out_dir=tmp_path,
        predictions_dir=None,
        skip_slow=True,
        no_figures=True,
    )
    assert exit_code == 3


# =====================================================================================
# The verdict table, declared before the analysis
# =====================================================================================


def test_the_architecture_stop_is_judged_against_the_mde_of_its_own_contrast():
    """The bug the dry run caught: sigma_Delta depends on the
    contrast, so the acc(A)~acc(E) row must be read against MDE(A-E), never MDE(A-B)."""
    source = (REPO / "scripts" / "run_a8_analysis.py").read_text()
    block = source.split("acc(A) ~ acc(E) while acc(B) < acc(E)")[0][-1200:]
    assert 'delta_ae["mde"]' in block
    assert 'delta_ab["mde"]' not in block.split("delta_ae = cell")[-1]


def test_the_below_mde_verdict_never_says_there_is_no_effect():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    diag = a8.diagnostics(rows, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    verdict_rows = a8.verdicts(analysis, diag, [], prov)
    below = [v for v in verdict_rows if "below MDE" in v["row"]]
    assert below, "the fixture has n = 2, so something must land below MDE"
    for verdict in below:
        assert "undecidable" in verdict["verdict"]
        assert "Do NOT write 'there is no effect'" in verdict["detail"]
        assert "Seeds needed" in verdict["detail"]


def test_delta_bd_verdicts_forbid_the_sentence_about_quantumness_being_useless():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    diag = a8.diagnostics(rows, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    verdict_rows = a8.verdicts(analysis, diag, [], prov)
    matched = [v for v in verdict_rows if "Delta_BD_matched ~ 0" in v["row"]]
    assert matched
    assert "NOT 'quantumness is useless'" in matched[0]["detail"]
    assert "exploratory" in matched[0]["verdict"]


def test_the_estimands_of_arm_d_are_flagged_exploratory():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    for cell in analysis["cells"].values():
        for name, point in cell["estimands"].items():
            assert point["exploratory"] == name.startswith("delta_BD")


def test_a_dilution_curve_is_never_called_a_test_of_h1():
    """Synthetic two-point axis: the verdict must say 'descriptive' and refuse to confirm
    H1.
    """
    rows = a8.load_rows(DRY_RUN_CSV)
    # Copy the linear rows onto a second axis point so a curve exists at all.
    h2_rows = []
    for row in rows:
        clone = dict(row)
        if clone["dilution"] != "linear" or clone["arm"] == "D_best":
            continue
        clone["dilution"] = "h2"
        clone["accuracy"] = str(float(clone["accuracy"]) * 0.99)
        clone["accuracy_float"] = float(clone["accuracy"])
        h2_rows.append(clone)
    combined = rows + h2_rows
    prov = a8.provenance(combined, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        combined, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    diag = a8.diagnostics(combined, analysis, {"gates": {}, "datasets": {}, "structural": {}})
    verdict_rows = a8.verdicts(analysis, diag, [], prov)
    # "P1 curve" is the prefix run_a8_analysis.py emits; assertion and emitter have to be
    # renamed in one step.
    curves = [v for v in verdict_rows if v["row"].startswith("P1 curve")]
    assert curves, "a two-point axis must produce a dilution-curve row"
    for verdict in curves:
        assert "descriptive" in verdict["verdict"]
        assert "does NOT 'confirm H1'" in verdict["detail"]


def test_the_three_uncertainty_accounts_are_never_summed():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    accounts = analysis["uncertainty_accounts"]
    assert accounts["2_binomial_se"] == pytest.approx(math.sqrt(0.25 / 1200))
    assert "Never summed" in accounts["rule"]
    source = (REPO / "scripts" / "run_a8_analysis.py").read_text()
    assert "total_uncertainty" not in source


def test_mcnemar_is_marked_mandatory_for_delta_ae_only():
    rows = a8.load_rows(DRY_RUN_CSV)
    prov = a8.provenance(rows, results_csv=DRY_RUN_CSV)
    analysis = a8.analyse(
        rows, predictions_dir=DRY_RUN_CSV.parent / "predictions",
        context={"datasets": {}, "gates": {}, "structural": {}}, prov=prov,
    )
    assert analysis["mcnemar"], "the fixture ships prediction files, so McNemar must run"
    for (_, _, name), record in analysis["mcnemar"].items():
        assert record["mandatory"] == (name == "delta_AE")
        if record["mandatory"]:
            assert "MANDATORY for Delta_AE" in record["mandatory_note"]


def test_the_predictions_directory_is_used_as_given_whatever_it_is_called(tmp_path):
    """--predictions-dir is documented as a free path, so it has to behave like one.

    The reader used to take the directory's PARENT and re-append "predictions" to it, so
    any directory not literally named that resolved to nothing and every vector came back
    missing — a silently empty third uncertainty account.
    """
    rows = a8.load_rows(DRY_RUN_CSV)
    original = DRY_RUN_CSV.parent / a7.PREDICTIONS_DIR
    found = a8.correctness_vectors(rows, original)
    assert found["vectors"], "the fixture must ship with prediction files"

    renamed = tmp_path / "somewhere_else"
    shutil.copytree(original, renamed)
    assert a8.correctness_vectors(rows, renamed)["vectors"].keys() == found["vectors"].keys()
    assert a8.correctness_vectors(rows, renamed)["missing"] == []


def test_mcnemar_reads_the_prediction_files_the_series_actually_wrote():
    rows = a8.load_rows(DRY_RUN_CSV)
    predictions = a8.correctness_vectors(rows, DRY_RUN_CSV.parent / "predictions")
    assert predictions["missing"] == [], predictions["missing"][:3]
    assert len(predictions["vectors"]) == 24
    for vector in predictions["vectors"].values():
        assert vector.dtype == bool
        assert vector.size == 1200
