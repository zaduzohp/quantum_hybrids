"""Tests for the data gates G1 and G2.

"""

from __future__ import annotations

import numpy as np
import pytest

from qsocket.gates import (
    G1_MIN_HEADROOM,
    G1_STRONG_ACCURACY_BAND,
    G2_MAX_COMPONENT_SHARE,
    ceiling,
    check_g1_headroom,
    check_g2_effective_dim,
    make_arm_e_linear_floor_model,
    make_mlp_strong_model,
    make_svc_strong_model,
)
from qsocket.training import TrainConfig

# --- G2 -----------------------------------------------------------------------------


def test_g2_passes_on_a_spread_spectrum():
    result = check_g2_effective_dim([0.30, 0.25, 0.20, 0.15, 0.10])
    assert result["passed"]
    assert result["top_component"] == 0
    assert result["top_share"] == pytest.approx(0.30)


def test_g2_fails_when_one_component_dominates():
    result = check_g2_effective_dim([0.85, 0.05, 0.04, 0.03, 0.03])
    assert not result["passed"]
    assert result["failures"] and "0.85" in result["failures"][0]


def test_g2_renormalises_over_the_retained_components():
    """explained_variance_ratio_ is a share of the ORIGINAL variance, so five components
    summing to 0.5 must be judged on their shares of that 0.5, not of 1.0."""
    result = check_g2_effective_dim([0.42, 0.02, 0.02, 0.02, 0.02])
    assert result["total_variance_explained"] == pytest.approx(0.5)
    assert result["top_share"] == pytest.approx(0.84)
    assert not result["passed"]
    assert result["max_share"] == G2_MAX_COMPONENT_SHARE


def test_g2_accepts_a_manifest(tmp_path):
    manifest = {
        "frozen_name": "toy",
        "pca": {"explained_variance_ratio_": [0.3, 0.25, 0.2, 0.15, 0.1]},
    }
    result = check_g2_effective_dim(manifest)
    assert result["passed"] and result["dataset"] == "toy"


def test_g2_rejects_unusable_input():
    with pytest.raises(ValueError):
        check_g2_effective_dim([0.0, 0.0])


# --- G1 with stub models ------------------------------------------------------------


def _models(strong, floor, *, contract_floor=True):
    return {
        "strong_model": lambda dataset: {"accuracy": strong},
        "floor_model": lambda dataset: {
            "accuracy": floor,
            "is_contract_arm_e": contract_floor,
        },
    }


def test_g1_passes_with_headroom_and_a_strong_model_in_band():
    result = check_g1_headroom({}, **_models(0.80, 0.70))
    assert result["passed"] and result["binding"]
    assert result["headroom"] == pytest.approx(0.10)


def test_g1_fails_when_there_is_no_headroom():
    result = check_g1_headroom({}, **_models(0.80, 0.78))
    assert not result["passed"]
    assert "headroom" in result["failures"][0]
    assert result["headroom"] < G1_MIN_HEADROOM


@pytest.mark.parametrize("strong", [0.60, 0.95])
def test_g1_fails_outside_the_accuracy_band(strong):
    result = check_g1_headroom({}, **_models(strong, strong - 0.20))
    assert not result["passed"] and not result["strong_in_band"]
    low, high = G1_STRONG_ACCURACY_BAND
    assert any(f"{low:.2f}" in failure for failure in result["failures"])


def test_g1_margin_is_the_signed_distance_to_the_threshold():
    """g1_margin = headroom - 0.05.

    The motivating case is the last one: a headroom printing as 0.050000 while sitting a
    float below the threshold fails the gate, and the margin makes that row readable.
    """
    for strong, floor in ((0.80, 0.70), (0.80, 0.78), (0.95, 0.60)):
        result = check_g1_headroom({}, **_models(strong, floor))
        assert result["g1_margin"] == pytest.approx(result["headroom"] - G1_MIN_HEADROOM)
        # The margin never overrides the verdict; it only mirrors the headroom criterion.
        assert (result["g1_margin"] < 0.0) == (result["headroom"] < G1_MIN_HEADROOM)

    # 0.30 - 0.25 is 0.04999999999999999 in binary floating point: headroom prints as
    # 0.050000, the gate fails, and g1_margin is negative rather than a printed zero.
    edge = check_g1_headroom({}, **_models(0.30, 0.25))
    assert f"{edge['headroom']:.6f}" == "0.050000"
    assert edge["headroom"] < G1_MIN_HEADROOM
    assert edge["g1_margin"] < 0.0
    assert not edge["passed"]


def test_g1_result_keys_are_additive_only():
    """Adding that key was allowed. This pins the full key set so a later edit cannot
    quietly rename or drop one of the keys the analysis pipeline reads."""
    result = check_g1_headroom({}, **_models(0.80, 0.70))
    assert set(result) == {
        "gate",
        "binding",
        "binding_note",
        "min_headroom",
        "strong_accuracy_band",
        "strong",
        "floor",
        "headroom",
        "g1_margin",
        "strong_in_band",
        "failures",
        "passed",
    }


def test_g1_marks_a_non_contract_floor_as_orientational():
    result = check_g1_headroom({}, **_models(0.80, 0.70, contract_floor=False))
    assert result["passed"] and not result["binding"]
    assert "ORIENTATIONAL" in result["binding_note"]


def test_g1_accepts_a_bare_float_and_rejects_a_dict_without_accuracy():
    result = check_g1_headroom(
        {}, strong_model=lambda d: 0.8, floor_model=lambda d: 0.7
    )
    assert result["passed"] and not result["binding"]
    with pytest.raises(ValueError, match="without an 'accuracy' key"):
        check_g1_headroom({}, strong_model=lambda d: {"acc": 0.8}, floor_model=lambda d: 0.7)


# --- G1 with the real models --------------------------------------------------------


def _separable_dataset(n=600, seed=0):
    """A toy set where a linear head is already near-perfect: strong and floor must both
    be high, so the gate has to FAIL on the headroom criterion."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.7, 0.7, size=(n, 5))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, 1, -1)
    cut_a, cut_b = n // 2, n // 2 + n // 4
    return {
        "train": (X[:cut_a], y[:cut_a]),
        "val": (X[cut_a:cut_b], y[cut_a:cut_b]),
        "test": (X[cut_b:], y[cut_b:]),
    }


def _xor_dataset(n=600, seed=1):
    """Nonlinear: an RBF SVM separates it, a linear head cannot."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.7, 0.7, size=(n, 5))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    cut_a, cut_b = n // 2, n // 2 + n // 4
    return {
        "train": (X[:cut_a], y[:cut_a]),
        "val": (X[cut_a:cut_b], y[cut_a:cut_b]),
        "test": (X[cut_b:], y[cut_b:]),
    }


def test_svc_strong_model_selects_on_validation_not_on_test():
    dataset = _xor_dataset()
    result = make_svc_strong_model()(dataset)
    assert len(result["cells"]) == 9
    best_on_test = max(cell["test_accuracy"] for cell in result["cells"])
    selected = max(cell["selection_accuracy"] for cell in result["cells"])
    assert result["selection_accuracy"] == pytest.approx(selected)
    # The reported accuracy is the selected cell's test accuracy, so it can only be at
    # or below the best test accuracy in the grid — never the grid maximum by
    # construction.
    assert result["accuracy"] <= best_on_test + 1e-12
    assert result["selection_split"] == "val"
    assert all(cell["fit_seconds"] > 0 for cell in result["cells"])


def test_svc_strong_model_honours_a_custom_grid():
    result = make_svc_strong_model(grid={"C": (1.0,), "gamma": ("scale",)})(_xor_dataset())
    assert len(result["cells"]) == 1
    assert result["selected"] == {"C": 1.0, "gamma": "scale"}


def test_arm_e_floor_model_is_contract_arm_e_and_binding():
    """The floor really runs arm E through training.train_model, which is what makes the
    G1 verdict binding rather than orientational."""
    floor = make_arm_e_linear_floor_model(
        lr_grid=(3e-3, 3e-2), seeds=(1,), cfg=TrainConfig(lr=0.0, max_epochs=15, patience=15)
    )
    result = floor(_separable_dataset())
    assert result["is_contract_arm_e"] and result["dilution"] == "linear"
    assert result["lr_selected"] in (3e-3, 3e-2)
    assert len(result["runs"]) == 2
    assert 0.0 <= result["accuracy"] <= 1.0

    verdict = check_g1_headroom(
        _separable_dataset(),
        strong_model=make_svc_strong_model(grid={"C": (1.0,), "gamma": ("scale",)}),
        floor_model=floor,
    )
    assert verdict["binding"]
    # A linearly separable toy set is exactly the "too easy" failure G1 exists to catch:
    # arm E already solves it, so there is no headroom for any socket.
    assert not verdict["passed"]
    assert verdict["headroom"] < G1_MIN_HEADROOM


def test_arm_e_floor_selects_lr_on_validation_per_cell():
    floor = make_arm_e_linear_floor_model(
        lr_grid=(1e-3, 3e-2), seeds=(1,), cfg=TrainConfig(lr=0.0, max_epochs=10, patience=10)
    )
    result = floor(_xor_dataset())
    per_lr = result["mean_val_accuracy_per_lr"]
    assert set(per_lr) == {"0.001", "0.03"}
    best = max(float(lr) for lr in per_lr if per_lr[lr] == max(per_lr.values()))
    # Ties resolve to the lower lr, matching training.select_lr.
    assert result["lr_selected"] <= best
    assert result["lr_selection"].startswith("per cell")


# --- the ceiling is a scale, not a gate ---------------------------------------------

_FAST = dict(lr_grid=(3e-2,), seeds=(1,), cfg=TrainConfig(lr=0.0, max_epochs=8, patience=8))


def test_mlp_strong_model_is_the_best_of_its_dilutions_and_marked_ceiling_only():
    result = make_mlp_strong_model(**_FAST)(_xor_dataset())
    assert result["is_ceiling_only"] is True
    assert result["which"] in ("mlp42", "mlp4285")
    assert set(result["per_dilution"]) == {"mlp42", "mlp4285"}
    assert result["accuracy"] == pytest.approx(
        max(r["accuracy"] for r in result["per_dilution"].values())
    )


def test_the_ceiling_model_cannot_gate_g1():
    """acc(strong MLP) sits above the band on perfectly good datasets, so gating on it
    would reject them as "too easy". It has to be refused rather than quietly accepted.
    """
    with pytest.raises(ValueError, match="CEILING reading"):
        check_g1_headroom(
            _xor_dataset(),
            strong_model=make_mlp_strong_model(**_FAST),
            floor_model=make_arm_e_linear_floor_model(**_FAST),
        )


def test_ceiling_takes_the_max_unconditionally_even_when_the_mlp_is_worse():
    """`max` is not 'the better one if it flatters the result': a ceiling that switched
    models depending on the outcome is exactly the selection asymmetry this closes."""
    result = ceiling(
        {},
        svc_model=lambda d: {"accuracy": 0.90, "label": "svm stub"},
        mlp_model=lambda d: {"accuracy": 0.70, "label": "mlp stub", "is_ceiling_only": True},
        floor_model=lambda d: {"accuracy": 0.60, "is_contract_arm_e": True},
    )
    assert result["strong_accuracy"] == pytest.approx(0.90)
    assert result["strong_which"] == "svm"
    assert result["ceiling"] == pytest.approx(0.30)
    assert result["ceiling_vs_mlp_only"] == pytest.approx(0.10)


def test_ceiling_reports_and_never_gates():
    result = ceiling(
        {},
        svc_model=lambda d: {"accuracy": 0.86},
        mlp_model=lambda d: {"accuracy": 0.976, "is_ceiling_only": True},
        floor_model=lambda d: {"accuracy": 0.764, "is_contract_arm_e": True},
    )
    assert result["gating"] is False
    assert "passed" not in result and "failures" not in result
    assert result["strong_which"] == "mlp"
    assert result["ceiling"] == pytest.approx(0.212, abs=1e-3)
    # The reason the split exists: the ceiling numerator is outside the gate's band.
    assert result["strong_in_g1_band"] is False


def test_the_floor_and_the_mlp_ceiling_come_off_one_code_path():
    """They must differ in the head and in nothing else, otherwise the ceiling is not a
    ceiling over the same chain."""
    dataset = _xor_dataset()
    floor = make_arm_e_linear_floor_model(**_FAST)(dataset)
    from qsocket.gates import _arm_e_with_head

    direct = _arm_e_with_head(
        dataset, dilution="linear", lr_grid=(3e-2,), seeds=(1,), cfg=_FAST["cfg"]
    )
    assert floor["accuracy"] == pytest.approx(direct["accuracy"])
    assert floor["lr_selected"] == direct["lr_selected"]
    assert floor["is_contract_arm_e"] and "is_contract_arm_e" not in direct
