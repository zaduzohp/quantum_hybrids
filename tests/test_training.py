"""Training loop: pairing, no leak, determinism, frozen vs trained socket.

The quantum arms are expensive on the simulator, so the quantum cases here run on
deliberately tiny data with a two-epoch budget. They test the mechanics of the loop.
The learning-capability check is test_sanity_toy in test_training_sanity.py, which runs
the full contract configuration and is slow by nature.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from qsocket.core import derive
from qsocket.head import make_head
from qsocket.socket import make_socket
from qsocket.training import (
    TrainConfig,
    TrainResult,
    batch_order_rng,
    macro_f1,
    ridge_accuracy,
    ridge_weights,
    select_lr,
    to_binary_labels,
    train_model,
)


def toy_data(n=120, seed=0):
    """Linearly separable, features inside FEATURE_RANGE."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi / 4, np.pi / 4, (n, 5))
    w = np.array([1.0, -0.8, 0.6, 0.3, -0.5])
    y = np.where(X @ w > 0, 1, -1)
    split = int(0.75 * n)
    return X[:split], y[:split], X[split:], y[split:]


def tiny_data(n=24, seed=0):
    return toy_data(n=n, seed=seed)


FAST_CFG = TrainConfig(lr=1e-2, batch_size=8, max_epochs=2, patience=30)


# --- labels and metrics -------------------------------------------------------------


def test_labels_are_mapped_from_pm1_to_01():
    mapped = to_binary_labels(np.array([-1, 1, 1, -1]))
    assert torch.equal(mapped, torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_labels_already_in_01_pass_through():
    assert torch.equal(to_binary_labels(np.array([0, 1, 1])), torch.tensor([0.0, 1.0, 1.0]))


def test_unexpected_labels_are_rejected():
    with pytest.raises(ValueError):
        to_binary_labels(np.array([0, 2]))


def test_the_two_codings_may_not_be_mixed():
    """{-1, 0, 1} passed the old per-value check and mapped -1 and 0 onto one class, so a
    three-class vector trained silently as a binary one."""
    with pytest.raises(ValueError, match="entirely"):
        to_binary_labels(np.array([-1, 0, 1]))


def test_macro_f1_is_the_unweighted_mean_of_both_classes():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert macro_f1(y, y) == pytest.approx(1.0)
    assert macro_f1(y, np.zeros(4)) == pytest.approx(1.0 / 3.0)


# --- pairing ------------------------------------------------------------------------


def test_batch_order_key_is_the_seed_alone():
    """derive(seed), no arm, no R, no dilution."""
    assert batch_order_rng(4).permutation(50).tolist() == (
        np.random.default_rng(derive(4)).permutation(50).tolist()
    )


def test_batch_order_is_identical_across_arms():
    """Compared on the orders that actually ran, not on a re-derivation of the recipe."""
    X_tr, y_tr, X_val, y_val = tiny_data()
    orders = {}
    for name, socket, dilution in [
        ("E", make_socket("identity", R=None, ansatz="", trainable=False, seed=1), "linear"),
        ("B", make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=1), "mlp42"),
    ]:
        log: list[np.ndarray] = []
        train_model(
            socket,
            make_head(dilution, seed=1),
            X_tr,
            y_tr,
            X_val,
            y_val,
            cfg=FAST_CFG,
            seed=3,
            order_log=log,
        )
        orders[name] = [o.tolist() for o in log]
    assert orders["E"] == orders["B"]


def test_batch_order_changes_with_the_seed():
    X_tr, y_tr, X_val, y_val = tiny_data()
    logs = []
    for seed in (1, 2):
        log: list[np.ndarray] = []
        train_model(
            make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
            make_head("linear", seed=seed),
            X_tr,
            y_tr,
            X_val,
            y_val,
            cfg=FAST_CFG,
            seed=seed,
            order_log=log,
        )
        logs.append([o.tolist() for o in log])
    assert logs[0] != logs[1]


# --- frozen vs trained socket -------------------------------------------------------


def test_frozen_socket_theta_is_unchanged_and_displacement_is_zero():
    X_tr, y_tr, X_val, y_val = tiny_data()
    socket = make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=2)
    theta_init = socket.theta_init.clone()
    result = train_model(
        socket, make_head("linear", seed=2), X_tr, y_tr, X_val, y_val, cfg=FAST_CFG, seed=2
    )
    assert torch.equal(socket.theta(), theta_init)
    assert result.theta_displacement == 0.0
    assert result.socket_convergence_epoch is None


def test_trained_socket_theta_moves_and_displacement_is_positive():
    X_tr, y_tr, X_val, y_val = tiny_data()
    socket = make_socket("quantum", R=1, ansatz="L1", trainable=True, seed=2)
    theta_init = socket.theta_init.clone()
    result = train_model(
        socket, make_head("linear", seed=2), X_tr, y_tr, X_val, y_val, cfg=FAST_CFG, seed=2
    )
    assert not torch.equal(socket.theta(), theta_init)
    assert result.theta_displacement > 0.0


def test_training_a_frozen_socket_still_trains_the_head():
    X_tr, y_tr, X_val, y_val = tiny_data()
    head = make_head("linear", seed=2)
    before = next(head.parameters()).detach().clone()
    train_model(
        make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=2),
        head,
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=FAST_CFG,
        seed=2,
    )
    assert not torch.equal(next(head.parameters()).detach(), before)


def test_everything_frozen_is_an_error_not_a_silent_noop():
    X_tr, y_tr, X_val, y_val = tiny_data()
    head = make_head("linear", seed=1)
    head.requires_grad_(False)
    with pytest.raises(ValueError):
        train_model(
            make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=1),
            head,
            X_tr,
            y_tr,
            X_val,
            y_val,
            cfg=FAST_CFG,
            seed=1,
        )


# --- no leak ------------------------------------------------------------------------


def test_train_model_cannot_see_a_test_set():
    """Structural, not behavioural: the signature has no place to put one."""
    parameters = set(inspect.signature(train_model).parameters)
    assert not {"X_test", "y_test", "test", "X_te", "y_te"} & parameters
    assert {"X_val", "y_val"} <= parameters


def test_early_stopping_reads_only_the_validation_set():
    """Changing the validation set changes best_epoch; the training set is untouched."""
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    common = dict(cfg=TrainConfig(lr=3e-2, batch_size=8, max_epochs=25, patience=5), seed=1)
    first = train_model(
        make_socket("identity", R=None, ansatz="", trainable=False, seed=1),
        make_head("mlp42", seed=1),
        X_tr,
        y_tr,
        X_val,
        y_val,
        **common,
    )
    flipped = train_model(
        make_socket("identity", R=None, ansatz="", trainable=False, seed=1),
        make_head("mlp42", seed=1),
        X_tr,
        y_tr,
        X_val,
        -y_val,
        **common,
    )
    assert (first.best_epoch, first.val_accuracy) != (flipped.best_epoch, flipped.val_accuracy)


# --- determinism --------------------------------------------------------------------


def _run_identity(seed=1, dilution="mlp42"):
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    return train_model(
        make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
        make_head(dilution, seed=seed),
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=TrainConfig(lr=1e-2, batch_size=8, max_epochs=30, patience=10),
        seed=seed,
    )


def test_two_runs_with_the_same_seed_agree_exactly():
    first, second = _run_identity(), _run_identity()
    assert first.val_accuracy == second.val_accuracy
    assert first.best_epoch == second.best_epoch
    assert first.train_accuracy == second.train_accuracy


def test_quantum_arm_is_deterministic_too():
    X_tr, y_tr, X_val, y_val = tiny_data()
    results = []
    for _ in range(2):
        results.append(
            train_model(
                make_socket("quantum", R=1, ansatz="L1", trainable=True, seed=5),
                make_head("linear", seed=5),
                X_tr,
                y_tr,
                X_val,
                y_val,
                cfg=FAST_CFG,
                seed=5,
            )
        )
    assert results[0].val_accuracy == results[1].val_accuracy
    assert results[0].best_epoch == results[1].best_epoch
    assert results[0].theta_displacement == pytest.approx(results[1].theta_displacement)


def test_result_carries_every_contract_field():
    result = _run_identity()
    assert isinstance(result, TrainResult)
    for field in (
        "best_epoch",
        "epochs_run",
        "train_accuracy",
        "val_accuracy",
        "val_macro_f1",
        "theta_displacement",
        "grad_rms_start",
        "grad_rms_end",
        "socket_convergence_epoch",
        "wall_seconds",
    ):
        assert hasattr(result, field)
    assert result.grad_rms_start > 0.0
    assert result.wall_seconds > 0.0


def test_best_weights_are_restored():
    """The reported accuracy is the best epoch's, not the last epoch's."""
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    log: list[np.ndarray] = []
    socket = make_socket("identity", R=None, ansatz="", trainable=False, seed=1)
    head = make_head("mlp42", seed=1)
    result = train_model(
        socket,
        head,
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=TrainConfig(lr=3e-2, batch_size=8, max_epochs=40, patience=5),
        seed=1,
        order_log=log,
    )
    with torch.no_grad():
        logits = head(socket(torch.tensor(X_val, dtype=torch.float32))).reshape(-1)
    restored = float(((logits > 0).float() == to_binary_labels(y_val)).float().mean())
    assert restored == pytest.approx(result.val_accuracy)
    assert result.epochs_run == len(log)


def test_socket_weights_take_part_in_the_state_dict_round_trip():
    """Restoring the best weights has to cover the socket too, not just the head —
    TorchConnector's parameter has to be reachable through state_dict for that."""
    from torch import nn

    socket = make_socket("quantum", R=1, ansatz="L1", trainable=True, seed=1)
    model = nn.Sequential(socket, make_head("linear", seed=1))
    keys = list(model.state_dict())
    assert any("quantum_layer" in k for k in keys)

    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        socket.quantum_layer.weight += 1.0
    assert not torch.equal(socket.theta(), snapshot["0.quantum_layer.weight"])
    model.load_state_dict(snapshot)
    assert torch.equal(socket.theta(), snapshot["0.quantum_layer.weight"])


def test_patience_stops_before_the_epoch_budget():
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    result = train_model(
        make_socket("identity", R=None, ansatz="", trainable=False, seed=1),
        make_head("linear", seed=1),
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=TrainConfig(lr=3e-2, batch_size=8, max_epochs=300, patience=3),
        seed=1,
    )
    assert result.epochs_run < 300
    assert result.epochs_run - result.best_epoch >= 3


# --- select_lr ----------------------------------------------------------------------


def test_select_lr_returns_a_value_from_the_grid_and_averages_over_arms():
    X_tr, y_tr, X_val, y_val = toy_data(n=80)

    def build_arms(seed):
        return {
            "E": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("linear", seed=seed),
            ),
            "E2": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("mlp42", seed=seed),
            ),
        }

    grid = (1e-3, 3e-2)
    best, table = select_lr(
        build_arms,
        X_tr,
        y_tr,
        X_val,
        y_val,
        grid=grid,
        seeds=(1, 2),
        cfg=TrainConfig(lr=1e-3, batch_size=8, max_epochs=12, patience=4),
        return_table=True,
    )
    assert best in grid
    assert set(table) == set(grid)
    assert best == max(table, key=lambda lr: (table[lr], -lr))


def test_select_lr_is_deterministic():
    X_tr, y_tr, X_val, y_val = toy_data(n=80)

    def build_arms(seed):
        return {
            "E": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("linear", seed=seed),
            )
        }

    kwargs = dict(
        grid=(1e-3, 1e-2),
        seeds=(1,),
        cfg=TrainConfig(lr=1e-3, batch_size=8, max_epochs=10, patience=4),
    )
    first = select_lr(build_arms, X_tr, y_tr, X_val, y_val, **kwargs)
    second = select_lr(build_arms, X_tr, y_tr, X_val, y_val, **kwargs)
    assert first == second


# --- ridge readout ------------------------------------------------------------------


def _ridge_val_accuracy(X_tr, y_tr, X_val, y_val, *, alpha: float) -> float:
    """The closed-form readout through the two functions PRODUCTION uses.

    ridge_readout was a third implementation of exactly this composition, reachable only
    from these tests; it is gone, and the properties it carried are asserted here on the
    code the driver actually runs.
    """
    return ridge_accuracy(ridge_weights(X_tr, y_tr, alpha=alpha), X_val, y_val)


def test_ridge_alpha_has_no_default():
    """alpha selects the model, so nobody may get one by accident: the grid is declared
    (CONTRACT_RIDGE_ALPHA_GRID) and the choice is made on validation, never defaulted."""
    parameter = inspect.signature(ridge_weights).parameters["alpha"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_ridge_readout_separates_a_separable_toy_set():
    """Plumbing check on the closed form, not a claim about ridge as a classifier: it
    fits squared error rather than a margin, so it sits below a logistic fit even on a
    perfectly separable set (measured 0.92 here on 50 validation samples)."""
    X_tr, y_tr, X_val, y_val = toy_data(n=200)
    assert _ridge_val_accuracy(X_tr, y_tr, X_val, y_val, alpha=1e-3) > 0.85


def test_ridge_readout_is_at_chance_on_random_labels():
    X_tr, y_tr, X_val, y_val = toy_data(n=200)
    rng = np.random.default_rng(3)
    scrambled = rng.choice([-1, 1], len(y_tr))
    assert _ridge_val_accuracy(X_tr, scrambled, X_val, y_val, alpha=1e-3) < 0.7


@pytest.mark.xfail(
    reason=(
        "The binding ridge control needs the alpha chosen by the project owner; alpha is "
        "an open decision. Reported, not guessed."
    ),
    strict=False,
)
def test_ridge_matches_the_adam_linear_head_at_the_contract_alpha():
    from qsocket.training import (
        CONTRACT_RIDGE_ALPHA,  # not defined until alpha is decided
    )

    X_tr, y_tr, X_val, y_val = toy_data(n=200)
    socket = make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=1)
    with torch.no_grad():
        features_tr = socket(torch.tensor(X_tr, dtype=torch.float32)).numpy()
        features_val = socket(torch.tensor(X_val, dtype=torch.float32)).numpy()
    adam = train_model(
        socket,
        make_head("linear", seed=1),
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=TrainConfig(lr=1e-2),
        seed=1,
    )
    ridge = _ridge_val_accuracy(
        features_tr, y_tr, features_val, y_val, alpha=CONTRACT_RIDGE_ALPHA
    )
    assert abs(ridge - adam.val_accuracy) <= 0.01


# --- select_lr: the lr x arm table and per-ansatz selection -------------------------


def _two_arm_builder():
    def build_arms(seed):
        return {
            "A": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("linear", seed=seed),
            ),
            "B": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("h2", seed=seed),
            ),
            "PASSENGER": (
                make_socket("identity", R=None, ansatz="", trainable=False, seed=seed),
                make_head("h4", seed=seed),
            ),
        }

    return build_arms


def _selection_kwargs(grid=(1e-3, 3e-2), seeds=(1, 2)):
    return dict(
        grid=grid,
        seeds=seeds,
        cfg=TrainConfig(lr=1e-3, batch_size=8, max_epochs=8, patience=3),
    )


def test_select_lr_detail_keeps_every_number_it_measured():
    """The point of keeping the whole table: per-arm accuracies are kept, not averaged away.
    Without them, adding an arm to the lr choice means repeating the whole series."""
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    grid, seeds = (1e-3, 3e-2), (1, 2)

    best, detail = select_lr(
        _two_arm_builder(),
        X_tr,
        y_tr,
        X_val,
        y_val,
        selection_arms=("A", "B"),
        return_detail=True,
        **_selection_kwargs(grid, seeds),
    )

    assert best in grid
    assert detail.best == best
    assert detail.arms == ("A", "B", "PASSENGER")
    assert detail.selection_arms == ("A", "B")
    assert set(detail.by_lr_arm_seed) == {
        (lr, arm, seed) for lr in grid for arm in detail.arms for seed in seeds
    }
    assert set(detail.by_lr_arm) == {(lr, arm) for lr in grid for arm in detail.arms}
    # The passenger is measured and reported but never enters the criterion.
    for lr in grid:
        expected = np.mean(
            [detail.by_lr_arm_seed[(lr, arm, seed)] for arm in ("A", "B") for seed in seeds]
        )
        assert detail.mean_by_lr[lr] == pytest.approx(expected)
    assert len(detail.rows()) == len(grid) * len(detail.arms) * len(seeds)
    assert {row["in_selection"] for row in detail.rows()} == {True, False}


def test_select_lr_averages_only_over_the_selection_arms():
    """The criterion is the mean over arms A and B: a passenger that is terrible at one lr
    must not move the choice.
    """
    X_tr, y_tr, X_val, y_val = toy_data(n=80)
    kwargs = _selection_kwargs()

    with_passenger = select_lr(
        _two_arm_builder(), X_tr, y_tr, X_val, y_val,
        selection_arms=("A", "B"), return_detail=True, **kwargs,
    )[1]

    def build_ab_only(seed):
        arms = _two_arm_builder()(seed)
        return {"A": arms["A"], "B": arms["B"]}

    without_passenger = select_lr(
        build_ab_only, X_tr, y_tr, X_val, y_val, return_detail=True, **kwargs
    )[1]

    assert with_passenger.best == without_passenger.best
    assert with_passenger.mean_by_lr == pytest.approx(without_passenger.mean_by_lr)


def test_select_lr_rejects_an_unknown_selection_arm():
    X_tr, y_tr, X_val, y_val = toy_data(n=40)
    with pytest.raises(ValueError, match="selection_arms"):
        select_lr(
            _two_arm_builder(), X_tr, y_tr, X_val, y_val,
            selection_arms=("A", "C"), **_selection_kwargs(grid=(1e-3,), seeds=(1,)),
        )


def test_select_lr_refuses_both_payloads_at_once():
    X_tr, y_tr, X_val, y_val = toy_data(n=40)
    with pytest.raises(ValueError):
        select_lr(
            _two_arm_builder(), X_tr, y_tr, X_val, y_val,
            return_table=True, return_detail=True,
            **_selection_kwargs(grid=(1e-3,), seeds=(1,)),
        )

