"""Sanity on a linearly separable toy set: every arm must actually learn.

Catches a training loop that "passes its tests" while learning nothing. It runs the
contract configuration (Adam, lr from the grid, batch 64, patience 30, best weights
restored) on 200 samples with 5 features.

Arm B's threshold is measured, not aspirational. It is a frozen random circuit under a
linear head, and the ceiling for any linear readout on those frozen features is 0.84 —
checked independently with LDA, a ridge classifier and the project's own `ridge_control`.
That is a construction limit, so raising `THRESHOLDS["B"]` above it would make the test
permanently red no matter what training does.

The lr has to be able to reach that ceiling: at the shared lr = 1e-2 of an earlier version
arm B reached only 0.580, under-trained rather than under-capable. lr = 0.1 reaches each
arm's own ceiling, so a shortfall reported here is about what an arm can represent.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsocket.head import make_head
from qsocket.socket import make_socket
from qsocket.training import TrainConfig, train_model

# lr chosen so EVERY arm is adequately trained on this toy set — see module docstring.
LR = 0.1

# Arm B's ceiling (0.84) is a construction limit — "frozen random circuit -> linear
# head" on this toy set — not a training one; see the module docstring. Arms A and E
# have no such ceiling here (the toy labels are linear in the raw features and E is a
# pass-through), hence the higher bar.
THRESHOLDS = {"A": 0.95, "E": 0.95, "B": 0.80}


def separable_toy(n=200, seed=0):
    """5 features drawn inside FEATURE_RANGE, labels from a linear rule."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi / 4, np.pi / 4, (n, 5))
    w = np.array([1.0, -0.8, 0.6, 0.3, -0.5])
    y = np.where(X @ w > 0, 1, -1)
    split = 150
    return X[:split], y[:split], X[split:], y[split:]


ARMS = {
    "E": lambda: make_socket("identity", R=None, ansatz="", trainable=False, seed=1),
    "B": lambda: make_socket("quantum", R=1, ansatz="L1", trainable=False, seed=1),
    "A": lambda: make_socket("quantum", R=1, ansatz="L1", trainable=True, seed=1),
}


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_every_arm_learns_a_separable_toy_set(arm):
    X_tr, y_tr, X_val, y_val = separable_toy()
    result = train_model(
        ARMS[arm](),
        make_head("linear", seed=1),
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg=TrainConfig(lr=LR),
        seed=1,
    )
    assert result.val_accuracy > THRESHOLDS[arm], (
        f"arm {arm} reached only {result.val_accuracy:.3f} validation accuracy "
        f"(best epoch {result.best_epoch} of {result.epochs_run}, "
        f"threshold {THRESHOLDS[arm]})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("arm,trainable", [("A", True), ("B", False)])
def test_a_whole_training_run_is_backend_independent(arm, trainable):
    """Backend equivalence on a complete training run rather than a single expectation
    value: same seed, same config, only the gradient backend swapped, so val_accuracy and
    best_epoch must come out identical.

    The epoch budget is cut to 20 to keep the Qiskit side inside a few minutes; the
    unabridged run was measured separately and agrees to 7 digits.
    """
    X_tr, y_tr, X_val, y_val = separable_toy()
    results = {}
    for backend in ("qiskit", "pennylane"):
        results[backend] = train_model(
            make_socket(
                "quantum", R=1, ansatz="L1", trainable=trainable, seed=1, backend=backend
            ),
            make_head("linear", seed=1),
            X_tr,
            y_tr,
            X_val,
            y_val,
            cfg=TrainConfig(lr=1e-2, max_epochs=20),
            seed=1,
        )
    assert results["qiskit"].val_accuracy == results["pennylane"].val_accuracy
    assert results["qiskit"].best_epoch == results["pennylane"].best_epoch
    assert results["qiskit"].epochs_run == results["pennylane"].epochs_run
    assert results["pennylane"].wall_seconds < results["qiskit"].wall_seconds
