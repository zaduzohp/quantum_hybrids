# VENDORED from qbanknote/confidence.py:163 (accuracy_from_z), qbanknote/confidence.py:101
# (interleave_by_class) and qbanknote/classification.py:9,15 (predictions_to_labels,
# evaluate_predictions) — commit 8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9, 2026-08-14.
# Upstream: ~/QC1_Quantum_Banknote_Classifier_on_ODRA_5
# Changes vs upstream: none in the bodies. accuracy_from_z calls evaluate_predictions,
# which calls predictions_to_labels, so both are copied as well — a copy of the top
# function alone would silently depend on the read-only repo. The `zip_longest` import
# moved from the module header of confidence.py into this file's header, and the
# upstream `Condition` alias and unrelated module docstring are not copied.
# Behaviour pinned by tests/test_vendored_metrics_cls.py.

"""VENDOR: accuracy_from_z, interleave_by_class (+ their two helpers).

Source: qbanknote/confidence.py:163 / :101 and qbanknote/classification.py:9 / :15.
Accuracy of sign(z) against labels in {-1, +1} at threshold 0, and the
class-interleaved hardware submission order.

shot_noise_std (qbanknote/confidence.py:66) is NOT copied here: it belongs to the
hardware variance decomposition (part B, task B3) and copying it now would mean a
vendored function with no caller and no reason to be pinned yet.

Threshold note: predictions_to_labels maps with `predictions > 0`, so an exact zero
is class -1. That matches the 0.5 threshold on a sigmoid used everywhere else in
this project (logit 0 -> label 0 -> -1), and it is copied rather than re-derived.
"""

from __future__ import annotations

from itertools import zip_longest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def predictions_to_labels(predictions: np.ndarray) -> np.ndarray:
    """Map continuous QNN outputs to labels in ``{-1, 1}``."""
    predictions = np.asarray(predictions).reshape(-1)
    return np.where(predictions > 0, 1, -1).astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return accuracy and F1 for binary labels in ``{-1, 1}``."""
    labels = predictions_to_labels(predictions)
    y_true = np.asarray(y_true).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "f1": float(f1_score(y_true, labels, pos_label=1)),
    }


def accuracy_from_z(z: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of ``sign(z)`` against labels in ``{-1, +1}`` (threshold 0)."""
    return evaluate_predictions(np.asarray(y), np.asarray(z))["accuracy"]


def interleave_by_class(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permutation that alternates the two classes in submission order.

    Slow QPU drift over a long submission would otherwise correlate with class
    if the test set were class-sorted. Submit ``X[order]`` to the hardware, then
    restore original sample order with ``z_submitted[inverse]``.

    Returns ``(order, inverse)`` where ``order[p]`` is the original index of the
    p-th submitted sample and ``inverse`` is its argsort.
    """
    y = np.asarray(y)
    idx_neg = np.where(y < 0)[0]
    idx_pos = np.where(y > 0)[0]
    order: list[int] = []
    for a, b in zip_longest(idx_neg.tolist(), idx_pos.tolist()):
        if a is not None:
            order.append(a)
        if b is not None:
            order.append(b)
    order_arr = np.asarray(order, dtype=int)
    inverse = np.argsort(order_arr)
    return order_arr, inverse
