"""Behaviour-pinning tests for vendored/metrics_cls.py.
A vendored copy loses the tests of the original, so its behaviour is pinned here
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qsocket.vendored.metrics_cls import (
    accuracy_from_z,
    evaluate_predictions,
    interleave_by_class,
    predictions_to_labels,
)

VENDORED_DIR = Path(__file__).resolve().parents[1] / "src" / "qsocket" / "vendored"
UPSTREAM_COMMIT = "8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9"


def test_accuracy_from_z_thresholds_at_zero():
    z = np.array([2.0, 0.5, -0.5, -3.0])
    y = np.array([1, 1, -1, -1])
    assert accuracy_from_z(z, y) == 1.0
    assert accuracy_from_z(-z, y) == 0.0
    assert accuracy_from_z(np.array([1.0, -1.0, 1.0, -1.0]), y) == 0.5


def test_exact_zero_counts_as_the_negative_class():
    """The upstream rule is `predictions > 0`, so z = 0 is class -1. Copied, not
    re-derived: it is the same convention as logit 0 -> label 0 in training.py."""
    assert predictions_to_labels(np.array([0.0])).tolist() == [-1.0]
    assert accuracy_from_z(np.array([0.0]), np.array([-1])) == 1.0
    assert accuracy_from_z(np.array([0.0]), np.array([1])) == 0.0


def test_accuracy_from_z_accepts_column_shaped_input():
    z = np.array([[1.0], [-1.0]])
    assert accuracy_from_z(z, np.array([[1], [-1]])) == 1.0


def test_evaluate_predictions_reports_accuracy_and_f1():
    scores = evaluate_predictions(np.array([1, 1, -1, -1]), np.array([1.0, -1.0, -1.0, -1.0]))
    assert scores["accuracy"] == 0.75
    assert scores["f1"] == pytest.approx(2 / 3)


def test_interleave_by_class_alternates_and_inverts():
    y = np.array([-1, -1, -1, 1, 1, 1])
    order, inverse = interleave_by_class(y)
    assert order.tolist() == [0, 3, 1, 4, 2, 5]
    assert y[order].tolist() == [-1, 1, -1, 1, -1, 1]
    # Submitting X[order] and restoring with [inverse] must be the identity.
    np.testing.assert_array_equal(y[order][inverse], y)


def test_interleave_by_class_handles_unequal_classes():
    y = np.array([-1, -1, -1, 1])
    order, inverse = interleave_by_class(y)
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(y[order][inverse], y)


def test_vendored_file_carries_provenance_header():
    text = (VENDORED_DIR / "metrics_cls.py").read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:12])
    assert "VENDORED" in head
    assert UPSTREAM_COMMIT in head
    assert "Changes vs upstream" in head
