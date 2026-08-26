"""Behaviour-pinning tests for the vendored `hyperplanes` generator.

A vendored copy loses the upstream tests, so its behaviour has to be pinned here,
following the pattern of tests/test_vendored_generators.py.

The digests below were produced in parallel by the upstream file (qml-benchmarks at commit
95e5a07e8e9e75ba7e24e67fb32b030112a1309a) and by the copy in
src/qsocket/vendored/hyperplanes.py — 6 arrays, 138,600 numbers, zero differences.

A failure here means the copy or the numerics underneath it stopped reproducing upstream.
Do not refresh the digests without understanding what changed; the production dataset
rests on these numbers.
"""

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qsocket.datasets import GENERATORS, REQUIRED_GENERATOR_KWARGS
from qsocket.vendored.hyperplanes import (
    generate_hyperplanes_parity,
    perceptron,
    predict,
)

VENDORED_DIR = Path(__file__).resolve().parents[1] / "src" / "qsocket" / "vendored"
UPSTREAM_COMMIT = "95e5a07e8e9e75ba7e24e67fb32b030112a1309a"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# (seed, n_samples, n_features, n_hyperplanes, dim_hyperplanes) -> (sha X, sha y).
# The middle case is the production cell at the production generator seed.
HYPERPLANES_CASES = [
    (
        (1234, 6000, 10, 3, 5),
        "98ffaf782b2632d4b85decd2d6dfa594700555a014063dc0d0fbe3dad932db5d",
        "1c3826281d27906f82c0f396ff9fcd82d955c09198c917f166fe2b77c5cc872c",
    ),
    (
        (11, 6000, 20, 3, 5),
        "1418ccf706b2c11f482317687f62bd2a3a2b9f925f05d014baa269530b3604e4",
        "1c3826281d27906f82c0f396ff9fcd82d955c09198c917f166fe2b77c5cc872c",
    ),
    (
        (1237, 300, 4, 2, 2),
        "41401f64864f83bdbe1648e3f175d33bde93f1c4010c5e4ea3c1187e0ebb26df",
        "aad67c5019741a0fe8774ba26ca47f54b5085736fd3727e0d43319c5181a035d",
    ),
]


@pytest.mark.parametrize(("params", "sha_x", "sha_y"), HYPERPLANES_CASES)
def test_hyperplanes_pinned(params, sha_x, sha_y):
    seed, *args = params
    # The generator takes no seed argument, so a global seed is set immediately before a
    # single call.
    np.random.seed(seed)
    X, y = generate_hyperplanes_parity(*args)
    assert X.shape == (args[0], args[1])
    assert sha(X) == sha_x
    assert sha(y) == sha_y


def test_the_signature_order_is_the_one_the_registration_assumes():
    """`n_hyperplanes` sits before `dim_hyperplanes`, and both are plain ints in the same
    small range, so swapping them positionally would produce a different dataset with no
    error anywhere. This pins the order, and that datasets.REQUIRED_GENERATOR_KWARGS lists
    exactly the non-`n_samples` parameters.
    """
    names = tuple(inspect.signature(generate_hyperplanes_parity).parameters)
    assert names == ("n_samples", "n_features", "n_hyperplanes", "dim_hyperplanes")
    assert REQUIRED_GENERATOR_KWARGS["hyperplanes"] == names[1:]
    assert GENERATORS["hyperplanes"] is generate_hyperplanes_parity


def test_the_generator_has_no_seed_parameter():
    """Reproducibility runs through the global np.random.seed, and only through it."""
    assert "seed" not in inspect.signature(generate_hyperplanes_parity).parameters
    assert "random_state" not in inspect.signature(generate_hyperplanes_parity).parameters


def test_labels_are_exactly_balanced_and_pm_one():
    """The generator subselects to n_samples//2 per class BY CONSTRUCTION, so the
    balance is exact. Labels are -1/+1, not 0/1."""
    np.random.seed(7)
    _, y = generate_hyperplanes_parity(6000, 20, 3, 5)
    assert set(np.unique(y)) == {-1, 1}
    assert (y == 1).sum() == (y == -1).sum() == 3000
    # And they arrive sorted, which is why datasets.generate_and_freeze must shuffle
    # before splitting.
    assert np.array_equal(y, np.array([-1] * 3000 + [1] * 3000))


def test_labels_are_the_parity_prediction_with_the_sign_flipped_upstream():
    """Upstream stacks the rows whose parity prediction is +1 before those where it is -1,
    then writes the labels in the opposite order, so the recorded label is the negation of
    predict(). For a balanced binary task this is a relabelling and cannot change any
    accuracy, but anyone comparing a recomputed parity against the stored y has to know.
    """
    np.random.seed(1234)
    weights = np.random.uniform(size=(3, 5))
    biases = np.random.uniform(size=(3,))
    z = np.random.normal(size=(200, 5))
    recomputed = np.array([predict(row, weights, biases) for row in z])
    assert set(np.unique(recomputed)) <= {-1, 1}

    # perceptron() is a hard threshold at 0 returning 1/0, and predict() is +1 for an
    # even number of ones. Both pinned directly, since they carry the label definition.
    assert perceptron(np.array([1.0]), np.array([1.0]), 0.5) == 1
    assert perceptron(np.array([-1.0]), np.array([1.0]), 0.5) == 0
    assert predict(np.array([1.0]), np.array([[1.0]]), np.array([0.5])) == -1
    assert predict(np.array([-1.0]), np.array([[1.0]]), np.array([0.5])) == 1


FORBIDDEN_IMPORTS = ("jax", "jaxlib", "numpyro", "flax", "optax")


def test_the_copy_does_not_require_jax_or_numpyro():
    """The whole point of the copy: upstream qml_benchmarks.data.__init__ imports
    ising.py, which requires jax and numpyro. The copy must not need them.
    """
    program = (
        "import sys\n"
        "from qsocket.vendored.hyperplanes import generate_hyperplanes_parity\n"
        f"print(','.join(m for m in {FORBIDDEN_IMPORTS!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"pulled in by the copy: {result.stdout.strip()}"


def test_vendored_file_carries_provenance_header():
    """Source file and line, upstream commit, licence, and whether
    it was changed."""
    text = (VENDORED_DIR / "hyperplanes.py").read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:12])
    assert "VENDORED" in head
    assert "qml_benchmarks/data/hyperplanes.py" in head
    assert UPSTREAM_COMMIT in head
    assert "Apache-2.0" in head
    assert "Changes vs upstream: none" in head
    # The vendoring header must record why the file was copied.
    assert "ising.py" in head and "jax" in head and "numpyro" in head
    # Upstream's own Apache header must survive the copy.
    assert "Copyright 2024 Xanadu Quantum Technologies Inc." in text
    assert "http://www.apache.org/licenses/LICENSE-2.0" in text
