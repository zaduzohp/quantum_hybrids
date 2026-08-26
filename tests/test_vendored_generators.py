"""Behaviour-pinning tests for the vendored generators.

A vendored copy loses the upstream tests, so its behaviour has to be pinned here.

The digests below were produced in parallel by the upstream qml_benchmarks package
(commit 95e5a07e8e9e75ba7e24e67fb32b030112a1309a, installed in a separate venv) and by the
copy in src/qsocket/vendored/ — 16 arrays, 399,000 numbers, zero differences.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qsocket.vendored.hidden_manifold import generate_hidden_manifold_model
from qsocket.vendored.two_curves import generate_two_curves

VENDORED_DIR = Path(__file__).resolve().parents[1] / "src" / "qsocket" / "vendored"
UPSTREAM_COMMIT = "95e5a07e8e9e75ba7e24e67fb32b030112a1309a"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# (seed, n_samples, n_features, degree, offset, noise) -> (sha X, sha y)
TWO_CURVES_CASES = [
    (
        (1234, 6000, 10, 2, 0.25, 0.01),
        "852addb157b794272188ea62f102e7e6a9a03c0e52eea0b293555c71560870b0",
        "1c3826281d27906f82c0f396ff9fcd82d955c09198c917f166fe2b77c5cc872c",
    ),
    (
        (1237, 300, 4, 3, 0.10, 0.00),
        "c61e65ba7de713cbcc2ea8e7d55ad93084b00d1bc20e017c297bab92225ca3bb",
        "aad67c5019741a0fe8774ba26ca47f54b5085736fd3727e0d43319c5181a035d",
    ),
]

# (seed, n_samples, n_features, manifold_dimension) -> (sha X, sha y)
HIDDEN_MANIFOLD_CASES = [
    (
        (1234, 6000, 10, 2),
        "426e06d0c086250aa583acdb551563f85d932d925a658885d28341ea0a6c9498",
        "60d9f772e47e78463d3fa359ef88fd6ebf894df093711357a2b28334cbe4e13a",
    ),
    (
        (1235, 6000, 10, 6),
        "e88ebe536ec1f4fde55d684f4ff422ab5f015332d1cfca6049103903dc02291c",
        "fec339f810cc061fd7d83493a7480f011373388b11e98504a18c532285b66d8e",
    ),
    (
        (1237, 300, 4, 3),
        "518468bb32ae18a01c830730a367541bb4244ae1efbc5ecb8a4509247ebe1c7b",
        "d791c04b00cdbfc0a3c3f1840ffa194c1397b472012198398ea949bf54a8dd95",
    ),
]


@pytest.mark.parametrize(("params", "sha_x", "sha_y"), TWO_CURVES_CASES)
def test_two_curves_pinned(params, sha_x, sha_y):
    seed, *args = params
    # The generators take no seed argument, so a global seed is set immediately
    # before a SINGLE call.
    np.random.seed(seed)
    X, y = generate_two_curves(*args)
    assert X.shape == (args[0], args[1])
    assert sha(X) == sha_x
    assert sha(y) == sha_y


@pytest.mark.parametrize(("params", "sha_x", "sha_y"), HIDDEN_MANIFOLD_CASES)
def test_hidden_manifold_pinned(params, sha_x, sha_y):
    seed, *args = params
    np.random.seed(seed)
    X, y = generate_hidden_manifold_model(*args)
    assert X.shape == (args[0], args[1])
    assert sha(X) == sha_x
    assert sha(y) == sha_y


def test_labels_are_balanced_and_pm_one():
    """Datasets must be 50/50 balanced — the 0.5 decision threshold assumes it.
    Upstream labels are -1/+1, not 0/1."""
    np.random.seed(7)
    _, y_tc = generate_two_curves(6000, 10, 5, 0.10, 0.01)
    np.random.seed(7)
    _, y_hm = generate_hidden_manifold_model(6000, 10, 6)
    for y in (y_tc, y_hm):
        assert set(np.unique(y)) == {-1, 1}
        assert (y == 1).sum() == (y == -1).sum() == 3000


FORBIDDEN_IMPORTS = ("jax", "jaxlib", "numpyro", "flax", "optax")


def test_generators_do_not_require_jax_or_numpyro():
    """The whole point of the copy: upstream qml_benchmarks.data.__init__ imports
    ising.py, which requires jax and numpyro. The copy must not need them.
    """
    program = (
        "import sys\n"
        "from qsocket.vendored.two_curves import generate_two_curves\n"
        "from qsocket.vendored.hidden_manifold import generate_hidden_manifold_model\n"
        f"print(','.join(m for m in {FORBIDDEN_IMPORTS!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"pulled in by the copy: {result.stdout.strip()}"


def test_the_jax_stack_is_not_installed_at_all():
    """The copy exists so that jax/numpyro never enter this environment."""
    for forbidden in FORBIDDEN_IMPORTS:
        with pytest.raises(ImportError):
            __import__(forbidden)


@pytest.mark.parametrize("filename", ["two_curves.py", "hidden_manifold.py"])
def test_vendored_files_carry_provenance_header(filename):
    """Every vendored file must state its source, its upstream commit and whether
    it was changed."""
    text = (VENDORED_DIR / filename).read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:12])
    assert "VENDORED" in head
    assert UPSTREAM_COMMIT in head
    assert "Changes vs upstream" in head
