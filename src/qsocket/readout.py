"""Readout from a measurement histogram."""

from __future__ import annotations

import numpy as np

from qsocket.vendored.counts import bitstring_qubit_value, qubit_expectation_from_counts


def expectations_from_counts(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """<Z_i> for i = 0..n_qubits-1, every one of them from the same histogram.

    Each qubit index is passed through to the vendored per-qubit estimator, unlike
    qbanknote/iqm.py:79 which ignores the requested observable and always returns <Z_0>.
    A bitstring whose length differs from n_qubits raises rather than being trimmed.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    return np.array(
        [qubit_expectation_from_counts(counts, i, n_qubits) for i in range(n_qubits)],
        dtype=float,
    )


def zz_expectation_from_counts(
    counts: dict[str, int], i: int, j: int, n_qubits: int
) -> float:
    """<Z_i Z_j> from the same histogram."""
    if not 0 <= i < n_qubits or not 0 <= j < n_qubits:
        raise ValueError(f"qubit indices {i}, {j} outside range for n_qubits={n_qubits}")
    if i == j:
        raise ValueError(f"zz_expectation_from_counts needs two distinct qubits, got {i} twice")

    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        bit_i = bitstring_qubit_value(bitstring, i, n_qubits)
        bit_j = bitstring_qubit_value(bitstring, j, n_qubits)
        eigenvalue = (1.0 if bit_i == "0" else -1.0) * (1.0 if bit_j == "0" else -1.0)
        expval += eigenvalue * count / shots
    return float(expval)


def connected_correlation(counts: dict[str, int], n_qubits: int) -> float:
    """max over pairs i<j of |<Z_i Z_j> - <Z_i><Z_j>|.

    The connected part, not the raw correlator: <Z_i Z_j> alone is nonzero for a
    product state whenever the single-qubit values are, so it would not distinguish
    an entangling circuit from the product control.
    """
    z = expectations_from_counts(counts, n_qubits)
    worst = 0.0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zz = zz_expectation_from_counts(counts, i, j, n_qubits)
            worst = max(worst, abs(zz - z[i] * z[j]))
    return float(worst)
