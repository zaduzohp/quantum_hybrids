# VENDORED from qbanknote/metrics.py:71 and qbanknote/metrics.py:77
# (commit 8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9, 2026-07-19)
# Upstream: ~/QC1_Quantum_Banknote_Classifier_on_ODRA_5
# Changes vs upstream: none — both functions are literal copies, this header and the
# module docstring are the only additions.
# Behaviour pinned by tests/test_vendored_counts.py.
"""VENDOR: bitstring_qubit_value, qubit_expectation_from_counts."""

from __future__ import annotations


def bitstring_qubit_value(bitstring: str, qubit: int, n_qubits: int) -> str:
    if len(bitstring) != n_qubits:
        raise ValueError(f"Expected bitstring length {n_qubits}, got {len(bitstring)!r}")
    return bitstring[-(qubit + 1)]


def qubit_expectation_from_counts(
    counts: dict[str, int], qubit: int, n_qubits: int
) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        bit = bitstring_qubit_value(bitstring, qubit, n_qubits)
        eigenvalue = 1.0 if bit == "0" else -1.0
        expval += eigenvalue * count / shots
    return float(expval)
