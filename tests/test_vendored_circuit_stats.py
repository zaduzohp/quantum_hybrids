"""Behaviour-pinning tests for the vendored circuit_stats module.

The copy lost the upstream test suite, so its behaviour is pinned here against
hand-computed values on a circuit whose gate inventory is known by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from qsocket.ansatzes import ansatz_L1, ansatz_L2, ansatz_product
from qsocket.vendored.circuit_stats import count_gate_types

VENDORED_FILE = Path(__file__).resolve().parents[1] / "src" / "qsocket" / "vendored" / "circuit_stats.py"
UPSTREAM_COMMIT = "8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9"


def test_counts_on_a_hand_built_circuit():
    qc = QuantumCircuit(3)
    qc.rx(0.1, 0)
    qc.rz(0.2, 0)
    qc.ry(0.3, 2)
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.barrier()
    qc.measure_all()

    stats = count_gate_types(qc)
    assert stats["Total Gates"] == 5
    assert stats["Single-Qubit Gates"] == 3
    assert stats["Two-Qubit Gates"] == 2
    assert stats["Single-Qubit Gates by Qubit"] == {0: 2, 1: 0, 2: 1}
    assert stats["Two-Qubit Gates by Pair"] == {(0, 1): 1, (0, 2): 0, (1, 2): 1}


def test_barriers_and_measurements_are_excluded():
    qc = QuantumCircuit(2)
    qc.barrier()
    qc.measure_all()
    stats = count_gate_types(qc)
    assert stats["Total Gates"] == 0
    assert stats["Two-Qubit Gates"] == 0


def test_pair_keys_are_sorted_and_cover_all_pairs():
    stats = count_gate_types(QuantumCircuit(4))
    assert set(stats["Two-Qubit Gates by Pair"]) == {
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    }


@pytest.mark.parametrize(
    ("builder", "R", "expected_single", "expected_two"),
    [
        (ansatz_L1, 1, 20, 4),
        (ansatz_L2, 1, 20, 4),
        (ansatz_product, 1, 20, 0),
        (ansatz_L1, 3, 50, 12),
        (ansatz_L2, 3, 50, 12),
        (ansatz_product, 3, 50, 0),
    ],
)
def test_counts_agree_with_the_ansatz_specification(builder, R, expected_single, expected_two):
    """Every parameter of these circuits sits on its own single-qubit rotation, so the
    single-qubit gate count must equal the parameter count."""
    stats = count_gate_types(builder(5, R))
    assert stats["Single-Qubit Gates"] == expected_single
    assert stats["Two-Qubit Gates"] == expected_two
    assert stats["Total Gates"] == expected_single + expected_two


def test_vendored_file_carries_provenance_header():
    head = "\n".join(VENDORED_FILE.read_text(encoding="utf-8").splitlines()[:12])
    assert "VENDORED" in head
    assert UPSTREAM_COMMIT in head
    assert "Changes vs upstream" in head
