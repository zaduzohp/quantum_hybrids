"""Pinning tests for the counts helpers vendored from qbanknote/metrics.py.

A vendored copy loses the upstream tests, so these exist to catch a silent drift in the
one convention the whole hardware readout stands on: Qiskit is little-endian, the
rightmost character of a bitstring is qubit 0.
"""

from __future__ import annotations

import pytest

from qsocket.vendored.counts import bitstring_qubit_value, qubit_expectation_from_counts


def test_bit_index_is_little_endian():
    # "10000": and leftmost is qubit 4.
    assert bitstring_qubit_value("10000", 4, 5) == "1"
    assert bitstring_qubit_value("10000", 0, 5) == "0"
    # "00001": rightmost is qubit 0.
    assert bitstring_qubit_value("00001", 0, 5) == "1"
    assert [bitstring_qubit_value("00001", i, 5) for i in range(1, 5)] == ["0"] * 4


def test_bit_index_covers_every_position():
    for i in range(5):
        bits = ["0"] * 5
        bits[5 - 1 - i] = "1"
        bitstring = "".join(bits)
        assert bitstring_qubit_value(bitstring, i, 5) == "1"
        assert sum(bitstring_qubit_value(bitstring, j, 5) == "1" for j in range(5)) == 1


def test_wrong_length_bitstring_raises():
    with pytest.raises(ValueError, match="Expected bitstring length 5"):
        bitstring_qubit_value("0000", 0, 5)


def test_expectation_signs():
    # eigenvalue +1 for bit "0", -1 for bit "1"
    assert qubit_expectation_from_counts({"00000": 100}, 0, 5) == pytest.approx(1.0)
    assert qubit_expectation_from_counts({"00001": 100}, 0, 5) == pytest.approx(-1.0)
    assert qubit_expectation_from_counts({"00001": 100}, 1, 5) == pytest.approx(1.0)


def test_expectation_is_shot_weighted():
    counts = {"00000": 750, "00001": 250}
    assert qubit_expectation_from_counts(counts, 0, 5) == pytest.approx(0.5)
    assert qubit_expectation_from_counts(counts, 3, 5) == pytest.approx(1.0)


def test_empty_histogram_returns_zero():
    assert qubit_expectation_from_counts({}, 0, 5) == 0.0
    assert qubit_expectation_from_counts({"00000": 0}, 0, 5) == 0.0
