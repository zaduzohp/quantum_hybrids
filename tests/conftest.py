"""Shared statevector helpers for the ansatz tests.

Everything here runs on an exact statevector, so the numbers are free of shot noise
and the tolerances in the tests mean what they say.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

N_QUBITS = 5
R_VALUES = (1, 2, 3)


def z_pauli(i: int, n_qubits: int = N_QUBITS) -> Pauli:
    """Pauli Z on qubit i. Qiskit is little-endian, so qubit i sits at position
    n_qubits - 1 - i in the label."""
    label = ["I"] * n_qubits
    label[n_qubits - 1 - i] = "Z"
    return Pauli("".join(label))


def zz_pauli(i: int, j: int, n_qubits: int = N_QUBITS) -> Pauli:
    label = ["I"] * n_qubits
    label[n_qubits - 1 - i] = "Z"
    label[n_qubits - 1 - j] = "Z"
    return Pauli("".join(label))


def bind(circuit: QuantumCircuit, theta: np.ndarray, x: np.ndarray | None = None) -> QuantumCircuit:
    """Bind theta (and x, when the circuit has a feature map) by parameter name."""
    values = {}
    for param in circuit.parameters:
        vector, index = param.name.split("[")
        index = int(index.rstrip("]"))
        if vector == "theta":
            values[param] = float(theta[index])
        elif vector == "x":
            if x is None:
                raise ValueError("circuit has feature-map parameters but no x was given")
            values[param] = float(x[index])
        else:
            raise ValueError(f"unexpected parameter vector {vector!r}")
    return circuit.assign_parameters(values)


def z_expectations(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    x: np.ndarray | None = None,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Exact <Z_i> for i = 0..n_qubits-1 from the statevector."""
    state = Statevector(bind(circuit, theta, x))
    return np.array(
        [np.real(state.expectation_value(z_pauli(i, n_qubits))) for i in range(n_qubits)]
    )


def zz_expectation(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    i: int,
    j: int,
    x: np.ndarray | None = None,
    n_qubits: int = N_QUBITS,
) -> float:
    state = Statevector(bind(circuit, theta, x))
    return float(np.real(state.expectation_value(zz_pauli(i, j, n_qubits))))


def max_connected_correlation(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    x: np.ndarray | None = None,
    n_qubits: int = N_QUBITS,
) -> float:
    """max over pairs i<j of |<Z_i Z_j> - <Z_i><Z_j>|. Zero exactly when the state is
    a product state in the computational basis correlations."""
    z = z_expectations(circuit, theta, x, n_qubits)
    worst = 0.0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zz = zz_expectation(circuit, theta, i, j, x, n_qubits)
            worst = max(worst, abs(zz - z[i] * z[j]))
    return worst


def jacobian_rank(
    circuit: QuantumCircuit,
    X: np.ndarray,
    theta: np.ndarray,
    *,
    tol: float = 1e-8,
    n_qubits: int = N_QUBITS,
) -> int:
    """Numerical rank of d(<Z_i> over the batch X)/d(theta).

    Thin wrapper over the package implementation.
    """
    from qsocket.rank import jacobian_matrix, numerical_rank

    return numerical_rank(jacobian_matrix(circuit, theta, X, n_qubits=n_qubits), tol=tol)
