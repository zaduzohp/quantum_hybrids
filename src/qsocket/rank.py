"""Numerical rank of the socket Jacobian — how many parameters change anything.

Rank here is the numerical rank of d(socket outputs)/d(theta), i.e. the number of
parameter directions that move the readout at all. Which outputs those are depends on
readout_order (qsocket.observables); widening the readout adds rows to the Jacobian, so
the rank is monotone in order. It is not the Fisher-information
effective dimension of Abbas et al. (Nature Comp. Sci. 1, 403).

Derivatives use the parameter-shift rule, exact for single-generator Rx/Ry/Rz gates:
df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2. Exactness matters — finite
differences would blur the singular-value spectrum the rank threshold reads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsocket.encoding import FEATURE_RANGE
from qsocket.observables import DEFAULT_READOUT_ORDER, pauli_z_chains, readout_size

DEFAULT_N_QUBITS = 5
DEFAULT_TOL = 1e-8
DEFAULT_BATCH = 50

CircuitBuilder = Callable[[int, int], QuantumCircuit]


def z_sign_table(n_qubits: int, *, order: int = DEFAULT_READOUT_ORDER) -> np.ndarray:
    """(n_outputs, 2**n_qubits) of +-1: eigenvalue of each Z chain on each basis state.

    All the chains are diagonal and commute, so every expectation value comes from one
    probability vector: <O> = signs @ p. The eigenvalue of a chain is the product of the
    eigenvalues of its members, which is why one table covers every order.

    Row order is the contract order of qsocket.observables, so row i of this table is
    column i of the socket output on both backends.
    """
    basis = np.arange(2**n_qubits)
    single = np.array([1.0 - 2.0 * ((basis >> i) & 1) for i in range(n_qubits)])
    return np.array(
        [np.prod(single[list(chain)], axis=0) for chain in pauli_z_chains(n_qubits, order=order)]
    )


def _binding_plan(circuit: QuantumCircuit) -> list[tuple[bool, int]]:
    """Per circuit parameter, in circuit order: whether it is a theta (True) or an x
    (False), and its index inside that vector."""
    plan = []
    for param in circuit.parameters:
        vector, index = param.name.split("[")
        plan.append((vector == "theta", int(index.rstrip("]"))))
    return plan


def z_expectation_batch(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    X: np.ndarray,
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    readout_order: int = DEFAULT_READOUT_ORDER,
) -> np.ndarray:
    """Exact Z-chain expectations for every row of X, shape (len(X), n_outputs).

    readout_order=1 is the main-series readout and the default, so every pre-probe call
    site is unchanged.
    """
    plan = _binding_plan(circuit)
    signs = z_sign_table(n_qubits, order=readout_order)
    out = np.empty((len(X), readout_size(n_qubits, order=readout_order)))
    for row, x in enumerate(X):
        values = [theta[index] if is_theta else x[index] for is_theta, index in plan]
        probabilities = np.abs(np.asarray(Statevector(circuit.assign_parameters(values)))) ** 2
        out[row] = signs @ probabilities
    return out


def jacobian_matrix(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    X: np.ndarray,
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    readout_order: int = DEFAULT_READOUT_ORDER,
) -> np.ndarray:
    """d(outputs over the batch)/d(theta), shape (len(X) * n_outputs, len(theta)).

    Widening the readout adds ROWS, never columns: theta is the same vector whatever is
    measured. The rank can therefore only rise or stay equal with order, and if it stays
    equal at order=2 the parameter space of the socket did not change and the entire
    difference the probe measures lies in the richness of the readout.
    """
    n_outputs = readout_size(n_qubits, order=readout_order)
    jac = np.empty((len(X) * n_outputs, len(theta)))
    for p in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[p] += np.pi / 2
        minus[p] -= np.pi / 2
        forward = z_expectation_batch(
            circuit, plus, X, n_qubits=n_qubits, readout_order=readout_order
        )
        backward = z_expectation_batch(
            circuit, minus, X, n_qubits=n_qubits, readout_order=readout_order
        )
        jac[:, p] = ((forward - backward) / 2).reshape(-1)
    return jac


def numerical_rank(jacobian: np.ndarray, *, tol: float = DEFAULT_TOL) -> int:
    """Count of singular values above tol * s_max."""
    s = np.linalg.svd(jacobian, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


def sample_inputs(batch: int, *, seed: int, n_qubits: int = DEFAULT_N_QUBITS) -> np.ndarray:
    lo, hi = FEATURE_RANGE
    return np.random.default_rng(seed).uniform(lo, hi, size=(batch, n_qubits))


def sample_theta(n_params: int, *, seed: int) -> np.ndarray:
    """Draw from U[0, 2*pi), the initialisation distribution of the quantum arms."""
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, n_params)


def effective_dimension(
    circuit_builder: CircuitBuilder,
    R: int,
    *,
    batch: int = DEFAULT_BATCH,
    tol: float = DEFAULT_TOL,
    n_qubits: int = DEFAULT_N_QUBITS,
    theta_seed: int = 20260812,
    x_seed: int = 31337,
    readout_order: int = DEFAULT_READOUT_ORDER,
) -> int:
    """Numerical rank of d(outputs)/d(theta) for a circuit built by circuit_builder.

    circuit_builder(n_qubits, R) must return the full socket circuit including the
    feature map.
    """
    circuit = circuit_builder(n_qubits, R)
    n_params = sum(1 for p in circuit.parameters if p.name.startswith("theta"))
    theta = sample_theta(n_params, seed=theta_seed + R)
    X = sample_inputs(batch, seed=x_seed, n_qubits=n_qubits)
    return numerical_rank(
        jacobian_matrix(circuit, theta, X, n_qubits=n_qubits, readout_order=readout_order),
        tol=tol,
    )


def singular_values(
    circuit_builder: CircuitBuilder,
    R: int,
    *,
    batch: int = DEFAULT_BATCH,
    n_qubits: int = DEFAULT_N_QUBITS,
    theta_seed: int = 20260812,
    x_seed: int = 31337,
) -> np.ndarray:
    circuit = circuit_builder(n_qubits, R)
    n_params = sum(1 for p in circuit.parameters if p.name.startswith("theta"))
    theta = sample_theta(n_params, seed=theta_seed + R)
    X = sample_inputs(batch, seed=x_seed, n_qubits=n_qubits)
    return np.linalg.svd(jacobian_matrix(circuit, theta, X, n_qubits=n_qubits), compute_uv=False)


def null_space_support(
    circuit_builder: CircuitBuilder,
    R: int,
    *,
    batch: int = DEFAULT_BATCH,
    tol: float = DEFAULT_TOL,
    support_tol: float = 1e-6,
    n_qubits: int = DEFAULT_N_QUBITS,
    theta_seed: int = 20260812,
    x_seed: int = 31337,
) -> list[list[int]]:
    """Theta indices carried by each dead direction.

    Each entry is the support of one null-space vector: parameters that trade off
    against each other without moving any output.
    """
    circuit = circuit_builder(n_qubits, R)
    n_params = sum(1 for p in circuit.parameters if p.name.startswith("theta"))
    theta = sample_theta(n_params, seed=theta_seed + R)
    X = sample_inputs(batch, seed=x_seed, n_qubits=n_qubits)
    jac = jacobian_matrix(circuit, theta, X, n_qubits=n_qubits)
    _, s, vt = np.linalg.svd(jac)
    rank = int(np.sum(s > tol * s[0]))
    return [sorted(np.flatnonzero(np.abs(v) > support_tol).tolist()) for v in vt[rank:]]


def label_supports(supports: Sequence[Sequence[int]], labels: Sequence[str]) -> list[list[str]]:
    """Translate theta indices into human-readable layer names."""
    return [[labels[i] for i in support] for support in supports]
