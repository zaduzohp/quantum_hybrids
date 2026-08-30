"""Which Z observables the socket reads out, and in which order.

The socket output is a vector of expectation values of diagonal Pauli strings, all read
from one state. `order` is the maximum Pauli weight:

    order=1  <Z_i>                     n outputs   -- the main series
    order=2  <Z_i> and <Z_i Z_j>       n + C(n,2)  -- the correlator probe, 15 at n=5
    order=n  every non-empty Z chain   2**n - 1    -- the diagonal readout ceiling

All of these commute, so a higher order is a richer readout of the same state and the
same run, not extra measurements. Nothing here is specific to the probe: order is a free
parameter so that widening the readout later is a flag rather than a refactor.

THE ORDER OF THE OUTPUT IS A CONTRACT. Ascending Pauli weight, then lexicographic in
qubit index:

    n=5, order=2 -> (0,) (1,) (2,) (3,) (4,) (0,1) (0,2) (0,3) (0,4) (1,2) ... (3,4)

Three places turn this list into something executable — Qiskit labels in
socket.pauli_z_observables, PennyLane operators in pennylane_socket.make_qnode, and the
eigenvalue table in rank.z_sign_table — and a paired difference between two backends is
meaningless if they disagree on which column is which. They all read the order from
pauli_z_chains, so there is exactly one definition of it.

The weight-ascending part of the contract carries a second guarantee that the tests pin:
the first n columns of an order>=1 readout are the order=1 readout, unchanged. A row of
the probe and a row of the main series can therefore be compared column by column.

No function here takes a default order. The default belongs at the API boundary
(socket.make_socket, socket.Socket), where it keeps every pre-probe call site meaning what
it meant before; below that boundary an implicit order=1 would be a silently truncated
readout rather than a convenience.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

# The readout order of the main series: one <Z_i> per qubit. Used as the default at the
# API boundary only.
DEFAULT_READOUT_ORDER = 1


def _validate(n_qubits: int, order: int) -> None:
    """One copy of the admissibility rule, because two copies drift apart."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order > n_qubits:
        raise ValueError(
            f"order must be <= n_qubits: a Pauli chain cannot be longer than the register "
            f"(got order={order}, n_qubits={n_qubits}). order={n_qubits} is already every "
            f"non-empty Z chain, i.e. the diagonal readout ceiling."
        )


def pauli_z_chains(n_qubits: int, *, order: int) -> tuple[tuple[int, ...], ...]:
    """Qubit-index tuples of every Z chain of weight 1..order, in the contract order.

    Returns tuples of qubit indices, not labels or operators: the callers disagree about
    endianness and about operator classes, but not about which qubits a chain covers.

    The identity is excluded. It is constant, so as a readout column it would only add a
    duplicate of the head bias.
    """
    _validate(n_qubits, order)
    # combinations() is lexicographic for a sorted input, so the contract order is just
    # weight-ascending iteration over it.
    return tuple(
        chain
        for weight in range(1, order + 1)
        for chain in combinations(range(n_qubits), weight)
    )


def readout_size(n_qubits: int, *, order: int) -> int:
    """How many outputs a readout of this order has: sum of C(n, w) for w in 1..order.

    Deliberately a closed form rather than len(pauli_z_chains(...)). Callers that only
    need a width — the head input dimension, or the width of the classical control that
    has to match it — must neither build operators nor hardcode 15. Being an independent
    computation of the same number, it also witnesses the enumeration: the two are
    asserted equal in tests, so an off-by-one in either one fails loudly.
    """
    _validate(n_qubits, order)
    return sum(comb(n_qubits, weight) for weight in range(1, order + 1))
