"""Seeding, thread pinning, the feature map and the readout contract.

derive() — deterministic seeds, blake2b rather than hash(), which is randomised across
processes. The pairing of delta = acc(A) - acc(B) rests on the keys excluding the arm:

    quantum socket init  derive(seed, ansatz_level, R)     head init  derive(seed, dilution)
    arm D projection     derive(seed, "D", scale)          batch order  derive(seed)
    dataset shuffle      derive(dataset_seed, "shuffle")

pin_blas_threads() — MUST run before numpy/torch import. A multi-threaded BLAS sums in a
split-dependent order and float addition is not associative; Adam turns those last bits
into a different trajectory over 300 epochs.

ry_feature_map() — angle encoding, one feature per qubit, over a range frozen and
identical in every arm.

pauli_z_chains() — WHICH Z observables are read out and IN WHICH ORDER: ascending Pauli
weight, then lexicographic (n=5, order=2 -> (0,)...(4,) (0,1)...(3,4)). THE ORDER IS A
CONTRACT: Qiskit labels, PennyLane operators and the eigenvalue table in rank.py all read
it from here, so the backends cannot disagree about which column is which. Weight-ascending
also means the first n columns of any order are the order=1 readout, unchanged.
"""

from __future__ import annotations

import hashlib
import os
from itertools import combinations
from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# The readout order of the main series: one <Z_i> per qubit.
DEFAULT_READOUT_ORDER = 1

FEATURE_RANGE: tuple[float, float] = (-np.pi / 4, np.pi / 4)
FEATURE_PARAM_NAME = "x"

BLAS_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def derive(*parts) -> int:
    """Deterministic seed, stable across processes, runs and Python versions.

    Keyed on repr(parts), so part order matters and derive(1, "a") differs from
    derive((1, "a")).
    """
    return int.from_bytes(hashlib.blake2b(repr(parts).encode("utf-8")).digest()[:8], "big")


def pin_blas_threads(threads: int = 1) -> dict[str, str]:
    """Pin every BLAS pool to `threads`, returning what is in force afterwards.

    setdefault: a cluster job that pins threads through its scheduler keeps its own value.
    """
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    return blas_thread_settings()


def blas_thread_settings() -> dict[str, str]:
    """What each pool is set to now; unset reads as "unset". Goes into env_hash, which
    would otherwise report one environment where there were two."""
    return {name: os.environ.get(name, "unset") for name in BLAS_THREAD_VARIABLES}


def ry_feature_map(n_qubits: int) -> QuantumCircuit:
    """Angle encoding: R_y(x_i) on qubit i.

    The returned circuit owns a ParameterVector named "x". To get re-uploading, compose
    this same circuit object more than once — composing reuses the identical Parameter
    objects, which is what makes every injection share one x.
    """
    x = ParameterVector(FEATURE_PARAM_NAME, n_qubits)
    qc = QuantumCircuit(n_qubits, name="ry_feature_map")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc


def pauli_z_chains(n_qubits: int, *, order: int) -> tuple[tuple[int, ...], ...]:
    """Qubit-index tuples of every Z chain of weight 1..order, in the contract order.

    Qubit indices rather than labels or operators: the callers disagree about endianness
    and about operator classes, but not about which qubits a chain covers. The identity
    is excluded — as a readout column it would only duplicate the head bias.
    """
    if order > n_qubits:
        raise ValueError(
            f"order must be <= n_qubits: a Pauli chain cannot be longer than the register "
            f"(got order={order}, n_qubits={n_qubits})."
        )
    # combinations() is lexicographic for a sorted input, so the contract order is just
    # weight-ascending iteration over it.
    return tuple(
        chain for weight in range(1, order + 1) for chain in combinations(range(n_qubits), weight)
    )


def readout_size(n_qubits: int, *, order: int) -> int:
    """How many outputs a readout of this order has: sum of C(n, w) for w in 1..order.

    A closed form rather than len(pauli_z_chains(...)): callers that only need a width
    must neither build operators nor hardcode 15. Being an independent computation of the
    same number, it also witnesses the enumeration — tests assert the two agree.
    """
    if order > n_qubits:
        raise ValueError(f"order must be <= n_qubits (got order={order}, n_qubits={n_qubits})")
    return sum(comb(n_qubits, weight) for weight in range(1, order + 1))
