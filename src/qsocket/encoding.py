"""R_y feature map (one feature per qubit) and the frozen feature range.

The feature range is frozen and identical in every arm, so it cannot differentially
favour one arm over another. Re-uploading is handled in ansatzes.build_socket_circuit:
for R > 1 the same feature map is injected before every ansatz block, reusing the
same Parameter objects rather than copies.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Frozen
FEATURE_RANGE: tuple[float, float] = (-np.pi / 4, np.pi / 4)

FEATURE_PARAM_NAME = "x"


def ry_feature_map(n_qubits: int) -> QuantumCircuit:
    """Angle encoding: R_y(x_i) on qubit i.
    The returned circuit owns a ParameterVector named "x" of length n_qubits. To get
    re-uploading, compose this same circuit object more than once — composing reuses
    the identical Parameter objects, which is what makes every injection share one x.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    x = ParameterVector(FEATURE_PARAM_NAME, n_qubits)
    qc = QuantumCircuit(n_qubits, name="ry_feature_map")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc
