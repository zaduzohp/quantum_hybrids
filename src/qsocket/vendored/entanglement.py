"""VENDORED — literal copy, do not modify.

Upstream:  QC1_Quantum_Banknote_Classifier_on_ODRA_5, src/qbanknote/metrics.py
Commit:    8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9  ("KL Done", 2026-07-19)
Copied:    2026-08-18
Lines:     50-56 (single_qubit_reduced_density), 59-68 (meyer_wallach_score)
Licence:   project-internal upstream repo, no LICENSE file — internal copy
Changes vs upstream: NONE (only unused-on-this-path code was dropped; the bodies
           of the copied functions are byte-for-byte identical, pinned by test)

Why copied: upstream metrics.py is 101 KB and pulls in qbanknote.iqm (IQM hardware),
tomography, evaluation and paths. These two functions are pure numpy; importing the
whole module would drag in the hardware stack this work does not use.
"""

from __future__ import annotations

import numpy as np

def single_qubit_reduced_density(
    state: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    arr = state.reshape((2,) * n_qubits)
    arr = np.moveaxis(arr, qubit, 0)
    psi_mat = arr.reshape(2, -1)
    return psi_mat @ psi_mat.conj().T

def meyer_wallach_score(state: np.ndarray, n_qubits: int) -> float:
    if n_qubits < 1:
        return 0.0
    acc = 0.0
    for i in range(n_qubits):
        rho_i = single_qubit_reduced_density(state, i, n_qubits)
        purity = float(np.real(np.trace(rho_i @ rho_i)))
        acc += 1.0 - purity
    q = (2.0 / n_qubits) * acc
    return float(max(0.0, min(1.0, q)))
