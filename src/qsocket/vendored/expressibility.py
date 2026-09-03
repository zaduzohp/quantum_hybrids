"""VENDORED.

Upstream:  QC1_Quantum_Banknote_Classifier_on_ODRA_5, src/qbanknote/metrics.py
Commit:    8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9  ("KL Done", 2026-07-19)
Copied: 2026-08-18
Lines:     935-936 (haar_pdf_fidelity), 939-956 (binned_distributions), 959-964 (kl_divergence), 1369-1371 (sample_haar_fidelities)
Changes: NONE (only unused code was deleted, rest is bite indentical)
"""

from __future__ import annotations

import numpy as np


def haar_pdf_fidelity(f: np.ndarray, dim: int) -> np.ndarray:
    return (dim - 1.0) * (1.0 - f) ** (dim - 2.0)

def binned_distributions(
    fid_values: np.ndarray,
    dim: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, edges = np.histogram(fid_values, bins=bins, density=False)
    p_emp = counts.astype(np.float64)
    if p_emp.sum() == 0:
        p_emp = np.ones_like(p_emp) / len(p_emp)
    else:
        p_emp /= p_emp.sum()

    mids = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    p_haar = haar_pdf_fidelity(mids, dim=dim) * width
    p_haar /= p_haar.sum()
    return edges, mids, p_emp, p_haar

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_s = p + eps
    q_s = q + eps
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))

def sample_haar_fidelities(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw fidelities F = |<psi|phi>|^2 from the Haar-random pair distribution."""
    return 1.0 - rng.random(int(n_samples)) ** (1.0 / (dim - 1))
