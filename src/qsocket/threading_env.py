"""One thread per BLAS pool. MUST run before numpy/torch import — the pools read these
variables when first loaded, so a later call is silently ineffective.

torch.set_num_threads(1) does not reach the BLAS under numpy/sklearn. A multi-threaded
BLAS sums in a split-dependent order, and float addition is not associative; Adam turns
those last bits into a different trajectory over 300 epochs.
"""

from __future__ import annotations

import os

# Every pool that can multi-thread a reduction under numpy/scipy/sklearn.
BLAS_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_blas_threads(threads: int = 1) -> dict[str, str]:
    """Pin every BLAS pool to `threads`, returning what is in force afterwards.

    setdefault: a cluster job that pins threads through its scheduler keeps its own
    value, and the return reports what the process actually runs with.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    return blas_thread_settings()


def blas_thread_settings() -> dict[str, str]:
    """What each pool is set to now; unset reads as "unset". Goes into env_hash, which
    would otherwise report one environment where there were two."""
    return {name: os.environ.get(name, "unset") for name in BLAS_THREAD_VARIABLES}
