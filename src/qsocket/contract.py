"""The experiment contract: what the main series IS, as data.

Every number here was fixed before the run and changing one changes what the results
mean. It lives in the package rather than in the driver script so that the analysis, the
probes and the supplementary tables read the same definition instead of importing a
2500-line driver to get at its constants.

The grid:

    generator seed   11 . 22 . 33          -- exactly these three pass the binding gates
    dilution         linear . h2 . h4 . h42  (6 / 15 / 29 / 295 head parameters)
    ansatz           L1 . L2               -- the ansatz is an axis, both levels
    training seed    1..10
    arms             A, F trained  .  B, E, D_matched, D_best frozen (feature cache)
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from qsocket.core import blas_thread_settings
from qsocket.datasets import (
    DEFAULT_DATA_DIR,
    N_COMPONENTS,
    N_SAMPLES_TOTAL,
    PRODUCTION_DATASET,
    dataset_paths,
    generate_and_freeze,
    load_manifest,
    load_splits,
    verify_frozen_identity,
)
from qsocket.gates import G1_LR_GRID
from qsocket.head import DILUTION_AXIS

# --- fixed configuration -------------------------------------------------------------

GENERATOR = "hyperplanes"
# The production cell. n_features = 20, for continuity with the retired two_curves.
GENERATOR_KWARGS: dict = {"n_features": 20, "n_hyperplanes": 3, "dim_hyperplanes": 5}

DATASET_SEEDS: tuple[int, ...] = (11, 22, 33)


A7_DATA_DIR = DEFAULT_DATA_DIR / "a7_generator_seeds"

GENERATED_HASH_PREFIXES: dict[int, tuple[str, str, str]] = {
    11: ("4360508611e0e896", "2bf856a6c49a9c38", "5e519b749c22b488"),
    22: ("daf152204bbd7292", "f087808ca635cff7", "3dc8d3e1e30004f1"),
    33: ("5d05157a8a23e915", "536dc33fc72f70f2", "648ab550df5c4fc9"),
}

DILUTIONS: tuple[str, ...] = DILUTION_AXIS
ANSATZ_LEVELS: tuple[str, ...] = ("L1", "L2")
# Arm F: the L1 skeleton with the CZ gates removed. Same parameter count, zero
# entanglement, so Delta_AF isolates entanglement with both arms trained.
PRODUCT_ANSATZ = "product"

# Which dilutions arm F runs on; all four by default, for control rather than symmetry.
# Delta_AF going to zero at high dilution the same way Delta_AB does is what shows the
# axis drives the zeroing rather than the absence of entanglement; at a single point
# "entanglement never mattered" cannot be told apart from "we only looked where the head
# is weakest". A constant and a CLI flag because arm F is ~120 of the expensive runs:
# restricting it to the two ends of the axis keeps the control argument at half the cost,
# and one point gives back the objection. Changing this changes what may be written about
# Delta_AF.
ARM_F_DILUTIONS: tuple[str, ...] = DILUTIONS

R_CONTRACT = 2
SEEDS: tuple[int, ...] = tuple(range(1, 11))
LR_SELECTION_SEEDS: tuple[int, ...] = (1, 2, 3)

# Imported and never redefined. Ties go to the lower lr.
CONTRACT_LR_GRID: tuple[float, ...] = tuple(G1_LR_GRID)
# Arm E gets one extra point upwards, as a separate constant rather than a modification
# of the contract grid: its validation optimum sits on the upper edge of that grid too,
# so Delta_AE is an upper bound on the socket's contribution at a lower bound on the
# baseline's quality.
ARM_E_LR_GRID: tuple[float, ...] = CONTRACT_LR_GRID + (1e-1,)

# The arms whose mean validation accuracy is the lr criterion. Passengers are measured
# at the selected lr and reported, but never move the choice.
LR_SELECTION_ARMS: tuple[str, ...] = ("A", "B")

TRAINED_ARMS: tuple[str, ...] = ("A", "F")
FROZEN_ARMS: tuple[str, ...] = ("B", "E", "D_matched", "D_best")
ARMS: tuple[str, ...] = ("A", "B", "E", "F", "D_matched", "D_best")
# Arms with no ansatz dimension: computed once per (dataset seed x dilution x seed).
ANSATZ_FREE_ARMS: tuple[str, ...] = ("E", "F", "D_matched", "D_best")

# patience 30 and a 300-epoch budget stay: patience = 300 costs ~4x the wall time for a
# difference in Delta smaller than the binomial SE. Both travel in every row, because
# they are load-bearing for the size of Delta and not only for the time.
PATIENCE = 30
MAX_EPOCHS = 300
BATCH_SIZE = 64
# Input-scale multiplier. 1.0 = the frozen FEATURE_RANGE scaling and nothing on top.
FEATURE_SCALE = 1.0

SPLITS_REPORTED: tuple[str, ...] = ("test", "val")

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "a7"
RESULTS_CSV = "a7_results.csv"
LR_TABLE_CSV = "a7_lr_table.csv"
LR_SELECTION_JSON = "a7_lr_selection.json"
GATES_JSON = "a7_gates.json"
SUMMARY_JSON = "a7_summary.json"
PREDICTIONS_DIR = "predictions"


# theta moved by less than this counts as "theta did not move".
THETA_STILL = 1e-6

# Above this fraction of runs hitting the epoch budget, the cell gets a note.
BUDGET_HIT_NOTE_FRACTION = 0.20


# --- provenance ----------------------------------------------------------------------


def git_commit() -> str:
    """Short commit of the working tree, marked dirty when it is. Read-only git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def env_hash() -> str:
    """Digest of everything that can change a number. Not the full pip freeze: the
    point is to detect "this row came from a different stack", and an unrelated package
    bumping would only add noise to that signal.

    The BLAS thread counts are IN the digest, not only the package versions. Two runs of
    the same code on the same versions with a different thread count give different
    accuracies for the trained arms (outputs/WAGI_do_odrobienia.md, 60 cells), so a
    digest that ignored them would report one environment where there were two — which is
    why that drift could not be seen from the rows."""
    import pandas
    import qiskit
    import scipy
    import sklearn

    try:
        import pennylane
        pennylane_version = pennylane.__version__
    except ImportError:  # pragma: no cover - pennylane is a hard dependency of the run
        pennylane_version = "absent"

    parts = [
        f"python=={platform.python_version()}",
        f"numpy=={np.__version__}",
        f"torch=={torch.__version__}",
        f"qiskit=={qiskit.__version__}",
        f"pennylane=={pennylane_version}",
        f"scikit-learn=={sklearn.__version__}",
        f"scipy=={scipy.__version__}",
        f"pandas=={pandas.__version__}",
        f"torch_threads=={torch.get_num_threads()}",
    ]
    parts += [f"{name}=={value}" for name, value in sorted(blas_thread_settings().items())]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- datasets ------------------------------------------------------------------------


def frozen_name_for(seed: int) -> str:
    """On-disk naming convention: generator, every knob, seed."""
    knobs = "_".join(f"{key}{GENERATOR_KWARGS[key]}" for key in
                     ("n_features", "n_hyperplanes", "dim_hyperplanes"))
    return f"{GENERATOR}_{knobs}_seed{seed}"


def dataset_location(seed: int) -> tuple[str, Path]:
    """(frozen name, directory) for one generator seed.

    Seed 11 resolves to the registered production artefact in data/, which this script
    never writes to.
    """
    if seed == 11:
        assert frozen_name_for(11) == PRODUCTION_DATASET, (
            "the naming convention drifted from the registered production dataset: "
            f"{frozen_name_for(11)!r} != {PRODUCTION_DATASET!r}"
        )
        return PRODUCTION_DATASET, DEFAULT_DATA_DIR
    return frozen_name_for(seed), A7_DATA_DIR


def ensure_dataset(seed: int, *, allow_generate: bool = True) -> dict:
    """Manifest for one generator seed, with every hash asserted, never trusted.

    Seed 11 goes through verify_frozen_identity. Seeds 22/33 are generated into
    A7_DATA_DIR if absent, then their three hashes are asserted against
    GENERATED_HASH_PREFIXES. Existing files are loaded and checked, never regenerated.
    """
    name, out_dir = dataset_location(seed)
    if seed == 11:
        return verify_frozen_identity(name, out_dir=out_dir)

    data_path, manifest_path = dataset_paths(name, out_dir=out_dir)
    if not (data_path.exists() and manifest_path.exists()):
        if not allow_generate:
            raise FileNotFoundError(
                f"generator seed {seed} is not on disk at {data_path} and generation is "
                "switched off"
            )
        generate_and_freeze(
            GENERATOR,
            n_samples=N_SAMPLES_TOTAL,
            generator_kwargs=dict(GENERATOR_KWARGS),
            dataset_seed=seed,
            out_dir=out_dir,
            frozen_name=name,
            n_components=N_COMPONENTS,
        )

    manifest = load_manifest(name, out_dir=out_dir)
    expected = GENERATED_HASH_PREFIXES[seed]
    actual = (manifest["dataset_hash"], manifest["pca_hash"], manifest["file_sha256"])
    for label, prefix, value in zip(("dataset_hash", "pca_hash", "file_sha256"), expected, actual):
        assert value.startswith(prefix), (
            f"generator seed {seed}: {label} {value} does not start with {prefix}. "
            "Either the generation chain changed or the file was edited; every number "
            "measured on this dataset would answer a different question."
        )
    # load_splits re-checks the content digest and the file digest against the manifest.
    load_splits(name, out_dir=out_dir)
    return manifest


# --- reading rows back ---------------------------------------------------------------


def optional_int(value):
    """An integer column that is allowed to be empty, read back from a CSV.

    pandas returns such a column as float64 as soon as one row is empty, so rff_width
    written as 32 comes back as "32.0": int("32.0") raises, and a resume key built from
    the raw text stops matching the planner's key built from the integer.
    """
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(float(value))



def dataset_seed_of(name) -> int:
    """Generator seed out of a frozen name. The name carries every knob and the seed, so
    the row does not need a redundant column that could disagree with it."""
    return int(str(name).rsplit("_seed", 1)[1])
