"""Quality gates the pipeline must make checkable rather than bypass.

  G1 dataset carrying capacity: a strong classical model on PCA-5 must beat the
     identity arm by at least 0.05 and land in the 0.65-0.90 band,
  G2 effective dimensionality: no single PCA component explains more than 80 %
     of the retained variance,
  G3 real entanglement: max connected correlation > 1e-3 for L1 and L2, < 1e-5
     for the product circuit,
  G4 ansatz-level matching: equal trainable parameter counts, zero SWAP gates
     after transpilation (both hard failures), 4 CZ per block as a spec check,
     depth and duration reported but not gated,
  G5 clbit <-> qubit mapping: a circuit putting |1> on qubit i flips bit i.

Two things here are load-bearing and easy to get wrong:

  * G3 is computed from exact statevector probabilities, never from sampling. Shot
    noise falls off as ~1/sqrt(shots) (1e-2 at 10 000 shots) while the negative control
    must be resolved below 1e-5, so sampling would report "no entanglement" everywhere
    — indistinguishable from a passing gate.
  * G4 statistics come from a circuit this module transpiled itself at
    optimization_level=1. The vendored get_circuit_stats transpiles at level 0 on its
    own, so count_gate_types is called directly on our own transpiled circuit instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsocket.ansatzes import (
    STAR_HUB_QUBIT,
    build_socket_circuit,
    socket_param_count,
)

# Reused rather than re-implemented: this is the code path a hardware run takes, and a
# second copy could drift from it without any test noticing.
from qsocket.hardware import transpile_for_backend
from qsocket.rank import DEFAULT_N_QUBITS, sample_inputs, sample_theta
from qsocket.readout import connected_correlation
from qsocket.vendored.circuit_stats import count_gate_types

# --- frozen gate constants ---------------------------------------------------------

G3_ENTANGLED_MIN = 1e-3
G3_PRODUCT_MAX = 1e-5

G4_OPTIMIZATION_LEVEL = 1
G4_SEED_TRANSPILER = 42
CZ_PER_BLOCK = 4

ANSATZ_LEVELS = ("L1", "L2")


# --- G3: real entanglement ---------------------------------------------------------


def exact_probability_histogram(circuit: QuantumCircuit) -> dict[str, float]:
    """Exact outcome probabilities in the shape of a histogram, free of shot noise.

    The readout functions only divide by the total, so probabilities work as counts
    summing to 1. Same readout code path as hardware, without a sampling step that would
    swamp the 1e-5 side of the gate.
    """
    if circuit.parameters:
        raise ValueError(
            f"circuit still has free parameters {sorted(p.name for p in circuit.parameters)}; "
            "bind theta and x before computing G3"
        )
    return Statevector(circuit).probabilities_dict()


def _bound_socket_circuit(
    ansatz_name: str,
    R: int,
    *,
    theta_seed: int,
    x_seed: int,
    n_qubits: int,
) -> tuple[QuantumCircuit, np.ndarray, np.ndarray]:
    """Socket circuit with theta ~ U[0, 2pi) and one x ~ U(FEATURE_RANGE) bound in.

    Two independent random streams: a connected correlation is a property of the state,
    i.e. of the pair (theta, x), not of the circuit alone. The samplers come from
    qsocket.rank so rank and entanglement draw their inputs the same way.
    """
    circuit = build_socket_circuit(ansatz_name, n_qubits, R)
    n_theta = sum(1 for p in circuit.parameters if p.name.startswith("theta"))
    theta = sample_theta(n_theta, seed=theta_seed)
    x = sample_inputs(1, seed=x_seed, n_qubits=n_qubits)[0]

    values = {}
    for param in circuit.parameters:
        vector, index = param.name.split("[")
        index = int(index.rstrip("]"))
        values[param] = float(theta[index] if vector == "theta" else x[index])
    return circuit.assign_parameters(values), theta, x


def check_g3_entanglement(
    ansatz_name: str,
    R: int,
    *,
    theta_seed: int,
    x_seed: int,
    n_qubits: int = DEFAULT_N_QUBITS,
) -> dict:
    """Gate G3: max_{i<j} |<Z_i Z_j> - <Z_i><Z_j>| from exact probabilities.

    A circuit carrying two-qubit gates must clear G3_ENTANGLED_MIN; a circuit with none
    — the product control — must stay below G3_PRODUCT_MAX. Which rule applies is read
    off the circuit itself rather than off the name, so a level that lost its CZ gates
    could not be judged by the entangling rule.

    The margin is large: over 20 draws of (theta, x) the smallest value seen for L1/L2
    is ~54x the threshold, so a result near 1e-3 is a symptom of a bug rather than of a
    tight gate.

    theta_seed and x_seed have no defaults: the pair (theta, x) is part of the
    measurement, so the caller states it.
    """
    bound, theta, x = _bound_socket_circuit(
        ansatz_name, R, theta_seed=theta_seed, x_seed=x_seed, n_qubits=n_qubits
    )
    cz_count = bound.count_ops().get("cz", 0)
    value = connected_correlation(exact_probability_histogram(bound), n_qubits)

    if cz_count:
        mode, threshold, passed = "entangling", G3_ENTANGLED_MIN, value > G3_ENTANGLED_MIN
        # How far above the threshold.
        margin = value / threshold
    else:
        mode, threshold, passed = "product_control", G3_PRODUCT_MAX, value < G3_PRODUCT_MAX
        # How far below it; infinite for a state with no correlation at all.
        margin = threshold / value if value > 0 else float("inf")

    return {
        "gate": "G3",
        "ansatz": ansatz_name,
        "R": R,
        "n_qubits": n_qubits,
        "theta_seed": theta_seed,
        "x_seed": x_seed,
        "cz_count": int(cz_count),
        "mode": mode,
        "threshold": threshold,
        "connected_correlation": float(value),
        "margin_factor": float(margin),
        "source": "exact statevector probabilities (no sampling)",
        "passed": bool(passed),
        "theta_norm": float(np.linalg.norm(theta)),
        "x": [float(v) for v in x],
    }


# --- G4: ansatz-level matching -----------------------------------------------------


def check_g4_parameter_parity(R: int, *, n_qubits: int = DEFAULT_N_QUBITS) -> dict:
    """Gate G4, static half: equal trainable parameter counts in L1 and L2.

    Separate from the transpilation half and without a backend argument: this is a
    property of the ansatz definition alone, which no transpiler can break.
    """
    expected = socket_param_count(n_qubits, R)
    counts = {
        level: sum(
            1
            for p in build_socket_circuit(level, n_qubits, R).parameters
            if p.name.startswith("theta")
        )
        for level in ANSATZ_LEVELS
    }
    equal = len(set(counts.values())) == 1
    matches_formula = all(count == expected for count in counts.values())

    failures = []
    if not equal:
        failures.append(f"parameter counts differ between levels: {counts}")
    if not matches_formula:
        failures.append(f"parameter counts do not match 15R+5 = {expected}: {counts}")

    return {
        "gate": "G4-parameters",
        "R": R,
        "n_qubits": n_qubits,
        "expected": expected,
        "counts": counts,
        "equal_across_levels": bool(equal),
        "matches_formula": bool(matches_formula),
        "failures": failures,
        "passed": not failures,
    }


def _estimate_duration_seconds(circuit: QuantumCircuit, backend) -> float | None:
    """Wall-clock duration of the transpiled circuit, or None when unavailable.

    A simulator has no instruction durations, so there is nothing to estimate. Never
    substitute a number that was not measured.
    """
    target = getattr(backend, "target", None)
    if target is None:
        return None
    try:
        return float(circuit.estimate_duration(target, unit="s"))
    except Exception:
        return None


def _transpiled_stats(
    ansatz_name: str, R: int, backend, *, optimization_level: int, n_qubits: int
) -> dict:
    """Transpile here, then count; the two steps are not separable by design.

    Accepting a circuit from outside would allow statistics from an
    optimization_level=0 transpilation to slip in unflagged. The level used is recorded
    in the result so a caller can assert on it.
    """
    logical = build_socket_circuit(ansatz_name, n_qubits, R)
    # Unbound parameters keep the gate independent of any particular theta.
    transpiled = transpile_for_backend(
        logical, backend, optimization_level, G4_SEED_TRANSPILER
    )
    assert transpiled is not logical, "statistics must come from a transpiled circuit"

    ops = transpiled.count_ops()
    gate_types = count_gate_types(transpiled)
    cz_pairs = sorted(
        {
            tuple(sorted(transpiled.find_bit(q).index for q in instruction.qubits))
            for instruction in transpiled.data
            if instruction.operation.num_qubits == 2 and instruction.operation.name != "barrier"
        }
    )

    return {
        "ansatz": ansatz_name,
        "R": R,
        "optimization_level": optimization_level,
        "seed_transpiler": G4_SEED_TRANSPILER,
        "depth": int(transpiled.depth()),
        "swap_count": int(ops.get("swap", 0)),
        "cz_count": int(ops.get("cz", 0) + ops.get("cx", 0)),
        "two_qubit_gates": int(gate_types["Two-Qubit Gates"]),
        "single_qubit_gates": int(gate_types["Single-Qubit Gates"]),
        "total_gates": int(gate_types["Total Gates"]),
        "two_qubit_pairs": [list(pair) for pair in cz_pairs],
        "all_two_qubit_gates_touch_hub": all(STAR_HUB_QUBIT in pair for pair in cz_pairs),
        "duration_seconds": _estimate_duration_seconds(transpiled, backend),
        "op_counts": {str(k): int(v) for k, v in ops.items()},
    }


def check_g4_transpilation(
    backend,
    R: int,
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    levels: Sequence[str] = ANSATZ_LEVELS,
) -> dict:
    """Gate G4, transpilation half, at optimization_level=1 and seed_transpiler=42 only.

    Hard failure: any SWAP gate — on a star topology it means the circuit stopped being
    the one that was specified.
    Spec failure: a CZ count other than 4 per block.
    Reported, never gated: depth, duration and single-qubit gate counts. L2 is deeper
    than L1 by construction, and single-qubit gates are cheap next to a CZ.

    The optimization_level=0 run in the result is a diagnostic showing that an L1/L2
    difference is not a transpiler artefact; it enters no verdict.
    """
    per_level = {
        level: _transpiled_stats(
            level, R, backend, optimization_level=G4_OPTIMIZATION_LEVEL, n_qubits=n_qubits
        )
        for level in levels
    }
    diagnostic = {
        level: _transpiled_stats(level, R, backend, optimization_level=0, n_qubits=n_qubits)
        for level in levels
    }

    failures = []
    for level, stats in per_level.items():
        if stats["swap_count"] != 0:
            failures.append(f"{level}: {stats['swap_count']} SWAP gates (hard failure)")
        if stats["cz_count"] != CZ_PER_BLOCK * R:
            failures.append(
                f"{level}: {stats['cz_count']} CZ gates, spec says {CZ_PER_BLOCK * R} "
                f"({CZ_PER_BLOCK} per block x R={R})"
            )
        if not stats["all_two_qubit_gates_touch_hub"]:
            failures.append(f"{level}: a two-qubit gate bypasses the hub qubit {STAR_HUB_QUBIT}")

    cz_counts = {level: stats["cz_count"] for level, stats in per_level.items()}
    single_counts = {level: stats["single_qubit_gates"] for level, stats in per_level.items()}

    return {
        "gate": "G4-transpilation",
        "R": R,
        "n_qubits": n_qubits,
        "backend": _backend_label(backend),
        "optimization_level": G4_OPTIMIZATION_LEVEL,
        "seed_transpiler": G4_SEED_TRANSPILER,
        "stats_source": "self-transpiled at optimization_level=1; get_circuit_stats not used",
        "levels": per_level,
        "cz_counts_equal": len(set(cz_counts.values())) == 1,
        "single_qubit_counts_equal": len(set(single_counts.values())) == 1,
        "depths": {level: stats["depth"] for level, stats in per_level.items()},
        "durations_seconds": {
            level: stats["duration_seconds"] for level, stats in per_level.items()
        },
        "diagnostic_optimization_level_0": diagnostic,
        "diagnostic_note": (
            "optimization_level=0 numbers are a non-gating diagnostic (CONTRACTS 7a); "
            "they enter no verdict"
        ),
        "failures": failures,
        "passed": not failures,
    }


def _backend_label(backend) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name) if name is not None else type(backend).__name__


# --- G5: clbit <-> qubit mapping ---------------------------------------------------

# A mapping error moves a value by 2.0, while shot noise is ~0.02 and readout error a
# few per cent, so 0.1 separates "noisy" from "wrong" without per-session tuning.
G5_TOLERANCE = 0.1


def check_g5_bit_mapping(expectations: np.ndarray, *, tol: float = G5_TOLERANCE) -> dict:
    """Gate G5: probe circuit i must flip bit i and nothing else.

    Input is the (n, n) output of running make_probe_circuits through
    expectations_on_backend: row i holds <Z_j> for the circuit that put |1> on qubit i,
    so it must be -1 at position i and +1 everywhere else.

    Pure function, so the verdict can be tested without a machine.
    """
    values = np.asarray(expectations, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square (n_qubits, n_qubits) array, got shape {values.shape}")
    n = values.shape[0]

    flipped_per_row: list[list[int]] = []
    ambiguous_rows: list[int] = []
    for i in range(n):
        low = [j for j in range(n) if values[i, j] < -1.0 + tol]
        high = [j for j in range(n) if values[i, j] > 1.0 - tol]
        if len(low) != 1 or len(low) + len(high) != n:
            # Neither a clean flip nor a clean non-flip: the mapping cannot be read off
            # this row.
            ambiguous_rows.append(i)
        flipped_per_row.append(low)

    permutation: list[int] | None = None
    if not ambiguous_rows:
        candidate = [row[0] for row in flipped_per_row]
        if sorted(candidate) == list(range(n)):
            permutation = candidate

    identity = permutation == list(range(n))
    diagnosis = _diagnose_permutation(permutation, ambiguous_rows, n)

    return {
        "gate": "G5",
        "n_qubits": n,
        "tolerance": tol,
        "flipped_qubit_per_circuit": flipped_per_row,
        "ambiguous_rows": ambiguous_rows,
        "permutation": permutation,
        "diagnosis": diagnosis,
        "expectations": values.tolist(),
        "failures": [] if identity else [diagnosis],
        "passed": bool(identity),
    }


def _diagnose_permutation(
    permutation: list[int] | None, ambiguous_rows: list[int], n: int
) -> str:
    """Name the failure mode, so a failed G5 says what to look at."""
    if ambiguous_rows:
        return (
            f"rows {ambiguous_rows} do not show exactly one flipped qubit; the histogram "
            "is too noisy or more than one qubit moved"
        )
    if permutation is None:
        return "the flipped qubits do not form a permutation; two circuits flipped the same bit"
    if permutation == list(range(n)):
        return "identity: circuit i flips bit i"
    if permutation == list(range(n - 1, -1, -1)):
        return "bit order reversed: circuit i flips bit n-1-i (endianness)"
    shifts = {(permutation[i] - i) % n for i in range(n)}
    if len(shifts) == 1:
        return f"cyclic shift by {shifts.pop()}: circuit i flips bit (i + shift) mod n"
    return f"mapping permuted: circuit i flips bit permutation[i] = {permutation}"


# --- G2: effective dimensionality of the representation ----------------------------

# Maximum share of the retained variance any single PCA component may carry. A dataset
# that fails is one-dimensional dressed up as five, and is reported as excluded.
G2_MAX_COMPONENT_SHARE = 0.80


def check_g2_effective_dim(source, *, max_share: float = G2_MAX_COMPONENT_SHARE) -> dict:
    """Gate G2: no single PCA component explains more than 80 % of the retained variance.

    `source` is either a manifest dict from datasets.generate_and_freeze / load_manifest,
    or the explained variance ratios themselves. Shares are renormalised over the
    retained components, since scikit-learn reports them as a share of the total variance
    of the original features.

    Computed after PCA, on the frozen representation, never on the raw features.
    """
    if isinstance(source, dict):
        ratios = np.asarray(source["pca"]["explained_variance_ratio_"], dtype=float)
        label = source.get("frozen_name", source.get("generator", "manifest"))
    else:
        ratios = np.asarray(source, dtype=float).reshape(-1)
        label = "explained_variance_ratio_"

    if ratios.size == 0 or not np.all(np.isfinite(ratios)) or ratios.sum() <= 0:
        raise ValueError(f"unusable explained variance ratios: {ratios.tolist()}")

    shares = ratios / ratios.sum()
    top_index = int(np.argmax(shares))
    top_share = float(shares[top_index])
    passed = top_share <= max_share

    return {
        "gate": "G2",
        "dataset": label,
        "n_components": int(ratios.size),
        "max_share": float(max_share),
        "explained_variance_ratio_": [float(v) for v in ratios],
        "share_of_retained": [float(v) for v in shares],
        "top_component": top_index,
        "top_share": top_share,
        "total_variance_explained": float(ratios.sum()),
        "failures": (
            []
            if passed
            else [
                f"component {top_index} carries {top_share:.3f} of the retained variance, "
                f"above the {max_share:.2f} limit"
            ]
        ),
        "passed": bool(passed),
    }


# --- G1: dataset carrying capacity --------------------------------------------------

# acc(strong classical on PCA-5) - acc(arm E with a linear head) must clear this, and
# the strong model must land inside the band. A dataset that is too easy has no headroom
# above arm E, one that is too hard puts every arm on the floor.
G1_MIN_HEADROOM = 0.05
G1_STRONG_ACCURACY_BAND = (0.65, 0.90)

# SVC(rbf) grid, selected on the validation split and evaluated on test.
G1_SVC_GRID: dict[str, tuple] = {"C": (0.1, 1.0, 10.0), "gamma": ("scale", 0.1, 1.0)}

# lr grid used per cell for arm E only. Gate-local selection, not the frozen select_lr
# of the main series.
G1_LR_GRID: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)


def _accuracy_of(model_output, y) -> float:
    """Accuracy at threshold 0 on a decision score, labels in {-1,+1}."""
    from qsocket.vendored.metrics_cls import accuracy_from_z

    return accuracy_from_z(np.asarray(model_output).reshape(-1), np.asarray(y).reshape(-1))


def _as_accuracy_record(value, *, role: str) -> dict:
    """Models may return a bare accuracy or a dict carrying it plus provenance."""
    if isinstance(value, dict):
        if "accuracy" not in value:
            raise ValueError(f"{role} model returned a dict without an 'accuracy' key")
        return dict(value)
    return {"accuracy": float(value)}


def check_g1_headroom(dataset, *, strong_model, floor_model) -> dict:
    """Gate G1: acc(strong classical) - acc(floor) >= 0.05, strong inside [0.65, 0.90].

    Both models arrive as arguments — nothing about them is hardwired here, so the same
    gate serves the binding reading and any reported-only variant. Each is a callable
    taking `dataset` = {"train": (X, y), "val": (X, y), "test": (X, y)} and returning
    either an accuracy or a dict with an "accuracy" key plus whatever provenance it
    wants recorded.

    Two properties of the binding reading:

      * the floor is arm E with a linear head. With an MLP head arm E is itself a strong
        nonlinear model on the same features, so the gate would be unsatisfiable;
      * the gating version runs on PCA-5, which is the carrying capacity the gate exists
        to measure. "Strong model on the full features" is reported, never gating.

    A floor model declaring `is_contract_arm_e` makes the verdict binding; anything else
    is recorded as orientational.
    """
    strong = _as_accuracy_record(strong_model(dataset), role="strong")
    # Ceiling and gate pull in opposite directions on the same number: a high accuracy
    # is a good denominator for the ceiling but "too easy, reject" for the band, so a
    # model built for the ceiling may not gate.
    if strong.get("is_ceiling_only"):
        raise ValueError(
            f"{strong.get('label', 'this strong model')} is a CEILING reading "
            "(D-28) and cannot gate G1: acc(strong MLP) lands above the band "
            f"{list(G1_STRONG_ACCURACY_BAND)} on datasets that are perfectly good, so "
            "gating on it would reject them as 'too easy'. Gate with "
            "make_svc_strong_model(); read the ceiling with ceiling()."
        )
    floor = _as_accuracy_record(floor_model(dataset), role="floor")

    headroom = float(strong["accuracy"]) - float(floor["accuracy"])
    low, high = G1_STRONG_ACCURACY_BAND
    in_band = low <= float(strong["accuracy"]) <= high

    failures = []
    if headroom < G1_MIN_HEADROOM:
        failures.append(
            f"headroom {headroom:+.3f} is below {G1_MIN_HEADROOM:.2f}: there is nothing "
            "for a socket to add over the linear head. Read together with the band "
            "check — a small gap at a high strong accuracy means the dataset is too "
            "easy, at a low one that it is too hard"
        )
    if not in_band:
        side = "above" if float(strong["accuracy"]) > high else "below"
        failures.append(
            f"strong model accuracy {strong['accuracy']:.3f} is {side} the band "
            f"[{low:.2f}, {high:.2f}]: the dataset is too {'easy' if side == 'above' else 'hard'}"
        )

    binding = bool(floor.get("is_contract_arm_e", False))
    # g1_margin is the signed distance to the threshold, reported so a rounded headroom
    # of 0.050000 next to `passed = False` does not read as self-contradictory. Purely
    # derived; the comparison above is still what decides.
    return {
        "gate": "G1",
        "binding": binding,
        "binding_note": (
            "binding: the floor is contract arm E (identity socket + linear head)"
            if binding
            else "ORIENTATIONAL: the floor is not contract arm E, this verdict decides nothing"
        ),
        "min_headroom": G1_MIN_HEADROOM,
        "strong_accuracy_band": [low, high],
        "strong": strong,
        "floor": floor,
        "headroom": headroom,
        "g1_margin": float(headroom - G1_MIN_HEADROOM),
        "strong_in_band": bool(in_band),
        "failures": failures,
        "passed": not failures,
    }


def make_svc_strong_model(*, grid: dict | None = None, split_for_selection: str = "val"):
    """Strong classical model for G1: SVC(rbf) over `grid`, selected on validation.

    Selection on validation and evaluation on test, so the reported number is not the
    grid maximum on the test set, which would be optimistically biased and make the gate
    easier to pass the larger the grid.

    Ties go to the earliest cell in the declared order, so the choice does not depend on
    dict iteration accidents. Per-cell fit times are recorded because high C with high
    gamma on a noisy dataset is the cell that can blow up.
    """
    import time

    from sklearn.svm import SVC

    grid = dict(grid or G1_SVC_GRID)

    def strong_model(dataset) -> dict:
        X_tr, y_tr = dataset["train"]
        X_sel, y_sel = dataset[split_for_selection]
        X_te, y_te = dataset["test"]

        cells = []
        for C in grid["C"]:
            for gamma in grid["gamma"]:
                started = time.perf_counter()
                estimator = SVC(kernel="rbf", C=C, gamma=gamma).fit(X_tr, y_tr)
                fit_seconds = time.perf_counter() - started
                cells.append(
                    {
                        "C": float(C),
                        "gamma": gamma if isinstance(gamma, str) else float(gamma),
                        "selection_accuracy": float(estimator.score(X_sel, y_sel)),
                        "test_accuracy": float(estimator.score(X_te, y_te)),
                        "train_accuracy": float(estimator.score(X_tr, y_tr)),
                        "n_support": [int(v) for v in estimator.n_support_],
                        "fit_seconds": float(fit_seconds),
                    }
                )

        best = max(
            enumerate(cells), key=lambda item: (item[1]["selection_accuracy"], -item[0])
        )[1]
        return {
            "accuracy": best["test_accuracy"],
            "label": f"SVC(rbf) selected on {split_for_selection}, evaluated on test",
            "selected": {"C": best["C"], "gamma": best["gamma"]},
            "selection_split": split_for_selection,
            "selection_accuracy": best["selection_accuracy"],
            "grid": {k: [v if isinstance(v, str) else float(v) for v in vals] for k, vals in grid.items()},
            "cells": cells,
            "slowest_cell_seconds": max(cell["fit_seconds"] for cell in cells),
            "total_fit_seconds": sum(cell["fit_seconds"] for cell in cells),
        }

    return strong_model


def _arm_e_with_head(
    dataset,
    *,
    dilution: str,
    lr_grid: Sequence[float],
    seeds: Sequence[int],
    cfg=None,
    ansatz: str = "L1",
) -> dict:
    """Arm E (identity socket + `dilution` head) through training.train_model.

    One code path for two readings: dilution="linear" is the G1 floor, an MLP head is the
    ceiling reading. Sharing the function makes "the two differ only in the head"
    checkable rather than asserted.

    lr is selected per cell on the validation split, ties to the lower lr, because one lr
    chosen on a single cell would handicap the others. Separate from training.select_lr,
    which is frozen once per (dataset x dilution) across arms.

    The caller decides what the record means: no flags are attached here.
    """
    import time

    import torch

    from qsocket.head import make_head
    from qsocket.socket import make_socket
    from qsocket.training import TrainConfig, train_model

    X_tr, y_tr = dataset["train"]
    X_val, y_val = dataset["val"]
    X_te, y_te = dataset["test"]
    X_te_t = torch.as_tensor(np.asarray(X_te), dtype=torch.float32)

    started = time.perf_counter()
    table = []
    for lr in lr_grid:
        for seed in seeds:
            socket = make_socket(
                "identity", R=None, ansatz=ansatz, trainable=False, seed=seed
            )
            head = make_head(dilution, seed=seed)
            config = (
                TrainConfig(lr=lr)
                if cfg is None
                else TrainConfig(
                    lr=lr,
                    batch_size=cfg.batch_size,
                    max_epochs=cfg.max_epochs,
                    patience=cfg.patience,
                    weight_decay=cfg.weight_decay,
                )
            )
            result = train_model(
                socket, head, X_tr, y_tr, X_val, y_val, cfg=config, seed=seed
            )
            with torch.no_grad():
                logits = head(socket(X_te_t)).reshape(-1).numpy()
            table.append(
                {
                    "lr": float(lr),
                    "seed": int(seed),
                    "val_accuracy": float(result.val_accuracy),
                    "test_accuracy": _accuracy_of(logits, y_te),
                    "train_accuracy": float(result.train_accuracy),
                    "best_epoch": int(result.best_epoch),
                    "epochs_run": int(result.epochs_run),
                }
            )

    per_lr = {
        lr: [row for row in table if row["lr"] == lr] for lr in (float(v) for v in lr_grid)
    }
    mean_val = {
        lr: float(np.mean([r["val_accuracy"] for r in rows])) for lr, rows in per_lr.items()
    }
    best_lr = max(mean_val, key=lambda lr: (mean_val[lr], -lr))

    return {
        "accuracy": float(np.mean([r["test_accuracy"] for r in per_lr[best_lr]])),
        "label": f"arm E: identity socket + {dilution} head, training.train_model",
        "dilution": dilution,
        "lr_selected": float(best_lr),
        "lr_grid": [float(v) for v in lr_grid],
        "lr_selection": "per cell, on the validation split; NOT the frozen select_lr",
        "val_accuracy": mean_val[best_lr],
        "mean_val_accuracy_per_lr": {str(lr): value for lr, value in mean_val.items()},
        "seeds": [int(s) for s in seeds],
        "runs": table,
        "wall_seconds": time.perf_counter() - started,
    }


def make_arm_e_linear_floor_model(
    *,
    lr_grid: Sequence[float] = G1_LR_GRID,
    seeds: Sequence[int] = (1,),
    cfg=None,
    ansatz: str = "L1",
):
    """Floor model for G1: contract arm E — identity socket, linear head, real training loop.

    Runs the same training loop as the main series on the identity socket, so the number
    is arm E rather than a stand-in for it — which is what makes G1 binding. The work
    happens in _arm_e_with_head, shared with the ceiling reading.

    Test accuracy is read at threshold 0 on the logit, the same rule as 0.5 on the
    sigmoid.
    """

    def floor_model(dataset) -> dict:
        record = _arm_e_with_head(
            dataset,
            dilution="linear",
            lr_grid=tuple(lr_grid),
            seeds=tuple(seeds),
            cfg=cfg,
            ansatz=ansatz,
        )
        # check_g1_headroom reads this flag to call its verdict binding. Attached here
        # rather than in the shared helper: only the linear head is the contract floor.
        record["is_contract_arm_e"] = True
        return record

    return floor_model


# --- ceiling: the reference scale for delta ----------------------------------------

# Heads the MLP ceiling reading is taken over. Both are dilution levels, so the ceiling
# asks what the best classical head does on these features rather than what an
# arbitrary network does.
CEILING_MLP_DILUTIONS: tuple[str, ...] = ("mlp42", "mlp4285")

# Training seeds for both ceiling readings and the floor they are compared against.
# More than one, because a single unlucky initialisation would move every delta reported
# against this scale.
CEILING_SEEDS: tuple[int, ...] = (1, 2, 3)


def make_mlp_strong_model(
    *,
    dilutions: Sequence[str] = CEILING_MLP_DILUTIONS,
    lr_grid: Sequence[float] = G1_LR_GRID,
    seeds: Sequence[int] = CEILING_SEEDS,
    cfg=None,
    ansatz: str = "L1",
):
    """Classical MLP reading of the ceiling: best of `dilutions` on the same PCA features.

    Same chain as the G1 floor — identity socket, same training loop, per-cell lr
    selection, same seeds — so the difference from the floor is the head and nothing else.

    Not a gate: the record carries `is_ceiling_only` and check_g1_headroom refuses it,
    because a strong MLP lands above the G1 band on datasets that are perfectly good.
    """

    def strong_model(dataset) -> dict:
        readings = {
            dilution: _arm_e_with_head(
                dataset,
                dilution=dilution,
                lr_grid=tuple(lr_grid),
                seeds=tuple(seeds),
                cfg=cfg,
                ansatz=ansatz,
            )
            for dilution in dilutions
        }
        # Ties go to the earliest dilution in the declared order.
        best = max(
            enumerate(readings.items()),
            key=lambda item: (item[1][1]["accuracy"], -item[0]),
        )[1]
        which, record = best
        return {
            "accuracy": float(record["accuracy"]),
            "label": f"strong classical MLP: best of {list(dilutions)} on PCA-5 features",
            "is_ceiling_only": True,
            "which": which,
            "lr_selected": record["lr_selected"],
            "seeds": [int(v) for v in seeds],
            "per_dilution": {
                name: {
                    "accuracy": rec["accuracy"],
                    "lr_selected": rec["lr_selected"],
                    "val_accuracy": rec["val_accuracy"],
                }
                for name, rec in readings.items()
            },
            "readings": readings,
        }

    return strong_model


def ceiling(dataset, *, svc_model=None, mlp_model=None, floor_model=None) -> dict:
    """max{acc(SVM), acc(MLP)} - acc(arm E, linear head).

    `max` is taken unconditionally, including when the MLP reading is the worse of the
    two: the ceiling is the best classical model on these features, not the one that
    happened to win. Taking the better only when it flatters the result is a selection
    asymmetry that can reverse the sign of a benchmark (arXiv:2403.07059).

    Reports, never gates: `gating` is False and no verdict is computed. G1 is decided by
    check_g1_headroom on make_svc_strong_model.

    All three models default to the contract construction and can be replaced for tests
    or a cheaper diagnostic run.
    """
    svc_model = svc_model or make_svc_strong_model()
    mlp_model = mlp_model or make_mlp_strong_model()
    floor_model = floor_model or make_arm_e_linear_floor_model(seeds=CEILING_SEEDS)

    svm = _as_accuracy_record(svc_model(dataset), role="strong SVM")
    mlp = _as_accuracy_record(mlp_model(dataset), role="strong MLP")
    floor = _as_accuracy_record(floor_model(dataset), role="floor")

    strong = max(float(svm["accuracy"]), float(mlp["accuracy"]))
    which = "svm" if float(svm["accuracy"]) >= float(mlp["accuracy"]) else "mlp"
    low, high = G1_STRONG_ACCURACY_BAND

    return {
        "quantity": "ceiling",
        "gating": False,
        "gating_note": (
            "REPORTED, never gating: G1 is decided on the SVM reading "
            "(SPEC section 6, point 3). Gating on this number would reject good "
            "datasets as 'too easy'"
        ),
        "ceiling": float(strong - float(floor["accuracy"])),
        "strong_accuracy": float(strong),
        "strong_which": which,
        "svm": svm,
        "mlp": mlp,
        "floor": floor,
        # Both single-model readings, so a report can show what the max cost or bought.
        "ceiling_vs_svm_only": float(float(svm["accuracy"]) - float(floor["accuracy"])),
        "ceiling_vs_mlp_only": float(float(mlp["accuracy"]) - float(floor["accuracy"])),
        # The reason band and ceiling are separate: the ceiling numerator is routinely
        # outside the band the gate uses.
        "strong_in_g1_band": bool(low <= strong <= high),
    }
