"""Signal-to-noise ratio of the socket gradient at initialisation.

Neither of the obvious quantities tests a barren plateau. theta_displacement does not,
because Adam normalises per parameter, so theta moves for any lr > 0, flat plateau
included. The gradient norm does not either, for the same reason: Adam divides by the
spread across batches, so a small consistent gradient gives a normal step while a large
sign-flipping one gives a step near zero. What does tell them apart:

    SNR_i = |mean_b(g_ib)| / std_b(g_ib),   std with ddof=1, b = 1..8 batches

Measured at the initialisation point of arm A, linear head, L1, R=2, frozen production
dataset. No optimiser step is taken — the model is bit-for-bit untouched afterwards, and
that is asserted, not commented.

Both thresholds are recomputed (per-parameter from the t distribution, per-block from the
binomial null) and compared against their declared values, so a mismatch stops the run
instead of redefining the test.

Asserted scope: train split only, so test rows are never read; socket block =
socket_param_count(5, 2) and head block = 6; the two blocks partition model.parameters()
— state_dict() would be wrong, because TorchConnector registers the socket weight twice.

    python scripts/probe_gradient_snr.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

# Before numpy and torch: the BLAS pools read these at import and never again.
from qsocket.core import pin_blas_threads

pin_blas_threads()

import numpy as np
import torch
from scipy.stats import binom, t
from torch import nn

from qsocket.ansatzes import socket_param_count
from qsocket.datasets import (
    DEFAULT_DATA_DIR,
    PRODUCTION_DATASET,
    SPLIT_SIZES,
    load_frozen,
    verify_frozen_identity,
)
from qsocket.head import HEAD_PARAM_COUNTS, make_head
from qsocket.socket import DEFAULT_BACKEND, DEFAULT_N_QUBITS, make_socket
from qsocket.training import batch_order_rng, to_binary_labels

# --- configuration ------------------------------------------------------------------

# The production dataset, loaded through datasets.load_frozen so the file, content and
# PCA digests are all re-verified on the way in. Its hashes are asserted against the
# frozen-dataset registry before the first gradient is taken.
DATASET = PRODUCTION_DATASET

# The only split this probe may touch. Not a parameter: it is the no-leak guarantee.
TRAIN_SPLIT = "train"

SEEDS: tuple[int, ...] = tuple(range(1, 11))
N_BATCHES = 8
BATCH_SIZE = 64  # CONTRACTS section 7.1
R = 2  # SPEC section 3.1, part A
ANSATZ = "L1"
DILUTION = "linear"
N_QUBITS = DEFAULT_N_QUBITS

# Two-sided significance level of the per-parameter test "mean gradient = 0", and the
# same alpha reused as the per-parameter false-positive rate of the binomial null.
ALPHA = 0.05

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "p1_gradient_snr"
PER_PARAMETER_CSV = "p1_per_parameter.csv"
SUMMARY_CSV = "p1_summary.csv"

BLOCKS = ("socket", "head")

# Declared in advance; recomputed below and compared against these. A mismatch is a
# stop-and-report condition, not something to adjust.
BRIEF_SNR_THRESHOLD = 0.836
BRIEF_BLOCK_HITS = 5
BRIEF_BLOCK_TAIL_PROBABILITY = 0.029

# 17 significant digits round-trips a float64 exactly, so two runs that compute the same
# numbers produce byte-identical CSVs and the determinism check is a file hash.
_FLOAT_FORMAT = "{:.17g}"


# --- thresholds, recomputed --------------------------------------------------------


def parameter_snr_threshold(n_batches: int = N_BATCHES, alpha: float = ALPHA) -> float:
    """SNR above which a single parameter's mean gradient differs from 0.

    |mean| / std > t_crit / sqrt(n) is the one-sample two-sided t test at level alpha
    rearranged, so the threshold depends only on the number of batches.
    """
    return float(t.ppf(1.0 - alpha / 2.0, n_batches - 1) / np.sqrt(n_batches))


PARAMETER_SNR_THRESHOLD_CALL = "t.ppf(0.975, n_batches - 1) / sqrt(n_batches)"


def block_hit_threshold(n_params: int, alpha: float = ALPHA) -> tuple[int, float, float]:
    """How many parameters of a block must clear the per-parameter threshold.

    Under the null "no parameter has signal" each of the n_params tests fires with
    probability alpha independently, so the number of hits is Binom(n_params, alpha).
    The threshold is the smallest k whose upper tail is still below alpha.

    Returns (k, P(X >= k), P(X >= k-1)); the second tail is returned so the report can
    show that k-1 would NOT have been significant.
    """
    for k in range(1, n_params + 1):
        tail = float(binom.sf(k - 1, n_params, alpha))
        if tail < alpha:
            return k, tail, float(binom.sf(k - 2, n_params, alpha))
    raise ValueError(f"no k <= {n_params} reaches tail probability below {alpha}")


BLOCK_HIT_THRESHOLD_CALL = "min k such that binom.sf(k - 1, n_params, 0.05) < 0.05"


# --- data and batches ----------------------------------------------------------------


def load_training_split(dataset: str, data_dir) -> tuple[np.ndarray, np.ndarray]:
    """The 4200 training rows, and only those.

    load_splits would pull all three splits including the test rows, which this
    measurement must never read. The split name is a module constant rather than an
    argument so no call site can widen it.
    """
    assert TRAIN_SPLIT == "train", "the gradient-SNR probe reads the training split and nothing else"
    X, y = load_frozen(dataset, TRAIN_SPLIT, out_dir=data_dir)
    assert len(X) == SPLIT_SIZES[TRAIN_SPLIT], (
        f"{dataset} train split has {len(X)} rows, expected {SPLIT_SIZES[TRAIN_SPLIT]}"
    )
    assert len(X) == len(y)
    return X, y


def batch_indices(
    seed: int, n_rows: int, *, n_batches: int = N_BATCHES, batch_size: int = BATCH_SIZE
) -> list[np.ndarray]:
    """The first `n_batches` batches of epoch 1, exactly as train_model would draw them.

    train_model permutes the training set once per epoch with batch_order_rng(seed) and
    walks it in contiguous slices, which is reproduced here so the measured gradient is
    the one the model would actually have seen.
    """
    if n_rows < n_batches * batch_size:
        raise ValueError(
            f"{n_rows} rows cannot supply {n_batches} batches of {batch_size}"
        )
    order = batch_order_rng(seed).permutation(n_rows)
    return [order[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]


# --- model ---------------------------------------------------------------------------


class _Model(nn.Module):
    """socket -> head, one logit out.

    A local copy of training._Model, private there. Identical by design: the gradient
    measured here has to be the one the training loop would compute.
    """

    def __init__(self, socket: nn.Module, head: nn.Module):
        super().__init__()
        self.socket = socket
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.socket(x)).reshape(-1)


def _flat_gradient(parameters: list[torch.nn.Parameter]) -> np.ndarray:
    """One float64 vector of the current .grad of `parameters`, in parameters() order."""
    pieces = []
    for p in parameters:
        assert p.grad is not None, "a trainable parameter received no gradient"
        pieces.append(p.grad.detach().reshape(-1).to(torch.float64).numpy())
    return np.concatenate(pieces)


def measure_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ansatz: str = ANSATZ,
    R_blocks: int = R,
    dilution: str = DILUTION,
    backend: str = DEFAULT_BACKEND,
    n_batches: int = N_BATCHES,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Raw per-batch gradients of one seed at the initialisation point.

    Built through make_socket / make_head so the derive keys match the main series;
    gradients are zeroed between batches and no optimiser step is taken, so the model is
    identical before and after.
    """
    socket = make_socket(
        "quantum",
        R=R_blocks,
        ansatz=ansatz,
        trainable=True,
        seed=seed,
        backend=backend,
    )
    head = make_head(dilution, seed=seed)
    model = _Model(socket, head)

    socket_params = list(socket.parameters())
    head_params = list(head.parameters())
    socket_n = sum(p.numel() for p in socket_params)
    head_n = sum(p.numel() for p in head_params)

    expected_socket = socket_param_count(N_QUBITS, R_blocks)
    expected_head = HEAD_PARAM_COUNTS[dilution]
    assert socket_n == expected_socket, (
        f"socket block has {socket_n} parameters, socket_param_count({N_QUBITS}, "
        f"{R_blocks}) says {expected_socket}"
    )
    assert head_n == expected_head, (
        f"head block has {head_n} parameters, HEAD_PARAM_COUNTS[{dilution!r}] says "
        f"{expected_head}"
    )
    # parameters(): TorchConnector registers the socket weight under
    # two keys, so state_dict would count the socket twice.
    model_n = sum(p.numel() for p in model.parameters())
    assert socket_n + head_n == model_n, (
        f"blocks {socket_n} + {head_n} do not partition the model's {model_n} parameters"
    )
    assert all(p.requires_grad for p in socket_params + head_params)

    theta_before = socket.theta().clone()

    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_t = to_binary_labels(y)
    criterion = nn.BCEWithLogitsLoss()

    batches = batch_indices(seed, len(X_t), n_batches=n_batches, batch_size=batch_size)
    gradients = {block: [] for block in BLOCKS}
    losses = []
    for index in batches:
        # Zero between batches
        model.zero_grad(set_to_none=True)
        loss = criterion(model(X_t[index]), y_t[index])
        loss.backward()
        gradients["socket"].append(_flat_gradient(socket_params))
        gradients["head"].append(_flat_gradient(head_params))
        losses.append(float(loss.detach()))

    theta_after = socket.theta()
    assert torch.equal(theta_before, theta_after), (
        "theta changed during the measurement — an optimiser step was taken somewhere"
    )

    return {
        "seed": seed,
        "backend": backend,
        "ansatz": ansatz,
        "R": R_blocks,
        "dilution": dilution,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "batch_indices": batches,
        "losses": losses,
        "gradients": {block: np.vstack(gradients[block]) for block in BLOCKS},
    }


def parameter_statistics(gradients: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """mean, std (ddof=1) and SNR per parameter, over the batch axis.

    gradients has shape (n_batches, n_params). A parameter whose gradient is constant
    across batches gets std = 0 and SNR = +inf, which is the correct verdict for a
    perfectly consistent non-zero gradient and is reported rather than patched with an
    epsilon.
    """
    mean = gradients.mean(axis=0)
    std = gradients.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(mean) / std
    # 0/0 — a parameter with a gradient that is identically zero has no signal.
    snr = np.where((std == 0.0) & (mean == 0.0), 0.0, snr)
    return {"mean": mean, "std": std, "snr": snr, "above": snr > threshold}


def grad_rms(gradients: np.ndarray) -> tuple[float, float]:
    """(RMS over the first batch, RMS over all batches), per parameter.
    Two numbers: the results row logs the pooled RMS of the first batch, which is the
    directly comparable one, while the all-batch value is the more stable estimate.
    Neither is derived from the other.
    """
    return (
        float(np.sqrt(np.mean(gradients[0] ** 2))),
        float(np.sqrt(np.mean(gradients**2))),
    )


def summarise_seed(measurement: dict, snr_threshold: float, block_hits: int) -> dict:
    """Per-seed aggregates: SNR median/min/max, hits, and grad_rms per block."""
    row: dict = {
        "seed": measurement["seed"],
        "backend": measurement["backend"],
        "ansatz": measurement["ansatz"],
        "R": measurement["R"],
        "dilution": measurement["dilution"],
        "n_batches": measurement["n_batches"],
        "batch_size": measurement["batch_size"],
        "snr_threshold": snr_threshold,
        "block_hit_threshold": block_hits,
        "loss_first_batch": measurement["losses"][0],
    }
    for block in BLOCKS:
        gradients = measurement["gradients"][block]
        statistics = parameter_statistics(gradients, snr_threshold)
        first_rms, all_rms = grad_rms(gradients)
        n_above = int(np.count_nonzero(statistics["above"]))
        row[f"{block}_n_params"] = gradients.shape[1]
        row[f"{block}_snr_median"] = float(np.median(statistics["snr"]))
        row[f"{block}_snr_min"] = float(np.min(statistics["snr"]))
        row[f"{block}_snr_max"] = float(np.max(statistics["snr"]))
        row[f"{block}_n_above_threshold"] = n_above
        row[f"{block}_grad_rms_first_batch"] = first_rms
        row[f"{block}_grad_rms_all_batches"] = all_rms
    # The block verdict applies to the socket; the head is the reference.
    row["socket_has_signal"] = row["socket_n_above_threshold"] >= block_hits
    row["head_has_signal"] = row["head_n_above_threshold"] >= block_hit_threshold(
        row["head_n_params"]
    )[0]
    socket_median = row["socket_snr_median"]
    row["snr_ratio_head_over_socket"] = (
        float("inf") if socket_median == 0.0 else row["head_snr_median"] / socket_median
    )
    return row


def per_parameter_rows(measurement: dict, snr_threshold: float) -> list[dict]:
    rows = []
    for block in BLOCKS:
        gradients = measurement["gradients"][block]
        statistics = parameter_statistics(gradients, snr_threshold)
        for index in range(gradients.shape[1]):
            rows.append(
                {
                    "seed": measurement["seed"],
                    "block": block,
                    "param_index": index,
                    "mean": float(statistics["mean"][index]),
                    "std": float(statistics["std"][index]),
                    "snr": float(statistics["snr"][index]),
                    "above_threshold": bool(statistics["above"][index]),
                }
            )
    return rows


def _cell(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return _FLOAT_FORMAT.format(value)
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> str:
    """Write rows and return the sha256 of the file, which is the determinism witness."""
    if not rows:
        raise ValueError(f"refusing to write an empty {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(rows[0]))
        for row in rows:
            writer.writerow([_cell(row[key]) for key in rows[0]])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value: float, width: int = 9, digits: int = 4) -> str:
    return f"{value:{width}.{digits}f}" if np.isfinite(value) else f"{'inf':>{width}}"


def report(summaries: list[dict], snr_threshold: float, block_hits: int) -> None:
    print("=" * 100)
    print("the gradient-SNR probe — gradient SNR of the socket at initialisation (arm A, L1, R=2, linear head)")
    print("=" * 100)
    print(f"per-parameter threshold  SNR > {snr_threshold:.6f}")
    print(f"    {PARAMETER_SNR_THRESHOLD_CALL} = t.ppf(0.975, 7) / sqrt(8)")
    k, tail, previous_tail = block_hit_threshold(summaries[0]["socket_n_params"])
    print(f"per-block threshold      >= {k} of {summaries[0]['socket_n_params']} parameters")
    print(f"    {BLOCK_HIT_THRESHOLD_CALL}")
    print(
        f"    binom.sf({k - 1}, {summaries[0]['socket_n_params']}, 0.05) = {tail:.6f} < 0.05"
        f"   ·   binom.sf({k - 2}, {summaries[0]['socket_n_params']}, 0.05)"
        f" = {previous_tail:.6f} >= 0.05"
    )
    print()

    header = (
        f"{'seed':>4s} | {'sock med':>9s} {'sock min':>9s} {'sock>thr':>8s} "
        f"{'sock rms1':>10s} {'sock rmsA':>10s} | {'head med':>9s} {'head min':>9s} "
        f"{'head>thr':>8s} {'head rms1':>10s} {'head rmsA':>10s} | {'head/sock':>9s}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{row['seed']:4d} | {_fmt(row['socket_snr_median'])} "
            f"{_fmt(row['socket_snr_min'])} {row['socket_n_above_threshold']:8d} "
            f"{row['socket_grad_rms_first_batch']:10.3e} "
            f"{row['socket_grad_rms_all_batches']:10.3e} | "
            f"{_fmt(row['head_snr_median'])} {_fmt(row['head_snr_min'])} "
            f"{row['head_n_above_threshold']:8d} "
            f"{row['head_grad_rms_first_batch']:10.3e} "
            f"{row['head_grad_rms_all_batches']:10.3e} | "
            f"{_fmt(row['snr_ratio_head_over_socket'])}"
        )
    print()

    print("across seeds — median [min, max]")
    for label, key in (
        ("socket SNR median", "socket_snr_median"),
        ("socket SNR min", "socket_snr_min"),
        ("socket params > threshold", "socket_n_above_threshold"),
        ("socket grad_rms first batch", "socket_grad_rms_first_batch"),
        ("socket grad_rms all batches", "socket_grad_rms_all_batches"),
        ("head SNR median", "head_snr_median"),
        ("head SNR min", "head_snr_min"),
        ("head params > threshold", "head_n_above_threshold"),
        ("head grad_rms first batch", "head_grad_rms_first_batch"),
        ("head grad_rms all batches", "head_grad_rms_all_batches"),
        ("head/socket SNR median ratio", "snr_ratio_head_over_socket"),
    ):
        values = np.array([float(row[key]) for row in summaries])
        print(
            f"  {label:30s} {np.median(values):12.4g}  "
            f"[{values.min():.4g}, {values.max():.4g}]"
        )
    print()

    passing = [row["seed"] for row in summaries if row["socket_has_signal"]]
    print(
        f"socket clears '>= {block_hits} of {summaries[0]['socket_n_params']}' on "
        f"{len(passing)} of {len(summaries)} seeds: {passing}"
    )
    head_passing = [row["seed"] for row in summaries if row["head_has_signal"]]
    print(
        f"head clears its own block threshold on {len(head_passing)} of "
        f"{len(summaries)} seeds: {head_passing}"
    )
    print()
    print(verdict(summaries, block_hits))


# Factor separating the last two verdict rows.
ORDER_OF_MAGNITUDE = 10.0


def verdict(summaries: list[dict], block_hits: int) -> str:
    n_passing = sum(1 for row in summaries if row["socket_has_signal"])
    majority = n_passing > len(summaries) / 2
    ratios = np.array([row["snr_ratio_head_over_socket"] for row in summaries])
    median_ratio = float(np.median(ratios))

    lines = ["VERDICT (thresholds declared before the measurement)"]
    if not majority:
        lines.append(
            f"  row 1 — the socket gradient is noise: it clears the block threshold on "
            f"{n_passing} of {len(summaries)} seeds, i.e. not on a majority."
        )
        lines.append(
            "  Consequence: the lr-curve probe is dropped. No lr helps; the project follows the "
            "'optimisation failure' path of SPEC section 9."
        )
    elif median_ratio >= ORDER_OF_MAGNITUDE:
        lines.append(
            f"  row 2 — the socket clears the threshold ({n_passing}/{len(summaries)} "
            f"seeds) but its SNR is an order of magnitude below the head's "
            f"(median head/socket ratio {median_ratio:.3g} >= {ORDER_OF_MAGNITUDE:g})."
        )
        lines.append(
            "  Consequence: the lr-curve probe is needed. A split between the lr optima of A and E is "
            "predicted and guard the lr rule is necessary."
        )
    else:
        lines.append(
            f"  row 3 — socket and head SNR are comparable "
            f"({n_passing}/{len(summaries)} seeds clear the threshold, median "
            f"head/socket ratio {median_ratio:.3g} < {ORDER_OF_MAGNITUDE:g}); nothing "
            "pathological."
        )
        lines.append(
            "  Consequence: the lr-curve probe is needed, but the grid may be trimmed with an a priori "
            "argument."
        )
    return "\n".join(lines)


def run(
    *,
    seeds=SEEDS,
    dataset: str = DATASET,
    data_dir=DEFAULT_DATA_DIR,
    out_dir=DEFAULT_OUT_DIR,
    ansatz: str = ANSATZ,
    R_blocks: int = R,
    dilution: str = DILUTION,
    backend: str = DEFAULT_BACKEND,
    n_batches: int = N_BATCHES,
    batch_size: int = BATCH_SIZE,
    write: bool = True,
) -> dict:
    snr_threshold = parameter_snr_threshold(n_batches)
    block_hits, tail, _ = block_hit_threshold(socket_param_count(N_QUBITS, R_blocks))

    if n_batches == N_BATCHES:
        assert abs(snr_threshold - BRIEF_SNR_THRESHOLD) < 5e-4, (
            f"recomputed per-parameter threshold {snr_threshold} disagrees with the "
            f"{BRIEF_SNR_THRESHOLD} declared before the measurement"
        )
    if R_blocks == R:
        assert block_hits == BRIEF_BLOCK_HITS, (
            f"recomputed block threshold {block_hits} disagrees with the "
            f"{BRIEF_BLOCK_HITS} declared before the measurement"
        )
        assert abs(tail - BRIEF_BLOCK_TAIL_PROBABILITY) < 5e-4, (
            f"recomputed tail probability {tail} disagrees with the "
            f"{BRIEF_BLOCK_TAIL_PROBABILITY} declared before the measurement"
        )

    manifest = verify_frozen_identity(dataset, out_dir=data_dir)
    X, y = load_training_split(dataset, data_dir)

    per_parameter: list[dict] = []
    summaries: list[dict] = []
    for seed in seeds:
        measurement = measure_seed(
            seed,
            X,
            y,
            ansatz=ansatz,
            R_blocks=R_blocks,
            dilution=dilution,
            backend=backend,
            n_batches=n_batches,
            batch_size=batch_size,
        )
        per_parameter.extend(per_parameter_rows(measurement, snr_threshold))
        summaries.append(summarise_seed(measurement, snr_threshold, block_hits))

    result = {
        "dataset": dataset,
        "manifest": manifest,
        "snr_threshold": snr_threshold,
        "block_hit_threshold": block_hits,
        "per_parameter": per_parameter,
        "summaries": summaries,
    }

    if write:
        out_dir = Path(out_dir)
        result["per_parameter_sha256"] = write_csv(
            out_dir / PER_PARAMETER_CSV, per_parameter
        )
        result["summary_sha256"] = write_csv(out_dir / SUMMARY_CSV, summaries)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ansatz", default=ANSATZ)
    parser.add_argument("--R", type=int, default=R)
    parser.add_argument("--dilution", default=DILUTION)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=("pennylane", "qiskit"))
    parser.add_argument("--n-batches", type=int, default=N_BATCHES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        dataset=args.dataset,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        ansatz=args.ansatz,
        R_blocks=args.R,
        dilution=args.dilution,
        backend=args.backend,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
    )

    print(f"dataset {args.dataset}  ·  split {TRAIN_SPLIT} only  ·  backend {args.backend}")
    print(f"  dataset_hash {result['manifest']['dataset_hash']}")
    print(f"  pca_hash     {result['manifest']['pca_hash']}")
    print()
    report(result["summaries"], result["snr_threshold"], result["block_hit_threshold"])
    print()
    print(f"{Path(args.out_dir) / PER_PARAMETER_CSV}  sha256 {result['per_parameter_sha256']}")
    print(f"{Path(args.out_dir) / SUMMARY_CSV}  sha256 {result['summary_sha256']}")


if __name__ == "__main__":
    main()
