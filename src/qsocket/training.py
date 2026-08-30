"""Training loop (simulator only; hardware runs evaluation of frozen weights).

Every value below was fixed before the experiment:

    loss        BCEWithLogitsLoss, no regularisation term, labels {-1,+1} -> {0,1}
    threshold   0.5 on the sigmoid, equivalently 0 on the logit
    optimiser   Adam, weight_decay = 0
    batch       64
    epochs      300, early stopping on validation, best weights restored
    patience    30 to start with, recalibrated from the best_epoch distribution
    lr          ONE shared lr for socket and head, in a single param_group

No batch norm, dropout, scheduler, gradient clipping or augmentation. Dropout in
particular would add randomness straight into sigma_seed, the yardstick the whole
study is measured against.

Early stopping uses the validation split only; train_model's signature does not accept
a test set, so the leak is impossible rather than merely forbidden.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from qsocket.seeding import derive

DEFAULT_SOCKET_CONVERGENCE_TOL = 1e-4


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 30
    weight_decay: float = 0.0


@dataclass
class TrainResult:
    best_epoch: int
    epochs_run: int
    train_accuracy: float
    val_accuracy: float
    val_macro_f1: float
    theta_displacement: float  # ||dtheta|| / sqrt(P), 0.0 when the socket is frozen
    grad_rms_start: float
    grad_rms_end: float
    socket_convergence_epoch: int | None
    wall_seconds: float


def to_binary_labels(y) -> torch.Tensor:
    """Map {-1, +1} to {0, 1}. Generators return +-1; BCE needs {0, 1}.

    One coding or the other, never both: {-1, 0, 1} used to pass, because every value was
    individually admissible, and -1 and 0 then collapsed into the same class. A three-class
    label vector would have trained silently as a binary one.
    """
    values = np.asarray(y).reshape(-1).astype(np.float64)
    unique = np.unique(values)
    admissible = np.all(np.isin(unique, (-1.0, 1.0))) or np.all(np.isin(unique, (0.0, 1.0)))
    if not admissible:
        raise ValueError(
            f"labels must be entirely in {{-1,+1}} or entirely in {{0,1}}, got "
            f"{unique.tolist()}. A mix of the two codings is not a binary problem: -1 and "
            "0 would both map to class 0."
        )
    return torch.as_tensor(values > 0, dtype=torch.float32)


def _as_features(X) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the per-class F1 over the two classes."""
    scores = []
    for cls in (0.0, 1.0):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        denominator = 2 * tp + fp + fn
        scores.append(1.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


class _Model(nn.Module):
    """socket -> head, one logit out."""

    def __init__(self, socket: nn.Module, head: nn.Module):
        super().__init__()
        self.socket = socket
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.socket(x)).reshape(-1)


@torch.no_grad()
def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """(accuracy, macro F1) at the 0.5 threshold on the sigmoid, i.e. 0 on the logit."""
    logits = model(X)
    predicted = (logits > 0).float()
    accuracy = float((predicted == y).float().mean())
    return accuracy, macro_f1(y.numpy(), predicted.numpy())


def batch_order_rng(seed: int) -> np.random.Generator:
    """Generator for the batch order, keyed by derive(seed) alone.

    No arm, R or dilution in the key: every arm walks the training samples in the same
    order, so no part of the paired difference is reshuffling.
    """
    return np.random.default_rng(derive(seed))


def _grad_rms(parameters: Sequence[torch.nn.Parameter]) -> float:
    """RMS gradient per parameter, pooled over every trainable tensor."""
    total, count = 0.0, 0
    for p in parameters:
        if p.grad is None:
            continue
        total += float((p.grad**2).sum())
        count += p.grad.numel()
    return float(np.sqrt(total / count)) if count else 0.0


def train_model(
    socket: nn.Module,
    head: nn.Module,
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    cfg: TrainConfig,
    seed: int,
    socket_convergence_tol: float = DEFAULT_SOCKET_CONVERGENCE_TOL,
    order_log: list[np.ndarray] | None = None,
) -> TrainResult:
    """Joint training of socket and head, one optimiser, one lr, one param_group.

    Adam normalises the step per parameter, so the gradient-scale difference between
    socket and head does not become a step-size difference and one shared lr suffices.

    There is deliberately no test-set argument.
    """
    started = time.perf_counter()

    X_tr_t, X_val_t = _as_features(X_tr), _as_features(X_val)
    y_tr_t, y_val_t = to_binary_labels(y_tr), to_binary_labels(y_val)

    model = _Model(socket, head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters: socket and head are both frozen")

    # One param_group: a separate socket lr would act on the trainable-socket arms only,
    # i.e. asymmetrically inside the A/B pair.
    optimiser = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    rng = batch_order_rng(seed)

    socket_theta = socket.theta() if hasattr(socket, "theta") else None
    socket_trainable = socket_theta is not None and any(
        p.requires_grad for p in socket.parameters()
    )
    theta_start = None if socket_theta is None else socket_theta.clone()
    theta_previous = None if theta_start is None else theta_start.clone()

    best_score: tuple[float, float] | None = None
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    grad_rms_start, grad_rms_end = 0.0, 0.0
    socket_convergence_epoch: int | None = None
    epochs_run = 0

    n = len(X_tr_t)
    for epoch in range(1, cfg.max_epochs + 1):
        epochs_run = epoch
        model.train()
        order = rng.permutation(n)
        if order_log is not None:
            order_log.append(order.copy())
        for start in range(0, n, cfg.batch_size):
            index = order[start : start + cfg.batch_size]
            optimiser.zero_grad()
            loss = criterion(model(X_tr_t[index]), y_tr_t[index])
            loss.backward()
            batch_grad_rms = _grad_rms(trainable)
            if epoch == 1 and start == 0:
                grad_rms_start = batch_grad_rms
            grad_rms_end = batch_grad_rms
            optimiser.step()

        if socket_trainable:
            theta_now = socket.theta().clone()
            step = float(torch.linalg.vector_norm(theta_now - theta_previous))
            if socket_convergence_epoch is None and step < socket_convergence_tol:
                socket_convergence_epoch = epoch
            theta_previous = theta_now

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_t), y_val_t))
        val_accuracy, _ = _evaluate(model, X_val_t, y_val_t)
        score = (val_accuracy, -val_loss)

        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= cfg.patience:
            break

    # Restoring the best weights makes an over-generous patience cost time only.
    model.load_state_dict(best_state)
    model.eval()

    train_accuracy, _ = _evaluate(model, X_tr_t, y_tr_t)
    val_accuracy, val_f1 = _evaluate(model, X_val_t, y_val_t)

    if theta_start is None or not socket_trainable:
        theta_displacement = 0.0
    else:
        theta_final = socket.theta()
        theta_displacement = float(
            torch.linalg.vector_norm(theta_final - theta_start) / np.sqrt(theta_start.numel())
        )

    return TrainResult(
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        val_macro_f1=val_f1,
        theta_displacement=theta_displacement,
        grad_rms_start=grad_rms_start,
        grad_rms_end=grad_rms_end,
        socket_convergence_epoch=None if not socket_trainable else socket_convergence_epoch,
        wall_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class LrSelection:
    """Everything select_lr computed.

    Keeping the per-arm accuracies costs nothing and lets the selection be recomputed
    over a different set of arms without repeating the series.

    mean_by_lr        lr -> mean validation accuracy over selection_arms x seeds
                      (this is the criterion, and the only thing the argmax reads)
    by_lr_arm         (lr, arm) -> mean over seeds
    by_lr_arm_seed    (lr, arm, seed) -> the single number that was measured
    """

    best: float
    grid: tuple[float, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    selection_arms: tuple[str, ...]
    mean_by_lr: dict[float, float]
    by_lr_arm: dict[tuple[float, str], float]
    by_lr_arm_seed: dict[tuple[float, str, int], float]

    def rows(self) -> list[dict]:
        """One row per (lr, arm, seed) — the shape the driver writes to CSV."""
        return [
            {
                "lr": float(lr),
                "arm": arm,
                "seed": int(seed),
                "val_accuracy": float(accuracy),
                "in_selection": arm in self.selection_arms,
                "lr_selected": float(self.best),
            }
            for (lr, arm, seed), accuracy in self.by_lr_arm_seed.items()
        ]


def select_lr(
    build_arms: Callable[[int], dict[str, tuple[nn.Module, nn.Module]]],
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    grid: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2),
    seeds: Sequence[int] = (1, 2, 3),
    cfg: TrainConfig | None = None,
    selection_arms: Sequence[str] | None = None,
    return_table: bool = False,
    return_detail: bool = False,
):
    """One lr per (dataset x dilution x ansatz), never per seed and never per arm.

    build_arms(seed) returns {arm_name: (socket, head)} — freshly built every call, so
    each (lr, seed) starts from the contract initialisation rather than from wherever
    the previous fit ended.

    Criterion: the best mean validation accuracy averaged over `selection_arms` at the
    given seeds; ties go to the lower lr. Choosing per seed would make every delta
    compare different hyperparameters. selection_arms defaults to every arm build_arms
    returns.

    Returns the lr; with return_table=True the pair (lr, {lr: mean accuracy}); with
    return_detail=True the pair (lr, LrSelection), which also carries the per-arm and
    per-(arm, seed) numbers. Requesting both raises, because the two payloads differ.
    """
    if return_table and return_detail:
        raise ValueError("ask for return_table or return_detail, not both")

    grid = tuple(float(lr) for lr in grid)
    seeds = tuple(int(seed) for seed in seeds)
    if not grid:
        raise ValueError("the lr grid is empty")
    if not seeds:
        raise ValueError("no seeds to select on")

    template = cfg if cfg is not None else TrainConfig(lr=grid[0])
    per_lr_arm_seed: dict[tuple[float, str, int], float] = {}
    arm_names: tuple[str, ...] | None = None

    for lr in grid:
        for seed in seeds:
            arms = build_arms(seed)
            if not arms:
                raise ValueError("build_arms returned no arms")
            names = tuple(arms)
            if arm_names is None:
                arm_names = names
            elif names != arm_names:
                raise ValueError(
                    f"build_arms returned {names} at (lr={lr}, seed={seed}) but {arm_names} "
                    "earlier; the selection average would be over a moving set of arms"
                )
            for arm, (socket, head) in arms.items():
                result = train_model(
                    socket,
                    head,
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    cfg=TrainConfig(
                        lr=lr,
                        batch_size=template.batch_size,
                        max_epochs=template.max_epochs,
                        patience=template.patience,
                        weight_decay=template.weight_decay,
                    ),
                    seed=seed,
                )
                per_lr_arm_seed[(lr, arm, seed)] = result.val_accuracy

    assert arm_names is not None
    selection = lr_selection_from_measurements(
        per_lr_arm_seed,
        grid=grid,
        seeds=seeds,
        arms=arm_names,
        selection_arms=selection_arms,
    )
    if return_detail:
        return selection.best, selection
    return (selection.best, selection.mean_by_lr) if return_table else selection.best


def lr_selection_from_measurements(
    by_lr_arm_seed: dict[tuple[float, str, int], float],
    *,
    grid: Sequence[float],
    seeds: Sequence[int],
    arms: Sequence[str],
    selection_arms: Sequence[str] | None = None,
) -> LrSelection:
    """The selection rule, applied to accuracies that already exist.

    select_lr measures and then calls this; a driver that measured the same cells in
    parallel calls it directly, so the two paths cannot drift apart.
    Criterion: mean validation accuracy over selection_arms x seeds, ties to the lower lr.
    """
    grid = tuple(float(lr) for lr in grid)
    seeds = tuple(int(seed) for seed in seeds)
    arms = tuple(arms)
    chosen = arms if selection_arms is None else tuple(selection_arms)

    unknown = [arm for arm in chosen if arm not in arms]
    if unknown:
        raise ValueError(f"selection_arms {unknown} are not among the measured arms {arms}")
    if not chosen:
        raise ValueError("selection_arms is empty; there would be nothing to average over")
    missing = [
        key
        for key in ((lr, arm, seed) for lr in grid for arm in chosen for seed in seeds)
        if key not in by_lr_arm_seed
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} (lr, arm, seed) cells of the selection are missing, e.g. "
            f"{missing[:3]}; selecting on a partial grid would compare different budgets"
        )

    by_lr_arm = {
        (lr, arm): float(np.mean([by_lr_arm_seed[(lr, arm, seed)] for seed in seeds]))
        for lr in grid
        for arm in arms
        if all((lr, arm, seed) in by_lr_arm_seed for seed in seeds)
    }
    # Averaged over (arm, seed) jointly: equal to the mean of per-arm means when every
    # arm has the same seeds, and still correct when it does not.
    table = {
        lr: float(np.mean([by_lr_arm_seed[(lr, arm, seed)] for arm in chosen for seed in seeds]))
        for lr in grid
    }
    return LrSelection(
        best=max(table, key=lambda lr: (table[lr], -lr)),
        grid=grid,
        seeds=seeds,
        arms=arms,
        selection_arms=chosen,
        mean_by_lr=table,
        by_lr_arm=by_lr_arm,
        by_lr_arm_seed=dict(by_lr_arm_seed),
    )


# Alpha grid of the closed-form readout control.
CONTRACT_RIDGE_ALPHA_GRID: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)


def _pm_one(y) -> np.ndarray:
    """Labels as {-1,+1}: ridge is a least-squares fit, so the symmetric coding is the
    natural target and the decision threshold is 0."""
    return np.where(np.asarray(y).reshape(-1) > 0, 1.0, -1.0)


def ridge_weights(features, y, *, alpha: float) -> np.ndarray:
    """(X'X + alpha I)^-1 X'y with an explicit intercept column, left unpenalised to
    match the bias of the trained head."""
    Phi = np.asarray(features, dtype=np.float64)
    A = np.hstack([Phi, np.ones((len(Phi), 1))])
    penalty = alpha * np.eye(A.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + penalty, A.T @ _pm_one(y))


def ridge_accuracy(weights: np.ndarray, features, y) -> float:
    """Accuracy of the closed-form readout at threshold 0 on the score."""
    Phi = np.asarray(features, dtype=np.float64)
    A = np.hstack([Phi, np.ones((len(Phi), 1))])
    return float(np.mean(np.where(A @ weights > 0, 1.0, -1.0) == _pm_one(y)))


def ridge_control(
    features_tr,
    y_tr,
    features_val,
    y_val,
    *,
    grid: Sequence[float] = CONTRACT_RIDGE_ALPHA_GRID,
    evaluation: dict[str, tuple] | None = None,
) -> dict:
    """Closed-form readout control: fit on train, pick alpha on validation, report
    accuracy per split.

    Frozen sockets only (arms B, E, D_matched, D_best). A trained socket's features
    change at every step, so the closed form has no defined argument there.

    The result is reported beside the Adam-trained head and never used to replace it:
    the gap between the two objectives is a finding.
    """
    grid = tuple(float(a) for a in grid)
    if not grid:
        raise ValueError("the ridge alpha grid is empty")

    fits = {alpha: ridge_weights(features_tr, y_tr, alpha=alpha) for alpha in grid}
    val_accuracy = {alpha: ridge_accuracy(w, features_val, y_val) for alpha, w in fits.items()}
    alpha = max(grid, key=lambda a: (val_accuracy[a], -a))

    accuracy_by_split = {"val": val_accuracy[alpha]}
    for split, (features, y) in (evaluation or {}).items():
        accuracy_by_split[split] = ridge_accuracy(fits[alpha], features, y)
    return {
        "alpha_selected": float(alpha),
        "alpha_grid": [float(a) for a in grid],
        "val_accuracy_per_alpha": {float(a): float(v) for a, v in val_accuracy.items()},
        "accuracy": accuracy_by_split,
    }
