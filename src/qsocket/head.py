"""Classifier head: linear (5->1) / MLP h = 2, 4, 42, 4285 (last one finally not used).

Architecture is fixed: one hidden layer, ReLU, biases everywhere, a single logit out,
and BCEWithLogitsLoss applied outside the module. Every dilution level is trained with
the same optimiser and loss, otherwise the dilution axis would mix capacity with how
the model is fitted. Parameter counts follow from 5+1 and 7h+1.

h42 and mlp42 are the same head under two names; the alias is enforced by
canonicalising the initialisation key, and asserted in tests.

Initialisation is keyed by derive(seed, dilution) with no arm in it, drawn from a
dedicated torch.Generator rather than the global RNG: with torch.manual_seed the arm
constructed second would consume a different stretch of the global stream and the two
arms would not share a head, breaking the pairing this file exists for.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from qsocket.core import derive

Dilution = Literal["linear", "h2", "h4", "h42", "mlp42", "mlp4285"]

# 5 + 1 for the linear head, 7h + 1 for the MLPs
HEAD_PARAM_COUNTS: dict[str, int] = {
    "linear": 6,
    "h2": 15,
    "h3": 22,
    "h4": 29,
    "h42": 295,
    "mlp42": 295,
    "mlp4285": 29996,
}

HIDDEN_WIDTHS: dict[str, int] = {
    "h2": 2, "h3": 3, "h4": 4, "h42": 42, "mlp42": 42, "mlp4285": 4285,
}

# The four points of the main-series dilution axis, in increasing capacity.
# mlp42, mlp4285 and h3 are reachable names but are not on the axis: mlp42 is h42 under
# its historical name, and mlp4285 added 0.0006 over h42, below the binomial SE.

DILUTION_AXIS: tuple[str, ...] = ("linear", "h2", "h4", "h42")

# Names that must build the same head. The initialisation key is derive(seed, dilution),
# so without this alias h42 and mlp42 would be two different draws with the same
# parameter count.
HEAD_NAME_ALIASES: dict[str, str] = {"h42": "mlp42"}

SOCKET_WIDTH = 5


def canonical_head_name(dilution: str) -> str:
    """The name the initialisation key is derived from, so aliases share their weights."""
    return HEAD_NAME_ALIASES.get(dilution, dilution)


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """torch's default nn.Linear initialisation, but drawn from `generator`."""
    fan_in = layer.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


def make_head(dilution: Dilution, *, seed: int) -> nn.Module:
    """Head for one dilution level.

    linear: Linear(5, 1) with bias.
    h*/mlp*: Linear(5, h) -> ReLU -> Linear(h, 1), both with bias.
    """
    if dilution not in HEAD_PARAM_COUNTS:
        raise ValueError(f"unknown dilution {dilution!r}; expected one of {sorted(HEAD_PARAM_COUNTS)}")

    generator = torch.Generator()
    # torch.Generator.manual_seed takes a signed 64-bit value; derive returns 64
    # unsigned bits, so the top bit is folded away.
    generator.manual_seed(derive(seed, canonical_head_name(dilution)) % (2**63))

    if dilution == "linear":
        layer = nn.Linear(SOCKET_WIDTH, 1, bias=True)
        _init_linear(layer, generator)
        head: nn.Module = nn.Sequential(layer)
    else:
        hidden = HIDDEN_WIDTHS[dilution]
        first = nn.Linear(SOCKET_WIDTH, hidden, bias=True)
        second = nn.Linear(hidden, 1, bias=True)
        _init_linear(first, generator)
        _init_linear(second, generator)
        head = nn.Sequential(first, nn.ReLU(), second)

    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == HEAD_PARAM_COUNTS[dilution], (
        f"head {dilution!r} has {n_params} parameters, expected {HEAD_PARAM_COUNTS[dilution]}"
    )
    return head


def make_linear_readout(in_features: int, *, seed: int) -> nn.Module:
    """Linear(in_features, 1) with bias — the readout of arm D_best."""
    generator = torch.Generator()
    generator.manual_seed(derive(seed, "linear_readout", int(in_features)) % (2**63))
    layer = nn.Linear(int(in_features), 1, bias=True)
    _init_linear(layer, generator)
    head = nn.Sequential(layer)
    assert sum(p.numel() for p in head.parameters()) == in_features + 1
    return head
