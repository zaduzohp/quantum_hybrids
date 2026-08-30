"""Socket variants for arms A-E: 5 inputs -> socket.out_features.

A/B: quantum circuit, trained / frozen at the same initialisation U[0, 2pi). The
     initialisation key is derive(seed, ansatz, R) with no arm in it, so for a given
     seed A and B start from an identical theta and the paired difference measures
     what training did from that starting point.
C:   trainable classical socket — not implemented yet.
D:   frozen classical random Fourier features, h = cos(Omega @ x + b), in two variants:
       D_matched  5 -> 5, on the dilution axis;
       D_best     width M in {32, 128, 512} selected on the validation split, linear
                  readout on M features, off the dilution axis.
     Both are exploratory, never part of the confirmatory family.
E:   identity pass-through.

Readout: the quantum arms read expectation values of diagonal Pauli strings, all from one
state. readout_order fixes the maximum Pauli weight -- 1 (the main series, 5 outputs) or
2 (the correlator probe, 15 outputs). The observables and their order come from
qsocket.observables, shared with the PennyLane path.

Two execution backends live side by side:

  pennylane  lightning.qubit + diff_method="adjoint". Default, ~25x faster, used for
             training runs.
  qiskit     EstimatorQNN + ReverseEstimatorGradient. Kept because hardware evaluation
             goes through Qiskit/IQM and it is the independent witness that the
             PennyLane path computes the same numbers.

The PennyLane path translates the Qiskit circuit rather than re-declaring it;
equivalence at 1e-10 is pinned by tests/test_backend_equivalence.py.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from qsocket.ansatzes import build_socket_circuit
from qsocket.observables import DEFAULT_READOUT_ORDER, pauli_z_chains, readout_size
from qsocket.pennylane_socket import PennyLaneQuantumLayer
from qsocket.seeding import derive

DEFAULT_N_QUBITS = 5

SocketKind = Literal["quantum", "identity", "classical", "random"]

D_MATCHED_WIDTH = 5
D_BEST_WIDTHS: tuple[int, ...] = (32, 128, 512)
SocketBackend = Literal["qiskit", "pennylane"]

DEFAULT_BACKEND: SocketBackend = "pennylane"


def pauli_z_observables(n_qubits: int, *, order: int) -> list[SparsePauliOp]:
    """Every Z chain of weight 1..order as a SparsePauliOp, in the contract order.

    The socket output is the vector of these expectation values, all read from the same
    state. Qiskit is little-endian, so qubit i sits at position n_qubits - 1 - i of the
    Pauli label.

    The chain list and its order come from qsocket.observables, not from a loop here:
    the PennyLane path builds its operators from the same list, and if the two ever
    disagreed on which column is which, no test in the suite would notice while every
    paired difference would be wrong.
    """
    observables = []
    for chain in pauli_z_chains(n_qubits, order=order):
        label = ["I"] * n_qubits
        for i in chain:
            label[n_qubits - 1 - i] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return observables


def initial_theta(*, ansatz: str, R: int, seed: int, n_params: int) -> np.ndarray:
    """theta ~ U[0, 2pi), keyed by derive(seed, ansatz, R).

    The wide range keeps the circuit a genuine random feature map; near theta = 0 it
    would sit close to the identity and arm B would degenerate into arm E. The arm is
    not part of the key, so arms A and B start from an identical theta.
    """
    rng = np.random.default_rng(derive(seed, ansatz, R))
    return rng.uniform(0.0, 2.0 * np.pi, n_params)


def sample_rff_weights(
    *, R: int, width: int, seed: int, n_inputs: int = DEFAULT_N_QUBITS
) -> tuple[np.ndarray, np.ndarray]:
    """(Omega, b) for arm D: Omega integer with |omega_i| <= R, b ~ U[0, 2pi).

    Frequencies are drawn from the circuit spectrum rather than a Gaussian: with R
    re-uploads of an Ry encoding the reachable frequencies per coordinate are the
    integers in [-R, R]. Arm D therefore has the same frequency support as arms A and B,
    and there is no weight scale to scan — Omega is integer and x lies in FEATURE_RANGE.

    The random phase b is mandatory; without it this is a random cosine-activation
    network rather than random Fourier features.
    """
    if R < 1:
        raise ValueError(f"R must be >= 1, got {R}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    rng = np.random.default_rng(derive(seed, "RFF", R, width))
    # Inclusive on both ends: the spectrum is {-R, ..., 0, ..., R}.
    omega = rng.integers(-R, R + 1, size=(width, n_inputs)).astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=width)
    return omega, phase


class RandomFourierFeatures(nn.Module):
    """h = cos(Omega @ x + b), one layer, weights held as BUFFERS, not parameters.

    Buffers rather than parameters with requires_grad=False: arm D has zero trainable
    socket parameters, and a buffer cannot be un-frozen by a stray requires_grad_(True)
    in a driver. Arm C is the same class with (Omega, b) as parameters.

    One layer, not R blocks: composed blocks are a cosine of a cosine, whose frequencies
    mix and multiply, so the output would leave the trigonometric class of the circuit
    and arm D would stop being a control.
    """

    def __init__(self, omega: np.ndarray, phase: np.ndarray):
        super().__init__()
        omega = np.asarray(omega, dtype=np.float64)
        phase = np.asarray(phase, dtype=np.float64)
        if omega.ndim != 2:
            raise ValueError(f"Omega must be 2-dimensional (width, n_inputs), got {omega.shape}")
        if phase.shape != (omega.shape[0],):
            raise ValueError(f"b must have shape ({omega.shape[0]},), got {phase.shape}")
        self.register_buffer("omega", torch.as_tensor(omega, dtype=torch.float32))
        self.register_buffer("phase", torch.as_tensor(phase, dtype=torch.float32))

    @property
    def width(self) -> int:
        return int(self.omega.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x @ self.omega.T + self.phase)


class Socket(nn.Module):
    """Socket of one arm. 5 -> out_features, which is 5 only in the main series.

    Two things widen the output: readout_order > 1 on a quantum socket (5 -> 15 at
    order=2, the correlator probe) and rff_width on arm D (5 -> M for D_best). Ask
    out_features; do not assume n_qubits.

    quantum: build_socket_circuit -> Z-chain observables -> a torch layer, either
             EstimatorQNN/TorchConnector (backend="qiskit") or a lightning.qubit
             QNode (backend="pennylane"). Output values are expectation values,
             hence in [-1, 1], and agree between the backends.
    identity: pass-through, no parameters.
    random:   RandomFourierFeatures, arm D. Parameterless (buffers), width 5 for
              D_matched and M for D_best.
    """

    def __init__(
        self,
        kind: SocketKind,
        *,
        n_qubits: int = DEFAULT_N_QUBITS,
        R: int | None = None,
        ansatz: str | None = None,
        layer: nn.Module | None = None,
        backend: SocketBackend | None = None,
        rff_width: int | None = None,
        readout_order: int = DEFAULT_READOUT_ORDER,
    ):
        super().__init__()
        self.kind = kind
        self.n_qubits = n_qubits
        self.R = R
        self.ansatz = ansatz
        self.backend = backend
        self.rff_width = rff_width
        # Readout order of a quantum socket.
        if kind != "quantum" and readout_order != DEFAULT_READOUT_ORDER:
            raise ValueError(
                f"readout_order={readout_order} is only defined for a quantum socket, "
                f"got kind={kind!r}. The classical controls match the quantum readout by "
                f"WIDTH (rff_width), not by Pauli weight."
            )
        self.readout_order = readout_order
        # Two names on purpose: theta() and theta_displacement mean the quantum parameter
        # vector, and arm D must not answer them with a buffer.
        self.classical_layer = layer if kind == "random" else None
        self.quantum_layer = layer if kind != "random" else None
        # Where training started, used for theta_displacement and the frozen-arm check.
        self.theta_init = (
            None if self.quantum_layer is None else self.quantum_layer.weight.detach().clone()
        )

    @property
    def is_quantum(self) -> bool:
        return self.kind == "quantum"

    @property
    def out_features(self) -> int:
        """Width of the socket output, i.e. the input width the head must be built for.

        Read from the socket rather than assumed to be n_qubits: at readout_order > 1 a
        quantum socket is 5 -> 15, and D_best is already 5 -> M. Callers that hardcode 5
        are the failure mode this property exists to remove."""
        if self.kind == "random":
            if self.rff_width is None:
                raise ValueError("a random socket has no width; rff_width was never set")
            return int(self.rff_width)
        if self.kind == "identity":
            return self.n_qubits
        return readout_size(self.n_qubits, order=self.readout_order)

    def theta(self) -> torch.Tensor | None:
        """Current socket parameters as a flat tensor, or None for a parameterless socket."""
        if self.quantum_layer is None:
            return None
        return self.quantum_layer.weight.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "random":
            return self.classical_layer(x)
        return self.quantum_layer(x)


# Bounds how many rows go into one lightning call; the output is asserted to be
# independent of it.
FEATURE_CACHE_BATCH = 512


def frozen_socket_features(socket: Socket, X, *, batch_size: int = FEATURE_CACHE_BATCH):
    """Socket output for every row of X, computed once. Only valid for a frozen socket.

    Arms B and E hold the socket fixed, so its output for a given input is constant over
    the run, yet train_model would recompute it every epoch: caching cuts a ~100x factor
    off those arms with a bit-for-bit identical trajectory.

    The caller keeps the metadata: a cached arm B is trained through an identity socket
    and would otherwise lose its identity in the results row.
    """
    trainable = [name for name, p in socket.named_parameters() if p.requires_grad]
    if trainable:
        raise ValueError(
            "frozen_socket_features refuses a trainable socket: its output changes after "
            f"every optimiser step, so a cache of it is stale immediately. Parameters "
            f"still requiring grad: {trainable}. For arm A there is nothing to cache."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    rows = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if rows.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (n_samples, n_features), got shape {tuple(rows.shape)}")

    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunks.append(socket(rows[start : start + batch_size]))
    return torch.cat(chunks, dim=0).numpy()


def make_socket(
    kind: SocketKind,
    *,
    R: int | None,
    ansatz: str,
    trainable: bool,
    seed: int,
    n_qubits: int = DEFAULT_N_QUBITS,
    backend: SocketBackend = DEFAULT_BACKEND,
    rff_width: int | None = None,
    readout_order: int = DEFAULT_READOUT_ORDER,
) -> Socket:
    """Build a socket.

    quantum  -- arms A (trainable=True) and B (trainable=False). Same circuit, same
                theta_init for a given (seed, ansatz, R); only requires_grad differs.
                backend selects how <Z_i> and its gradient are computed, not what.
    identity -- arm E. No parameters, forward returns its input unchanged.
    random   -- arm D, both variants; rff_width picks which (5 = D_matched). ansatz is
                ignored: the frequency support depends on R only.
    classical -- arm C, not implemented yet.

    readout_order applies to the quantum kinds only, and defaults to the main-series
    value, so every pre-probe call site is unchanged. Passing it to a classical or
    identity socket raises in Socket.__init__ rather than being ignored: the classical
    controls match the quantum readout by WIDTH (rff_width), not by Pauli weight.
    """
    if kind == "classical":
        raise NotImplementedError(
            "arm C (classical socket) is blocked on open decision O5b (architecture of "
            "the classical arms, C_matched vs C_best, and the shared nonlinearity)"
        )
    if kind == "random":
        if R is None:
            raise ValueError("R is required for arm D: it sets the frequency support")
        # Trainability is checked before width: a trainable RFF socket is arm C whatever
        # width came with it.
        if trainable:
            raise NotImplementedError(
                "a TRAINABLE classical socket is arm C, which is still blocked on O5b "
                "(width, and how to come back to the 5 -> 5 shape). Arm D is frozen."
            )
        if rff_width is None:
            raise ValueError(
                "rff_width is required for arm D: D_matched is 5 (on the dilution axis) "
                f"and D_best sweeps {list(D_BEST_WIDTHS)} (off it). There is no default, "
                "because the two variants answer two different questions."
            )
        omega, phase = sample_rff_weights(R=R, width=rff_width, seed=seed, n_inputs=n_qubits)
        socket = Socket(
            "random",
            n_qubits=n_qubits,
            R=R,
            ansatz=None,
            layer=RandomFourierFeatures(omega, phase),
            rff_width=int(rff_width),
            readout_order=readout_order,
        )
        assert not list(socket.parameters()), "arm D must have zero socket parameters"
        return socket

    if kind == "identity":
        socket = Socket("identity", n_qubits=n_qubits, readout_order=readout_order)
        if trainable:
            raise ValueError("the identity socket has no parameters; trainable=True is meaningless")
        return socket

    if kind != "quantum":
        raise ValueError(f"unknown socket kind {kind!r}")

    if R is None:
        raise ValueError("R is required for a quantum socket")
    if backend not in ("qiskit", "pennylane"):
        raise ValueError(f"unknown backend {backend!r}; expected 'qiskit' or 'pennylane'")

    circuit = build_socket_circuit(ansatz, n_qubits, R)
    # Split by name rather than position; circuit.parameters already sorts
    # ParameterVector elements by numeric index.
    input_params = [p for p in circuit.parameters if p.name.startswith("x[")]
    weight_params = [p for p in circuit.parameters if p.name.startswith("theta[")]
    assert len(input_params) == n_qubits
    assert len(input_params) + len(weight_params) == len(circuit.parameters)

    theta_init = initial_theta(ansatz=ansatz, R=R, seed=seed, n_params=len(weight_params))

    if backend == "pennylane":
        layer: nn.Module = PennyLaneQuantumLayer(
            circuit, n_qubits, theta_init, readout_order=readout_order
        )
        socket = Socket(
            "quantum", n_qubits=n_qubits, R=R, ansatz=ansatz, layer=layer, backend=backend,
            readout_order=readout_order,
        )
        if not trainable:
            socket.requires_grad_(False)
        return socket

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=pauli_z_observables(n_qubits, order=readout_order),
        input_params=input_params,
        weight_params=weight_params,
        estimator=StatevectorEstimator(),
        gradient=ReverseEstimatorGradient(),
        default_precision=0.0,
    )
    socket = Socket(
        "quantum",
        n_qubits=n_qubits,
        R=R,
        ansatz=ansatz,
        layer=TorchConnector(qnn, initial_weights=theta_init),
        backend=backend,
        readout_order=readout_order,
    )
    if not trainable:
        socket.requires_grad_(False)
    return socket
