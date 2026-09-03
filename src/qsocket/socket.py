"""Socket variants for arms A-E: 5 inputs -> socket.out_features.

A/B  quantum circuit, trained / frozen at the same initialisation U[0, 2pi). The key is
     derive(seed, ansatz, R) with NO arm in it, so A and B start from an identical theta
     and the paired difference measures what training did from that starting point.
C    trainable classical socket — not implemented yet.
D    frozen random Fourier features, h = cos(Omega @ x + b): D_matched is 5 -> 5 on the
     dilution axis, D_best is width M in {32, 128, 512} picked on val. Both exploratory.
E    identity pass-through.

The quantum arms read diagonal Pauli strings, all from one state; readout_order fixes the
maximum weight (1 = main series, 5 outputs; 2 = correlator probe, 15). Which observables,
and in which order, comes from qsocket.core.pauli_z_chains for both backends.

Two backends live side by side: `pennylane` (lightning.qubit + adjoint) is the default and
~25x faster; `qiskit` (EstimatorQNN + ReverseEstimatorGradient) is kept because hardware
evaluation goes through Qiskit/IQM and it is the independent witness that the PennyLane
path computes the same numbers. That path TRANSLATES the Qiskit circuit gate by gate
rather than re-declaring it, so the two cannot drift; equivalence at 1e-10 is pinned by
tests/test_backend_equivalence.py.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pennylane as qml
import torch
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn

from qsocket.ansatzes import build_socket_circuit
from qsocket.core import DEFAULT_READOUT_ORDER, derive, pauli_z_chains, readout_size

DEFAULT_N_QUBITS = 5

SocketKind = Literal["quantum", "identity", "classical", "random"]
SocketBackend = Literal["qiskit", "pennylane"]
DEFAULT_BACKEND: SocketBackend = "pennylane"

D_MATCHED_WIDTH = 5
D_BEST_WIDTHS: tuple[int, ...] = (32, 128, 512)

LIGHTNING_DEVICE = "lightning.qubit"
DIFF_METHOD = "adjoint"

# Every gate the socket ansatzes emit. Anything outside these tables is a translation
# error rather than something to skip. Barriers carry no physics.
_ROTATIONS = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}
_TWO_QUBIT = {"cz": qml.CZ}


class _Gate(NamedTuple):
    """One translated gate: `vector` is "x"/"theta" for a rotation and None for CZ,
    `index` points into that vector."""

    name: str
    wires: tuple[int, ...]
    vector: str | None
    index: int | None


# --- observables ---------------------------------------------------------------------


def pauli_z_observables(n_qubits: int, *, order: int) -> list[SparsePauliOp]:
    """Every Z chain of weight 1..order as a SparsePauliOp, in the contract order.

    Qiskit is little-endian, so qubit i sits at position n_qubits - 1 - i of the label.
    """
    observables = []
    for chain in pauli_z_chains(n_qubits, order=order):
        label = ["I"] * n_qubits
        for i in chain:
            label[n_qubits - 1 - i] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return observables


def _z_chain_operator(chain: tuple[int, ...]):
    """Product of PauliZ over the qubits of one chain; PauliZ itself for weight 1.

    reduce over @ rather than qml.prod(*ops): a single-element prod wraps the operator in
    a Prod, which is the same observable but not the same object the order=1 path has
    always handed to adjoint.
    """
    operator = qml.PauliZ(chain[0])
    for wire in chain[1:]:
        operator = operator @ qml.PauliZ(wire)
    return operator


# --- the PennyLane execution path ----------------------------------------------------


def translate_circuit(circuit: QuantumCircuit) -> tuple[_Gate, ...]:
    """Qiskit socket circuit -> the ordered gates PennyLane replays.

    `vector` is "x" or "theta" for a rotation and None for CZ; `index` points into that
    vector. Binding is by parameter NAME, so it cannot drift with Qiskit's parameter
    ordering. Qiskit qubit indices are used directly as PennyLane wires.
    """
    instructions = []
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        wires = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)
        if name in _TWO_QUBIT:
            instructions.append(_Gate(name, wires, None, None))
            continue
        if name not in _ROTATIONS:
            raise ValueError(
                f"cannot translate gate {name!r} to PennyLane; the socket ansatzes are "
                f"expected to use only {sorted(_ROTATIONS) + sorted(_TWO_QUBIT)}"
            )
        (parameter,) = item.operation.params
        vector, index = str(parameter.name).split("[")
        instructions.append(_Gate(name, wires, vector, int(index.rstrip("]"))))
    return tuple(instructions)


def make_qnode(
    circuit: QuantumCircuit,
    n_qubits: int,
    *,
    interface: str,
    readout_order: int = DEFAULT_READOUT_ORDER,
):
    """QNode returning the Z-chain expectations, differentiated by the adjoint method.

    shots=None is explicit: a sampled default would add an unseeded term to every
    expectation value, breaking the pairing between arms holding identical theta without
    failing a run-against-itself determinism test.

    x may carry a leading batch dimension; PennyLane broadcasts the angles and lightning
    evaluates the batch in one call. Cost note: adjoint differentiates once per
    observable, so wall time grows roughly linearly in the number of outputs.
    """
    instructions = translate_circuit(circuit)
    # Built once here, not per trace: the operators are stateless.
    observables = [_z_chain_operator(c) for c in pauli_z_chains(n_qubits, order=readout_order)]

    def _circuit(x, theta):
        for name, wires, vector, index in instructions:
            if vector is None:
                _TWO_QUBIT[name](wires=list(wires))
            else:
                source = x if vector == "x" else theta
                _ROTATIONS[name](source[..., index], wires=wires[0])
        return [qml.expval(observable) for observable in observables]

    device = qml.device(LIGHTNING_DEVICE, wires=n_qubits, shots=None)
    return qml.QNode(_circuit, device, interface=interface, diff_method=DIFF_METHOD)


def expectations_pennylane(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    X: np.ndarray,
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    readout_order: int = DEFAULT_READOUT_ORDER,
) -> np.ndarray:
    """Exact Z-chain expectations for every row of X, float64.

    PennyLane counterpart of rank.z_expectation_batch. The equivalence test needs it
    because the torch path is float32 on both backends, so agreement at 1e-10 can only
    be established where nothing is downcast.
    """
    qnode = make_qnode(circuit, n_qubits, interface="numpy", readout_order=readout_order)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    values = qnode(X, np.asarray(theta, dtype=np.float64))
    n_outputs = readout_size(n_qubits, order=readout_order)
    return np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=-1).reshape(
        len(X), n_outputs
    )


class PennyLaneQuantumLayer(nn.Module):
    """(B, n_qubits) -> (B, n_outputs) of Z-chain expectations, theta the trainable weight.

    Mirrors the surface of Qiskit's TorchConnector — parameter named `weight`, forward
    takes a batch — so callers stay backend-agnostic. float32 matches TorchConnector, so
    theta is bit-for-bit identical on both backends for a given seed.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_qubits: int,
        initial_weights: np.ndarray,
        *,
        readout_order: int = DEFAULT_READOUT_ORDER,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.readout_order = readout_order
        self.n_outputs = readout_size(n_qubits, order=readout_order)
        self.diff_method = DIFF_METHOD
        self.qnode = make_qnode(circuit, n_qubits, interface="torch", readout_order=readout_order)
        self.weight = nn.Parameter(torch.as_tensor(np.asarray(initial_weights), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.qnode(x, self.weight)
        return torch.stack(list(values), dim=-1).to(x.dtype).reshape(*x.shape[:-1], self.n_outputs)


# --- initialisation ------------------------------------------------------------------


def initial_theta(*, ansatz: str, R: int, seed: int, n_params: int) -> np.ndarray:
    """theta ~ U[0, 2pi), keyed by derive(seed, ansatz, R)."""
    return np.random.default_rng(derive(seed, ansatz, R)).uniform(0.0, 2.0 * np.pi, n_params)


def sample_rff_weights(
    *, R: int, width: int, seed: int, n_inputs: int = DEFAULT_N_QUBITS
) -> tuple[np.ndarray, np.ndarray]:
    """(Omega, b) for arm D: Omega integer with |omega_i| <= R, b ~ U[0, 2pi).

    Frequencies come from the circuit spectrum rather than a Gaussian: with R re-uploads
    of an Ry encoding the reachable frequencies per coordinate are the integers in
    [-R, R]. Arm D therefore has the same frequency support as arms A and B, and there is
    no weight scale to scan. The random phase b is mandatory; without it this is a random
    cosine-activation network rather than random Fourier features.
    """
    rng = np.random.default_rng(derive(seed, "RFF", R, width))
    # Inclusive on both ends: the spectrum is {-R, ..., 0, ..., R}.
    omega = rng.integers(-R, R + 1, size=(width, n_inputs)).astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=width)
    return omega, phase


class RandomFourierFeatures(nn.Module):
    """h = cos(Omega @ x + b), one layer, weights held as BUFFERS, not parameters."""

    def __init__(self, omega: np.ndarray, phase: np.ndarray):
        super().__init__()
        self.register_buffer("omega", torch.as_tensor(omega, dtype=torch.float32))
        self.register_buffer("phase", torch.as_tensor(phase, dtype=torch.float32))

    @property
    def width(self) -> int:
        return int(self.omega.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x @ self.omega.T + self.phase)


# --- the socket ----------------------------------------------------------------------


class Socket(nn.Module):
    """Socket of one arm. 5 -> out_features, which is 5 only in the main series.

    Two things widen the output: readout_order > 1 on a quantum socket (5 -> 15 at
    order=2, the correlator probe) and rff_width on arm D (5 -> M for D_best). Ask
    out_features; do not assume n_qubits.
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
        if kind != "quantum" and readout_order != DEFAULT_READOUT_ORDER:
            raise ValueError(
                f"readout_order={readout_order} is only defined for a quantum socket, got "
                f"kind={kind!r}. The classical controls match the quantum readout by WIDTH "
                f"(rff_width), not by Pauli weight."
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
        quantum socket is 5 -> 15, and D_best is already 5 -> M."""
        if self.kind == "random":
            return int(self.rff_width)
        if self.kind == "identity":
            return self.n_qubits
        return readout_size(self.n_qubits, order=self.readout_order)

    def theta(self) -> torch.Tensor | None:
        """Socket parameters as a flat tensor, or None for a parameterless socket."""
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

    Arms B, D and E hold the socket fixed, so its output for a given input is constant
    over the run, yet train_model would recompute it every epoch: caching cuts a ~100x
    factor off those arms with a bit-for-bit identical trajectory.

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

    rows = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if rows.ndim != 2:
        # A single row without its batch dimension would come back with the wrong shape
        # and be trained on, silently.
        raise ValueError(f"X must be 2-dimensional (n_samples, n_features), got {tuple(rows.shape)}")
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
    readout_order applies to the quantum kinds only. Passing it to a classical or
    identity socket raises in Socket.__init__ rather than being ignored.
    """
    if kind == "classical":
        raise NotImplementedError(
            "arm C (classical socket) is blocked on open decision O5b (architecture of the "
            "classical arms, C_matched vs C_best, and the shared nonlinearity)"
        )
    if kind == "random":
        # Trainability is checked before width: a TRAINABLE RFF socket is arm C whatever
        # width came with it.
        if trainable:
            raise NotImplementedError(
                "a TRAINABLE classical socket is arm C, which is still blocked on O5b "
                "(width, and how to come back to the 5 -> 5 shape). Arm D is frozen."
            )
        if rff_width is None:
            raise ValueError(
                "rff_width is required for arm D: D_matched is 5 (on the dilution axis) and "
                f"D_best sweeps {list(D_BEST_WIDTHS)} (off it). There is no default, because "
                "the two variants answer two different questions."
            )
        omega, phase = sample_rff_weights(R=R, width=rff_width, seed=seed, n_inputs=n_qubits)
        socket = Socket(
            "random",
            n_qubits=n_qubits,
            R=R,
            layer=RandomFourierFeatures(omega, phase),
            rff_width=int(rff_width),
            readout_order=readout_order,
        )
        assert not list(socket.parameters()), "arm D must have zero socket parameters"
        return socket

    if kind == "identity":
        if trainable:
            raise ValueError("the identity socket has no parameters; trainable=True is meaningless")
        return Socket("identity", n_qubits=n_qubits, readout_order=readout_order)

    if backend not in ("qiskit", "pennylane"):
        raise ValueError(f"unknown backend {backend!r}; expected 'qiskit' or 'pennylane'")

    circuit = build_socket_circuit(ansatz, n_qubits, R)
    # Split by name rather than position; circuit.parameters already sorts
    # ParameterVector elements by numeric index.
    input_params = [p for p in circuit.parameters if p.name.startswith("x[")]
    weight_params = [p for p in circuit.parameters if p.name.startswith("theta[")]
    theta_init = initial_theta(ansatz=ansatz, R=R, seed=seed, n_params=len(weight_params))

    if backend == "pennylane":
        layer: nn.Module = PennyLaneQuantumLayer(
            circuit, n_qubits, theta_init, readout_order=readout_order
        )
    else:
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=pauli_z_observables(n_qubits, order=readout_order),
            input_params=input_params,
            weight_params=weight_params,
            estimator=StatevectorEstimator(),
            gradient=ReverseEstimatorGradient(),
            default_precision=0.0,
        )
        layer = TorchConnector(qnn, initial_weights=theta_init)

    socket = Socket(
        "quantum",
        n_qubits=n_qubits,
        R=R,
        ansatz=ansatz,
        layer=layer,
        backend=backend,
        readout_order=readout_order,
    )
    if not trainable:
        socket.requires_grad_(False)
    return socket
