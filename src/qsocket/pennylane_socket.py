"""PennyLane execution path for the quantum socket: lightning.qubit + adjoint.

Faster alternative to the Qiskit path. Same circuit, encoding, theta and observables — only the way the
derivative is obtained changes.

The circuit is translated gate by gate from the QuantumCircuit that
ansatzes.build_socket_circuit produces, never re-declared here, so the two backends
cannot drift apart; an unknown instruction raises instead of being dropped.

Precision: the device is created with shots=None explicitly. A sampled default would
add an unseeded term to every expectation value, breaking the pairing between arms
holding identical theta without failing a run-against-itself determinism test.

Dtype: the torch interface returns float32, matching Qiskit's TorchConnector. The
float64 witness for the equivalence test is expectations_pennylane below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from qiskit import QuantumCircuit

LIGHTNING_DEVICE = "lightning.qubit"
DIFF_METHOD = "adjoint"

# Every rotation the socket ansatzes emit. A gate outside this table is a translation
# error rather than something to skip.
_ROTATIONS = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}
_TWO_QUBIT = {"cz": qml.CZ}
# Barriers carry no physics; build_socket_circuit emits them only for drawing.
_IGNORED = {"barrier"}


@dataclass(frozen=True)
class _Instruction:
    """One translated gate: which PennyLane op, on which wires, fed by which parameter.

    vector is "x" or "theta" for a rotation and None for CZ; index points into that
    vector. Qiskit qubit indices are used directly as PennyLane wires.
    """

    name: str
    wires: tuple[int, ...]
    vector: str | None
    index: int | None


def translate_circuit(circuit: QuantumCircuit) -> tuple[_Instruction, ...]:
    """Qiskit socket circuit -> ordered list of PennyLane instructions.

    Binds by parameter name ("x[3]", "theta[12]") rather than position, so the binding
    cannot drift with Qiskit's parameter ordering.
    """
    instructions: list[_Instruction] = []
    for item in circuit.data:
        name = item.operation.name
        if name in _IGNORED:
            continue
        wires = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)
        if name in _TWO_QUBIT:
            instructions.append(_Instruction(name, wires, None, None))
            continue
        if name not in _ROTATIONS:
            raise ValueError(
                f"cannot translate gate {name!r} to PennyLane; the socket ansatzes are "
                f"expected to use only {sorted(_ROTATIONS) + sorted(_TWO_QUBIT)}"
            )
        (parameter,) = item.operation.params
        vector, index = str(parameter.name).split("[")
        if vector not in ("x", "theta"):
            raise ValueError(f"unexpected parameter vector {vector!r} in {parameter.name!r}")
        instructions.append(_Instruction(name, wires, vector, int(index.rstrip("]"))))
    return tuple(instructions)


def make_device(n_qubits: int):
    """lightning.qubit with shots=None set explicitly ."""
    return qml.device(LIGHTNING_DEVICE, wires=n_qubits, shots=None)


def make_qnode(circuit: QuantumCircuit, n_qubits: int, *, interface: str):
    """QNode returning the five <Z_i>, differentiated by the adjoint method.

    x may carry a leading batch dimension; PennyLane broadcasts the rotation angles and
    lightning evaluates the batch in one call.
    """
    instructions = translate_circuit(circuit)

    def _circuit(x, theta):
        for instruction in instructions:
            if instruction.vector is None:
                _TWO_QUBIT[instruction.name](wires=list(instruction.wires))
                continue
            source = x if instruction.vector == "x" else theta
            _ROTATIONS[instruction.name](source[..., instruction.index], wires=instruction.wires[0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.QNode(_circuit, make_device(n_qubits), interface=interface, diff_method=DIFF_METHOD)


def expectations_pennylane(
    circuit: QuantumCircuit,
    theta: np.ndarray,
    X: np.ndarray,
    *,
    n_qubits: int = 5,
) -> np.ndarray:
    """Exact <Z_i> for every row of X, shape (len(X), n_qubits), in float64.

    PennyLane counterpart of rank.z_expectation_batch. The equivalence test needs it
    because the torch path is float32 on both backends, so agreement at 1e-10 can only
    be established where nothing is downcast.
    """
    qnode = make_qnode(circuit, n_qubits, interface="numpy")
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    values = qnode(X, np.asarray(theta, dtype=np.float64))
    return np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=-1).reshape(
        len(X), n_qubits
    )


class PennyLaneQuantumLayer(nn.Module):
    """(B, n_qubits) -> (B, n_qubits) of <Z_i>, with theta as the trainable weight.

    Mirrors the surface of Qiskit's TorchConnector — parameter named `weight`, forward
    takes a batch — so callers stay backend-agnostic.
    """

    def __init__(self, circuit: QuantumCircuit, n_qubits: int, initial_weights: np.ndarray):
        super().__init__()
        self.n_qubits = n_qubits
        self.diff_method = DIFF_METHOD
        self.qnode = make_qnode(circuit, n_qubits, interface="torch")
        # float32 matches TorchConnector, so theta is bit-for-bit identical on both
        # backends for a given seed.
        self.weight = nn.Parameter(torch.as_tensor(np.asarray(initial_weights), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.qnode(x, self.weight)
        return torch.stack(list(values), dim=-1).to(x.dtype).reshape(*x.shape[:-1], self.n_qubits)
