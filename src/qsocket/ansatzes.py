"""Quantum socket ansatzes: two ansatz levels plus a product-circuit control.

All three circuits share the same skeleton: 5 qubits, hub = qubit 2, two-qubit gates
placed only on native IQM Spark hub-spoke edges, zero SWAP gates, exactly
n_qubits*R*3 + n_qubits parameters (20 / 35 / 50 at R = 1 / 2 / 3), and 4 CZ gates per
block (zero for the product circuit).

  L1  parallel fan            per block: Rz(all) Rx(all), then CZ(hub, s) for every
                              spoke s, then Rz(all).
  L2  sequential entangler    per block: Rz(all) Ry(all), then
                              Rx(hub) CZ(hub,s0) Rx(hub) CZ(hub,s1) Rx(hub) CZ(hub,s2)
                              Rx(hub) Ry(hub) CZ(hub,s_last).

  product                     L1 with the CZ gates removed. This is NOT an experiment
                              arm, it's only the negative control for the
                              entanglement gate.

Both levels close with a single Rx(all) layer, which is where the trailing n_qubits
parameters come from.
"""

from __future__ import annotations

from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qsocket.encoding import ry_feature_map

# Hub qubit of the star topology (QB2 on IQM Spark)
STAR_HUB_QUBIT: int = 2

THETA_PARAM_NAME = "theta"


def socket_param_count(n_qubits: int, R: int) -> int:
    """Number of socket parameters: n_qubits*R*3 + n_qubits.

    For n_qubits = 5 this is 20 / 35 / 50 at R = 1 / 2 / 3: 3*n_qubits per block plus a
    closing Rx layer. R is the degree of the trigonometric polynomial, and it is the
    only thing the count may grow with.

    This count is both nominal and real: the L1/L2 pair has zero dead parameters, so
    the numerical rank of d<Z_i>/d(theta) equals it exactly at every R.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if R < 1:
        raise ValueError(f"R must be >= 1, got {R}")
    return n_qubits * R * 3 + n_qubits


def spoke_qubits(n_qubits: int, hub: int = STAR_HUB_QUBIT) -> list[int]:
    """Spokes in ascending qubit order, hub excluded. For n=5, hub=2: [0, 1, 3, 4]."""
    if not 0 <= hub < n_qubits:
        raise ValueError(f"hub {hub} outside range for n_qubits={n_qubits}")
    return [q for q in range(n_qubits) if q != hub]


def _rotation_layer(qc: QuantumCircuit, gate: str, theta, p: int, n_qubits: int) -> int:
    """Apply one single-qubit rotation layer across all qubits; return the new offset."""
    for i in range(n_qubits):
        getattr(qc, gate)(theta[p + i], i)
    return p + n_qubits


def _block_L1(qc: QuantumCircuit, theta, p: int, n_qubits: int, hub: int) -> int:
    """One L1 block: Rz(all) Rx(all), the CZ fan from the hub, then Rz(all)."""
    p = _rotation_layer(qc, "rz", theta, p, n_qubits)
    p = _rotation_layer(qc, "rx", theta, p, n_qubits)
    for spoke in spoke_qubits(n_qubits, hub):
        qc.cz(hub, spoke)
    p = _rotation_layer(qc, "rz", theta, p, n_qubits)
    return p


def _block_L2(qc: QuantumCircuit, theta, p: int, n_qubits: int, hub: int) -> int:
    """One L2 block: Rz(all) Ry(all), then hub rotations interleaved with the CZ gates,
    ending on a CZ: Rx(hub) CZ ... Rx(hub) Ry(hub) CZ.
    """
    p = _rotation_layer(qc, "rz", theta, p, n_qubits)
    p = _rotation_layer(qc, "ry", theta, p, n_qubits)

    spokes = spoke_qubits(n_qubits, hub)
    for spoke in spokes[:-1]:
        qc.rx(theta[p], hub)
        p += 1
        qc.cz(hub, spoke)
    qc.rx(theta[p], hub)
    p += 1
    qc.ry(theta[p], hub)
    p += 1
    qc.cz(hub, spokes[-1])
    return p


def _block_product(qc: QuantumCircuit, theta, p: int, n_qubits: int, hub: int) -> int:
    """One product block: the L1 skeleton with the CZ fan removed, hence no
    entanglement."""
    p = _rotation_layer(qc, "rz", theta, p, n_qubits)
    p = _rotation_layer(qc, "rx", theta, p, n_qubits)
    p = _rotation_layer(qc, "rz", theta, p, n_qubits)
    return p


def _final_layer(qc: QuantumCircuit, theta, p: int, n_qubits: int) -> int:
    """Closing Rx(all): n_qubits parameters, no trailing Rz.

    An Rz here would commute with the Z measurement and be dead on arrival, and a
    second layer would exceed the two angles that survive after the last entangling
    gate anyway.
    """
    return _rotation_layer(qc, "rx", theta, p, n_qubits)


def ansatz_L1(n_qubits: int, R: int) -> QuantumCircuit:
    """Parallel fan. Per block: Rz(all) Rx(all), CZ(hub, s) per spoke, then Rz(all).
    Closing layer: Rx(all). Nominal count and Jacobian rank both 15R+5 = 20/35/50."""
    hub = STAR_HUB_QUBIT
    theta = ParameterVector(THETA_PARAM_NAME, socket_param_count(n_qubits, R))
    qc = QuantumCircuit(n_qubits, name=f"L1_R{R}")
    p = 0
    for _ in range(R):
        p = _block_L1(qc, theta, p, n_qubits, hub)
    p = _final_layer(qc, theta, p, n_qubits)
    assert p == len(theta)
    return qc


def ansatz_L2(n_qubits: int, R: int) -> QuantumCircuit:
    """Sequential entangler through the hub. Per block: Rz(all) Ry(all), then
    Rx(hub) CZ(hub,s0) Rx(hub) CZ(hub,s1) Rx(hub) CZ(hub,s2) Rx(hub) Ry(hub) CZ(hub,s_last).
    Closing layer: Rx(all). Nominal count and Jacobian rank both 15R+5 = 20/35/50."""
    hub = STAR_HUB_QUBIT
    theta = ParameterVector(THETA_PARAM_NAME, socket_param_count(n_qubits, R))
    qc = QuantumCircuit(n_qubits, name=f"L2_R{R}")
    p = 0
    for _ in range(R):
        p = _block_L2(qc, theta, p, n_qubits, hub)
    p = _final_layer(qc, theta, p, n_qubits)
    assert p == len(theta)
    return qc


def ansatz_product(n_qubits: int, R: int) -> QuantumCircuit:
    """The L1 skeleton with NO CZ gates. Same nominal count, 20/35/50."""
    hub = STAR_HUB_QUBIT
    theta = ParameterVector(THETA_PARAM_NAME, socket_param_count(n_qubits, R))
    qc = QuantumCircuit(n_qubits, name=f"product_R{R}")
    p = 0
    for _ in range(R):
        p = _block_product(qc, theta, p, n_qubits, hub)
    p = _final_layer(qc, theta, p, n_qubits)
    assert p == len(theta)
    return qc


ANSATZ_REGISTRY: dict[str, Callable[[int, int], QuantumCircuit]] = {
    "L1": ansatz_L1,
    "L2": ansatz_L2,
    "product": ansatz_product,
}

_BLOCK_BUILDERS: dict[str, Callable[..., int]] = {
    "L1": _block_L1,
    "L2": _block_L2,
    "product": _block_product,
}


def build_socket_circuit(
    ansatz_name: str,
    n_qubits: int,
    R: int,
    *,
    annotate: bool = False,
) -> QuantumCircuit:
    """Full socket circuit with data re-uploading.

    Layout: feature_map -> block_1 -> feature_map -> block_2 -> ... -> feature_map ->
    block_R -> closing Rx(all). The feature map is injected before every block and
    every injection shares one x vector, i.e. the identical Parameter objects, not
    copies.

    The closing layer is Rx(all) ALONE -- five parameters, no trailing Rz. An Rz there
    would commute with the Z measurement and be dead on arrival. That is where 15R+5
    comes from rather than 15R+10.

    annotate=True inserts labelled barriers marking each re-upload and the closing
    layer. They exist for drawing only and are emitted identically for every ansatz
    level, so they cannot make the levels differ under transpilation. Leave it False
    for anything that runs on hardware or gets measured.
    """
    if ansatz_name not in _BLOCK_BUILDERS:
        raise ValueError(f"unknown ansatz {ansatz_name!r}; expected one of {sorted(_BLOCK_BUILDERS)}")

    hub = STAR_HUB_QUBIT
    build_block = _BLOCK_BUILDERS[ansatz_name]
    theta = ParameterVector(THETA_PARAM_NAME, socket_param_count(n_qubits, R))
    feature_map = ry_feature_map(n_qubits)

    qc = QuantumCircuit(n_qubits, name=f"socket_{ansatz_name}_R{R}")
    p = 0
    for block_index in range(R):
        if annotate:
            qc.barrier(label=f"reupload {block_index + 1}/{R}")
        # The same feature_map object every time, so all injections share one x.
        qc.compose(feature_map, inplace=True)
        if annotate:
            qc.barrier(label=f"block {block_index + 1}/{R}")
        p = build_block(qc, theta, p, n_qubits, hub)
    if annotate:
        qc.barrier(label="closing Rx")
    p = _final_layer(qc, theta, p, n_qubits)

    assert p == len(theta)
    return qc