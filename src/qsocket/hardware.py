"""IQM hardware estimator.

expectations_on_backend transpiles, executes in batches and returns <Z_i> in input
order together with session metadata. calibration_set_id is required to decompose the
hardware variance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from qsocket.readout import expectations_from_counts

try: 
    from iqm.qiskit_iqm import transpile_to_IQM as _iqm_transpile
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase as _IQMBackendBase
except ImportError:
    _iqm_transpile = None
    _IQMBackendBase = None


def make_probe_circuits(n_qubits: int) -> list[QuantumCircuit]:
    """One circuit per qubit, putting |1> on that qubit only.

    Circuit i must flip bit i of the histogram and nothing else, so
    expectations_from_counts gives -1 at position i and +1 elsewhere. This is the only
    end-to-end check that the clbit <-> qubit mapping survives transpilation, routing
    and the provider's own bit ordering.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    circuits = []
    for i in range(n_qubits):
        qc = QuantumCircuit(n_qubits, name=f"probe_q{i}")
        qc.x(i)
        qc.measure_all()
        circuits.append(qc)
    return circuits


def _backend_name(backend) -> str:
    name = getattr(backend, "name", None)
    if callable(name):  # BackendV1 exposes name() as a method
        name = name()
    return str(name) if name is not None else type(backend).__name__


def _is_iqm_backend(backend) -> bool:
    return _IQMBackendBase is not None and isinstance(backend, _IQMBackendBase)


def transpile_for_backend(
    circuit: QuantumCircuit,
    backend,
    optimization_level: int,
    seed_transpiler: int | None,
) -> QuantumCircuit:
    """IQM backends go through transpile_to_IQM, everything else through qiskit.transpile."""
    kwargs: dict[str, Any] = {"optimization_level": optimization_level}
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = seed_transpiler
    if _iqm_transpile is not None and _is_iqm_backend(backend):
        return _iqm_transpile(circuit, backend, **kwargs)
    return transpile(circuit, backend, **kwargs)


def _normalize_counts(counts) -> dict[str, int]:
    """Counts as {bitstring: int}, with Qiskit's register separators removed.

    Registers are rendered most significant first, the same ordering as within a
    register, so dropping the spaces yields one little-endian bitstring.
    """
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError(f"Expected one counts dict, got {len(counts)}")
        counts = counts[0]
    return {str(key).replace(" ", ""): int(value) for key, value in counts.items()}


def _calibration_set_id(result) -> str | None:
    """Calibration set id if the provider exposes it.

    Never fatal — a simulator has no calibration set. On hardware its absence breaks the
    analysis, so the caller checks meta and refuses the row.
    """
    for path in (
        lambda r: getattr(r, "parameters", None),
        lambda r: getattr(r, "metadata", None),
        lambda r: getattr(r, "_metadata", None),
    ):
        try:
            obj = path(result)
            if obj is None:
                continue
            if isinstance(obj, dict):
                cid = obj.get("calibration_set_id") or obj.get("calibration_set")
                if cid is not None:
                    return str(cid)
            cid = getattr(obj, "calibration_set_id", None)
            if cid is not None:
                return str(cid)
        except Exception:
            continue
    return None


def _extract_timestamps(result) -> dict[str, str] | None:
    """Provider-side timeline, if any."""
    try:
        timeline = result._metadata.get("timeline", [])
        if not timeline:
            return None
        return {str(entry.status): str(entry.timestamp) for entry in timeline}
    except Exception:
        return None


def _job_id(job) -> str | None:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        try:
            job_id = job_id()
        except Exception:
            return None
    return str(job_id) if job_id is not None else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expectations_on_backend(
    backend,
    circuits: list[QuantumCircuit],
    *,
    shots: int,
    optimization_level: int = 1,
    seed_transpiler: int | None = 42,
    max_circuits_per_job: int = 100,
) -> tuple[np.ndarray, dict]:
    """Transpile, execute, return (len(circuits), n_qubits) of <Z_i> plus session meta.

    Output row k belongs to input circuit k; batching is the only place that could
    silently break, so every batch asserts the histogram count it got back.
    """
    if not circuits:
        raise ValueError("no circuits given")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    if max_circuits_per_job < 1:
        raise ValueError(f"max_circuits_per_job must be >= 1, got {max_circuits_per_job}")

    n_qubits = circuits[0].num_qubits
    for index, qc in enumerate(circuits):
        if qc.num_qubits != n_qubits:
            raise ValueError(
                f"circuit {index} has {qc.num_qubits} qubits, circuit 0 has {n_qubits}; "
                "one call must describe one register layout"
            )
        if qc.parameters:
            raise ValueError(
                f"circuit {index} still has free parameters {sorted(p.name for p in qc.parameters)}; "
                "bind them before calling — this function never assigns parameters positionally"
            )

    prepared = []
    for qc in circuits:
        if qc.num_clbits == 0:
            qc = qc.copy()
            qc.measure_all()
        prepared.append(qc)

    transpiled = []
    for index, qc in enumerate(prepared):
        tqc = transpile_for_backend(qc, backend, optimization_level, seed_transpiler)
        # Measurement is added before transpilation, so the classical register keeps one
        # bit per logical qubit and the transpiler carries the qubit -> clbit map through
        # routing. Otherwise histogram strings would be silently permuted.
        if tqc.num_clbits != n_qubits:
            raise RuntimeError(
                f"circuit {index}: transpilation produced {tqc.num_clbits} clbits for "
                f"{n_qubits} qubits; the clbit <-> qubit mapping is no longer implicit"
            )
        if tqc.parameters:
            raise RuntimeError(
                f"circuit {index}: transpiled circuit has free parameters "
                f"{sorted(p.name for p in tqc.parameters)}"
            )
        transpiled.append(tqc)

    all_counts: list[dict[str, int]] = []
    job_ids: list[str] = []
    calibration_ids: list[str] = []
    timelines: list[dict[str, str]] = []
    submitted_at = _utc_now()

    for start in range(0, len(transpiled), max_circuits_per_job):
        batch = transpiled[start : start + max_circuits_per_job]
        job = backend.run(batch, shots=shots)
        result = job.result()

        job_id = _job_id(job)
        if job_id is not None:
            job_ids.append(job_id)
        cid = _calibration_set_id(result)
        if cid is not None:
            calibration_ids.append(cid)
        timeline = _extract_timestamps(result)
        if timeline is not None:
            timelines.append(timeline)

        counts_list = result.get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
        if len(counts_list) != len(batch):
            raise RuntimeError(
                f"expected {len(batch)} histograms, backend returned {len(counts_list)}; "
                "input order can no longer be trusted"
            )
        all_counts.extend(_normalize_counts(c) for c in counts_list)

    completed_at = _utc_now()
    if len(all_counts) != len(transpiled):
        raise RuntimeError(
            f"expected {len(transpiled)} histograms in total, collected {len(all_counts)}"
        )

    expectations = np.array(
        [expectations_from_counts(counts, n_qubits) for counts in all_counts], dtype=float
    )

    unique_calibration = sorted(set(calibration_ids))
    meta = {
        "backend_name": _backend_name(backend),
        "is_iqm_backend": _is_iqm_backend(backend),
        "shots": shots,
        "n_circuits": len(circuits),
        "n_qubits": n_qubits,
        "calibration_set_id": unique_calibration[0] if len(unique_calibration) == 1 else None,
        "calibration_set_ids": unique_calibration,
        "job_ids": job_ids,
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
        "max_circuits_per_job": max_circuits_per_job,
        "n_jobs": (len(transpiled) + max_circuits_per_job - 1) // max_circuits_per_job,
        "transpiled_depth": int(transpiled[0].depth()),
        "transpiled_cz_count": int(transpiled[0].count_ops().get("cz", 0)),
        "transpiled_depth_max": int(max(qc.depth() for qc in transpiled)),
        "transpiled_cz_count_max": int(max(qc.count_ops().get("cz", 0) for qc in transpiled)),
        "submitted_at_utc": submitted_at,
        "completed_at_utc": completed_at,
        "provider_timelines": timelines,
    }
    return expectations, meta
