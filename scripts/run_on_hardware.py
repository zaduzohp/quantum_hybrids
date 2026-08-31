"""Gate G5 on hardware: does a circuit that puts |1> on qubit i flip bit i?

Run by hand at the start of every hardware session. The clbit <-> qubit mapping is
produced by transpilation against the current calibration, and calibration_set_id
changes between sessions, so a session evaluated against a verdict from a different
calibration is silently unusable. Each run appends a row, so every session can be
paired with its own verdict.

calibration_set_id is a hard requirement: if the backend does not return exactly one,
this script aborts and writes nothing. Without it the calibration term of sigma_hw
cannot be separated from the shot term afterwards.

The token comes from the IQM_TOKEN environment variable and from nowhere else — never
from a file, the command line, or a log.

    export IQM_TOKEN=...        # e.g. read -rs IQM_TOKEN && export IQM_TOKEN
    python scripts/run_on_hardware.py --shots 4096
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from qsocket.gates import check_g5_bit_mapping
from qsocket.hardware import expectations_on_backend, make_probe_circuits
from qsocket.rank import DEFAULT_N_QUBITS
from qsocket.results import PLACEHOLDER_CALIBRATION_IDS

DEFAULT_URL = "https://odra5.e-science.pl/"
# outputs/ is already git-ignored, so a session log lands there.
DEFAULT_OUT = Path("outputs/g5_bit_mapping.jsonl")

# This script's default, not a project constant. A mapping error moves a value by 2.0
# while shot noise here is ~0.016, i.e. two orders of magnitude of margin.
DEFAULT_SHOTS = 4096

# Strings a provider might return in place of a real identifier; none may reach a row.
# The list comes from qsocket.results, which enforces the same rule when a results row is
# written: two copies drifted apart once already (this one had "0" and lacked "todo",
# "tbd" and "placeholder"), so a session could pass here and be refused there.
PLACEHOLDER_IDS = frozenset(PLACEHOLDER_CALIBRATION_IDS) | {"0"}


def _is_placeholder(calibration_set_id) -> bool:
    if calibration_set_id is None:
        return True
    return str(calibration_set_id).strip().lower() in PLACEHOLDER_IDS


def run_g5(
    backend,
    *,
    shots: int,
    out_path: Path,
    n_qubits: int = DEFAULT_N_QUBITS,
    note: str | None = None,
) -> dict:
    """Run the probe circuits, judge the mapping, append one row.

    The backend is injected rather than built here, so the whole path is testable
    without hardware. Raises RuntimeError before writing anything if calibration_set_id
    is missing, a placeholder, or not unique across the jobs of this run.
    """
    expectations, meta = expectations_on_backend(
        backend, make_probe_circuits(n_qubits), shots=shots
    )

    calibration_set_id = meta.get("calibration_set_id")
    if _is_placeholder(calibration_set_id):
        seen = meta.get("calibration_set_ids", [])
        raise RuntimeError(
            "no usable calibration_set_id from the backend "
            f"(value: {calibration_set_id!r}, ids seen across jobs: {seen}). "
            "Nothing was written: a G5 verdict that cannot be attributed to a "
            "calibration is unusable, and sigma_hw stops being decomposable "
            "(CONTRACTS 5)."
        )

    verdict = check_g5_bit_mapping(expectations)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "G5",
        "backend_name": meta.get("backend_name"),
        "is_iqm_backend": meta.get("is_iqm_backend"),
        "calibration_set_id": str(calibration_set_id),
        "job_ids": meta.get("job_ids", []),
        "shots": shots,
        "n_qubits": n_qubits,
        "optimization_level": meta.get("optimization_level"),
        "seed_transpiler": meta.get("seed_transpiler"),
        "transpiled_depth": meta.get("transpiled_depth"),
        "transpiled_cz_count": meta.get("transpiled_cz_count"),
        "passed": verdict["passed"],
        "diagnosis": verdict["diagnosis"],
        "permutation": verdict["permutation"],
        "tolerance": verdict["tolerance"],
        "expectations": verdict["expectations"],
        "note": note,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append, never overwrite: every session keeps its own verdict.
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def connect(url: str):
    """IQM backend from IQMProvider, token strictly from IQM_TOKEN."""
    token = os.environ.get("IQM_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "IQM_TOKEN is not set. The token is read from the environment and from "
            "nowhere else: `read -rs IQM_TOKEN && export IQM_TOKEN`."
        )
    from iqm.qiskit_iqm import IQMProvider

    return IQMProvider(url, token=token).get_backend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("IQM_URL", DEFAULT_URL).strip())
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--note", default=None, help="free-text session label stored in the row")
    args = parser.parse_args()

    backend = connect(args.url)
    row = run_g5(
        backend, shots=args.shots, out_path=args.out, n_qubits=args.n_qubits, note=args.note
    )

    print(f"backend            {row['backend_name']}")
    print(f"calibration_set_id {row['calibration_set_id']}")
    print(f"shots              {row['shots']}")
    for i, values in enumerate(row["expectations"]):
        print(f"  probe q{i}: " + "  ".join(f"{v:+.3f}" for v in values))
    print(f"diagnosis          {row['diagnosis']}")
    print(f"G5                 {'PASS' if row['passed'] else 'FAIL'}")
    print(f"appended to        {args.out}")
    if not row["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
