"""KL expressibility (vs Haar) and Meyer-Wallach entanglement of the socket.

Tests the QELM assumption that the reservoir must generate a rich Fourier spectrum,
which requires a chaotic/mixing regime, against thresholds declared before the run.

Both metrics are vendored verbatim (see the headers in qsocket/vendored/); what lives
here is only the wiring to this project's sockets. The theta/x samplers come from
qsocket.rank, so rank, entanglement and this probe draw their inputs the same way.

Haar is measured rather than cited: the same pipeline is fed Haar-distributed
fidelities, giving the finite-sample KL floor and the empirical Q_Haar the verdict
compares to.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qsocket.ansatzes import build_socket_circuit
from qsocket.core import FEATURE_RANGE
from qsocket.rank import DEFAULT_N_QUBITS, sample_inputs
from qsocket.vendored.entanglement import meyer_wallach_score
from qsocket.vendored.expressibility import (
    binned_distributions,
    kl_divergence,
    sample_haar_fidelities,
)

# Fixed before the run.
N_PAIRS = 5000
N_MW = 2000
N_BINS = 75
SEEDS = (1, 2, 3)
CELLS: tuple[tuple[str, int], ...] = (
    ("product", 2),
    ("L1", 1),
    ("L1", 2),   # <-- arm B
    ("L1", 3),
    ("L2", 2),
)
# Verdict bands.
RICH_Q_RATIO, POOR_Q_RATIO = 0.8, 0.4
RICH_KL_FACTOR, POOR_KL_FACTOR = 3.0, 10.0


def x_settings(n_qubits: int, x_scale: float = 1.0) -> dict[str, np.ndarray]:
    """x = 0 (encoding is the identity) plus two draws from FEATURE_RANGE, times x_scale.

    x_scale = 2.0 is exactly the pi/2 encoding range — MinMaxScaler is linear and the
    clip bound scales with the values, so clip and scale commute for a positive
    multiplier. It separates a property of the ansatz from one of the encoding range.
    """
    return {
        "x_zero": np.zeros(n_qubits),
        "x_draw1": x_scale * sample_inputs(1, seed=101, n_qubits=n_qubits)[0],
        "x_draw2": x_scale * sample_inputs(1, seed=202, n_qubits=n_qubits)[0],
    }


def _bind(circuit, theta: np.ndarray, x: np.ndarray):
    values = {}
    for param in circuit.parameters:
        vector, index = param.name.split("[")
        index = int(index.rstrip("]"))
        values[param] = float(theta[index] if vector == "theta" else x[index])
    return circuit.assign_parameters(values)


def state_of(circuit, theta, x) -> np.ndarray:
    return Statevector.from_instruction(_bind(circuit, theta, x)).data


def measure_cell(ansatz: str, R: int, *, x_label: str, x: np.ndarray, seed: int,
                 n_qubits: int, n_pairs: int, n_mw: int, n_bins: int = N_BINS) -> dict:
    circuit = build_socket_circuit(ansatz, n_qubits, R)
    n_theta = sum(1 for p in circuit.parameters if p.name.startswith("theta"))
    dim = 2 ** n_qubits
    started = time.perf_counter()

    # One stream of theta draws: pairs for fidelity, singles for MW.
    rng = np.random.default_rng(seed)
    fids = np.empty(n_pairs)
    for k in range(n_pairs):
        ta = rng.uniform(0.0, 2.0 * np.pi, n_theta)
        tb = rng.uniform(0.0, 2.0 * np.pi, n_theta)
        pa, pb = state_of(circuit, ta, x), state_of(circuit, tb, x)
        fids[k] = float(abs(np.vdot(pa, pb)) ** 2)

    mw = np.empty(n_mw)
    for k in range(n_mw):
        t = rng.uniform(0.0, 2.0 * np.pi, n_theta)
        mw[k] = meyer_wallach_score(state_of(circuit, t, x), n_qubits)

    _, _, p_emp, p_haar = binned_distributions(fids, dim, n_bins)
    return {
        "ansatz": ansatz, "R": R, "x_setting": x_label, "seed": seed,
        "n_theta": n_theta, "n_pairs": n_pairs, "n_mw": n_mw, "n_bins": n_bins,
        "kl": kl_divergence(p_emp, p_haar),
        "mw_mean": float(mw.mean()), "mw_sd": float(mw.std(ddof=1)),
        "fid_mean": float(fids.mean()),
        "wall_seconds": time.perf_counter() - started,
        "_fids": fids, "_mw": mw,
    }


def measure_haar(*, seed: int, n_qubits: int, n_pairs: int, n_mw: int, n_bins: int = N_BINS) -> dict:
    """Upper anchor: Haar fidelities through the same pipeline, Haar states for MW."""
    dim = 2 ** n_qubits
    rng = np.random.default_rng(10_000 + seed)
    fids = sample_haar_fidelities(n_pairs, dim, rng)
    mw = np.empty(n_mw)
    for k in range(n_mw):
        z = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        mw[k] = meyer_wallach_score(z / np.linalg.norm(z), n_qubits)
    _, _, p_emp, p_haar = binned_distributions(fids, dim, n_bins)
    return {
        "ansatz": "haar", "R": 0, "x_setting": "none", "seed": seed,
        "n_theta": 0, "n_pairs": n_pairs, "n_mw": n_mw, "n_bins": n_bins,
        "kl": kl_divergence(p_emp, p_haar),
        "mw_mean": float(mw.mean()), "mw_sd": float(mw.std(ddof=1)),
        "fid_mean": float(fids.mean()), "wall_seconds": 0.0,
        "_fids": fids, "_mw": mw,
    }


def verdict(q_ratio: float, kl_factor: float) -> str:
    if q_ratio >= RICH_Q_RATIO and kl_factor <= RICH_KL_FACTOR:
        return "MET (rich reservoir)"
    if q_ratio < POOR_Q_RATIO or kl_factor > POOR_KL_FACTOR:
        return "NOT MET (outside the mixing regime)"
    return "INTERMEDIATE REGIME"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "b11"))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--mw", type=int, default=N_MW)
    ap.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    ap.add_argument(
        "--x-scale", type=float, default=1.0,
        help="multiplier on x; 2.0 == the pi/2 encoding range. FEATURE_RANGE is never modified.",
    )
    ap.add_argument(
        "--bins", type=int, default=N_BINS,
        help="histogram bins; KL depends on this, so cross-study comparison needs it matched",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "b11_raw_rows.csv"
    fid_dir = out_dir / "raw"
    fid_dir.mkdir(exist_ok=True)

    xs = x_settings(args.n_qubits, args.x_scale)
    print("=" * 100)
    print("the expressibility study -- KL expressibility vs Haar, and Meyer-Wallach entanglement")
    print("=" * 100)
    print(f"n_qubits {args.n_qubits}   pairs {args.pairs}   MW samples {args.mw}   bins {args.bins}")
    print(f"seeds    {args.seeds}")
    half = args.x_scale * FEATURE_RANGE[1]
    tag = "pi/4" if abs(half - np.pi / 4) < 1e-9 else "pi/2" if abs(half - np.pi / 2) < 1e-9 else "?"
    print(f"x        {list(xs)}   FEATURE_RANGE = {FEATURE_RANGE} (UNTOUCHED)")
    print(f"x_scale  {args.x_scale} -> effective half-width {half:.4f} ({tag})")
    print("metrics  VENDORED from QC1 commit 8855e2ec (qsocket/vendored/)")
    print(f"bands    rich: Q/Q_haar >= {RICH_Q_RATIO} and KL <= {RICH_KL_FACTOR}x floor | "
          f"poor: Q/Q_haar < {POOR_Q_RATIO} or KL > {POOR_KL_FACTOR}x floor")
    print()

    rows, written = [], False

    def emit(r: dict) -> None:
        nonlocal written
        fids, mw = r.pop("_fids"), r.pop("_mw")
        tag = f"{r['ansatz']}_R{r['R']}_{r['x_setting']}_s{r['seed']}"
        np.savez_compressed(fid_dir / f"{tag}.npz", fidelities=fids, mw=mw)
        with raw.open("a" if written else "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(r))
            if not written:
                w.writeheader()
            w.writerow(r)
        written = True
        rows.append(r)

    for seed in args.seeds:
        r = measure_haar(seed=seed, n_qubits=args.n_qubits, n_pairs=args.pairs, n_mw=args.mw,
                         n_bins=args.bins)
        print(f"  haar          seed {seed}: KL {r['kl']:.5f}  Q {r['mw_mean']:.4f} "
              f"(sd {r['mw_sd']:.4f})", flush=True)
        emit(r)

    for ansatz, R in CELLS:
        for x_label, x in xs.items():
            for seed in args.seeds:
                r = measure_cell(ansatz, R, x_label=x_label, x=x, seed=seed,
                                 n_qubits=args.n_qubits, n_pairs=args.pairs, n_mw=args.mw,
                                 n_bins=args.bins)
                mark = "  <-- ARM B" if (ansatz, R) == ("L1", 2) else ""
                print(f"  {ansatz:<8} R={R} {x_label:<8} seed {seed}: "
                      f"KL {r['kl']:.5f}  Q {r['mw_mean']:.4f} (sd {r['mw_sd']:.4f})  "
                      f"{r['wall_seconds']:.0f} s{mark}", flush=True)
                emit(r)

    # --- aggregation ---
    def mean_of(pred, key):
        v = [r[key] for r in rows if pred(r)]
        return float(np.mean(v)) if v else float("nan")

    kl_floor = mean_of(lambda r: r["ansatz"] == "haar", "kl")
    q_haar = mean_of(lambda r: r["ansatz"] == "haar", "mw_mean")

    print(f"\n  KL floor {kl_floor:.5f}   Q_haar {q_haar:.4f}")
    print("\n=== agregacja po x i ziarnach ===")
    print(f"  {'cell':<12} {'KL':>9} {'KL/floor':>9} {'Q':>8} {'Q/Q_haar':>9}   verdict")
    summary = {}
    for ansatz, R in CELLS:
        sel = lambda r, a=ansatz, d=R: r["ansatz"] == a and r["R"] == d
        kl, q = mean_of(sel, "kl"), mean_of(sel, "mw_mean")
        kf, qr = kl / kl_floor, q / q_haar
        summary[f"{ansatz}_R{R}"] = {"kl": kl, "kl_over_floor": kf, "q": q,
                                     "q_over_haar": qr, "verdict": verdict(qr, kf)}
        print(f"  {ansatz+' R='+str(R):<12} {kl:9.5f} {kf:9.2f} {q:8.4f} {qr:9.3f}   "
              f"{verdict(qr, kf)}")

    arm_b = summary["L1_R2"]
    print("\nVERDICT for arm B (L1, R=2); the bands were declared before the measurement:")
    print(f"  {arm_b['verdict']}   (Q/Q_haar = {arm_b['q_over_haar']:.3f}, "
          f"KL = {arm_b['kl_over_floor']:.2f}x podlogi)")
    if summary["L1_R2"]["q"] < 1e-3 and summary["product_R2"]["q"] < 1e-3:
        print("  STOP: Q(L1) ~ Q(product) ~ 0 -- gniazdo praktycznie nie splata, a G3 przechodzi.")

    # x-dependence check
    spread_x = {}
    for ansatz, R in CELLS:
        vals = [mean_of(lambda r, a=ansatz, d=R, xl=xl: r["ansatz"] == a and r["R"] == d
                        and r["x_setting"] == xl, "kl") for xl in xs]
        spread_x[f"{ansatz}_R{R}"] = float(np.nanmax(vals) - np.nanmin(vals))
    spread_ansatz = float(np.nanmax([summary[k]["kl"] for k in summary])
                          - np.nanmin([summary[k]["kl"] for k in summary]))
    worst_x = max(spread_x.values())
    print(f"\n  KL spread over x (max over cells): {worst_x:.5f}   "
          f"rozrzut KL po ansatzach: {spread_ansatz:.5f}")
    if worst_x > spread_ansatz:
        print("  STOP: KL zalezy od x mocniej niz od ansatzu -- mierzymy nie to, co nazywamy.")

    (out_dir / "b11_summary.json").write_text(
        json.dumps({"n_qubits": args.n_qubits, "n_pairs": args.pairs, "n_mw": args.mw,
                    "n_bins": args.bins, "seeds": args.seeds,
                    "kl_floor_haar": kl_floor, "q_haar": q_haar,
                    "bands": {"rich_q": RICH_Q_RATIO, "poor_q": POOR_Q_RATIO,
                              "rich_kl": RICH_KL_FACTOR, "poor_kl": POOR_KL_FACTOR},
                    "per_cell": summary,
                    "kl_spread_over_x": spread_x,
                    "kl_spread_over_ansatz": spread_ansatz,
                    "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\n  {raw}\n  {out_dir / 'b11_summary.json'}\n  {fid_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
