"""sigma_Delta with its own confidence interval, and the sample size equivalence needs.

A figure about the PRECISION of the series, not its effect. Two questions the tables
answer only cell by cell:

  (a) how noisy is one paired difference, and how well is that noise itself known?
      sigma_Delta comes from n - J degrees of freedom, so it carries a chi-square interval.
      Reported bare, a factor-of-two gap between cells reads as real when the intervals
      overlap completely.
  (b) how many seeds would equivalence at delta = 0.02 need? TOST is not a verdict for
      Delta_AB because its power here is nil; the honest form of that is a number, and
      drawn against the n the series has it shows a gap of an order of magnitude.

Both quantities are read off the A8 tables, never recomputed from raw results: this script
draws what the pipeline decided. The seeds-needed column is recomputed through the same
function A8 uses and checked against diagnostics.csv — not an independent implementation,
just a guard that both CSVs came from one run.

NOT a power curve for the reported contrast: Delta_AB is a CI excluding zero in six of
eight cells and nothing here revises it. The sample-size panel is about equivalence only.

    python scripts/plot_sigma_power.py --tables outputs/stats/main/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_a8_analysis as a8

from qsocket.head import DILUTION_AXIS

# The same visual vocabulary as the A8 figures: colour is never the only carrier.
MARKERS = a8.MARKERS
LINESTYLES = a8.LINESTYLES
CENSORED_AT = 400  # the n_max of seeds_needed_for_tost; above it the answer is "> 400"


def load(tables: Path, estimands: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(tables / "estimands.csv")
    missing = [name for name in estimands if name not in set(frame["estimand"])]
    if missing:
        raise SystemExit(f"estimands.csv has no rows for {missing}")
    status = str(frame["status"].iloc[0])
    frame = frame[frame["estimand"].isin(estimands)].copy()
    frame["dilution"] = pd.Categorical(
        frame["dilution"], categories=[d for d in DILUTION_AXIS], ordered=True)
    return frame.sort_values(["estimand", "ansatz", "dilution"]), status


def seeds_needed(frame: pd.DataFrame, delta: float) -> pd.Series:
    """seeds needed for 80 % TOST power at `delta`, one per row, None above n_max.

    n_blocks comes from the row, not from a default: the blocked design spends one degree
    of freedom per generator seed, and computing this on n - 1 while A8 computed it on
    n - J makes cross_check() below fire on a difference the pipeline never had.
    """
    return pd.Series(
        [
            a8.seeds_needed_for_tost(
                sigma=float(row.sigma_delta), delta=delta, blocks=int(row.n_blocks)
            )
            for row in frame.itertuples()
        ],
        index=frame.index,
    )


def cross_check(tables: Path, frame: pd.DataFrame, delta: float) -> list[str]:
    """The A8 diagnostics rows for the same quantity must agree with what we recomputed.

    Only checkable at A8's own delta: at any other delta A8 wrote no such row, and the
    check is skipped rather than faked.
    """
    path = tables / "diagnostics.csv"
    if delta != a8.TOST_DELTA or not path.exists():
        return []
    diagnostics = pd.read_csv(path)
    rows = diagnostics[diagnostics["quantity"] == "seeds needed for 80% TOST power"]
    written = {str(scope): value for scope, value in zip(rows["scope"], rows["value"])}
    complaints = []
    for _, row in frame.iterrows():
        scope = f"{row['dilution']}|{row['ansatz']}|{row['estimand']}"
        if scope not in written:
            continue
        theirs, ours = written[scope], row["seeds_needed"]
        theirs = None if pd.isna(theirs) else int(theirs)
        ours = None if ours is None or pd.isna(ours) else int(ours)
        if theirs != ours:
            complaints.append(
                f"{scope}: diagnostics.csv says {theirs}, sigma in estimands.csv gives {ours}")
    return complaints


def draw(frame: pd.DataFrame, status: str, delta: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dilutions = [d for d in DILUTION_AXIS if d in set(frame["dilution"])]
    x = np.arange(len(dilutions), dtype=float)
    series = list(dict.fromkeys(zip(frame["estimand"], frame["ansatz"])))
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0), "hspace": 0.12},
    )

    n_values, mde_constants = set(), set()
    for i, (estimand, ansatz) in enumerate(series):
        block = frame[(frame["estimand"] == estimand) & (frame["ansatz"] == ansatz)]
        label = f"{estimand.replace('delta_', 'Δ_')}, {ansatz}"
        offset = 0.06 * (i - (len(series) - 1) / 2)
        xs, sigma, low, high, needed = [], [], [], [], []
        for k, dilution in enumerate(dilutions):
            row = block[block["dilution"] == dilution]
            if row.empty:
                continue
            row = row.iloc[0]
            xs.append(x[k] + offset)
            sigma.append(float(row["sigma_delta"]))
            low.append(float(row["sigma_delta"]) - float(row["sigma_ci95_low"]))
            high.append(float(row["sigma_ci95_high"]) - float(row["sigma_delta"]))
            needed.append(row["seeds_needed"])
            n_values.add(int(row["n"]))
            mde_constants.add(round(float(row["mde_constant"]), 6))
        if not xs:
            continue
        style = {
            "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)],
            "markersize": 4.5,
            "linewidth": 1.1,
        }
        bars = top.errorbar(xs, sigma, yerr=[low, high], capsize=3, label=label, **style)
        colour = bars.lines[0].get_color()
        # Censored points are drawn at the ceiling with an upward marker, never dropped:
        # "more seeds than we searched for" is the finding, and a gap in the line would
        # read as missing data.
        finite = [(xi, v) for xi, v in zip(xs, needed) if v is not None and not pd.isna(v)]
        censored = [xi for xi, v in zip(xs, needed) if v is None or pd.isna(v)]
        if finite:
            bottom.plot([xi for xi, _ in finite], [int(v) for _, v in finite],
                        color=colour, **style)
        if censored:
            bottom.plot(censored, [CENSORED_AT] * len(censored), marker="^",
                        linestyle="none", markersize=7, markerfacecolor="none",
                        color=colour)

    top.set_yscale("log")
    # Headroom above the data so the legend sits on empty axes instead of on the h2 peak.
    lower, upper = top.get_ylim()
    top.set_ylim(lower, upper * 2.6)
    # sigma and MDE differ by the constant c(n) = (t_.975 + t_.80)/sqrt(n) exactly, so the
    # right-hand axis is the same axis relabelled and no second curve is needed. It is
    # drawn only when every cell shares one n, since otherwise the constant is not one
    # number and a single relabelling would be a lie.
    if len(mde_constants) == 1:
        constant = mde_constants.pop()
        twin = top.twinx()
        twin.set_yscale("log")
        twin.set_ylim(*(v * constant for v in top.get_ylim()))
        twin.set_ylabel(f"MDE = {constant:.3f}·σ_Δ  (accuracy)", fontsize=8)
        twin.tick_params(labelsize=7)
    top.set_ylabel("σ_Δ, paired difference (accuracy)", fontsize=9)
    top.set_title(
        "σ of one paired difference, with its own 95 % CI — and what equivalence would cost",
        fontsize=10,
    )
    top.legend(fontsize=7, ncol=2, loc="upper left")
    top.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    achieved = sorted(n_values)
    if achieved:
        bottom.axhline(achieved[0], color="black", linestyle="--", linewidth=1.0)
        bottom.annotate(
            f"n of this series = {achieved[0]}", xy=(0.02, achieved[0]),
            xycoords=bottom.get_yaxis_transform(), xytext=(0, 3),
            textcoords="offset points", fontsize=7, va="bottom")
    bottom.axhline(CENSORED_AT, color="0.7", linewidth=0.8)
    # Right-aligned: the censored markers sit on the left half of this line.
    bottom.annotate(
        f"search stopped at n = {CENSORED_AT}   (△ = more than this)",
        xy=(0.98, CENSORED_AT), xycoords=bottom.get_yaxis_transform(), xytext=(0, -9),
        textcoords="offset points", fontsize=6.5, color="0.4", va="top", ha="right")
    bottom.set_yscale("log")
    bottom.set_ylim(top=CENSORED_AT * 1.6)
    bottom.set_ylabel(
        f"seeds for 80 % TOST power at δ = {delta:g}", fontsize=9)
    bottom.set_xlabel("dilution (head parameters increase →)", fontsize=9)
    bottom.set_xticks(x)
    bottom.set_xticklabels(dilutions, fontsize=8)
    bottom.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    for panel in (top, bottom):
        panel.tick_params(labelsize=7)
    figure.text(
        0.5, 0.015,
        "Lower panel is the cost of an EQUIVALENCE claim at this δ, not of the reported "
        "contrast. Nothing here revises Δ_AB.",
        ha="center", fontsize=6.5, color="0.35",
    )
    if status != a8.STATUS_COMPLETE:
        a8._stamp(figure, a8.STATUS_PROVISIONAL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tables", required=True, type=Path,
        help="the A8 tables directory (contains estimands.csv and diagnostics.csv)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output PDF; defaults to <tables>/../figures/fig_sigma_power.pdf")
    parser.add_argument(
        "--estimands", nargs="+", default=["delta_AB", "delta_AE"],
        help="which estimands to draw (default: the two the report reads along the axis)")
    parser.add_argument(
        "--delta", type=float, default=a8.TOST_DELTA,
        help=f"equivalence margin for the sample-size panel (default {a8.TOST_DELTA}). "
             "At any other value the cross-check against diagnostics.csv is skipped.")
    args = parser.parse_args()

    frame, status = load(args.tables, tuple(args.estimands))
    frame["seeds_needed"] = seeds_needed(frame, args.delta)
    complaints = cross_check(args.tables, frame, args.delta)
    if complaints:
        raise SystemExit(
            "estimands.csv and diagnostics.csv disagree — the tables are not from one "
            "run, and the figure is not drawn:\n  " + "\n  ".join(complaints))

    out_path = args.out or (args.tables.parent / "figures" / "fig_sigma_power.pdf")
    draw(frame, status, args.delta, out_path)

    print(f"status  {status}")
    print(f"delta   {args.delta:g}   (TOST margin for the sample-size panel)")
    columns = ["estimand", "ansatz", "dilution", "n", "sigma_delta",
               "sigma_ci95_low", "sigma_ci95_high", "mde_this_series", "seeds_needed"]
    print(frame[columns].to_string(index=False))
    print(f"figure  {out_path}")


if __name__ == "__main__":
    main()
