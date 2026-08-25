"""Exact paired tests: the sign test and the Wilcoxon signed-rank test.

Assertions carried over in `tests/test_analysis_pipeline.py`:
  * exact zeros are DROPPED before ranking, never counted as ties on either side,
  * ties take average ranks,
  * the two-sided p-value is the fraction of sign patterns at or below the observed
    statistic, with a 1e-12 tolerance on the comparison.
"""

from __future__ import annotations

import math

import numpy as np

# Differences closer to zero than this are dropped rather than ranked.
ZERO_TOLERANCE = 1e-12


def average_ranks(values: list[float]) -> list[float]:
    """Ranks of `values`, ties sharing their average rank. 1-based, ascending."""
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    next_rank = 1
    while start < len(order):
        end = start
        while end + 1 < len(order) and math.isclose(
            values[order[end + 1]], values[order[start]], rel_tol=0.0, abs_tol=ZERO_TOLERANCE
        ):
            end += 1
        avg_rank = (next_rank + next_rank + (end - start)) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        next_rank += end - start + 1
        start = end + 1
    return ranks


def _signed_rank_sum_counts(scaled_ranks: list[int], total: int) -> list[int]:
    """counts[s] = how many of the 2**n sign patterns put exactly `s` on the positive side.

    A subset-sum count over the rank multiset, in exact integer arithmetic. The states are
    the reachable positive-side sums, so the table is `total + 1` long rather than 2**n.
    """
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in scaled_ranks:
        # Descending, so each rank is used at most once per pattern.
        for value in range(total - rank, -1, -1):
            if counts[value]:
                counts[value + rank] += counts[value]
    return counts


def wilcoxon_signed_rank_exact(differences: list[float]) -> dict[str, float | int | None]:
    """Two-sided exact Wilcoxon signed-rank test on paired differences.

    The p-value is the exact permutation probability over the 2**n assignments of signs,
    computed by counting rather than enumeration.
    """
    diffs = [
        float(value)
        for value in differences
        if not math.isclose(float(value), 0.0, abs_tol=ZERO_TOLERANCE)
    ]
    if not diffs:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "pvalue": 1.0,
            "rank_biserial": 0.0,
            "median_difference": 0.0,
        }

    ranks = average_ranks([abs(v) for v in diffs])
    total_rank = float(sum(ranks))
    positive_rank_sum = float(sum(rank for diff, rank in zip(diffs, ranks) if diff > 0))
    negative_rank_sum = total_rank - positive_rank_sum
    statistic = float(min(positive_rank_sum, negative_rank_sum))
    rank_biserial = (
        float((positive_rank_sum - negative_rank_sum) / total_rank) if total_rank else 0.0
    )

    # Average ranks are integers or half-integers, so doubling them makes the DP exact in integers.
    scaled = [int(round(rank * 2)) for rank in ranks]
    total_scaled = sum(scaled)
    counts = _signed_rank_sum_counts(scaled, total_scaled)
    statistic_scaled = statistic * 2
    favourable = sum(
        count
        for positive_side, count in enumerate(counts)
        if min(positive_side, total_scaled - positive_side) <= statistic_scaled + ZERO_TOLERANCE
    )
    pvalue = favourable / (2 ** len(ranks))

    return {
        "n_nonzero": int(len(diffs)),
        "statistic": statistic,
        "pvalue": float(pvalue),
        "rank_biserial": rank_biserial,
        "median_difference": float(np.median(diffs)),
    }


def sign_test_exact(differences: list[float]) -> dict[str, float | int]:
    """Two-sided exact sign test on paired differences."""
    diffs = [
        float(value)
        for value in differences
        if not math.isclose(float(value), 0.0, abs_tol=ZERO_TOLERANCE)
    ]
    n = len(diffs)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "negative": 0, "pvalue": 1.0}
    positive = sum(1 for value in diffs if value > 0)
    negative = n - positive
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    pvalue = min(1.0, 2.0 * probability)
    return {
        "n_nonzero": int(n),
        "positive": int(positive),
        "negative": int(negative),
        "pvalue": float(pvalue),
    }
