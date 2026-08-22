# VENDORED from qbanknote/stats.py:38, :57 and :90
# (commit 8855e2ec99bdc39b5b94dec0b18ea87b9fd50fe9, 2026-07-19)
# Upstream: ~/QC1_Quantum_Banknote_Classifier_on_ODRA_5
# Changes vs upstream: none — the three functions are literal copies, this header and the
# module docstring are the only additions.
#
# average_ranks came along because wilcoxon_signed_rank_exact calls it: copying the
# caller without the callee would have meant reimplementing tie handling.
#
# The 2**n enumeration is the point: at n = 10 it is 1024
# combinations and the p-value is the exact permutation distribution over signs, not a
# normal approximation. Accuracy differences on 1200 test rows are discrete with step
# 1/1200, so the normality the t-test assumes is not available.
#
# Behaviour pinned by tests/test_analysis_pipeline.py (including the assertions carried over
# from the upstream tests/test_smoke.py:194-198).
"""VENDOR: average_ranks, wilcoxon_signed_rank_exact, sign_test_exact."""

from __future__ import annotations

import itertools
import math

import numpy as np


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    next_rank = 1
    while start < len(order):
        end = start
        while end + 1 < len(order) and math.isclose(
            values[order[end + 1]], values[order[start]], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        avg_rank = (next_rank + next_rank + (end - start)) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        next_rank += end - start + 1
        start = end + 1
    return ranks


def wilcoxon_signed_rank_exact(differences: list[float]) -> dict[str, float | int | None]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
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
    rank_biserial = float((positive_rank_sum - negative_rank_sum) / total_rank) if total_rank else 0.0

    distribution = []
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        signed_positive_sum = sum(rank for sign, rank in zip(signs, ranks) if sign == 1)
        distribution.append(min(signed_positive_sum, total_rank - signed_positive_sum))
    pvalue = sum(1 for value in distribution if value <= statistic + 1e-12) / len(distribution)

    return {
        "n_nonzero": int(len(diffs)),
        "statistic": statistic,
        "pvalue": float(pvalue),
        "rank_biserial": rank_biserial,
        "median_difference": float(np.median(diffs)),
    }


def sign_test_exact(differences: list[float]) -> dict[str, float | int]:
    diffs = [float(value) for value in differences if not math.isclose(float(value), 0.0, abs_tol=1e-12)]
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
