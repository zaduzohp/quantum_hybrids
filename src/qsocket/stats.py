"""The paired-design statistics, in one place.
Everything here is a pure function of (sigma, n, delta). None of it reads a result, and
none of it carries a value measured in this project.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.stats import chi2, norm
from scipy.stats import t as student_t


def mde_constant(n: int) -> float:
    """(t_.975,n-1 + t_.80,n-1) / sqrt(n) — the MDE per unit of sigma."""
    if n < 2:
        return float("nan")
    df = n - 1
    return float((student_t.ppf(0.975, df) + student_t.ppf(0.80, df)) / math.sqrt(n))


def mde(sigma: float, n: int) -> float:
    """Minimum detectable effect of the two-sided paired test at 80 % power."""
    return float(mde_constant(n) * sigma)


def sigma_confidence_interval(sd: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """CI for sigma itself; at n = 10 the factor is 1.83/0.69.

    sigma is estimated, so a two-fold difference between two cells is not a difference
    until their intervals stop overlapping.
    """
    if n < 2 or not np.isfinite(sd):
        return (float("nan"), float("nan"))
    df = n - 1
    tail = (1.0 - level) / 2.0
    return (
        float(sd * math.sqrt(df / chi2.ppf(1.0 - tail, df))),
        float(sd * math.sqrt(df / chi2.ppf(tail, df))),
    )


def binomial_se(n_eval: int, p: float = 0.5) -> float:
    """SE of a single accuracy on n_eval rows — the second of the three uncertainty
    accounts, never added to the others."""
    return float(math.sqrt(p * (1.0 - p) / n_eval))


def tost_power(*, sigma: float, n: int, delta: float, true_delta: float = 0.0,
               alpha: float = 0.05, df: int | None = None) -> float:
    """Exact power of the paired TOST, by quadrature — no Monte Carlo, so it is stable.

    TOST rejects exactly when |mean| + t_{1-alpha,df} * s / sqrt(n) < delta. The sample
    mean and s are independent, mean ~ N(true_delta, sigma^2/n) and s^2 df / sigma^2 ~
    chi2(df), so the power is a one-dimensional integral over the chi2 density.

    `df` must be the degrees of freedom the TEST is run on: a blocked design spends one
    per block, and n - 1 there is the power of a test nobody performed.
    """
    if n < 2 or not np.isfinite(sigma) or sigma <= 0 or delta <= 0:
        return float("nan") if delta > 0 else 0.0
    df = n - 1 if df is None else int(df)
    if df < 1:
        return float("nan")
    t_crit = float(student_t.ppf(1.0 - alpha, df))
    se = sigma / math.sqrt(n)
    # c(w) > 0 bounds the region where rejection is possible at all.
    w_max = df * (delta * math.sqrt(n) / (t_crit * sigma)) ** 2

    def integrand(w: float) -> float:
        s = sigma * math.sqrt(w / df)
        c = delta - t_crit * s / math.sqrt(n)
        if c <= 0:
            return 0.0
        probability = norm.cdf((c - true_delta) / se) - norm.cdf((-c - true_delta) / se)
        return float(probability * chi2.pdf(w, df))

    value, _ = integrate.quad(integrand, 0.0, w_max, limit=400)
    return float(min(max(value, 0.0), 1.0))
