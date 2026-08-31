"""The estimators of the paired design: point estimates, blocking, TOST, McNemar.

Pure functions over lists of paired differences and over the blocked structure
(dataset seed as a FIXED effect). Nothing here reads a CSV, knows an arm name or decides
what a verdict means -- that stays in the analysis driver. Split out of it so the main
series and the exploratory probes cannot end up estimating the same quantity two ways.

The three uncertainty accounts are computed separately and NEVER summed:
the CI over the paired differences, the binomial SE on the test rows (qsocket.stats),
and McNemar on the discordant pairs.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t

from qsocket import stats
from qsocket.head import DILUTION_AXIS as CONTRACT_DILUTIONS
from qsocket.stats import (
    mde_constant,
    sigma_confidence_interval,
    sign_test_exact,
    wilcoxon_signed_rank_exact,
)

TOST_DELTA = 0.02


TOST_ALPHA = 0.05


TOST_POWER_FLOOR = 0.80


POOLING_DIVERGENCE_RULE = (
    "any per-dataset 95% CI failing to overlap the pooled 95% CI. On divergence the "
    "PER-DATASET reading wins and the divergence is a finding, not a defect."
)


MIXEDLM_EQUIVALENCE_TOL = 1e-6


PairKey = namedtuple("PairKey", "dataset_seed seed")


def paired_differences(left: dict, right: dict, *, label: str) -> dict:
    """left - right, one difference per (generator seed, training seed) shared by both.

    Both key sets must be PairKey: a series keyed by a bare seed would pair across
    generator seeds and produce a plausible number for a quantity nobody defined.

    Keys present on one side only are reported as unpaired, never dropped into the mean —
    an unbalanced pairing makes sigma_Delta a different quantity.
    """
    for name, mapping in (("left", left), ("right", right)):
        bad = [k for k in mapping if not isinstance(k, PairKey)]
        if bad:
            raise ValueError(
                f"{label}: {name} series is keyed by {bad[:3]!r}, not PairKey. A paired "
                "difference must be taken within one training seed AND one generator "
                "seed; keying by seed alone would silently pair across datasets."
            )
    shared = sorted(set(left) & set(right))
    return {
        "label": label,
        "keys": shared,
        "differences": [float(left[k] - right[k]) for k in shared],
        "left_values": [float(left[k]) for k in shared],
        "right_values": [float(right[k]) for k in shared],
        "unpaired_left": sorted(set(left) - set(right)),
        "unpaired_right": sorted(set(right) - set(left)),
    }


def estimate(differences: list[float]) -> dict:
    """The paired contrast: point estimate, 95% and 90% CI, sigma with its own CI, MDE.

    The 90% CI is included because TOST is equivalently "the 90% CI lies inside
    (-delta, +delta)", which is the reported form.
    """
    values = np.asarray(differences, dtype=float)
    n = int(values.size)
    out: dict = {
        "n": n,
        "mean": float(values.mean()) if n else float("nan"),
        "sd": float(values.std(ddof=1)) if n > 1 else float("nan"),
        "min": float(values.min()) if n else float("nan"),
        "max": float(values.max()) if n else float("nan"),
        "n_positive": int(np.sum(values > 0)),
        "n_negative": int(np.sum(values < 0)),
        "n_zero": int(np.sum(values == 0)),
    }
    if n > 1:
        df = n - 1
        se = out["sd"] / math.sqrt(n)
        out["se"] = float(se)
        for level, key in ((0.95, "ci95"), (0.90, "ci90")):
            half = float(student_t.ppf(0.5 + level / 2.0, df) * se)
            out[f"{key}_low"] = out["mean"] - half
            out[f"{key}_high"] = out["mean"] + half
            out[f"{key}_half_width"] = half
        low, high = sigma_confidence_interval(out["sd"], n)
        out["sigma_ci95_low"], out["sigma_ci95_high"] = low, high
        out["mde_constant"] = mde_constant(n)
        out["mde"] = out["mde_constant"] * out["sd"]
        out["above_mde"] = bool(abs(out["mean"]) >= out["mde"])
        out["ci95_excludes_zero"] = bool(out["ci95_low"] > 0 or out["ci95_high"] < 0)
    else:
        for key in (
            "se", "ci95_low", "ci95_high", "ci95_half_width", "ci90_low", "ci90_high",
            "ci90_half_width", "sigma_ci95_low", "sigma_ci95_high", "mde_constant", "mde",
        ):
            out[key] = float("nan")
        out["above_mde"] = False
        out["ci95_excludes_zero"] = False

    # Both exact tests, plus t for comparison.
    sign = sign_test_exact(list(values))
    wilcoxon = wilcoxon_signed_rank_exact(list(values))
    out["p_sign_exact"] = float(sign["pvalue"])
    out["sign_positive"] = int(sign["positive"])
    out["sign_negative"] = int(sign["negative"])
    out["p_wilcoxon_exact"] = float(wilcoxon["pvalue"])
    out["wilcoxon_statistic"] = float(wilcoxon["statistic"])
    if n > 1 and out["sd"] > 0:
        t_stat = out["mean"] / out["se"]
        out["t_statistic"] = float(t_stat)
        out["p_t"] = float(2.0 * student_t.sf(abs(t_stat), n - 1))
    else:
        out["t_statistic"] = float("nan")
        out["p_t"] = float("nan")
    return out


def in_reference_units(
    point: dict,
    *,
    sigma_seed_left,
    sigma_seed_right,
    ceilings: dict | None = None,
) -> dict:
    """Delta expressed in sigma_seed and in ceiling units, the form the thesis is stated in.

    "In units of sigma_seed" does not name which arm's sigma_seed, so both arms of the
    contrast are reported plus sigma_Delta of the contrast itself.

    The ceiling denominator is per generator seed, never pooled: the ceiling differs
    between datasets, so an average of it describes none of them. `ceilings` maps
    dataset_seed -> ceiling and the ratio is formed block by block. The single-number
    `in_ceiling` is emitted only when there is exactly one block.
    """

    def ratio(numerator, denominator):
        if denominator is None or not np.isfinite(denominator) or denominator == 0:
            return float("nan")
        return float(numerator / denominator)

    ceilings = {str(k): float(v) for k, v in (ceilings or {}).items()
                if v is not None and np.isfinite(float(v))}
    block_means = point.get("block_means") or {}
    if not block_means and len(ceilings) == 1 and point.get("n"):
        # No blocking information and exactly one ceiling: the estimate is that one
        # block. `estimate_blocked` supplies block_means; this is the bare `estimate()`
        # path.
        block_means = {next(iter(ceilings)): float(point["mean"])}
    in_ceiling_by_seed = {
        seed: ratio(mean, ceilings.get(seed))
        for seed, mean in sorted(block_means.items())
        if seed in ceilings
    }
    # A single pooled ratio is only meaningful when there is a single block.
    single = (
        next(iter(in_ceiling_by_seed.values()))
        if len(in_ceiling_by_seed) == 1 else float("nan")
    )
    return {
        "in_sigma_seed_left": ratio(point["mean"], sigma_seed_left),
        "in_sigma_seed_right": ratio(point["mean"], sigma_seed_right),
        "in_sigma_delta": ratio(point["mean"], point["sd"]),
        "in_ceiling": single,
        "in_ceiling_by_seed": in_ceiling_by_seed,
        "ceiling_by_seed": {k: v for k, v in sorted(ceilings.items())},
        "sigma_seed_left": sigma_seed_left,
        "sigma_seed_right": sigma_seed_right,
        "ceiling_used": (
            next(iter(ceilings.values())) if len(ceilings) == 1 else float("nan")
        ),
    }


def arm_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    sd = float(array.std(ddof=1)) if n > 1 else float("nan")
    low, high = sigma_confidence_interval(sd, n)
    return {
        "n": n,
        "mean": float(array.mean()) if n else float("nan"),
        "sd": sd,
        "sigma_ci95_low": low,
        "sigma_ci95_high": high,
        "min": float(array.min()) if n else float("nan"),
        "max": float(array.max()) if n else float("nan"),
    }


def tost(differences: list[float], *, delta: float = TOST_DELTA, alpha: float = TOST_ALPHA,
         se: float | None = None, df: int | None = None, sd: float | None = None) -> dict:
    """Two one-sided t tests. Equivalently: the 90% CI inside (-delta, +delta).

    `se`, `df` and `sd` may be supplied to run the test on the blocked residual instead
    of the iid spread of `differences`; they must be passed together, or the result is a
    hybrid nobody declared. Without them the iid reading is used, which is right for a
    single block.
    """
    values = np.asarray(differences, dtype=float)
    n = int(values.size)
    if n < 2:
        return {"n": n, "computable": False}
    mean = float(values.mean())
    supplied = [x is not None for x in (se, df, sd)]
    if any(supplied) and not all(supplied):
        raise ValueError("tost: pass se, df and sd together or none of them")
    if all(supplied):
        df = int(df)
        se = float(se)
        sd = float(sd)
    else:
        df = n - 1
        sd = float(values.std(ddof=1))
        se = sd / math.sqrt(n)
    if se == 0:
        p_lower = 0.0 if mean > -delta else 1.0
        p_upper = 0.0 if mean < delta else 1.0
    else:
        p_lower = float(student_t.sf((mean + delta) / se, df))
        p_upper = float(student_t.cdf((mean - delta) / se, df))
    half = float(student_t.ppf(1.0 - alpha, df) * se)
    return {
        "n": n,
        "computable": True,
        "delta": float(delta),
        "alpha": float(alpha),
        # The df the test was actually run on, so its power can be computed on the same
        # one rather than on an iid n - 1 nobody used.
        "df": int(df),
        "mean": mean,
        "sd": sd,
        "ci90_low": mean - half,
        "ci90_high": mean + half,
        "ci90_half_width": half,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "pvalue": float(max(p_lower, p_upper)),
        "equivalent": bool(max(p_lower, p_upper) < alpha),
        # When the half width exceeds delta, no result — not even Delta = 0 exactly —
        # could declare equivalence.
        "half_width_exceeds_delta": bool(half > delta),
    }


def tost_power(*, sigma: float, n: int, delta: float = TOST_DELTA, true_delta: float = 0.0,
               alpha: float = TOST_ALPHA, df: int | None = None) -> float:
    """Exact power of the paired TOST — qsocket.stats.tost_power at this study's delta.

    `df` must be the degrees of freedom the TEST is run on. Defaulting to n - 1 while the
    test itself ran on the blocked residual (n - J) reports the power of a test nobody
    performed; at this series' numbers the two differ by ~0.002, which is small but sits
    directly under the 0.80 floor a STOP row is decided on.
    """
    return stats.tost_power(
        sigma=sigma, n=n, delta=delta, true_delta=true_delta, alpha=alpha, df=df
    )


def seeds_needed_for_mde(*, sigma: float, effect: float, n_max: int = 2000) -> int | None:
    """Smallest n whose MDE = c(n)*sigma is no larger than `effect`.

    "|Delta| below MDE" must never read as "there is no effect", so it is accompanied by
    the number of seeds that would decide it. Solved by scanning n: c(n) contains two t
    quantiles and the inversion is not closed-form.
    """
    if not (np.isfinite(sigma) and np.isfinite(effect)) or sigma <= 0 or effect <= 0:
        return None
    for n in range(3, n_max + 1):
        if mde_constant(n) * sigma <= effect:
            return n
    return None


def seeds_needed_for_tost(*, sigma: float, delta: float = TOST_DELTA,
                          target_power: float = TOST_POWER_FLOOR, n_max: int = 400,
                          blocks: int = 1) -> int | None:
    """How many training seeds equivalence at this delta would need.

    Reported instead of a verdict when the power is nil — a decision fixed before the
    analysis ran.

    `blocks` is the number of generator seeds: the blocked design spends one degree of
    freedom per block, so the answer at n differences is the power on n - blocks df, the
    same df the test itself would run on.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    blocks = max(1, int(blocks))
    for n in range(blocks + 2, n_max + 1):
        if tost_power(sigma=sigma, n=n, delta=delta, df=n - blocks) >= target_power:
            return n
    return None


def pooled_blocked_estimate(pairs: dict) -> dict:
    """Paired differences blocked by dataset, with the dataset a fixed effect.

    Three levels are not enough for a variance component, so the model is
    d_ij = mu + alpha_j: mu is estimated by the unweighted mean of the block means, and
    the standard error comes from the within-block residual variance on N - J degrees of
    freedom. More power on the same estimand, and not a random effect.
    """
    by_block: dict[int, list[float]] = {}
    for key, value in zip(pairs["keys"], pairs["differences"]):
        by_block.setdefault(key.dataset_seed, []).append(float(value))
    blocks = sorted(by_block)
    if not blocks:
        return {"computable": False, "n": 0}
    counts = np.array([len(by_block[b]) for b in blocks], dtype=float)
    means = np.array([float(np.mean(by_block[b])) for b in blocks], dtype=float)
    n_total = int(counts.sum())
    n_blocks = len(blocks)
    df = n_total - n_blocks
    point = float(means.mean())
    residual_ss = float(
        sum(((np.asarray(by_block[b], dtype=float) - means[i]) ** 2).sum()
            for i, b in enumerate(blocks))
    )
    out: dict = {
        "computable": True,
        "blocks": blocks,
        "n": n_total,
        "n_blocks": n_blocks,
        "block_means": {str(b): float(means[i]) for i, b in enumerate(blocks)},
        "block_counts": {str(b): int(counts[i]) for i, b in enumerate(blocks)},
        "mean": point,
        "df_residual": int(df),
        "model": "d_ij = mu + alpha_j (dataset = FIXED effect); estimate = unweighted mean of block means",
        "not_a_variance_component": (
            "three levels do not support a variance component (SPEC 7.6: the CI on sigma "
            "at n=3 spans a factor of 12). This is a blocking factor."
        ),
    }
    if df > 0:
        mse = residual_ss / df
        # SE of the unweighted mean of block means, correct under imbalance too.
        se = math.sqrt(mse * float(np.sum(1.0 / counts)) / n_blocks**2)
        half95 = float(student_t.ppf(0.975, df) * se)
        out.update(
            {
                "mse_within": float(mse),
                "sigma_within": float(math.sqrt(mse)),
                "se": float(se),
                "ci95_low": point - half95,
                "ci95_high": point + half95,
                "ci95_half_width": half95,
            }
        )
    else:
        out.update({k: float("nan") for k in
                    ("mse_within", "sigma_within", "se", "ci95_low", "ci95_high", "ci95_half_width")})
    # Exact tests on the pooled differences. No size guard: the sign-test tail is a sum
    # of binomial coefficients, O(n), not an enumeration of 2**n.
    every = [float(v) for v in pairs["differences"]]
    out["p_sign_exact"] = float(sign_test_exact(every)["pvalue"])
    return out


def estimate_blocked(pairs: dict) -> dict:
    """`estimate()` with the uncertainty taken from the blocked model.

    Treating the generator seeds as one iid sample folds between-dataset difficulty into
    sigma_Delta — a variance component in disguise on a fixed effect with too few levels —
    and sigma_Delta is the denominator of the headline result.

      * the point estimate is unchanged: the design is balanced, so the unweighted mean of
        block means equals the naive mean exactly;
      * sd / se / ci95 / ci90 / mde come from the within-block residual, N - J df;
      * the exact sign and Wilcoxon tests stay on the pooled differences, which test
        "median difference = 0" across datasets and need no blocking;
      * between-block spread is its own field, not hidden inside the interval, because
        variation across datasets is a finding.

    The tighter interval is what lets divergence_check notice that blocks disagree; where
    it fires, the per-dataset reading wins and this pooled number is not the headline.
    """
    values = [float(v) for v in pairs["differences"]]
    out = estimate(values)
    blocked = pooled_blocked_estimate(pairs)
    out["pooling"] = "blocked (generator seed = FIXED effect, SPEC 7.3)"
    if not blocked.get("computable") or not np.isfinite(blocked.get("se", float("nan"))):
        # One block, or no residual degrees of freedom: keep the iid numbers and say so,
        # rather than emitting a blocked-looking interval that is not one.
        out["pooling"] = "iid (blocking not possible: %s block(s))" % blocked.get("n_blocks", 0)
        out["blocked_computable"] = False
        return out

    block_means = [float(v) for v in blocked["block_means"].values()]
    df = int(blocked["df_residual"])
    se = float(blocked["se"])
    out.update(
        {
            "blocked_computable": True,
            "n_blocks": blocked["n_blocks"],
            "block_means": blocked["block_means"],
            "df_residual": df,
            # sigma_Delta is the within-dataset residual sd, the intended denominator.
            "sd": float(blocked["sigma_within"]),
            "sd_pooled_iid": out["sd"],
            "between_block_spread": (
                float(np.std(block_means, ddof=1)) if len(block_means) > 1 else 0.0
            ),
            "se": se,
        }
    )
    for level, key in ((0.95, "ci95"), (0.90, "ci90")):
        half = float(student_t.ppf(0.5 + level / 2.0, df) * se)
        out[f"{key}_low"] = out["mean"] - half
        out[f"{key}_high"] = out["mean"] + half
        out[f"{key}_half_width"] = half
    low, high = sigma_confidence_interval(out["sd"], df + 1)
    out["sigma_ci95_low"], out["sigma_ci95_high"] = low, high
    # MDE in the blocked design: the same (t_0.975 + t_0.80) * se as the iid case, but on
    # the blocked df and se.
    out["mde_constant"] = float(
        (student_t.ppf(0.975, df) + student_t.ppf(0.80, df)) / math.sqrt(out["n"])
    )
    out["mde"] = float((student_t.ppf(0.975, df) + student_t.ppf(0.80, df)) * se)
    out["above_mde"] = bool(abs(out["mean"]) >= out["mde"])
    out["ci95_excludes_zero"] = bool(out["ci95_low"] > 0 or out["ci95_high"] < 0)
    return out


def ols_block_crosscheck(pairs: dict) -> dict:
    """The same blocked fit through statsmodels OLS, as a check on the hand computation.

    Two implementations of one estimator, required to agree. A disagreement is a defect
    in this file, not a choice.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as error:  # pragma: no cover - statsmodels is a hard dependency
        return {"computable": False, "error": repr(error)}
    frame = pd.DataFrame(
        {
            "d": [float(v) for v in pairs["differences"]],
            "dataset_seed": [k.dataset_seed for k in pairs["keys"]],
            "seed": [k.seed for k in pairs["keys"]],
        }
    )
    if frame.empty:
        return {"computable": False}
    if frame["dataset_seed"].nunique() < 2:
        model = smf.ols("d ~ 1", data=frame).fit()
        return {
            "computable": True,
            "formula": "d ~ 1",
            "mean": float(model.params["Intercept"]),
            "mse_resid": float(model.mse_resid),
            "df_resid": float(model.df_resid),
        }
    model = smf.ols("d ~ C(dataset_seed)", data=frame).fit()
    return {
        "computable": True,
        "formula": "d ~ C(dataset_seed)",
        "mse_resid": float(model.mse_resid),
        "df_resid": float(model.df_resid),
        "fitted_block_means": {
            str(seed): float(value)
            for seed, value in frame.assign(fit=model.fittedvalues)
            .groupby("dataset_seed")["fit"].first().items()
        },
    }


def divergence_check(per_dataset: dict, pooled: dict) -> dict:
    """The rule declared in POOLING_DIVERGENCE_RULE, applied.

    Per dataset wins on divergence, and the divergence is a finding: the datasets have
    measurably different difficulty, so Delta need not be comparable across them while
    pooling assumes it is.
    """
    if not pooled.get("computable") or not np.isfinite(pooled.get("ci95_low", float("nan"))):
        return {"checkable": False, "rule": POOLING_DIVERGENCE_RULE}
    diverging = []
    for dataset_seed, point in sorted(per_dataset.items()):
        low, high = point.get("ci95_low"), point.get("ci95_high")
        if not (np.isfinite(low) and np.isfinite(high)):
            continue
        overlaps = not (high < pooled["ci95_low"] or low > pooled["ci95_high"])
        if not overlaps:
            diverging.append(int(dataset_seed))
    span = [
        float(min(p["mean"] for p in per_dataset.values())),
        float(max(p["mean"] for p in per_dataset.values())),
    ] if per_dataset else [float("nan"), float("nan")]
    return {
        "checkable": True,
        "rule": POOLING_DIVERGENCE_RULE,
        "diverging_dataset_seeds": diverging,
        "diverged": bool(diverging),
        "per_dataset_span": span,
        "verdict": (
            "per-dataset reading WINS and the divergence is a FINDING: Delta depends on "
            "the draw of the dataset. Do not report pooled as the main result."
            if diverging
            else "no divergence under the declared rule; pooled and per-dataset agree"
        ),
    }


def mixedlm_check(pairs_by_dilution: dict) -> dict:
    """MixedLM as a check, never the main analysis.

    Delta ~ dilution + (1 | seed), groups = seed. Formally redundant here — the design is
    balanced and the random part is a single intercept — so it must come out equivalent to
    the paired contrast. A disagreement is reported as a finding about the model.
    """
    records = []
    for dilution, pairs in sorted(pairs_by_dilution.items()):
        for key, value in zip(pairs["keys"], pairs["differences"]):
            records.append(
                {"d": float(value), "dilution": dilution, "seed": key.seed,
                 "dataset_seed": key.dataset_seed}
            )
    frame = pd.DataFrame(records)
    if not frame.empty:
        # The dilution factor is ordered along the axis, so the reference level is
        # `linear` and every coefficient reads as a shift from the linear head. Left to
        # patsy the reference would be the alphabetically first name.
        present = [d for d in CONTRACT_DILUTIONS if d in set(frame["dilution"])]
        frame["dilution"] = pd.Categorical(frame["dilution"], categories=present, ordered=True)
    out: dict = {
        "n": int(len(frame)),
        "reference_level": (
            str(frame["dilution"].cat.categories[0]) if not frame.empty else None
        ),
        "note": (
            "reported as a CHECK. In part A the mixed model is formally redundant: the "
            "design is balanced and the random part is one intercept, so the fixed effect "
            "equals the within-subject contrast (SPEC 7.3)."
        ),
    }
    if frame.empty or frame["seed"].nunique() < 2:
        out["computable"] = False
        out["reason"] = "fewer than two training seeds — nothing for a random intercept to group"
        return out
    try:
        import warnings

        import statsmodels.formula.api as smf

        formula = "d ~ 1" if frame["dilution"].nunique() < 2 else "d ~ C(dilution)"
        # Captured rather than left to stderr: "the MLE may be on the boundary" is the
        # expected message for a balanced design with one random intercept, and belongs
        # in the record as evidence rather than in the console as noise.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, data=frame, groups=frame["seed"]).fit(reml=True)
        fit_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
        # The intercept is the REFERENCE LEVEL's mean, so that is what it must be compared
        # against. The mean over every dilution is a different quantity as soon as the axis
        # has more than one point, and comparing to it made the check pass by accident.
        reference = str(frame["dilution"].cat.categories[0])
        paired_mean = float(frame.loc[frame["dilution"] == reference, "d"].mean())
        intercept = float(model.params["Intercept"])
        out.update(
            {
                "computable": True,
                "formula": f"{formula}, groups = seed",
                "intercept": intercept,
                "intercept_se": float(model.bse["Intercept"]),
                "group_variance": float(model.cov_re.iloc[0, 0]),
                "scale": float(model.scale),
                "params": {str(k): float(v) for k, v in model.params.items()},
                "reference_level_mean_for_comparison": paired_mean,
                "intercept_minus_reference_mean": float(intercept - paired_mean),
                # Always a verdict, never None: the check exists to be able to fail, and a
                # None here made the verdict row unconditionally green whenever the axis
                # had more than one point — i.e. in every real run.
                "equivalent_to_paired_contrast": bool(
                    abs(intercept - paired_mean) < MIXEDLM_EQUIVALENCE_TOL
                ),
                "equivalence_tolerance": MIXEDLM_EQUIVALENCE_TOL,
                "converged": bool(getattr(model, "converged", True)),
                "fit_warnings": fit_warnings,
            }
        )
    except Exception as error:
        out["computable"] = False
        out["reason"] = f"MixedLM did not fit: {error!r}"
    return out


def mcnemar_from_vectors(left: np.ndarray, right: np.ndarray) -> dict:
    """McNemar on the discordant pairs, exact binomial, via statsmodels."""
    from statsmodels.stats.contingency_tables import mcnemar

    both = int(np.sum(left & right))
    b = int(np.sum(left & ~right))
    c = int(np.sum(right & ~left))
    neither = int(np.sum(~left & ~right))
    table = [[both, b], [c, neither]]
    result = mcnemar(table, exact=True)
    return {
        "n": int(left.size),
        "both_correct": both,
        "b_left_only": b,
        "c_right_only": c,
        "neither": neither,
        "discordant": b + c,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "delta_from_counts": float((b - c) / left.size),
    }
