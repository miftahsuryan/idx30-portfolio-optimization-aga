"""Statistical hypothesis testing, effect size estimation, and confidence intervals.

Implements:
1. Automated Shapiro-Wilk normality test routing to Paired t-test or Wilcoxon Signed-Rank test.
2. Two-sided exact sign-flip permutation / randomization tests.
3. Percentile bootstrap confidence intervals for paired differences.
4. Cohen's dz paired effect sizes.
5. Holm-Bonferroni family-wise error rate corrections.
"""

import itertools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


def compare_significance(
    df: pd.DataFrame,
    metric: str = "Sharpe",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compare performance metrics between SGA and AGA across WFO phases with automatic normality testing.

    Checks normality of paired differences (AGA - SGA) using Shapiro-Wilk:
    - If p > alpha: assumes normal distribution, applies Paired Student t-test.
    - If p <= alpha: assumes non-normal distribution, applies Wilcoxon Signed-Rank test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['Phase', 'Algorithm', 'Run', metric].
    metric : str
        Metric column to evaluate (e.g. 'Sharpe', 'Fitness', 'Runtime').
    alpha : float
        Significance threshold level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Summary table with test statistics, p-values, distributions, and conclusions.
    """
    results = []

    for phase in sorted(df["Phase"].unique()):
        df_phase = df[df["Phase"] == phase]
        sga = df_phase[df_phase["Algorithm"] == "SGA"].sort_values("Run")[metric].values
        aga = df_phase[df_phase["Algorithm"] == "AGA"].sort_values("Run")[metric].values

        difference = aga - sga
        shapiro_stat, shapiro_p = shapiro(difference)

        if shapiro_p > alpha:
            normality = "Normal"
            method = "Paired t-test"
            stat, p_val = ttest_rel(aga, sga)
        else:
            normality = "Tidak Normal"
            method = "Wilcoxon Signed-Rank"
            stat, p_val = wilcoxon(aga, sga)

        conclusion = (
            "Terdapat perbedaan signifikan antara SGA dan AGA"
            if p_val < alpha
            else "Tidak terdapat perbedaan signifikan antara SGA dan AGA"
        )

        results.append({
            "Phase": phase,
            "Metric": metric,
            "Mean SGA": float(np.mean(sga)),
            "Std SGA": float(np.std(sga, ddof=1)),
            "Mean AGA": float(np.mean(aga)),
            "Std AGA": float(np.std(aga, ddof=1)),
            "Normality Test": "Shapiro-Wilk",
            "Normality p-value": float(shapiro_p),
            "Distribution": normality,
            "Statistical Test": method,
            "Statistic": float(stat),
            "p-value": float(p_val),
            "Significant (p<0.05)": bool(p_val < alpha),
            "Conclusion": conclusion
        })

    return pd.DataFrame(results)


def exact_sign_flip_pvalue(differences: Union[List[float], np.ndarray]) -> float:
    """Compute two-sided exact randomization (sign-flip) p-value for paired mean differences.

    Parameters
    ----------
    differences : array-like
        Vector of paired differences (AGA - SGA).

    Returns
    -------
    float
        Exact permutation p-value.
    """
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan

    observed = abs(diffs.mean())
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)))
    permuted = np.abs((signs * diffs).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def bootstrap_mean_ci(
    differences: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 2026
) -> Tuple[float, float]:
    """Calculate percentile bootstrap confidence interval for the mean paired difference.

    Parameters
    ----------
    differences : array-like
        Vector of paired differences.
    confidence : float
        Confidence level (default 0.95 = 95%).
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    diffs = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(samples, [alpha / 2.0, 1.0 - (alpha / 2.0)])
    return float(ci_low), float(ci_high)


def holm_adjust(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction for multiple testing hypothesis control.

    Parameters
    ----------
    p_values : array-like
        Array of unadjusted p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values maintaining family-wise error rate control.
    """
    p_vals = np.asarray(p_values, dtype=float)
    order = np.argsort(p_vals)
    adjusted = np.empty_like(p_vals)
    running_max = 0.0
    m = len(p_vals)

    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_vals[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max

    return adjusted


def paired_oos_statistics(df_paired: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired out-of-sample statistics: mean difference, 95% CI, Cohen's dz, exact sign-flip, and Holm p-values.

    Parameters
    ----------
    df_paired : pd.DataFrame
        DataFrame with columns ['Run', 'Algorithm', 'Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'].

    Returns
    -------
    pd.DataFrame
        Comprehensive publication-grade statistics table.
    """
    metric_names = [
        "Total Return",
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    ]
    rows = []

    for metric in metric_names:
        pivot = df_paired.pivot(index="Run", columns="Algorithm", values=metric)
        differences = (pivot["AGA"] - pivot["SGA"]).to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(differences)
        sd_diff = differences.std(ddof=1)
        cohen_dz = differences.mean() / sd_diff if sd_diff > 0 else np.nan

        rows.append({
            "Metric": metric,
            "Mean AGA": float(pivot["AGA"].mean()),
            "Mean SGA": float(pivot["SGA"].mean()),
            "Mean Difference (AGA-SGA)": float(differences.mean()),
            "95% CI Low": ci_low,
            "95% CI High": ci_high,
            "Cohen dz": float(cohen_dz),
            "Exact Sign-Flip p": exact_sign_flip_pvalue(differences),
        })

    result = pd.DataFrame(rows)
    result["Holm-adjusted p"] = holm_adjust(result["Exact Sign-Flip p"].to_numpy())
    return result

