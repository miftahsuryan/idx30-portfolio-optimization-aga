"""Statistical hypothesis testing for multi-run stochastic optimization evaluation.

Implements the exact two-stage statistical evaluation pipeline from
Implementasi-fix.ipynb (Cells 140-149):
1. Multi-run paired experimental design across 4 Walk-Forward retraining phases
   (N_RUNS = 10 paired seeds per phase with shared initial populations under an
   equal 44,000 NFE budget -> 80 total optimization runs).
2. Shapiro-Wilk normality testing (alpha = 0.05) on paired differences d_r = X_{AGA,r} - X_{SGA,r}.
3. Conditional routing:
   - Parametric Paired Student's t-test (scipy.stats.ttest_rel) when p_Shapiro > 0.05 (Normal).
   - Non-parametric Wilcoxon Signed-Rank test (scipy.stats.wilcoxon) when p_Shapiro <= 0.05 (Tidak Normal).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


def summarize_multi_run_performance(
    df_phase_results: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate 80-run optimization performance overall and per WFO phase (Cells 140-142 of Implementasi-fix.ipynb).

    Parameters
    ----------
    df_phase_results : pd.DataFrame
        DataFrame with columns ['Phase', 'Algorithm', 'Run', 'Fitness', 'Sharpe', 'Runtime', 'ConvergenceGen'].

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (comparison_algorithm, comparison_per_phase)
    """
    comparison_algorithm = (
        df_phase_results
        .groupby("Algorithm")
        .agg(
            Mean_Fitness=("Fitness", "mean"),
            Std_Fitness=("Fitness", "std"),
            Mean_Sharpe=("Sharpe", "mean"),
            Std_Sharpe=("Sharpe", "std"),
            Mean_Runtime=("Runtime", "mean"),
            Mean_Convergence=("ConvergenceGen", "mean"),
        )
        .reset_index()
    )

    comparison_per_phase = (
        df_phase_results
        .groupby(["Phase", "Algorithm"])
        .agg(
            Mean_Fitness=("Fitness", "mean"),
            Std_Fitness=("Fitness", "std"),
            Mean_Sharpe=("Sharpe", "mean"),
            Std_Sharpe=("Sharpe", "std"),
            Mean_Runtime=("Runtime", "mean"),
            Mean_Convergence=("ConvergenceGen", "mean"),
        )
        .reset_index()
    )

    return comparison_algorithm, comparison_per_phase


def compare_significance(
    df: pd.DataFrame,
    metric: str = "Sharpe",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compare performance metrics between SGA and AGA across WFO phases with automatic normality testing.

    Matches Cell 145 of Implementasi-fix.ipynb:
    - Computes paired differences `difference = aga - sga` across the 10 shared-seed runs per phase.
    - Tests normality of `difference` using Shapiro-Wilk (`scipy.stats.shapiro`):
      * If p > alpha: assumes normal distribution, applies Paired Student's t-test (`ttest_rel(aga, sga)`).
      * If p <= alpha: assumes non-normal distribution, applies Wilcoxon Signed-Rank test (`wilcoxon(aga, sga)`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['Phase', 'Algorithm', 'Run', metric].
    metric : str
        Metric column to evaluate ('Sharpe', 'Fitness', or 'Runtime').
    alpha : float
        Significance threshold level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Summary table with Means, Stds, Shapiro-Wilk p-values, distribution classifications,
        selected test statistics, p-values, and conclusions.
    """
    results = []

    for phase in sorted(df["Phase"].unique()):
        df_phase = df[df["Phase"] == phase]

        sga = (
            df_phase[df_phase["Algorithm"] == "SGA"]
            .sort_values("Run")[metric]
            .values
        )
        aga = (
            df_phase[df_phase["Algorithm"] == "AGA"]
            .sort_values("Run")[metric]
            .values
        )

        difference = aga - sga
        shapiro_stat, shapiro_p = shapiro(difference)

        if shapiro_p > alpha:
            normality = "Normal"
            method = "Paired t-test"
            statistic, p_value = ttest_rel(aga, sga)
        else:
            normality = "Tidak Normal"
            method = "Wilcoxon Signed-Rank"
            statistic, p_value = wilcoxon(aga, sga)

        if p_value < alpha:
            conclusion = "Terdapat perbedaan signifikan antara SGA dan AGA"
        else:
            conclusion = "Tidak terdapat perbedaan signifikan antara SGA dan AGA"

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
            "Statistic": float(statistic),
            "p-value": float(p_value),
            "Significant (p<0.05)": bool(p_value < alpha),
            "Conclusion": conclusion,
        })

    return pd.DataFrame(results)
