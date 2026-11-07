"""Objective and fitness functions for portfolio optimization.

Implements the regularized Mean-Variance fitness function, diversification penalty
(Herfindahl-Hirschman Index), constraint repairs (proportional excess redistribution,
max_iters=15), and in-sample Sharpe ratio calculations aligned 100% with
Implementasi-fix.ipynb (Cells 43, 49, and 92-93).
"""

from typing import Optional, Union
import numpy as np


def validate_weights(
    weights: np.ndarray,
    max_weight: Optional[float] = 0.10,
    atol: float = 1e-3
) -> bool:
    """Validate whether portfolio weights satisfy investment constraints:

    1. Sum of weights equals 1.0 (within atol).
    2. All weights are non-negative (no short-selling).
    3. No single weight exceeds max_weight (within atol, if specified).

    Parameters
    ----------
    weights : np.ndarray
        Array of portfolio asset weights.
    max_weight : float, optional
        Maximum allowed weight per asset (default 0.10 = 10% cap).
    atol : float
        Absolute tolerance for checks.

    Returns
    -------
    bool
        True if all constraints are satisfied, False otherwise.
    """
    w = np.asarray(weights, dtype=float)
    if not np.isclose(np.sum(w), 1.0, atol=atol):
        return False
    if np.any(w < -atol):
        return False
    if max_weight is not None and np.any(w > max_weight + atol):
        return False
    return True


def repair_weights(
    w: np.ndarray,
    max_weight: float = 0.30,
    max_iters: int = 15
) -> np.ndarray:
    """Repair portfolio weights to satisfy budget (sum=1) and upper-bound constraints.

    Matches Cell 49 of Implementasi-fix.ipynb:
    - Clips weights to [0, max_weight]
    - Reinitializes randomly if sum(w) == 0
    - Normalizes initial weights so sum(w) == 1.0
    - Iteratively redistributes excess allocation above max_weight (up to max_iters=15)
      proportionally among assets below max_weight
    - Performs final normalization w / w.sum()

    Parameters
    ----------
    w : np.ndarray
        Candidate portfolio weight vector.
    max_weight : float
        Upper bound threshold used by the repair operator.
    max_iters : int
        Maximum number of iterative excess redistribution cycles (default 15).

    Returns
    -------
    np.ndarray
        Repaired, normalized weight vector.
    """
    w = np.clip(w, 0, max_weight)
    total = w.sum()

    if total == 0:
        w = np.clip(np.random.rand(len(w)), 0, max_weight)
        total = w.sum()

    w = w / total

    iters = 0
    while np.any(w > max_weight) and iters < max_iters:
        excess = w - max_weight
        excess[excess < 0] = 0
        total_excess = excess.sum()

        w = np.minimum(w, max_weight)

        idx_under_max = w < max_weight
        sum_under_max = w[idx_under_max].sum()
        count_under_max = np.sum(idx_under_max)

        if count_under_max > 0:
            if sum_under_max > 0:
                w[idx_under_max] += total_excess * (w[idx_under_max] / sum_under_max)
            else:
                w[idx_under_max] += total_excess / count_under_max
        else:
            break

        iters += 1

    return w / w.sum()


def fitness_function(
    weights: Union[list, np.ndarray],
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    lambda_div: float = 0.90,
    max_weight: Optional[float] = 0.10
) -> float:
    """Compute the minimization objective fitness value for a portfolio.

    Matches Cells 43 & 92-93 of Implementasi-fix.ipynb:
        minimize Fitness = -Sharpe_Ratio + lambda_div * HHI + Penalty_MaxWeight

    Where:
        - E = weights @ mu (expected daily portfolio return)
        - V = weights.T @ cov_matrix @ weights
        - sigma = sqrt(V + 1e-6)
        - sharpe = E / sigma
        - diversification = sum(weights^2) (Herfindahl-Hirschman Index)
        - penalty_max = 1000 * sum(max(0, weights - max_weight)^2)

    Parameters
    ----------
    weights : list or np.ndarray
        Portfolio weight vector.
    mu : np.ndarray
        Expected daily asset returns vector.
    cov_matrix : np.ndarray
        Daily asset return covariance matrix.
    lambda_div : float
        Diversification penalty coefficient (tuned thesis value = 0.90; initial = 0.10).
    max_weight : float, optional
        Maximum weight constraint threshold (tuned thesis value = 0.10; initial = 0.30).

    Returns
    -------
    float
        Computed fitness value (lower is better).
    """
    weights = np.array(weights)

    E = weights @ mu
    V = weights.T @ cov_matrix @ weights
    sigma = np.sqrt(V + 1e-6)

    sharpe = E / sigma
    diversification = np.sum(weights ** 2)

    penalty_max = 0.0
    if max_weight is not None:
        violations = np.maximum(0, weights - max_weight)
        penalty_max = 1000 * np.sum(violations ** 2)

    fitness = -sharpe + (lambda_div * diversification) + penalty_max
    return float(fitness)


def calculate_insample_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.06,
    trading_days: int = 244
) -> float:
    """Calculate the annualized in-sample Sharpe Ratio (as in Cell 80 of Implementasi-fix.ipynb).

    Parameters
    ----------
    weights : np.ndarray
        Asset weight vector.
    mu : np.ndarray
        Daily expected returns.
    cov_matrix : np.ndarray
        Daily covariance matrix.
    risk_free_rate : float
        Annualized risk-free benchmark rate (default 0.06 = 6%).
    trading_days : int
        Number of trading days per year (default 244).

    Returns
    -------
    float
        Annualized in-sample Sharpe ratio.
    """
    w = np.asarray(weights, dtype=float)
    daily_ret = float(np.dot(w, mu))
    daily_var = float(w.T @ cov_matrix @ w)
    daily_risk = np.sqrt(max(0.0, daily_var))

    annual_ret = daily_ret * trading_days
    annual_risk = daily_risk * np.sqrt(trading_days)

    if annual_risk == 0:
        return np.nan

    return (annual_ret - risk_free_rate) / annual_risk
