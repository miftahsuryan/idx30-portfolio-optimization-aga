"""Objective and fitness functions for portfolio optimization.

Implements the regularized Mean-Variance fitness function, diversification penalty
(Herfindahl-Hirschman Index), constraint repairs, and in-sample Sharpe ratio calculations.
"""

from typing import Optional, Tuple, Union
import numpy as np


def validate_weights(
    weights: np.ndarray,
    max_weight: Optional[float] = 0.20,
    atol: float = 1e-6
) -> bool:
    """Validate whether portfolio weights satisfy investment constraints:

    1. Sum of weights equals 1.0 (within atol).
    2. All weights are non-negative (no short-selling).
    3. No single weight exceeds max_weight (if specified).

    Parameters
    ----------
    weights : np.ndarray
        Array of portfolio asset weights.
    max_weight : float, optional
        Maximum allowed weight per asset.
    atol : float
        Absolute tolerance for equality checks.

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
    weights: np.ndarray,
    max_weight: float = 0.20,
    max_iters: int = 20
) -> np.ndarray:
    """Repair portfolio weights to strictly satisfy budget (sum=1) and upper-bound constraints.

    Uses an iterative excess redistribution mechanism:
    - Clips weights to [0, max_weight]
    - Redistributes excess allocation proportionally among assets below max_weight
    - Renormalizes so sum(weights) == 1.0

    Parameters
    ----------
    weights : np.ndarray
        Candidate portfolio weight vector.
    max_weight : float
        Maximum allowed weight per individual asset.
    max_iters : int
        Maximum number of iterative redistribution cycles.

    Returns
    -------
    np.ndarray
        Repaired, feasible weight vector.
    """
    w = np.asarray(weights, dtype=float).copy()
    w = np.clip(w, 0.0, max_weight)

    total = np.sum(w)
    if total <= 0:
        w = np.clip(np.random.rand(len(w)), 0.0, max_weight)
        total = np.sum(w)

    w = w / total

    iters = 0
    while np.any(w > max_weight) and iters < max_iters:
        excess = np.maximum(0.0, w - max_weight)
        total_excess = np.sum(excess)

        w = np.minimum(w, max_weight)

        under_max_mask = w < max_weight
        sum_under_max = np.sum(w[under_max_mask])
        count_under_max = np.sum(under_max_mask)

        if count_under_max > 0:
            if sum_under_max > 0:
                w[under_max_mask] += total_excess * (w[under_max_mask] / sum_under_max)
            else:
                w[under_max_mask] += total_excess / count_under_max
        else:
            break
        iters += 1

    sum_final = np.sum(w)
    return w / sum_final if sum_final > 0 else np.ones_like(w) / len(w)


def fitness_function(
    weights: Union[list, np.ndarray],
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    lambda_div: float = 0.05,
    max_weight: Optional[float] = 0.20,
    penalty_multiplier: float = 1000.0
) -> float:
    """Compute the minimization objective fitness value for a portfolio.

    Objective:
        minimize Fitness = -Sharpe_Ratio + lambda_div * HHI + Penalty_MaxWeight

    Where:
        - Sharpe_Ratio = E / sigma
        - E = weights @ mu (expected daily portfolio return)
        - sigma = sqrt(weights.T @ cov_matrix @ weights + 1e-6)
        - HHI = sum(weights^2) (Herfindahl-Hirschman Index diversification penalty)
        - Penalty_MaxWeight = penalty_multiplier * sum(max(0, weights - max_weight)^2)

    Parameters
    ----------
    weights : list or np.ndarray
        Portfolio weight vector.
    mu : np.ndarray
        Expected asset returns vector.
    cov_matrix : np.ndarray
        Asset covariance matrix.
    lambda_div : float
        Diversification penalty coefficient.
    max_weight : float, optional
        Maximum weight constraint threshold.
    penalty_multiplier : float
        Penalty severity factor for exceeding max_weight.

    Returns
    -------
    float
        Computed fitness value (lower is better).
    """
    w = np.asarray(weights, dtype=float)

    # Expected portfolio return and volatility
    e_ret = float(np.dot(w, mu))
    variance = float(w.T @ cov_matrix @ w)
    sigma = np.sqrt(max(0.0, variance) + 1e-6)

    # In-sample ratio
    sharpe = e_ret / sigma

    # Diversification penalty (HHI)
    diversification = float(np.sum(w ** 2))

    # Upper bound penalty
    penalty_max = 0.0
    if max_weight is not None:
        violations = np.maximum(0.0, w - max_weight)
        penalty_max = penalty_multiplier * float(np.sum(violations ** 2))

    return -sharpe + (lambda_div * diversification) + penalty_max


def calculate_insample_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.06,
    trading_days: int = 244
) -> float:
    """Calculate the annualized in-sample Sharpe Ratio for evaluated portfolio weights.

    Parameters
    ----------
    weights : np.ndarray
        Asset weight vector.
    mu : np.ndarray
        Daily expected returns.
    cov_matrix : np.ndarray
        Daily covariance matrix.
    risk_free_rate : float
        Annualized risk-free benchmark rate (e.g., 0.06 = 6%).
    trading_days : int
        Number of trading days per year for annualization.

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

