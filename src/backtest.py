"""Walk-Forward Optimization (WFO) backtesting engine with transaction fees and rebalancing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def eval_wfo_2phase(
    w1: np.ndarray,
    w2: np.ndarray,
    ret_p1: pd.DataFrame,
    ret_p2: pd.DataFrame,
    fee_buy: float = 0.0015,
    fee_sell: float = 0.0025,
    risk_free_rate: float = 0.06,
    trading_days: int = 244
) -> Tuple[float, float, float, float]:
    """Evaluate 2-phase walk-forward validation on validation dataset.

    Parameters
    ----------
    w1, w2 : np.ndarray
        Optimized weights for validation Phase 1 and Phase 2.
    ret_p1, ret_p2 : pd.DataFrame
        Daily return matrices for Phase 1 and Phase 2.
    fee_buy : float
        Broker fee on asset purchases.
    fee_sell : float
        Broker/tax fee on asset sales.
    risk_free_rate : float
        Annual risk-free rate.
    trading_days : int
        Trading days per year.

    Returns
    -------
    Tuple[float, float, float, float]
        (cumulative_sharpe, annual_return, annual_risk, total_return)
    """
    # Phase 1
    capital = 1.0 - (1.0 * fee_buy)
    cr1 = (1.0 + ret_p1).cumprod()
    port_val_1 = cr1.dot(w1) * capital
    capital_end_p1 = port_val_1.iloc[-1]

    # Drifted weights at end of Phase 1
    asset_vals_end = w1 * cr1.iloc[-1].values
    drifted_weights = asset_vals_end / np.sum(asset_vals_end)

    # Rebalance cost into Phase 2
    diff = w2 - drifted_weights
    cost = capital_end_p1 * (
        np.sum(diff[diff > 0] * fee_buy) + np.sum(np.abs(diff[diff < 0]) * fee_sell)
    )
    capital_p2 = capital_end_p1 - cost

    # Phase 2
    cr2 = (1.0 + ret_p2).cumprod()
    port_val_2 = cr2.dot(w2) * capital_p2

    # Combined performance series
    combined_val = pd.concat([port_val_1, port_val_2])
    daily_rets = combined_val.pct_change().dropna()
    n_days = len(daily_rets)

    total_return = (combined_val.iloc[-1] / (1.0 - fee_buy)) - 1.0
    annual_return = ((1.0 + total_return) ** (trading_days / n_days)) - 1.0 if n_days > 0 else 0.0
    annual_risk = daily_rets.std() * np.sqrt(trading_days)
    sharpe = (annual_return - risk_free_rate) / annual_risk if annual_risk > 0 else -999.0

    return sharpe, annual_return * 100.0, annual_risk * 100.0, total_return * 100.0


def run_rebalancing(
    weights_list: List[np.ndarray],
    returns_list: List[pd.DataFrame],
    name: str = "WFO Strategy",
    risk_free_rate: float = 0.06,
    fee_buy: float = 0.0015,
    fee_sell: float = 0.0025,
    trading_days: int = 252
) -> Dict[str, Any]:
    """Execute multi-phase Walk-Forward backtest with periodic rebalancing and realistic transaction fees.

    Parameters
    ----------
    weights_list : List[np.ndarray]
        List of optimized weight vectors, one per phase.
    returns_list : List[pd.DataFrame]
        List of daily return DataFrames corresponding to each out-of-sample phase.
    name : str
        Strategy display name.
    risk_free_rate : float
        Annual risk-free benchmark rate.
    fee_buy : float
        Purchase transaction fee rate (e.g., 0.0015 = 0.15%).
    fee_sell : float
        Sale transaction fee rate (e.g., 0.0025 = 0.25%).
    trading_days : int
        Annualization trading days factor.

    Returns
    -------
    Dict[str, Any]
        Dictionary with performance metrics and full time-series series.
    """
    combined_portfolio_value = pd.Series(dtype="float64")
    capital = 1.0
    end_weights = None
    total_fees_paid = 0.0

    for phase_index, phase_returns in enumerate(returns_list):
        current_weights = np.asarray(weights_list[phase_index], dtype=float)

        # Calculate transaction / rebalancing cost
        if end_weights is not None:
            weight_diff = current_weights - end_weights
            rebalance_cost = capital * (
                np.sum(weight_diff[weight_diff > 0] * fee_buy)
                + np.sum(np.abs(weight_diff[weight_diff < 0]) * fee_sell)
            )
        else:
            rebalance_cost = capital * fee_buy

        total_fees_paid += rebalance_cost
        capital -= rebalance_cost

        # Compound daily returns inside phase
        cumulative_returns = (1.0 + phase_returns).cumprod()
        start_date = phase_returns.index[0] - pd.Timedelta(days=1)
        initial_row = pd.DataFrame(
            np.ones((1, len(phase_returns.columns))),
            index=[start_date],
            columns=phase_returns.columns
        )
        cumulative_returns = pd.concat([initial_row, cumulative_returns])
        phase_portfolio_value = cumulative_returns.dot(current_weights) * capital

        # Compute price-drifted end weights
        asset_values_end = current_weights * cumulative_returns.iloc[-1]
        end_weights = asset_values_end / np.sum(asset_values_end)
        capital = phase_portfolio_value.iloc[-1]

        # Stitch together continuous portfolio value series
        if phase_index == 0:
            combined_portfolio_value = phase_portfolio_value
        else:
            combined_portfolio_value = pd.concat([
                combined_portfolio_value,
                phase_portfolio_value.iloc[1:]
            ])

    daily_returns = combined_portfolio_value.pct_change().dropna()
    n_days = len(daily_returns)
    cumulative_return = combined_portfolio_value.iloc[-1] - 1.0

    if n_days > 0:
        annual_return = ((1.0 + cumulative_return) ** (trading_days / n_days)) - 1.0
    else:
        annual_return = 0.0

    annual_volatility = daily_returns.std() * np.sqrt(trading_days)
    running_max = combined_portfolio_value.cummax()
    drawdown = (combined_portfolio_value / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0 else 0.0
    )

    return {
        "Method": name,
        "Portfolio Value": combined_portfolio_value,
        "Daily Returns": daily_returns,
        "Drawdown": drawdown,
        "Total Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Total Fees Paid": total_fees_paid,
        "Final Portfolio Value": combined_portfolio_value.iloc[-1]
    }


def run_isolated(
    weight_initial: np.ndarray,
    returns_full: pd.DataFrame,
    name: str = "Buy and Hold",
    risk_free_rate: float = 0.06,
    fee_buy: float = 0.0015,
    trading_days: int = 252
) -> Dict[str, Any]:
    """Execute Buy-and-Hold backtest on a single continuous time window without rebalancing.

    Parameters
    ----------
    weight_initial : np.ndarray
        Fixed initial portfolio asset weights.
    returns_full : pd.DataFrame
        Daily returns for the period.
    name : str
        Strategy display name.
    risk_free_rate : float
        Annual risk-free rate.
    fee_buy : float
        Initial entry transaction fee.
    trading_days : int
        Trading days per year.

    Returns
    -------
    Dict[str, Any]
        Performance metrics dictionary.
    """
    weights = np.asarray(weight_initial, dtype=float)
    capital = 1.0
    buy_cost = capital * fee_buy
    total_fees_paid = buy_cost
    capital -= buy_cost

    cumulative_returns = (1.0 + returns_full).cumprod()
    start_date = returns_full.index[0] - pd.Timedelta(days=1)
    initial_row = pd.DataFrame(
        np.ones((1, len(returns_full.columns))),
        index=[start_date],
        columns=returns_full.columns
    )
    cumulative_returns = pd.concat([initial_row, cumulative_returns])
    portfolio_value = cumulative_returns.dot(weights) * capital

    daily_returns = portfolio_value.pct_change().dropna()
    n_days = len(daily_returns)
    total_return = portfolio_value.iloc[-1] - 1.0

    if n_days > 0:
        annual_return = ((1.0 + total_return) ** (trading_days / n_days)) - 1.0
    else:
        annual_return = 0.0

    annual_volatility = daily_returns.std() * np.sqrt(trading_days)
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0 else 0.0
    )

    return {
        "Method": name,
        "Portfolio Value": portfolio_value,
        "Daily Returns": daily_returns,
        "Drawdown": drawdown,
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Total Fees Paid": total_fees_paid,
        "Final Portfolio Value": portfolio_value.iloc[-1]
    }


def run_isolated_per_phase(
    ret_test_p: pd.DataFrame,
    ret_idx_p: pd.DataFrame,
    w_aga: np.ndarray,
    w_ga: np.ndarray,
    w_idx: np.ndarray,
    phase_name: str
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Execute isolated comparative backtest for an individual WFO phase.

    Evaluates AGA, SGA, Market Benchmark (IDX30), and 1/N Equal Weight.

    Parameters
    ----------
    ret_test_p : pd.DataFrame
        Phase daily returns for selected 22 constituent assets.
    ret_idx_p : pd.DataFrame
        Phase daily returns for official IDX30 assets.
    w_aga : np.ndarray
        Optimized AGA weights.
    w_ga : np.ndarray
        Optimized SGA weights.
    w_idx : np.ndarray
        Official IDX30 weights.
    phase_name : str
        Phase identifier (e.g. 'Phase 1').

    Returns
    -------
    Tuple[List[Dict[str, Any]], pd.DataFrame]
        List of strategy metric dicts and a summary DataFrame.
    """
    equal_weight = np.ones(len(ret_test_p.columns)) / len(ret_test_p.columns)

    res_aga = run_isolated(w_aga, ret_test_p, f"AGA ({phase_name})")
    res_ga = run_isolated(w_ga, ret_test_p, f"SGA ({phase_name})")
    res_idx = run_isolated(w_idx, ret_idx_p, f"IDX30 ({phase_name})")
    res_equal = run_isolated(equal_weight, ret_test_p, f"Equal Weight ({phase_name})")

    results = [res_aga, res_ga, res_idx, res_equal]

    summary_rows = [
        {
            "Strategi": r["Method"],
            "Total Return (%)": r["Total Return"] * 100.0,
            "Annual Return (%)": r["Annual Return"] * 100.0,
            "Annual Volatility (%)": r["Annual Volatility"] * 100.0,
            "Sharpe Ratio": r["Sharpe Ratio"],
            "Max Drawdown (%)": r["Max Drawdown"] * 100.0,
            "Transaction Cost (%)": r["Total Fees Paid"] * 100.0,
            "Final Value": r["Final Portfolio Value"],
            "Phase": phase_name,
        }
        for r in results
    ]
    df_result = pd.DataFrame(summary_rows).sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
    df_result.insert(0, "Rank", range(1, len(df_result) + 1))
    return results, df_result

