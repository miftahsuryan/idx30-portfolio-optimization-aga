"""Data loading, preprocessing, asset filtering, and financial statistics pipeline for IDX30 stocks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_raw_stock_data(filepath: str = "data/raw/IDX30-Scraping.csv") -> pd.DataFrame:
    """Load historical stock price data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the scraped stock prices CSV file.

    Returns
    -------
    pd.DataFrame
        Raw stock price DataFrame with columns ['Date', 'ticker', 'Close'].
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess historical stock dataset.

    - Converts 'Date' column to datetime.
    - Coerces 'Close' prices to numeric and drops NaNs.
    - Sorts chronologically by ticker and date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """
    df_clean = df.copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["Close"] = pd.to_numeric(df_clean["Close"], errors="coerce")
    df_clean = df_clean.dropna().sort_values(["ticker", "Date"]).reset_index(drop=True)
    return df_clean


def get_consistent_idx30_tickers(list_filepath: str = "data/raw/IDX30-List.csv") -> List[str]:
    """Identify tickers that remained consistently in the IDX30 index across all review periods.

    Parameters
    ----------
    list_filepath : str
        Path to the IDX30 historical constituents CSV.

    Returns
    -------
    List[str]
        Alphabetically sorted list of consistent ticker symbols.
    """
    df_idx30 = pd.read_csv(list_filepath)
    period_groups = df_idx30.groupby(["tahun", "periode"])["ticker"].apply(set).tolist()
    consistent_tickers = sorted(list(set.intersection(*period_groups)))
    return consistent_tickers


def prepare_study_dataset(
    df_preprocessed: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """Filter preprocessed data to consistent tickers and compute daily simple returns (Rit).

    Parameters
    ----------
    df_preprocessed : pd.DataFrame
        Cleaned stock dataframe.
    tickers : List[str]
        List of target ticker symbols.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with added 'Rit' return column.
    """
    df_selected = df_preprocessed[df_preprocessed["ticker"].isin(tickers)].copy()
    df_selected = df_selected.sort_values(["ticker", "Date"]).reset_index(drop=True)
    df_selected["Rit"] = df_selected.groupby("ticker")["Close"].pct_change()
    return df_selected


def compute_statistics(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Pivot daily returns, drop missing dates, and compute expected returns (mu) and covariance matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'ticker', and 'Rit'.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame, pd.DataFrame]
        mu: Series of mean daily returns per asset.
        cov_matrix: DataFrame of return covariances.
        returns: Pivoted daily returns matrix (Date x Ticker).
    """
    returns = df.pivot(index="Date", columns="ticker", values="Rit").dropna()
    mu = returns.mean()
    cov_matrix = returns.cov()
    return mu, cov_matrix, returns


def split_dataset(
    df_selected: pd.DataFrame,
    train_start: str = "2023-01-01",
    train_end: str = "2024-07-31",
    val_start: str = "2024-08-01",
    val_end: str = "2025-01-27",
    test_start: str = "2025-01-28",
    test_end: str = "2026-01-26"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into Train, Validation, and Test sets.

    Parameters
    ----------
    df_selected : pd.DataFrame
        Full selected dataset with 'Date'.
    train_start, train_end : str
        In-sample training date range.
    val_start, val_end : str
        Validation date range for hyperparameter tuning.
    test_start, test_end : str
        Out-of-sample testing date range for walk-forward evaluation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_val, df_test)
    """
    df_train = df_selected[(df_selected["Date"] >= train_start) & (df_selected["Date"] <= train_end)].copy()
    df_val = df_selected[(df_selected["Date"] >= val_start) & (df_selected["Date"] <= val_end)].copy()
    df_test = df_selected[(df_selected["Date"] >= test_start) & (df_selected["Date"] <= test_end)].copy()
    return df_train, df_val, df_test


def create_idx_benchmark(
    df_idx: pd.DataFrame,
    idx_map: Dict[str, float],
    start_date: str,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construct benchmark portfolio returns and official IDX30 index weights.

    Parameters
    ----------
    df_idx : pd.DataFrame
        Price/return dataframe.
    idx_map : Dict[str, float]
        Dictionary mapping ticker to official index weight.
    start_date : str
        Phase start date.
    end_date : str, optional
        Phase end date.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (returns_phase, weights_phase)
    """
    tickers = list(idx_map.keys())
    mask = (df_idx["ticker"].isin(tickers)) & (df_idx["Date"] >= start_date)
    if end_date is not None:
        mask = mask & (df_idx["Date"] <= end_date)

    df_phase = df_idx[mask]
    _, _, returns_phase = compute_statistics(df_phase)
    weights_phase = np.array([idx_map[ticker] for ticker in returns_phase.columns])
    weights_phase = weights_phase / np.sum(weights_phase)
    return returns_phase, weights_phase

