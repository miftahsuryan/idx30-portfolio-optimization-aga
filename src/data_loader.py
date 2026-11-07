"""Data loading, preprocessing, asset filtering, and financial statistics pipeline for IDX30 stocks.

Aligned 100% with Implementasi-fix.ipynb (Cells 6-32, 107-108, 124-130).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import IDX30_WEIGHT_MAPS, WFO_TEST_PHASES


def load_raw_stock_data(filepath: str = "data/raw/IDX30-Scraping.csv") -> pd.DataFrame:
    """Load historical stock price data from CSV (Cell 6 of Implementasi-fix.ipynb)."""
    return pd.read_csv(filepath)


def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess historical stock dataset (Cell 12 of Implementasi-fix.ipynb).

    - Converts 'Date' column to datetime.
    - Coerces 'Close' prices to numeric and drops NaNs.
    - Sorts chronologically by ticker and date.
    """
    df_clean = df.copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["Close"] = pd.to_numeric(df_clean["Close"], errors="coerce")
    df_clean = df_clean.dropna().sort_values(["ticker", "Date"]).reset_index(drop=True)
    return df_clean


def get_consistent_idx30_tickers(list_filepath: str = "data/raw/IDX30-List.csv") -> List[str]:
    """Identify tickers that remained consistently in the IDX30 index across all review periods (Cell 19)."""
    df_idx30 = pd.read_csv(list_filepath)
    period_groups = df_idx30.groupby(["tahun", "periode"])["ticker"].apply(set).tolist()
    consistent_tickers = sorted(list(set.intersection(*period_groups)))
    return consistent_tickers


def prepare_study_dataset(
    df_preprocessed: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """Filter preprocessed data to consistent tickers and compute daily simple returns Rit (Cells 20-21)."""
    df_selected = df_preprocessed[df_preprocessed["ticker"].isin(tickers)].copy()
    df_selected = df_selected.sort_values(["ticker", "Date"])
    df_selected["Rit"] = df_selected.groupby("ticker")["Close"].pct_change()
    return df_selected


def compute_statistics(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Pivot daily returns, drop missing dates, and compute expected returns (mu) and covariance matrix (Cell 27)."""
    returns = df.pivot(index="Date", columns="ticker", values="Rit").dropna()
    mu = returns.mean()
    cov_matrix = returns.cov()
    return mu, cov_matrix, returns


def split_dataset(
    df_selected: pd.DataFrame,
    train_start: str = "2023-01-01",
    train_end: str = "2024-07-31",
    val_end: str = "2025-01-27",
    test_end: str = "2026-01-26"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into Train, Validation, and Test sets (Cell 23 of Implementasi-fix.ipynb)."""
    df_train = df_selected[
        (df_selected["Date"] >= train_start) & (df_selected["Date"] <= train_end)
    ].copy()
    df_val = df_selected[
        (df_selected["Date"] > train_end) & (df_selected["Date"] <= val_end)
    ].copy()
    df_test = df_selected[
        (df_selected["Date"] > val_end) & (df_selected["Date"] <= test_end)
    ].copy()
    return df_train, df_val, df_test


def get_wfo_retraining_statistics(
    df_selected: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Compute expanding-window in-sample statistics (mu, cov, returns) for the 4 WFO retraining phases (Cells 107-108)."""
    mu_data: Dict[str, pd.Series] = {}
    cov_data: Dict[str, pd.DataFrame] = {}
    returns_data: Dict[str, pd.DataFrame] = {}

    for p_cfg in WFO_TEST_PHASES:
        phase_name = p_cfg["phase"]
        df_phase = df_selected[
            (df_selected["Date"] >= p_cfg["is_start"]) &
            (df_selected["Date"] <= p_cfg["is_end"])
        ]
        mu, cov, returns = compute_statistics(df_phase)
        mu_data[phase_name] = mu
        cov_data[phase_name] = cov
        returns_data[phase_name] = returns

    return mu_data, cov_data, returns_data


def get_wfo_test_phase_returns(returns_test: pd.DataFrame) -> List[pd.DataFrame]:
    """Slice out-of-sample returns_test into the 4 WFO test phases (Cell 130 of Implementasi-fix.ipynb)."""
    ret_test_p1 = returns_test[
        (returns_test.index >= "2025-01-27") & (returns_test.index < "2025-05-01")
    ]
    ret_test_p2 = returns_test[
        (returns_test.index >= "2025-05-01") & (returns_test.index < "2025-08-01")
    ]
    ret_test_p3 = returns_test[
        (returns_test.index >= "2025-08-01") & (returns_test.index < "2025-11-01")
    ]
    ret_test_p4 = returns_test[
        returns_test.index >= "2025-11-01"
    ]
    return [ret_test_p1, ret_test_p2, ret_test_p3, ret_test_p4]


def create_idx_benchmark(
    df_idx: pd.DataFrame,
    idx_map: Dict[str, float],
    start_date: str,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construct benchmark portfolio returns and official IDX30 index weights (Cell 125 of Implementasi-fix.ipynb).

    Uses strict `< end_date` boundary and direct constituent weights from IDX30-List.csv
    exactly as implemented in the thesis notebook.
    """
    tickers = list(idx_map.keys())
    df_phase = df_idx[
        (df_idx["ticker"].isin(tickers)) &
        (df_idx["Date"] >= start_date)
    ]

    if end_date is not None:
        df_phase = df_phase[df_phase["Date"] < end_date]

    _, _, returns_phase = compute_statistics(df_phase)
    weights_phase = np.array([
        idx_map[ticker] for ticker in returns_phase.columns
    ])
    return returns_phase, weights_phase


def get_wfo_idx30_benchmark_phases(
    df_preprocessed: pd.DataFrame
) -> Tuple[List[pd.DataFrame], List[np.ndarray]]:
    """Build the 4-phase IDX30 benchmark return matrices and official weight arrays (Cells 124-129)."""
    df_idx = df_preprocessed.copy()
    if "Rit" not in df_idx.columns:
        df_idx["Rit"] = df_idx.groupby("ticker")["Close"].pct_change()

    ret_idx_p1, w_idx_p1 = create_idx_benchmark(
        df_idx, IDX30_WEIGHT_MAPS["Fase 1"], start_date="2025-01-27", end_date="2025-05-01"
    )
    ret_idx_p2, w_idx_p2 = create_idx_benchmark(
        df_idx, IDX30_WEIGHT_MAPS["Fase 2"], start_date="2025-05-01", end_date="2025-08-01"
    )
    ret_idx_p3, w_idx_p3 = create_idx_benchmark(
        df_idx, IDX30_WEIGHT_MAPS["Fase 3"], start_date="2025-08-01", end_date="2025-11-01"
    )
    ret_idx_p4, w_idx_p4 = create_idx_benchmark(
        df_idx, IDX30_WEIGHT_MAPS["Fase 4"], start_date="2025-11-01", end_date=None
    )

    return [ret_idx_p1, ret_idx_p2, ret_idx_p3, ret_idx_p4], [w_idx_p1, w_idx_p2, w_idx_p3, w_idx_p4]
