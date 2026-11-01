"""IDX30 Portfolio Optimization using Adaptive Genetic Algorithm (AGA).

A modular Python framework for metaheuristic portfolio optimization,
walk-forward out-of-sample backtesting, and rigorous statistical hypothesis validation.
"""

__version__ = "1.0.0"
__author__ = "Miftahullah Suryanugraha"

from src import config
from src.data_loader import (
    load_raw_stock_data,
    preprocess_stock_data,
    get_consistent_idx30_tickers,
    prepare_study_dataset,
    compute_statistics,
    split_dataset,
    create_idx_benchmark,
)
from src.objectives import (
    validate_weights,
    repair_weights,
    fitness_function,
    calculate_insample_sharpe,
)
from src.optimizers import (
    initialize_population,
    tournament_selection,
    crossover,
    mutation,
    calculate_adaptive_rates,
    genetic_algorithm,
    adaptive_genetic_algorithm,
)
from src.backtest import (
    eval_wfo_2phase,
    run_rebalancing,
    run_isolated,
    run_isolated_per_phase,
)
from src.statistics import (
    compare_significance,
    exact_sign_flip_pvalue,
    bootstrap_mean_ci,
    holm_adjust,
    paired_oos_statistics,
)
from src.visualization import (
    plot_evaluation_metrics,
    plot_convergence,
    plot_diversity,
    plot_adaptive_rates,
)

__all__ = [
    "load_raw_stock_data",
    "preprocess_stock_data",
    "get_consistent_idx30_tickers",
    "prepare_study_dataset",
    "compute_statistics",
    "split_dataset",
    "create_idx_benchmark",
    "validate_weights",
    "repair_weights",
    "fitness_function",
    "calculate_insample_sharpe",
    "initialize_population",
    "tournament_selection",
    "crossover",
    "mutation",
    "calculate_adaptive_rates",
    "genetic_algorithm",
    "adaptive_genetic_algorithm",
    "eval_wfo_2phase",
    "run_rebalancing",
    "run_isolated",
    "run_isolated_per_phase",
    "compare_significance",
    "exact_sign_flip_pvalue",
    "bootstrap_mean_ci",
    "holm_adjust",
    "paired_oos_statistics",
    "plot_evaluation_metrics",
    "plot_convergence",
    "plot_diversity",
    "plot_adaptive_rates",
]

