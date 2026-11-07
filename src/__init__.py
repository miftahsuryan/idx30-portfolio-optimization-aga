"""IDX30 Portfolio Optimization using Adaptive Genetic Algorithm (AGA).

A modular Python framework for metaheuristic portfolio optimization,
walk-forward out-of-sample backtesting, and paired multi-run statistical evaluation
aligned 100% with the original undergraduate thesis (Implementasi-fix.ipynb).
"""

__version__ = "1.1.0"
__author__ = "Miftahullah Suryanugraha"

from src import config
from src.data_loader import (
    load_raw_stock_data,
    preprocess_stock_data,
    get_consistent_idx30_tickers,
    prepare_study_dataset,
    compute_statistics,
    split_dataset,
    get_wfo_retraining_statistics,
    get_wfo_test_phase_returns,
    create_idx_benchmark,
    get_wfo_idx30_benchmark_phases,
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
    get_final_ga_weights,
    get_final_aga_weights,
    evaluate_algorithm_performance,
)
from src.backtest import (
    eval_wfo_2phase,
    run_rebalancing,
    run_isolated,
    run_isolated_per_phase,
)
from src.statistics import (
    summarize_multi_run_performance,
    compare_significance,
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
    "get_wfo_retraining_statistics",
    "get_wfo_test_phase_returns",
    "create_idx_benchmark",
    "get_wfo_idx30_benchmark_phases",
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
    "get_final_ga_weights",
    "get_final_aga_weights",
    "evaluate_algorithm_performance",
    "eval_wfo_2phase",
    "run_rebalancing",
    "run_isolated",
    "run_isolated_per_phase",
    "summarize_multi_run_performance",
    "compare_significance",
    "plot_evaluation_metrics",
    "plot_convergence",
    "plot_diversity",
    "plot_adaptive_rates",
]
