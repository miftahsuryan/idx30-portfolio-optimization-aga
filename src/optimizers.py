"""Evolutionary optimization algorithms for portfolio weight optimization.

Aligned 100% with the original undergraduate thesis (Implementasi-fix.ipynb):
1. Standard Genetic Algorithm (SGA) with fixed crossover (Pc=0.5) and mutation (Pm=0.9),
   population=440, generations=100 (44,000 NFE).
2. Adaptive Genetic Algorithm (AGA) with bounded linear adaptive crossover and mutation rates
   (k1=0.9, k2=0.9, k3=0.1, k4=0.9, k5=0.9, k6=0.3), population=500, generations=88 (44,000 NFE).
3. Final WFO weight selection (get_final_ga_weights, get_final_aga_weights) and
   multi-run paired algorithm evaluation (evaluate_algorithm_performance).
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    AGA_BEST_PARAMS,
    LAMBDA_DIV_DEFAULT,
    MAX_WEIGHT_DEFAULT,
    MAX_WEIGHT_INITIAL,
    SEED,
    SGA_BEST_PARAMS,
    WFO_WEIGHT_SEED_START,
)
from src.objectives import fitness_function, repair_weights


def initialize_population(
    pop_size: int,
    n_assets: int,
    max_weight: float = MAX_WEIGHT_DEFAULT
) -> np.ndarray:
    """Initialize a random initial population of candidate portfolio weight vectors.

    Matches Cell 47 of Implementasi-fix.ipynb.

    Parameters
    ----------
    pop_size : int
        Number of candidate individuals (chromosomes) in the population.
    n_assets : int
        Number of assets in the investment universe.
    max_weight : float
        Upper bound allocation per asset prior to normalization.

    Returns
    -------
    np.ndarray
        Array of shape (pop_size, n_assets) containing normalized weights.
    """
    pop = []
    for _ in range(pop_size):
        w = np.random.rand(n_assets)
        w = np.minimum(w, max_weight)
        w /= w.sum()
        pop.append(w)
    return np.array(pop)


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int = 3
) -> int:
    """Select the best individual from k randomly chosen candidates (tournament selection).

    Matches Cell 48 of Implementasi-fix.ipynb. Since fitness is formulated as a
    minimization problem, the candidate with the minimum fitness value wins.
    """
    idx = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return int(best)


def crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    pc: float = 0.9
) -> np.ndarray:
    """Perform arithmetic (convex combination) crossover between two parent individuals.

    Matches Cell 50 of Implementasi-fix.ipynb.
    """
    if np.random.rand() > pc:
        return p1.copy()

    alpha = np.random.rand()
    child = alpha * p1 + (1 - alpha) * p2
    return child


def mutation(
    individual: np.ndarray,
    pm: float = 0.1,
    scale: float = 0.05,
    max_weight: float = MAX_WEIGHT_INITIAL
) -> np.ndarray:
    """Perform Gaussian mutation on an individual weight vector.

    Matches Cell 51 of Implementasi-fix.ipynb.
    """
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < pm:
            child[i] += np.random.normal(0, scale)
    return child


def calculate_adaptive_rates(
    fc: float,
    f_ind: float,
    f_min: float,
    f_max: float,
    f_avg: float,
    k1: float = 0.9,
    k2: float = 0.9,
    k3: float = 0.1,
    k4: float = 0.9,
    k5: float = 0.9,
    k6: float = 0.3
) -> Tuple[float, float]:
    """Calculate adaptive crossover probability (Pc) and mutation probability (Pm).

    Matches Cell 55 & Cell 135 of Implementasi-fix.ipynb (minimization formulation):
    - If f_avg - f_min <= 1e-4: returns (k1, k4)
    - Crossover:
        If fc <= f_avg:
            pc = max(k1 - ((k1 - k3) * (f_avg - fc) / (f_avg - f_min)), k3)
        Else:
            pc = k2
    - Mutation:
        If f_ind <= f_avg:
            pm = max(k4 - ((k4 - k6) * (f_avg - f_ind) / (f_avg - f_min)), k6)
        Else:
            pm = k5
    """
    denom_crossover = f_avg - f_min
    denom_mutation = f_avg - f_min

    if denom_crossover <= 1e-4:
        return k1, k4

    if fc <= f_avg:
        pc_adaptive = k1 - ((k1 - k3) * (f_avg - fc) / denom_crossover)
        pc = max(pc_adaptive, k3)
    else:
        pc = k2

    if f_ind <= f_avg:
        pm_adaptive = k4 - ((k4 - k6) * (f_avg - f_ind) / denom_mutation)
        pm = max(pm_adaptive, k6)
    else:
        pm = k5

    return pc, pm


def genetic_algorithm(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    pop_size: int = 440,
    generations: int = 100,
    max_weight: float = MAX_WEIGHT_DEFAULT,
    seed: Optional[int] = SEED,
    init_pop: Optional[np.ndarray] = None,
    save_pop_history: bool = False,
    pc: float = 0.5,
    pm: float = 0.9,
    lambda_div: float = LAMBDA_DIV_DEFAULT,
    operator_max_weight: float = MAX_WEIGHT_INITIAL
) -> Dict[str, Any]:
    """Execute Standard Genetic Algorithm (SGA) with fixed Pc and Pm.

    Matches Cell 53 & Cell 92-93 of Implementasi-fix.ipynb.
    - Evaluates candidate fitness using tuned parameters (lambda_div=0.90, max_weight=0.10).
    - Uses operator_max_weight=0.30 for default population initialization and repair_weights
      matching the exact function binding in Implementasi-fix.ipynb.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(mu)
    if init_pop is not None:
        population = init_pop.copy()
    else:
        population = initialize_population(pop_size, n_assets, operator_max_weight)

    best_hist, avg_hist, diversity_hist = [], [], []
    pc_hist, pm_hist = [], []
    best_weights_history = []
    pop_history = []

    global_best_fitness = np.inf
    global_best_weights = None
    last_improvement_gen = None
    tol = 1e-6

    for gen in range(generations):
        fitness = np.array([
            fitness_function(w, mu, cov_matrix, lambda_div=lambda_div, max_weight=max_weight)
            for w in population
        ])

        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]

        if best_fit < global_best_fitness - tol:
            global_best_fitness = float(best_fit)
            global_best_weights = population[best_idx].copy()
            last_improvement_gen = gen

        best_hist.append(float(best_fit))
        avg_hist.append(float(fitness.mean()))
        diversity_hist.append(float(np.mean(np.std(population, axis=0))))
        best_weights_history.append(population[best_idx].copy())

        if save_pop_history:
            pop_history.append(population.copy())

        new_pop = []
        pc_list, pm_list = [], []

        while len(new_pop) < pop_size:
            idx1 = tournament_selection(population, fitness)
            idx2 = tournament_selection(population, fitness)

            p1 = population[idx1]
            p2 = population[idx2]

            child = crossover(p1, p2, pc=pc)
            child = mutation(child, pm=pm, scale=0.05, max_weight=operator_max_weight)
            child = repair_weights(child, operator_max_weight)

            new_pop.append(child)
            pc_list.append(pc)
            pm_list.append(pm)

        population = np.array(new_pop)
        pc_hist.append(float(np.mean(pc_list)))
        pm_hist.append(float(np.mean(pm_list)))

    if last_improvement_gen is None:
        convergence_gen = generations
    else:
        convergence_gen = last_improvement_gen

    port_return = float(np.dot(global_best_weights, mu))
    port_risk = float(np.sqrt(global_best_weights @ cov_matrix @ global_best_weights))
    sharpe = port_return / port_risk if port_risk != 0 else 0.0
    runtime = time.time() - start_time

    return {
        "best_weights": global_best_weights,
        "best_fitness": global_best_fitness,
        "best_return": port_return,
        "best_risk": port_risk,
        "best_sharpe": sharpe,
        "history_best": best_hist,
        "history_avg": avg_hist,
        "history_diversity": diversity_hist,
        "history_pc": pc_hist,
        "history_pm": pm_hist,
        "best_weights_history": best_weights_history,
        "population_history": pop_history if save_pop_history else None,
        "convergence_generation": convergence_gen,
        "runtime": runtime,
    }


def adaptive_genetic_algorithm(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    pop_size: int = 500,
    generations: int = 88,
    max_weight: float = MAX_WEIGHT_DEFAULT,
    seed: Optional[int] = None,
    init_pop: Optional[np.ndarray] = None,
    save_pop_history: bool = False,
    adaptive_func: Optional[Callable] = None,
    k1: float = 0.9,
    k2: float = 0.9,
    k3: float = 0.1,
    k4: float = 0.9,
    k5: float = 0.9,
    k6: float = 0.3,
    lambda_div: float = LAMBDA_DIV_DEFAULT,
    operator_max_weight: float = MAX_WEIGHT_INITIAL
) -> Dict[str, Any]:
    """Execute Adaptive Genetic Algorithm (AGA) with dynamic parameter adaptation.

    Matches Cell 56 & Cell 92-93 of Implementasi-fix.ipynb.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(mu)
    if init_pop is not None:
        population = init_pop.copy()
    else:
        population = initialize_population(pop_size, n_assets, operator_max_weight)

    best_hist, avg_hist, diversity_hist = [], [], []
    pc_hist, pm_hist = [], []
    best_weights_history, pop_history = [], []

    global_best_fitness = np.inf
    global_best_weights = None
    last_improvement_gen = None
    tol = 1e-6

    for gen in range(generations):
        fitness = np.array([
            fitness_function(w, mu, cov_matrix, lambda_div=lambda_div, max_weight=max_weight)
            for w in population
        ])

        f_max = float(np.max(fitness))
        f_min = float(np.min(fitness))
        f_avg = float(np.mean(fitness))

        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]

        if best_fit < global_best_fitness - tol:
            global_best_fitness = float(best_fit)
            global_best_weights = population[best_idx].copy()
            last_improvement_gen = gen

        best_hist.append(float(best_fit))
        avg_hist.append(f_avg)
        diversity_hist.append(float(np.mean(np.std(population, axis=0))))
        best_weights_history.append(population[best_idx].copy())

        if save_pop_history:
            pop_history.append(population.copy())

        new_pop = []
        pc_list, pm_list = [], []

        while len(new_pop) < pop_size:
            idx1 = tournament_selection(population, fitness)
            idx2 = tournament_selection(population, fitness)

            p1, p2 = population[idx1], population[idx2]
            f1, f2 = fitness[idx1], fitness[idx2]

            if f2 < f1:
                p1, p2 = p2, p1
                f1, f2 = f2, f1

            fc = min(f1, f2)
            f_child_est = (f1 + f2) / 2

            if adaptive_func is not None:
                pc, pm = adaptive_func(fc, f_child_est, f_min, f_max, f_avg)
            else:
                pc, pm = calculate_adaptive_rates(
                    fc, f_child_est, f_min, f_max, f_avg,
                    k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6
                )

            child = crossover(p1, p2, pc=pc)
            child = mutation(child, pm=pm, scale=0.05, max_weight=operator_max_weight)
            child = repair_weights(child, operator_max_weight)

            new_pop.append(child)
            pc_list.append(pc)
            pm_list.append(pm)

        population = np.array(new_pop)
        pc_hist.append(float(np.mean(pc_list)))
        pm_hist.append(float(np.mean(pm_list)))

    if last_improvement_gen is None:
        convergence_gen = generations
    else:
        convergence_gen = last_improvement_gen

    port_return = float(np.dot(global_best_weights, mu))
    port_risk = float(np.sqrt(global_best_weights @ cov_matrix @ global_best_weights))
    sharpe = port_return / port_risk if port_risk != 0 else 0.0
    runtime = time.time() - start_time

    return {
        "best_weights": global_best_weights,
        "best_fitness": global_best_fitness,
        "best_return": port_return,
        "best_risk": port_risk,
        "best_sharpe": sharpe,
        "history_best": best_hist,
        "history_avg": avg_hist,
        "history_diversity": diversity_hist,
        "history_pc": pc_hist,
        "history_pm": pm_hist,
        "best_weights_history": best_weights_history,
        "population_history": pop_history if save_pop_history else None,
        "convergence_generation": convergence_gen,
        "runtime": runtime,
    }


def get_final_ga_weights(
    mu: Union[pd.Series, np.ndarray],
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    n_runs: int = 1,
    pop_size: int = 440,
    generations: int = 100
) -> Dict[str, Any]:
    """Generate final SGA portfolio weights for a WFO retraining phase (Cell 109 of Implementasi-fix.ipynb)."""
    mu_arr = mu.values if hasattr(mu, "values") else np.asarray(mu)
    cov_arr = cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix)

    best_fitness = float("inf")
    best_weights = None

    for run in range(n_runs):
        current_seed = run + WFO_WEIGHT_SEED_START
        res = genetic_algorithm(
            mu=mu_arr,
            cov_matrix=cov_arr,
            pop_size=pop_size,
            generations=generations,
            pc=SGA_BEST_PARAMS["pc"],
            pm=SGA_BEST_PARAMS["pm"],
            seed=current_seed,
        )
        if res["best_fitness"] < best_fitness:
            best_fitness = res["best_fitness"]
            best_weights = res["best_weights"]

    if hasattr(mu, "index"):
        best_weights = pd.Series(best_weights, index=mu.index, name="Weight")

    return {"weights": best_weights, "fitness": best_fitness}


def get_final_aga_weights(
    mu: Union[pd.Series, np.ndarray],
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    n_runs: int = 1,
    pop_size: int = 500,
    generations: int = 88
) -> Dict[str, Any]:
    """Generate final AGA portfolio weights for a WFO retraining phase (Cell 110 of Implementasi-fix.ipynb)."""
    mu_arr = mu.values if hasattr(mu, "values") else np.asarray(mu)
    cov_arr = cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix)

    best_fitness = float("inf")
    best_weights = None

    for run in range(n_runs):
        current_seed = run + WFO_WEIGHT_SEED_START
        res = adaptive_genetic_algorithm(
            mu=mu_arr,
            cov_matrix=cov_arr,
            pop_size=pop_size,
            generations=generations,
            seed=current_seed,
            k1=AGA_BEST_PARAMS["k1"],
            k2=AGA_BEST_PARAMS["k2"],
            k3=AGA_BEST_PARAMS["k3"],
            k4=AGA_BEST_PARAMS["k4"],
            k5=AGA_BEST_PARAMS["k5"],
            k6=AGA_BEST_PARAMS["k6"],
        )
        if res["best_fitness"] < best_fitness:
            best_fitness = res["best_fitness"]
            best_weights = res["best_weights"]

    if hasattr(mu, "index"):
        best_weights = pd.Series(best_weights, index=mu.index, name="Weight")

    return {"weights": best_weights, "fitness": best_fitness}


def evaluate_algorithm_performance(
    phase_data: Dict[str, Tuple[Union[pd.Series, np.ndarray], Union[pd.DataFrame, np.ndarray]]],
    n_runs: int = 10,
    seed: int = SEED
) -> Tuple[pd.DataFrame, list, list, list, list, list, list]:
    """Run paired multi-start evaluation of SGA vs. AGA across WFO retraining phases.

    Matches Cells 135-136 of Implementasi-fix.ipynb:
    - 4 phases x 10 runs x 2 algorithms = 80 total optimization runs.
    - Uses shared initial population (initialized with max_weight=0.10) per seed_run = seed + run
      so that SGA and AGA form exact paired observations under equal 44,000 NFE budget.
    """
    raw_results = []
    sga_best_histories, sga_avg_histories = [], []
    aga_best_histories, aga_avg_histories = [], []
    pc_histories, pm_histories = [], []

    pop_ga = SGA_BEST_PARAMS["pop_size"]
    gen_ga = SGA_BEST_PARAMS["generations"]
    pc_ga = SGA_BEST_PARAMS["pc"]
    pm_ga = SGA_BEST_PARAMS["pm"]

    pop_aga = AGA_BEST_PARAMS["pop_size"]
    gen_aga = AGA_BEST_PARAMS["generations"]

    for phase_name, (mu, cov_matrix) in phase_data.items():
        mu_arr = mu.values if hasattr(mu, "values") else np.asarray(mu)
        cov_arr = cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix)

        for run in range(n_runs):
            seed_run = seed + run
            np.random.seed(seed_run)

            shared_population = initialize_population(
                pop_aga,
                len(mu_arr),
                max_weight=MAX_WEIGHT_DEFAULT,
            )

            result_sga = genetic_algorithm(
                mu=mu_arr,
                cov_matrix=cov_arr,
                pop_size=pop_ga,
                generations=gen_ga,
                pc=pc_ga,
                pm=pm_ga,
                seed=seed_run,
                init_pop=shared_population[:pop_ga],
            )

            result_aga = adaptive_genetic_algorithm(
                mu=mu_arr,
                cov_matrix=cov_arr,
                pop_size=pop_aga,
                generations=gen_aga,
                seed=seed_run,
                init_pop=shared_population,
                k1=AGA_BEST_PARAMS["k1"],
                k2=AGA_BEST_PARAMS["k2"],
                k3=AGA_BEST_PARAMS["k3"],
                k4=AGA_BEST_PARAMS["k4"],
                k5=AGA_BEST_PARAMS["k5"],
                k6=AGA_BEST_PARAMS["k6"],
            )

            raw_results.extend([
                {
                    "Phase": phase_name,
                    "Algorithm": "SGA",
                    "Run": run + 1,
                    "Fitness": result_sga["best_fitness"],
                    "Sharpe": result_sga["best_sharpe"],
                    "Runtime": result_sga["runtime"],
                    "ConvergenceGen": result_sga["convergence_generation"],
                },
                {
                    "Phase": phase_name,
                    "Algorithm": "AGA",
                    "Run": run + 1,
                    "Fitness": result_aga["best_fitness"],
                    "Sharpe": result_aga["best_sharpe"],
                    "Runtime": result_aga["runtime"],
                    "ConvergenceGen": result_aga["convergence_generation"],
                },
            ])

            sga_best_histories.append(result_sga["history_best"])
            sga_avg_histories.append(result_sga["history_avg"])
            aga_best_histories.append(result_aga["history_best"])
            aga_avg_histories.append(result_aga["history_avg"])
            pc_histories.append(result_aga["history_pc"])
            pm_histories.append(result_aga["history_pm"])

    return (
        pd.DataFrame(raw_results),
        sga_best_histories,
        sga_avg_histories,
        aga_best_histories,
        aga_avg_histories,
        pc_histories,
        pm_histories,
    )
