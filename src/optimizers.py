"""Evolutionary optimization algorithms for portfolio weight optimization.

Provides implementation of:
1. Standard Genetic Algorithm (SGA) with fixed crossover (Pc) and mutation (Pm).
2. Adaptive Genetic Algorithm (AGA) with Srinivas & Patnaik inspired adaptive rates
   dynamically tuned per individual and population fitness distribution.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.objectives import fitness_function, repair_weights


def initialize_population(
    pop_size: int,
    n_assets: int,
    max_weight: float = 0.20
) -> np.ndarray:
    """Initialize a random initial population of candidate portfolio weight vectors.

    Parameters
    ----------
    pop_size : int
        Number of candidate individuals (chromosomes) in the population.
    n_assets : int
        Number of assets in the investment universe.
    max_weight : float
        Upper bound allocation per asset.

    Returns
    -------
    np.ndarray
        Array of shape (pop_size, n_assets) containing valid normalized weights.
    """
    pop = []
    for _ in range(pop_size):
        w = np.random.rand(n_assets)
        w = np.minimum(w, max_weight)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            w = np.ones(n_assets) / n_assets
        pop.append(w)
    return np.array(pop)


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int = 3
) -> int:
    """Select the best individual from k randomly chosen candidates (tournament selection).

    Since fitness is formulated as a minimization problem, the candidate with
    the minimum fitness value wins.

    Parameters
    ----------
    population : np.ndarray
        Current population array of shape (pop_size, n_assets).
    fitness : np.ndarray
        1D array of fitness values corresponding to each individual.
    k : int
        Tournament size (number of candidates compared).

    Returns
    -------
    int
        Index of the winning individual in the population array.
    """
    idx = np.random.choice(len(population), size=k, replace=False)
    best_idx = idx[np.argmin(fitness[idx])]
    return int(best_idx)


def crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    pc: float = 0.9
) -> np.ndarray:
    """Perform arithmetic (convex combination) crossover between two parent individuals.

    Parameters
    ----------
    p1 : np.ndarray
        Weight vector of parent 1.
    p2 : np.ndarray
        Weight vector of parent 2.
    pc : float
        Crossover probability.

    Returns
    -------
    np.ndarray
        Offspring weight vector.
    """
    if np.random.rand() > pc:
        return p1.copy()

    alpha = np.random.rand()
    child = (alpha * p1) + ((1.0 - alpha) * p2)
    return child


def mutation(
    individual: np.ndarray,
    pm: float = 0.1,
    scale: float = 0.05,
    max_weight: float = 0.20
) -> np.ndarray:
    """Perform Gaussian mutation on an individual weight vector.

    Parameters
    ----------
    individual : np.ndarray
        Candidate weight vector.
    pm : float
        Per-gene mutation probability.
    scale : float
        Standard deviation of the zero-mean Gaussian perturbation.
    max_weight : float
        Maximum asset weight cap.

    Returns
    -------
    np.ndarray
        Mutated individual.
    """
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < pm:
            child[i] += np.random.normal(0.0, scale)
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

    Based on the relative fitness of parents and individuals with respect to
    the population's minimum, average, and maximum fitness (minimization formulation):

    For Crossover:
        If fc <= f_avg:
            Pc = k1 - ((k1 - k3) * (f_avg - fc) / (f_avg - f_min))
            Pc = max(Pc, k3)
        Else:
            Pc = k2

    For Mutation:
        If f_ind <= f_avg:
            Pm = k4 - ((k4 - k6) * (f_avg - f_ind) / (f_avg - f_min))
            Pm = max(Pm, k6)
        Else:
            Pm = k5

    Parameters
    ----------
    fc : float
        Fitness of the better parent (min(f1, f2)).
    f_ind : float
        Estimated fitness of the offspring candidate.
    f_min : float
        Minimum (best) fitness across current population.
    f_max : float
        Maximum (worst) fitness across current population.
    f_avg : float
        Mean fitness of the population.
    k1, k2, k3 : float
        Crossover rate bounds and scaling coefficients.
    k4, k5, k6 : float
        Mutation rate bounds and scaling coefficients.

    Returns
    -------
    Tuple[float, float]
        (pc, pm) adaptive probability rates.
    """
    denom = f_avg - f_min
    if denom <= 1e-4:
        return k1, k4

    # Adaptive Crossover
    if fc <= f_avg:
        pc_adaptive = k1 - ((k1 - k3) * (f_avg - fc) / denom)
        pc = max(float(pc_adaptive), k3)
    else:
        pc = k2

    # Adaptive Mutation
    if f_ind <= f_avg:
        pm_adaptive = k4 - ((k4 - k6) * (f_avg - f_ind) / denom)
        pm = max(float(pm_adaptive), k6)
    else:
        pm = k5

    return pc, pm


def genetic_algorithm(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    pop_size: int = 440,
    generations: int = 100,
    max_weight: float = 0.20,
    seed: Optional[int] = 29,
    init_pop: Optional[np.ndarray] = None,
    save_pop_history: bool = False,
    pc: float = 0.5,
    pm: float = 0.9,
    lambda_div: float = 0.05
) -> Dict[str, Union[np.ndarray, float, int, list]]:
    """Execute Standard Genetic Algorithm (SGA) with fixed Pc and Pm for portfolio optimization.

    Parameters
    ----------
    mu : np.ndarray
        Vector of expected daily returns.
    cov_matrix : np.ndarray
        Covariance matrix of daily returns.
    pop_size : int
        Population size.
    generations : int
        Number of generations (iterations).
    max_weight : float
        Upper bound cap per asset.
    seed : int, optional
        Random seed for reproducibility.
    init_pop : np.ndarray, optional
        Pre-initialized starting population.
    save_pop_history : bool
        Whether to store full population arrays per generation.
    pc : float
        Fixed crossover probability.
    pm : float
        Fixed mutation probability.
    lambda_div : float
        Diversification penalty weight.

    Returns
    -------
    dict
        Comprehensive run results including best weights, fitness, in-sample Sharpe,
        convergence curves, and runtime.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(mu)
    if init_pop is not None:
        population = init_pop.copy()
    else:
        population = initialize_population(pop_size, n_assets, max_weight)

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
        avg_hist.append(float(np.mean(fitness)))
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

            child = crossover(p1, p2, pc=pc)
            child = mutation(child, pm=pm, scale=0.05, max_weight=max_weight)
            child = repair_weights(child, max_weight=max_weight)

            new_pop.append(child)
            pc_list.append(pc)
            pm_list.append(pm)

        population = np.array(new_pop)
        pc_hist.append(float(np.mean(pc_list)))
        pm_hist.append(float(np.mean(pm_list)))

    convergence_gen = generations if last_improvement_gen is None else last_improvement_gen

    port_return = float(np.dot(global_best_weights, mu))
    port_risk = float(np.sqrt(global_best_weights @ cov_matrix @ global_best_weights))
    sharpe = port_return / port_risk if port_risk != 0 else 0.0

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
        "runtime": time.time() - start_time,
    }


def adaptive_genetic_algorithm(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    pop_size: int = 500,
    generations: int = 88,
    max_weight: float = 0.20,
    seed: Optional[int] = None,
    init_pop: Optional[np.ndarray] = None,
    save_pop_history: bool = False,
    k1: float = 0.9,
    k2: float = 0.9,
    k3: float = 0.1,
    k4: float = 0.9,
    k5: float = 0.9,
    k6: float = 0.3,
    lambda_div: float = 0.05
) -> Dict[str, Union[np.ndarray, float, int, list]]:
    """Execute Adaptive Genetic Algorithm (AGA) with dynamic parameter adaptation.

    Adjusts crossover (Pc) and mutation (Pm) rates per reproduction pair based on
    the fitness landscape to prevent premature convergence and sustain diversity.

    Parameters
    ----------
    mu : np.ndarray
        Vector of expected daily returns.
    cov_matrix : np.ndarray
        Covariance matrix of daily returns.
    pop_size : int
        Population size.
    generations : int
        Number of generations.
    max_weight : float
        Upper bound cap per asset.
    seed : int, optional
        Random seed for reproducibility.
    init_pop : np.ndarray, optional
        Pre-initialized starting population.
    save_pop_history : bool
        Whether to store full population arrays per generation.
    k1 to k6 : float
        Adaptive rate scaling parameters.
    lambda_div : float
        Diversification penalty weight.

    Returns
    -------
    dict
        Comprehensive run results including best weights, fitness, in-sample Sharpe,
        convergence curves, adaptive rate trajectories, and runtime.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(mu)
    if init_pop is not None:
        population = init_pop.copy()
    else:
        population = initialize_population(pop_size, n_assets, max_weight)

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
            f_child_est = (f1 + f2) / 2.0

            pc_adapt, pm_adapt = calculate_adaptive_rates(
                fc=fc,
                f_ind=f_child_est,
                f_min=f_min,
                f_max=f_max,
                f_avg=f_avg,
                k1=k1, k2=k2, k3=k3,
                k4=k4, k5=k5, k6=k6
            )

            child = crossover(p1, p2, pc=pc_adapt)
            child = mutation(child, pm=pm_adapt, scale=0.05, max_weight=max_weight)
            child = repair_weights(child, max_weight=max_weight)

            new_pop.append(child)
            pc_list.append(pc_adapt)
            pm_list.append(pm_adapt)

        population = np.array(new_pop)
        pc_hist.append(float(np.mean(pc_list)))
        pm_hist.append(float(np.mean(pm_list)))

    convergence_gen = generations if last_improvement_gen is None else last_improvement_gen

    port_return = float(np.dot(global_best_weights, mu))
    port_risk = float(np.sqrt(global_best_weights @ cov_matrix @ global_best_weights))
    sharpe = port_return / port_risk if port_risk != 0 else 0.0

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
        "runtime": time.time() - start_time,
    }

