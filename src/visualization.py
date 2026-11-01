"""Visualization utilities for portfolio backtesting, convergence, diversity, and adaptive rates.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# Consistent Color Palette
COLOR_AGA = "#1f77b4"      # Professional Blue
COLOR_SGA = "#2ca02c"      # Green
COLOR_IDX30 = "#d62728"    # Red
COLOR_EQUAL = "#7f7f7f"    # Neutral Gray


def plot_evaluation_metrics(
    results: List[Dict[str, Any]],
    title: str = "Comprehensive IDX30 Portfolio Performance Evaluation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot grouped comparative bar charts across 6 core financial metrics.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of strategy backtest result dictionaries.
    title : str
        Figure suptitle.
    save_path : str, optional
        Path to save rendered image.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    strategy_names = [res["Method"] for res in results]
    metrics = {
        "Total Return (%)": [res["Total Return"] * 100.0 for res in results],
        "Annual Return (%)": [res["Annual Return"] * 100.0 for res in results],
        "Annual Volatility (%)": [res["Annual Volatility"] * 100.0 for res in results],
        "Sharpe Ratio": [res["Sharpe Ratio"] for res in results],
        "Max Drawdown (%)": [res["Max Drawdown"] * 100.0 for res in results],
        "Transaction Cost (%)": [res["Total Fees Paid"] * 100.0 for res in results],
    }

    colors = [COLOR_AGA, COLOR_SGA, COLOR_IDX30, COLOR_EQUAL][:len(strategy_names)]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, (metric_name, metric_values) in zip(axes.flat, metrics.items()):
        bars = ax.bar(
            strategy_names,
            metric_values,
            color=colors,
            edgecolor="black",
            linewidth=0.8
        )
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.axhline(y=0, color="black", linewidth=1, alpha=0.6)

        if "%" in metric_name:
            if "Cost" in metric_name:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            else:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        max_abs = max(np.abs(metric_values)) if len(metric_values) > 0 else 1.0
        offset = max_abs * 0.03 if max_abs > 0 else 0.01

        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (offset if val >= 0 else -offset),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
                fontweight="semibold"
            )
        ax.tick_params(axis="x", rotation=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_convergence(
    sga_history_best: List[float],
    aga_history_best: List[float],
    title: str = "Optimization Convergence Trajectory (SGA vs. AGA)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot best fitness convergence trajectories across generations.

    Parameters
    ----------
    sga_history_best : List[float]
        SGA best fitness per generation.
    aga_history_best : List[float]
        AGA best fitness per generation.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sga_history_best, linestyle="--", linewidth=2.5, color=COLOR_SGA, label="Standard GA (SGA)", alpha=0.9)
    ax.plot(aga_history_best, linewidth=2.5, color=COLOR_AGA, label="Adaptive GA (AGA)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Objective Fitness (Lower is Better)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_diversity(
    sga_diversity: List[float],
    aga_diversity: List[float],
    title: str = "Population Diversity Trajectory Across Generations",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot population diversity (mean standard deviation across chromosomes).

    Parameters
    ----------
    sga_diversity : List[float]
        SGA diversity per generation.
    aga_diversity : List[float]
        AGA diversity per generation.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sga_diversity, linestyle="--", linewidth=2.5, color=COLOR_SGA, label="Standard GA Diversity", alpha=0.9)
    ax.plot(aga_diversity, linewidth=2.5, color=COLOR_AGA, label="Adaptive GA Diversity")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Population Diversity (Mean Std Dev)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_adaptive_rates(
    history_pc: List[float],
    history_pm: List[float],
    title: str = "AGA Dynamic Operator Rate Evolution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot dynamic crossover rate (Pc) and mutation rate (Pm) across generations.

    Parameters
    ----------
    history_pc : List[float]
        Mean Pc per generation.
    history_pm : List[float]
        Mean Pm per generation.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(history_pc, linewidth=2.5, color=COLOR_AGA, label="Adaptive $P_c$")
    ax1.set_title("Crossover Probability ($P_c$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=10)
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history_pm, linewidth=2.5, color="#e6550d", label="Adaptive $P_m$")
    ax2.set_title("Mutation Probability ($P_m$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Generation", fontsize=10)
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

