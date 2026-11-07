"""IDX30 Portfolio Analytics Platform.

Quantitative Research & Allocation Framework:
- Regularized Mean-Variance Optimization using Adaptive Genetic Algorithm (AGA).
- 4-Phase Rolling Walk-Forward Backtesting with Transaction Cost Frictions.
- Dynamic Operator Control ($P_c, P_m$) and Constraint Enforcement.
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import src
from src.config import (
    AGA_BEST_PARAMS,
    FEE_BUY,
    FEE_SELL,
    IDX30_WEIGHT_MAPS,
    LAMBDA_DIV_DEFAULT,
    MAX_WEIGHT_DEFAULT,
    RISK_FREE_RATE,
    SEED,
    SGA_BEST_PARAMS,
    TRADING_DAYS_PER_YEAR,
)

# ==============================================================================
# Page Configuration & Institutional Theme
# ==============================================================================
st.set_page_config(
    page_title="IDX30 Portfolio Analytics | AGA Terminal",
    page_icon="https://cdn-icons-png.flaticon.com/512/2620/2620611.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist, High-Finance Typography & Components
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }

    .kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .terminal-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }

    .terminal-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    .meta-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .meta-badge {
        font-family: 'JetBrains Mono', monospace;
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 500;
        border-radius: 4px;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.1rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }

    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .stat-delta-positive {
        display: inline-block;
        padding: 0.15rem 0.45rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 3px;
        background-color: #ecfdf5;
        color: #065f46;
    }

    .stat-delta-neutral {
        display: inline-block;
        padding: 0.15rem 0.45rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 3px;
        background-color: #f1f5f9;
        color: #475569;
    }

    .callout-box {
        background-color: #f8fafc;
        border-left: 3px solid #1e40af;
        border-radius: 0 4px 4px 0;
        padding: 0.85rem 1rem;
        font-size: 0.875rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    /* Style Streamlit Tabs for clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #1e40af !important;
        border-bottom: 2px solid #1e40af !important;
    }

    /* Dataframe styling */
    .dataframe {
        font-size: 0.82rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Data Loaders
# ==============================================================================
@st.cache_data
def load_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame, pd.Series]:
    raw_path = "data/raw/IDX30-Scraping.csv"
    list_path = "data/raw/IDX30-List.csv"
    df_raw = src.load_raw_stock_data(raw_path)
    df_clean = src.preprocess_stock_data(df_raw)
    consistent_tickers = src.get_consistent_idx30_tickers(list_path)
    df_selected = src.prepare_study_dataset(df_clean, consistent_tickers)
    last_prices = df_clean.groupby("ticker")["Close"].last()
    return df_raw, df_clean, consistent_tickers, df_selected, last_prices


@st.cache_data
def load_wfo_weights() -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_aga = pd.read_csv("results/tables/wfo_weights_aga.csv", index_col="ticker")
    w_sga = pd.read_csv("results/tables/wfo_weights_sga.csv", index_col="ticker")
    return w_aga, w_sga


@st.cache_data
def compute_wfo_time_series(capital_idr: float) -> Tuple[Dict[str, any], Dict[str, any], Dict[str, any], Dict[str, any]]:
    df_raw, df_clean, consistent_tickers, df_selected, _ = load_all_datasets()
    w_aga_df, w_sga_df = load_wfo_weights()

    _, _, df_test = src.split_dataset(df_selected)
    _, _, returns_test = src.compute_statistics(df_test)
    returns_list = src.get_wfo_test_phase_returns(returns_test)
    returns_idx_list, w_idx_list = src.get_wfo_idx30_benchmark_phases(df_clean)

    phases = ["Fase 1", "Fase 2", "Fase 3", "Fase 4"]
    w_aga_list = [w_aga_df.loc[returns_list[i].columns, p].values for i, p in enumerate(phases)]
    w_sga_list = [w_sga_df.loc[returns_list[i].columns, p].values for i, p in enumerate(phases)]
    equal_weight = np.ones(len(consistent_tickers)) / len(consistent_tickers)

    res_aga = src.run_rebalancing(w_aga_list, returns_list, "Adaptive GA (AGA, 4×)")
    res_sga = src.run_rebalancing(w_sga_list, returns_list, "Standard GA (SGA, 4×)")
    res_eq = src.run_isolated(equal_weight, returns_test, "Equal Weight (1/N)")
    res_idx = src.run_rebalancing(w_idx_list, returns_idx_list, "IDX30 Benchmark")

    return res_aga, res_sga, res_eq, res_idx


# ==============================================================================
# Helper Formatting Functions
# ==============================================================================
def format_idr(value: float) -> str:
    """Format numeric values into formatted Indonesian Rupiah (IDR)."""
    return f"Rp {value:,.0f}".replace(",", ".")


# Institutional Plotly Layout Configuration
PLOTLY_THEME = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=11, color="#334155"),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        tickcolor="#cbd5e1",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        tickcolor="#cbd5e1",
        zeroline=False,
    ),
)


# ==============================================================================
# Sidebar Navigation & Execution Controls
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="kicker">Execution Environment</div>', unsafe_allow_html=True)
    st.markdown("### Portfolio Sizing")

    capital_idr = st.number_input(
        "Initial Capital (IDR):",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=100_000_000,
        step=10_000_000,
        format="%d",
        help="Initial capital pool used to calculate terminal portfolio values and lot-sized allocations.",
    )
    st.markdown(f"**Base Capital:** `{format_idr(capital_idr)}`")

    st.markdown("---")
    st.markdown('<div class="kicker">Market Frictions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Asset Universe:** 22 Consistently-Listed IDX30 Equities
        - **Risk-Free Rate:** 6.00% p.a. (Bank Indonesia BI-Rate Benchmark)
        - **Transaction Slippage / Brokerage:**
          - Acquisition (Buy): `0.15%`
          - Liquidation (Sell): `0.25%` (incl. Final Income Tax)
        - **Rebalancing Schedule:** Quarterly Expanding Window
        """
    )

    st.markdown("---")
    st.markdown('<div class="kicker">Attribution & Codebase</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Author:** Miftahullah Suryanugraha  
        *Undergraduate Thesis in Informatics Engineering*  
        [Source Repository (GitHub)](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)
        """
    )

# Load underlying dataset
df_raw, df_clean, consistent_tickers, df_selected, last_prices = load_all_datasets()
w_aga_df, w_sga_df = load_wfo_weights()

# ==============================================================================
# Header Section
# ==============================================================================
st.markdown('<div class="kicker">Quantitative Research & Investment Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-title">IDX30 Adaptive Portfolio Allocation Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="terminal-subtitle">Regularized Mean-Variance Optimization using Adaptive Genetic Algorithm (AGA) with 4-Phase Out-of-Sample Walk-Forward Validation.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="meta-badge-container">
        <span class="meta-badge">UNIVERSE: 22 IDX30 EQUITIES</span>
        <span class="meta-badge">MODEL: MEAN-VARIANCE + HHI PENALTY</span>
        <span class="meta-badge">OPERATOR CONTROL: DYNAMIC (SRINIVAS & PATNAIK)</span>
        <span class="meta-badge">VALIDATION: 4-PHASE WALK-FORWARD (2025-2026)</span>
        <span class="meta-badge">BENCHMARK: IDX30 & 1/N EQUAL WEIGHT</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "Walk-Forward Performance (OOS)",
    "Algorithm Optimization Simulation",
    "Asset Universe & Risk Diagnostics",
])

# ==============================================================================
# TAB 1: WALK-FORWARD PERFORMANCE (OUT-OF-SAMPLE)
# ==============================================================================
with tab1:
    st.markdown(
        """
        <div class="callout-box">
            <strong>Methodology Note:</strong> This backtest evaluates portfolio allocations strictly out-of-sample over 4 quarterly rebalancing phases (January 2025 – January 2026). The model incorporates continuous intra-phase asset price drift, capital replenishment, and asymmetric transaction costs (0.15% acquisition, 0.25% liquidation).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Reconstruct multi-phase portfolio trajectories
    res_aga, res_sga, res_eq, res_idx = compute_wfo_time_series(capital_idr)

    # Metric Cards Top Row
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)

    with col_c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Out-of-Sample Return</div>
                <div class="stat-value">{res_aga['Total Return'] * 100:.2f}%</div>
                <span class="stat-delta-positive">+{(res_aga['Total Return'] - res_idx['Total Return']) * 100:.2f}% vs IDX30</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Annualized Sharpe Ratio</div>
                <div class="stat-value">{res_aga['Sharpe Ratio']:.4f}</div>
                <span class="stat-delta-positive">+{res_aga['Sharpe Ratio'] - res_idx['Sharpe Ratio']:.4f} vs IDX30</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Annualized Return</div>
                <div class="stat-value">{res_aga['Annual Return'] * 100:.2f}%</div>
                <span class="stat-delta-positive">+{(res_aga['Annual Return'] - res_sga['Annual Return']) * 100:.2f}% vs SGA</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Maximum Drawdown</div>
                <div class="stat-value">{res_aga['Max Drawdown'] * 100:.2f}%</div>
                <span class="stat-delta-positive">{abs(res_idx['Max Drawdown'] - res_aga['Max Drawdown']) * 100:.2f}% Less Loss</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c5:
        final_aga_rp = capital_idr * res_aga["Final Portfolio Value"]
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Terminal Portfolio Value</div>
                <div class="stat-value">{format_idr(final_aga_rp)}</div>
                <span class="stat-delta-neutral">Base: {format_idr(capital_idr)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Cumulative Equity and Drawdown
    fig_curves = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Portfolio Trajectory (IDR)", "Peak-to-Trough Drawdown (%)"),
        row_heights=[0.68, 0.32],
    )

    strategies = [
        ("Adaptive GA (AGA, 4×)", res_aga, "#1e40af", "solid", 2.5),
        ("Standard GA (SGA, 4×)", res_sga, "#059669", "dash", 1.8),
        ("Equal Weight (1/N)", res_eq, "#64748b", "dot", 1.5),
        ("IDX30 Market Benchmark", res_idx, "#dc2626", "solid", 1.8),
    ]

    for label, res, color, dash, width in strategies:
        equity_series = res["Portfolio Value"] * capital_idr
        drawdown_series = res["Drawdown"] * 100.0

        fig_curves.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%d %b %Y}<br>Portfolio: Rp %{y:,.0f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )

        fig_curves.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                mode="lines",
                name=label,
                showlegend=False,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra>" + label + "</extra>",
            ),
            row=2,
            col=1,
        )

    # Phase boundary vertical lines
    rebalance_dates = [
        ("2025-05-02", "Rebalance: Phase 2"),
        ("2025-08-01", "Rebalance: Phase 3"),
        ("2025-11-03", "Rebalance: Phase 4"),
    ]
    for r_date, r_label in rebalance_dates:
        fig_curves.add_vline(
            x=r_date,
            line_width=1,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=r_label,
            annotation_position="top left",
            annotation_font=dict(size=9, color="#64748b", family="JetBrains Mono, monospace"),
        )

    fig_curves.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        **PLOTLY_THEME,
    )
    fig_curves.update_yaxes(title_text="Portfolio Value (IDR)", row=1, col=1)
    fig_curves.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig_curves, use_container_width=True)

    # Comparative Performance Table
    st.markdown("#### Cumulative Walk-Forward Performance Summary (4 Phases Combined)")
    ordered_strategies = [
        (1, "Adaptive GA (AGA, 4× Rebalanced)", res_aga),
        (2, "Equal Weight (1/N Buy & Hold)", res_eq),
        (3, "Standard GA (SGA, 4× Rebalanced)", res_sga),
        (4, "IDX30 Market Benchmark (4× Rebalanced)", res_idx),
    ]
    summary_data = [
        {
            "Rank": rank,
            "Strategy": label,
            "Total Return (%)": f"{res['Total Return'] * 100:.4f}%",
            "Annualized Return (%)": f"{res['Annual Return'] * 100:.4f}%",
            "Annualized Volatility (%)": f"{res['Annual Volatility'] * 100:.4f}%",
            "Sharpe Ratio": f"{res['Sharpe Ratio']:.4f}",
            "Max Drawdown (%)": f"{res['Max Drawdown'] * 100:.4f}%",
            "Transaction Cost (%)": f"{res['Total Fees Paid'] * 100:.4f}%",
            "Terminal Wealth": format_idr(res["Final Portfolio Value"] * capital_idr),
        }
        for rank, label, res in ordered_strategies
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with st.expander("Isolated Per-Phase Walk-Forward Evaluation (Fase 1 – Fase 4)"):
        st.markdown(
            "Out-of-sample evaluation of each individual 3-month WFO testing window (initial entry cost `0.15%`, ranked by annualized Sharpe Ratio within each phase):"
        )
        iso_df = pd.read_csv("results/tables/wfo_isolated_phase_performance.csv")
        st.dataframe(iso_df, use_container_width=True, hide_index=True)

    # Phase-by-Phase Weight Allocation Analysis
    st.markdown("#### Walk-Forward Asset Weights & Rebalancing Allocations")
    col_w_sel, _ = st.columns([1, 2])
    with col_w_sel:
        phase_selected = st.selectbox(
            "Evaluation Horizon:",
            ["All Phases (Horizontal Comparison)", "Fase 1", "Fase 2", "Fase 3", "Fase 4"],
        )

    if phase_selected == "All Phases (Horizontal Comparison)":
        fig_w = px.bar(
            w_aga_df.reset_index().melt(id_vars="ticker", var_name="Phase", value_name="Weight"),
            x="ticker",
            y="Weight",
            color="Phase",
            barmode="group",
            color_discrete_sequence=["#1e40af", "#3b82f6", "#60a5fa", "#93c5fd"],
        )
        fig_w.update_layout(
            height=380,
            xaxis_title="Constituent Stock Ticker",
            yaxis_title="Portfolio Weight (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOTLY_THEME,
        )
        fig_w.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            comp_weights = pd.DataFrame({
                "Ticker": w_aga_df.index,
                "AGA Weight (%)": (w_aga_df[phase_selected] * 100.0).round(2),
                "SGA Weight (%)": (w_sga_df[phase_selected] * 100.0).round(2),
                "Capital Allocation": (w_aga_df[phase_selected] * capital_idr).round(0),
            })

            # Calculate Lots (1 Lot = 100 shares) based on current close price
            comp_weights["Last Price"] = comp_weights["Ticker"].map(last_prices)
            comp_weights["Suggested Lots"] = np.floor(
                comp_weights["Capital Allocation"] / (comp_weights["Last Price"] * 100.0)
            ).astype(int)

            comp_weights = comp_weights.sort_values("AGA Weight (%)", ascending=False).reset_index(drop=True)

            st.dataframe(
                comp_weights.style.format({
                    "AGA Weight (%)": "{:.2f}%",
                    "SGA Weight (%)": "{:.2f}%",
                    "Capital Allocation": "Rp {:,.0f}",
                    "Last Price": "Rp {:,.0f}",
                    "Suggested Lots": "{:,} lot",
                }),
                height=380,
                use_container_width=True,
                hide_index=True,
            )

        with col_w2:
            fig_donut = px.pie(
                comp_weights[comp_weights["AGA Weight (%)"] > 0],
                names="Ticker",
                values="AGA Weight (%)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_donut.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="v", y=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # Statistical Inference & Multi-Run Stability
    with st.expander("Extended Multi-Run Stochastic Stability & Statistical Evaluation (4 Phases × 10 Paired Runs = 80 Runs)"):
        st.markdown(
            r"""
            **Methodological Context:** Because SGA and AGA are stochastic population-based metaheuristics, an extended multi-run statistical evaluation is conducted across the 4 Walk-Forward retraining windows ($N_{\text{runs}} = 10$ paired seeds per phase, `SEED = 29..38`, total $80$ optimization runs).
            - **Paired Experimental Design:** For each seed $r \in \{1, \dots, 10\}$, both SGA ($\text{Pop}=440, \text{Gen}=100$) and AGA ($\text{Pop}=500, \text{Gen}=88$) start from the **exact same initial population (`shared_population`)** under an identical $44,000$ function evaluation (NFE) budget.
            - **Two-Stage Statistical Testing Pipeline:**
              1. **Shapiro-Wilk Normality Test ($\alpha = 0.05$):** Applied to the paired differences $d_r = X_{\text{AGA}, r} - X_{\text{SGA}, r}$ ($n = 10$ pairs per phase).
              2. **Automatic Test Routing:** Uses the parametric **Paired Student's t-test** (`scipy.stats.ttest_rel`) when $p_{\text{Shapiro}} > 0.05$ (*Normal*), and the non-parametric **Wilcoxon Signed-Rank Test** (`scipy.stats.wilcoxon`) when $p_{\text{Shapiro}} \le 0.05$ (*Tidak Normal*).
            """
        )
        stats_df = pd.read_csv("results/tables/statistical_significance_tests.csv")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: ALGORITHM OPTIMIZATION SIMULATION
# ==============================================================================
with tab2:
    st.markdown('<div class="kicker">Evolutionary Simulation Engine</div>', unsafe_allow_html=True)
    st.markdown("#### Real-Time Portfolio Optimization (AGA vs. SGA)")
    st.caption("Customize constituent assets, penalty constraints (tuned thesis defaults: w_max = 10%, λ_div = 0.90), and genetic hyperparameters to run live comparative convergence.")

    with st.form("optimizer_form"):
        col_param1, col_param2, col_param3 = st.columns(3)

        with col_param1:
            st.markdown("**1. Asset Universe**")
            selected_assets = st.multiselect(
                "Select Tickers:",
                options=consistent_tickers,
                default=consistent_tickers,
                help="Subset of IDX30 stocks to form the portfolio optimization space (default: all 22 consistent IDX30 stocks).",
            )

        with col_param2:
            st.markdown("**2. Risk & Regularization**")
            max_w_input = st.slider("Max Single-Asset Weight (%):", 10, 50, int(MAX_WEIGHT_DEFAULT * 100), 5) / 100.0
            lambda_div_input = st.slider("HHI Diversification Penalty (λ):", 0.0, 1.0, float(LAMBDA_DIV_DEFAULT), 0.05)

        with col_param3:
            st.markdown("**3. Hyperparameters**")
            pop_size_input = st.slider("Population Size (M):", 40, 500, 100, 20)
            generations_input = st.slider("Generations (G):", 15, 100, 50, 5)
            seed_input = st.number_input("RNG Seed:", value=SEED, step=1)

        btn_run = st.form_submit_button("Execute Optimization Simulation", use_container_width=True)

    if btn_run:
        if len(selected_assets) < 3:
            st.warning("A minimum of 3 assets is required to construct a valid investment portfolio.")
        else:
            with st.spinner("Executing evolutionary search..."):
                t_start = time.time()

                df_sub = df_selected[df_selected["ticker"].isin(selected_assets)]
                df_train_sub, df_val_sub, df_test_sub = src.split_dataset(df_sub)
                mu_sub, cov_sub, _ = src.compute_statistics(df_train_sub)

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Evaluating Standard Genetic Algorithm (SGA)...")
                progress_bar.progress(35)
                res_live_sga = src.genetic_algorithm(
                    mu=mu_sub.values,
                    cov_matrix=cov_sub.values,
                    pop_size=pop_size_input,
                    generations=generations_input,
                    max_weight=max_w_input,
                    lambda_div=lambda_div_input,
                    seed=seed_input,
                )

                status_text.text("Evaluating Adaptive Genetic Algorithm (AGA)...")
                progress_bar.progress(75)
                res_live_aga = src.adaptive_genetic_algorithm(
                    mu=mu_sub.values,
                    cov_matrix=cov_sub.values,
                    pop_size=pop_size_input,
                    generations=generations_input,
                    max_weight=max_w_input,
                    lambda_div=lambda_div_input,
                    seed=seed_input,
                )

                progress_bar.progress(100)
                elapsed = time.time() - t_start
                status_text.text(f"Simulation completed in {elapsed:.2f} seconds.")
                time.sleep(0.4)
                progress_bar.empty()
                status_text.empty()

                st.success(f"Evolutionary optimization finished ({elapsed:.2f}s elapsed).")

                # Metrics Row
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("AGA Best Fitness", f"{res_live_aga['best_fitness']:.5f}")
                with col_m2:
                    st.metric("SGA Best Fitness", f"{res_live_sga['best_fitness']:.5f}")
                with col_m3:
                    st.metric("AGA In-Sample Sharpe", f"{res_live_aga['best_sharpe']:.4f}")
                with col_m4:
                    st.metric("SGA In-Sample Sharpe", f"{res_live_sga['best_sharpe']:.4f}")

                # Evolutionary Curves
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(y=res_live_sga["history_best"], mode="lines", name="Standard GA (SGA)", line=dict(color="#059669", dash="dash", width=1.8)))
                    fig_conv.add_trace(go.Scatter(y=res_live_aga["history_best"], mode="lines", name="Adaptive GA (AGA)", line=dict(color="#1e40af", width=2.5)))
                    fig_conv.update_layout(title="Objective Fitness Minimization Trajectory", xaxis_title="Generation", yaxis_title="Fitness Score", height=350, **PLOTLY_THEME)
                    st.plotly_chart(fig_conv, use_container_width=True)

                with col_p2:
                    fig_div = go.Figure()
                    fig_div.add_trace(go.Scatter(y=res_live_sga["history_diversity"], mode="lines", name="Standard GA Diversity", line=dict(color="#059669", dash="dash", width=1.8)))
                    fig_div.add_trace(go.Scatter(y=res_live_aga["history_diversity"], mode="lines", name="Adaptive GA Diversity", line=dict(color="#1e40af", width=2.5)))
                    fig_div.update_layout(title="Population Chromosome Diversity", xaxis_title="Generation", yaxis_title="Mean Gene Standard Deviation", height=350, **PLOTLY_THEME)
                    st.plotly_chart(fig_div, use_container_width=True)

                # Dynamic Operator Rates
                fig_rates = go.Figure()
                fig_rates.add_trace(go.Scatter(y=res_live_aga["history_pc"], mode="lines", name="Adaptive Crossover Rate (Pc)", line=dict(color="#1e40af", width=2)))
                fig_rates.add_trace(go.Scatter(y=res_live_aga["history_pm"], mode="lines", name="Adaptive Mutation Rate (Pm)", line=dict(color="#ea580c", width=2)))
                fig_rates.update_layout(title="Dynamic Operator Probability Trajectory ($P_c$ vs. $P_m$)", xaxis_title="Generation", yaxis_title="Operator Probability", height=300, **PLOTLY_THEME)
                st.plotly_chart(fig_rates, use_container_width=True)

                # Allocation Table & Lot Sizing
                st.markdown("#### Recommended Portfolio Allocation & Execution Sizing")
                w_live = res_live_aga["best_weights"]
                df_alloc = pd.DataFrame({
                    "Ticker": selected_assets,
                    "Weight (%)": (w_live * 100.0).round(2),
                    "Capital (IDR)": (w_live * capital_idr).round(0),
                })
                df_alloc["Last Close"] = df_alloc["Ticker"].map(last_prices)
                df_alloc["Execution Lots"] = np.floor(
                    df_alloc["Capital (IDR)"] / (df_alloc["Last Close"] * 100.0)
                ).astype(int)
                df_alloc = df_alloc.sort_values("Weight (%)", ascending=False).reset_index(drop=True)

                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.dataframe(
                        df_alloc.style.format({
                            "Weight (%)": "{:.2f}%",
                            "Capital (IDR)": "Rp {:,.0f}",
                            "Last Close": "Rp {:,.0f}",
                            "Execution Lots": "{:,} lot",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col_t2:
                    fig_bar = px.bar(
                        df_alloc[df_alloc["Weight (%)"] > 0],
                        x="Ticker",
                        y="Weight (%)",
                        title="Optimal Capital Allocation by Asset",
                        color_discrete_sequence=["#1e40af"],
                    )
                    fig_bar.update_layout(height=350, **PLOTLY_THEME)
                    st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# TAB 3: ASSET UNIVERSE & RISK DIAGNOSTICS
# ==============================================================================
with tab3:
    st.markdown('<div class="kicker">Cross-Sectional Analytics</div>', unsafe_allow_html=True)
    st.markdown("#### IDX30 Risk-Return Characteristics & Pairwise Dependencies")

    df_train, _, _ = src.split_dataset(df_selected)
    mu_all, cov_all, rets_all = src.compute_statistics(df_train)

    annual_ret = mu_all * TRADING_DAYS_PER_YEAR * 100.0
    annual_vol = np.sqrt(np.diag(cov_all)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    sharpe_ratios = (annual_ret - (RISK_FREE_RATE * 100.0)) / annual_vol

    df_assets = pd.DataFrame({
        "Ticker": mu_all.index,
        "Annual Return (%)": annual_ret.values,
        "Annual Volatility (%)": annual_vol,
        "Sharpe Ratio": sharpe_ratios.values,
    })

    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        st.markdown("##### Cross-Sectional Risk vs. Return Tradeoff")
        fig_scatter = px.scatter(
            df_assets,
            x="Annual Volatility (%)",
            y="Annual Return (%)",
            text="Ticker",
            color="Sharpe Ratio",
            color_continuous_scale="Blues",
            size=np.maximum(df_assets["Sharpe Ratio"] + 1.0, 0.4),
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#1e40af")))
        fig_scatter.update_layout(height=420, **PLOTLY_THEME)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_a2:
        st.markdown("##### Pairwise Daily Return Correlation Matrix")
        corr_matrix = rets_all.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Rebased Historical Equity Trajectories
    st.markdown("##### Normalized Historical Price Evolution (Base = 100)")
    tickers_to_plot = st.multiselect(
        "Select Assets to Rebase:",
        options=consistent_tickers,
        default=["BBCA", "BBRI", "BMRI", "ASII", "TLKM"],
    )

    if tickers_to_plot:
        df_p_sub = df_clean[df_clean["ticker"].isin(tickers_to_plot)].copy()
        pivot_p = df_p_sub.pivot(index="Date", columns="ticker", values="Close").dropna()
        pivot_norm = (pivot_p / pivot_p.iloc[0]) * 100.0

        fig_norm = px.line(
            pivot_norm.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Normalized Price"),
            x="Date",
            y="Normalized Price",
            color="Ticker",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig_norm.update_layout(
            height=380,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Rebased Price Index (Base 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_norm, use_container_width=True)
