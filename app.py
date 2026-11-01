"""Interactive Streamlit Dashboard for IDX30 Portfolio Optimization.

Dual-Mode Web Application:
1. Thesis Walk-Forward Optimization (WFO) Showcase (Instant pre-computed results).
2. Live Portfolio Optimizer & Backtest Simulator (Real-time AGA vs. SGA execution).
3. IDX30 Constituent Asset Explorer & Correlation Analytics.
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
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="IDX30 Portfolio Optimization | Adaptive GA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Quantitative Dashboard
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-tag {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e3f2fd;
        color: #0d47a1;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Data Loaders
# ==============================================================================
@st.cache_data
def load_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    raw_path = "data/raw/IDX30-Scraping.csv"
    list_path = "data/raw/IDX30-List.csv"
    df_raw = src.load_raw_stock_data(raw_path)
    df_clean = src.preprocess_stock_data(df_raw)
    consistent_tickers = src.get_consistent_idx30_tickers(list_path)
    df_selected = src.prepare_study_dataset(df_clean, consistent_tickers)
    return df_raw, df_clean, consistent_tickers, df_selected


@st.cache_data
def load_wfo_weights() -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_aga = pd.read_csv("results/tables/wfo_weights_aga.csv", index_col="ticker")
    w_sga = pd.read_csv("results/tables/wfo_weights_sga.csv", index_col="ticker")
    return w_aga, w_sga


@st.cache_data
def compute_wfo_time_series(capital_idr: float) -> Tuple[Dict[str, any], Dict[str, any], Dict[str, any], Dict[str, any]]:
    df_raw, _, consistent_tickers, df_selected = load_all_datasets()
    w_aga_df, w_sga_df = load_wfo_weights()

    phases = [
        ("Fase 1", "2025-01-30", "2025-04-30"),
        ("Fase 2", "2025-05-02", "2025-07-31"),
        ("Fase 3", "2025-08-01", "2025-10-31"),
        ("Fase 4", "2025-11-03", "2026-01-26"),
    ]

    returns_list = []
    w_aga_list = []
    w_sga_list = []
    w_idx_list = []
    returns_idx_list = []

    df_idx_raw = src.preprocess_stock_data(df_raw)
    df_idx_raw["Rit"] = df_idx_raw.groupby("ticker")["Close"].pct_change()

    for phase_name, start_d, end_d in phases:
        df_p = df_selected[(df_selected["Date"] >= start_d) & (df_selected["Date"] <= end_d)]
        _, _, ret_p = src.compute_statistics(df_p)
        returns_list.append(ret_p)

        w_aga_list.append(w_aga_df.loc[ret_p.columns, phase_name].values)
        w_sga_list.append(w_sga_df.loc[ret_p.columns, phase_name].values)

        idx_map = IDX30_WEIGHT_MAPS[phase_name]
        ret_idx_p, w_idx = src.create_idx_benchmark(df_idx_raw, idx_map, start_d, end_d)
        w_idx_list.append(w_idx)
        returns_idx_list.append(ret_idx_p)

    w_eq_list = [np.ones(len(r.columns)) / len(r.columns) for r in returns_list]

    res_aga = src.run_rebalancing(w_aga_list, returns_list, "Adaptive GA (AGA, 4×)")
    res_sga = src.run_rebalancing(w_sga_list, returns_list, "Standard GA (SGA, 4×)")
    res_eq = src.run_rebalancing(w_eq_list, returns_list, "Equal Weight (1/N)")
    res_idx = src.run_rebalancing(w_idx_list, returns_idx_list, "IDX30 Benchmark")

    return res_aga, res_sga, res_eq, res_idx


# ==============================================================================
# Helper Formatting Functions
# ==============================================================================
def format_idr(value: float) -> str:
    """Format floating currency values into Indonesian Rupiah (IDR)."""
    return f"Rp {value:,.0f}".replace(",", ".")


# ==============================================================================
# Sidebar UI & Parameters
# ==============================================================================
with st.sidebar:
    st.markdown("### 📊 Portfolio Settings")
    st.markdown("Customize simulation capital & trading parameters.")

    capital_idr = st.number_input(
        "Initial Capital (IDR):",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=100_000_000,
        step=10_000_000,
        format="%d",
        help="Nominal starting capital used to scale portfolio values and share allocations.",
    )
    st.caption(f"Nominal: **{format_idr(capital_idr)}**")

    st.markdown("---")
    st.markdown("### 🔍 Model & Benchmark Info")
    st.markdown(
        """
        - **Asset Universe:** 22 Consistent IDX30 Stocks
        - **Objective:** Maximize Sharpe Ratio with HHI Penalty
        - **Evaluation:** 4-Phase Walk-Forward Optimization
        - **Trading Frictions:** Buy 0.15%, Sell 0.25%
        - **Risk-Free Rate:** 6.0% p.a.
        """
    )
    st.markdown("---")
    st.markdown("### 👨‍💻 Research Attribution")
    st.markdown(
        """
        **Author:** Miftahullah Suryanugraha  
        *Undergraduate Thesis Project (Skripsi)*  
        [GitHub Repository](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)
        """
    )

# Load base data
df_raw, df_clean, consistent_tickers, df_selected = load_all_datasets()
w_aga_df, w_sga_df = load_wfo_weights()

# ==============================================================================
# Main Application Header
# ==============================================================================
st.markdown('<div class="main-header">📈 IDX30 Stock Portfolio Optimization Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Adaptive Genetic Algorithm (AGA) with 4-Phase Walk-Forward Backtesting and Dynamic Operator Control</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🏆 Thesis 4-Phase WFO Showcase",
    "⚡ Live Portfolio Optimizer & Simulator",
    "📊 IDX30 Asset Explorer & Correlation",
])

# ==============================================================================
# TAB 1: THESIS WFO SHOWCASE (Instant Load)
# ==============================================================================
with tab1:
    st.markdown("### 🎯 Out-of-Sample Walk-Forward Optimization (Jan 2025 – Jan 2026)")
    st.info(
        "💡 **Walk-Forward Optimization (WFO)** evaluates the portfolio strictly out-of-sample across 4 distinct rebalancing phases with realistic price drift and transaction fees (0.15% buy, 0.25% sell). Results below show pre-computed thesis benchmarks."
    )

    # Compute continuous curves
    res_aga, res_sga, res_eq, res_idx = compute_wfo_time_series(capital_idr)

    # KPI Top Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric(
            label="AGA Total Return",
            value=f"{res_aga['Total Return'] * 100:.2f}%",
            delta=f"+{(res_aga['Total Return'] - res_idx['Total Return']) * 100:.2f}% vs IDX30",
        )
    with kpi_col2:
        st.metric(
            label="AGA Sharpe Ratio",
            value=f"{res_aga['Sharpe Ratio']:.4f}",
            delta=f"+{res_aga['Sharpe Ratio'] - res_idx['Sharpe Ratio']:.4f} vs IDX30",
        )
    with kpi_col3:
        st.metric(
            label="AGA Annualized Return",
            value=f"{res_aga['Annual Return'] * 100:.2f}%",
            delta=f"+{(res_aga['Annual Return'] - res_sga['Annual Return']) * 100:.2f}% vs SGA",
        )
    with kpi_col4:
        st.metric(
            label="AGA Max Drawdown",
            value=f"{res_aga['Max Drawdown'] * 100:.2f}%",
            delta=f"{abs(res_idx['Max Drawdown'] - res_aga['Max Drawdown']) * 100:.2f}% less drawdown",
        )
    with kpi_col5:
        final_aga_rp = capital_idr * res_aga["Final Portfolio Value"]
        st.metric(
            label="Final Portfolio Value",
            value=format_idr(final_aga_rp),
            delta=format_idr(final_aga_rp - (capital_idr * res_idx["Final Portfolio Value"])),
        )

    # Interactive Cumulative Equity & Drawdown Chart
    st.markdown("#### 📈 Cumulative Portfolio Wealth & Drawdown Dynamics")

    fig_curves = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Portfolio Value (IDR)", "Drawdown Trajectory (%)"),
        row_heights=[0.65, 0.35],
    )

    strategies = [
        ("Adaptive GA (AGA)", res_aga, "#1f77b4", "solid", 3),
        ("Standard GA (SGA)", res_sga, "#2ca02c", "dash", 2),
        ("Equal Weight (1/N)", res_eq, "#7f7f7f", "dot", 2),
        ("IDX30 Benchmark", res_idx, "#d62728", "solid", 2),
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
                hovertemplate="%{x|%d %b %Y}<br>Value: Rp %{y:,.0f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )

        fig_curves.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                mode="lines",
                name=label + " DD",
                showlegend=False,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra>" + label + "</extra>",
            ),
            row=2,
            col=1,
        )

    # Add phase boundary vertical lines
    rebalance_dates = ["2025-05-02", "2025-08-01", "2025-11-03"]
    for r_date in rebalance_dates:
        fig_curves.add_vline(x=r_date, line_width=1, line_dash="dash", line_color="gray", opacity=0.7)

    fig_curves.update_layout(
        height=620,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_curves.update_yaxes(title_text="Portfolio Value (Rp)", row=1, col=1)
    fig_curves.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig_curves, use_container_width=True)

    # Summary Performance Table
    st.markdown("#### 📋 Performance Summary Across Strategies")
    summary_data = [
        {
            "Strategy": label,
            "Total Return (%)": f"{res['Total Return'] * 100:.2f}%",
            "Annual Return (%)": f"{res['Annual Return'] * 100:.2f}%",
            "Annual Volatility (%)": f"{res['Annual Volatility'] * 100:.2f}%",
            "Sharpe Ratio": f"{res['Sharpe Ratio']:.4f}",
            "Max Drawdown (%)": f"{res['Max Drawdown'] * 100:.2f}%",
            "Fees Paid": format_idr(res["Total Fees Paid"] * capital_idr),
            "Final Wealth": format_idr(res["Final Portfolio Value"] * capital_idr),
        }
        for label, res, _, _, _ in strategies
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Phase-by-Phase Weight Allocation Explorer
    st.markdown("#### ⚖️ Phase-by-Phase Optimal Asset Weights")
    phase_selected = st.selectbox("Select Walk-Forward Phase:", ["All Phases Side-by-Side", "Fase 1", "Fase 2", "Fase 3", "Fase 4"])

    if phase_selected == "All Phases Side-by-Side":
        fig_w = px.bar(
            w_aga_df.reset_index().melt(id_vars="ticker", var_name="Phase", value_name="Weight"),
            x="ticker",
            y="Weight",
            color="Phase",
            barmode="group",
            title="AGA Optimal Asset Weights Across All 4 WFO Phases",
            color_discrete_sequence=["#1f77b4", "#4292c6", "#6baed6", "#9ecae1"],
        )
        fig_w.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        fig_w.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.markdown(f"**Asset Allocation for {phase_selected} (AGA vs SGA)**")
            comp_weights = pd.DataFrame({
                "AGA (%)": w_aga_df[phase_selected] * 100.0,
                "SGA (%)": w_sga_df[phase_selected] * 100.0,
                "Allocation (Rp)": w_aga_df[phase_selected] * capital_idr,
            }).sort_values("AGA (%)", ascending=False)
            st.dataframe(
                comp_weights.style.format({
                    "AGA (%)": "{:.2f}%",
                    "SGA (%)": "{:.2f}%",
                    "Allocation (Rp)": "Rp {:,.0f}",
                }),
                height=380,
                use_container_width=True,
            )

        with col_w2:
            fig_pie = px.pie(
                comp_weights[comp_weights["AGA (%)"] > 0].reset_index(),
                names="ticker",
                values="AGA (%)",
                title=f"AGA Portfolio Composition ({phase_selected})",
                hole=0.4,
            )
            fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    # Statistical Significance Gallery
    with st.expander("🔬 Statistical Hypothesis Testing & Significance Results"):
        st.markdown(
            """
            Multi-run verification comparing **Standard GA (SGA)** and **Adaptive GA (AGA)** across 10 independent runs per phase:
            - **Normality Check:** Evaluated via **Shapiro-Wilk test** on paired differences.
            - **Significance Test:** Automatically routed to **Paired Student's t-test** or **Wilcoxon Signed-Rank test**.
            """
        )
        stats_df = pd.read_csv("results/tables/statistical_significance_tests.csv")
        st.dataframe(stats_df, use_container_width=True)


# ==============================================================================
# TAB 2: LIVE PORTFOLIO OPTIMIZER & SIMULATOR
# ==============================================================================
with tab2:
    st.markdown("### ⚡ Live Portfolio Optimizer & Simulation Engine")
    st.markdown("Select stocks, tweak constraints and algorithm parameters, and run real-time evolutionary optimization.")

    with st.form("optimizer_form"):
        col_param1, col_param2, col_param3 = st.columns(3)

        with col_param1:
            st.markdown("**1. Asset Universe Selection**")
            selected_assets = st.multiselect(
                "Constituent Stocks:",
                options=consistent_tickers,
                default=consistent_tickers[:12],
                help="Select stocks to include in the portfolio optimization universe.",
            )

        with col_param2:
            st.markdown("**2. Investment Constraints**")
            max_w_input = st.slider("Max Weight per Asset (%):", 10, 50, int(MAX_WEIGHT_DEFAULT * 100), 5) / 100.0
            lambda_div_input = st.slider("Diversification Penalty (λ):", 0.0, 0.20, LAMBDA_DIV_DEFAULT, 0.01)

        with col_param3:
            st.markdown("**3. Algorithm Hyperparameters**")
            pop_size_input = st.slider("Population Size:", 30, 300, 100, 10)
            generations_input = st.slider("Generations:", 10, 100, 40, 5)
            seed_input = st.number_input("Random Seed:", value=SEED, step=1)

        btn_run = st.form_submit_button("🚀 Run Live Optimization (AGA vs. SGA)", use_container_width=True)

    if btn_run:
        if len(selected_assets) < 3:
            st.error("Please select at least 3 stocks to form a diversified portfolio.")
        else:
            with st.spinner("Executing Adaptive Genetic Algorithm & Standard GA in real time..."):
                start_exec = time.time()

                # Filter dataset
                df_sub = df_selected[df_selected["ticker"].isin(selected_assets)]
                df_train_sub, df_val_sub, df_test_sub = src.split_dataset(df_sub)
                mu_sub, cov_sub, _ = src.compute_statistics(df_train_sub)

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Running Standard Genetic Algorithm (SGA)...")
                progress_bar.progress(30)
                res_live_sga = src.genetic_algorithm(
                    mu=mu_sub.values,
                    cov_matrix=cov_sub.values,
                    pop_size=pop_size_input,
                    generations=generations_input,
                    max_weight=max_w_input,
                    lambda_div=lambda_div_input,
                    seed=seed_input,
                )

                status_text.text("Running Adaptive Genetic Algorithm (AGA)...")
                progress_bar.progress(70)
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
                status_text.text(f"Optimization completed in {time.time() - start_exec:.2f} seconds!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.success("Optimization and simulation executed successfully!")

                # Live Metric Comparison
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("AGA Best Fitness", f"{res_live_aga['best_fitness']:.5f}")
                with col_m2:
                    st.metric("SGA Best Fitness", f"{res_live_sga['best_fitness']:.5f}")
                with col_m3:
                    st.metric("AGA In-Sample Sharpe", f"{res_live_aga['best_sharpe']:.4f}")
                with col_m4:
                    st.metric("SGA In-Sample Sharpe", f"{res_live_sga['best_sharpe']:.4f}")

                # Live Evolutionary Plots
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(y=res_live_sga["history_best"], mode="lines", name="Standard GA (SGA)", line=dict(color="#2ca02c", dash="dash")))
                    fig_conv.add_trace(go.Scatter(y=res_live_aga["history_best"], mode="lines", name="Adaptive GA (AGA)", line=dict(color="#1f77b4", width=2.5)))
                    fig_conv.update_layout(title="Objective Fitness Convergence", xaxis_title="Generation", yaxis_title="Fitness (Lower is Better)", height=380, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_conv, use_container_width=True)

                with col_p2:
                    fig_div = go.Figure()
                    fig_div.add_trace(go.Scatter(y=res_live_sga["history_diversity"], mode="lines", name="SGA Diversity", line=dict(color="#2ca02c", dash="dash")))
                    fig_div.add_trace(go.Scatter(y=res_live_aga["history_diversity"], mode="lines", name="AGA Diversity", line=dict(color="#1f77b4", width=2.5)))
                    fig_div.update_layout(title="Population Diversity Trajectory", xaxis_title="Generation", yaxis_title="Diversity (Mean Std)", height=380, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_div, use_container_width=True)

                # Live Dynamic Operators Trajectory
                fig_rates = go.Figure()
                fig_rates.add_trace(go.Scatter(y=res_live_aga["history_pc"], mode="lines", name="Adaptive Crossover Rate (Pc)", line=dict(color="#1f77b4", width=2)))
                fig_rates.add_trace(go.Scatter(y=res_live_aga["history_pm"], mode="lines", name="Adaptive Mutation Rate (Pm)", line=dict(color="#ff7f0e", width=2)))
                fig_rates.update_layout(title="AGA Dynamic Operator Probability Evolution Across Generations", xaxis_title="Generation", yaxis_title="Probability", height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_rates, use_container_width=True)

                # Recommended Allocation Table
                st.markdown("#### 💰 Recommended Asset Allocation")
                w_live_aga = res_live_aga["best_weights"]
                df_alloc = pd.DataFrame({
                    "Ticker": selected_assets,
                    "Weight (%)": w_live_aga * 100.0,
                    "Capital Allocation": w_live_aga * capital_idr,
                }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.dataframe(
                        df_alloc.style.format({
                            "Weight (%)": "{:.2f}%",
                            "Capital Allocation": "Rp {:,.0f}",
                        }),
                        use_container_width=True,
                    )
                with col_t2:
                    fig_alloc = px.pie(df_alloc[df_alloc["Weight (%)"] > 0.01], names="Ticker", values="Weight (%)", hole=0.35, title="Optimized Asset Weight Distribution")
                    fig_alloc.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_alloc, use_container_width=True)


# ==============================================================================
# TAB 3: ASSET EXPLORER & CORRELATION ANALYTICS
# ==============================================================================
with tab3:
    st.markdown("### 📊 IDX30 Constituent Asset Analytics")
    st.markdown("Explore return distributions, risk vs. return trade-offs, and inter-asset correlations.")

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
        st.markdown("#### 🎯 Risk vs. Return Profile")
        fig_scatter = px.scatter(
            df_assets,
            x="Annual Volatility (%)",
            y="Annual Return (%)",
            text="Ticker",
            color="Sharpe Ratio",
            color_continuous_scale="Viridis",
            size=np.maximum(df_assets["Sharpe Ratio"] + 1.0, 0.5),
            title="Asset Risk vs. Return (In-Sample Training Period)",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_a2:
        st.markdown("#### 🔗 Return Correlation Heatmap")
        corr_matrix = rets_all.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pairwise Daily Return Correlation Matrix",
        )
        fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Normalized Price Trends
    st.markdown("#### 📉 Normalized Price Historical Trajectories (Base = 100)")
    tickers_to_plot = st.multiselect(
        "Select tickers to compare:",
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
            title="Normalized Stock Price Evolution (Base 100)",
        )
        fig_norm.update_layout(height=400, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_norm, use_container_width=True)

