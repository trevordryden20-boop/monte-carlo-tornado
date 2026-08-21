import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Institutional Monte Carlo Engine", layout="wide")

st.title("Advanced Multi-Asset Monte Carlo Simulation Engine")
st.markdown("""
This model simulates joint asset trajectories using a **Geometric Brownian Motion (GBM)** framework 
driven by a **Multivariate Normal Distribution** via **Cholesky Decomposition** to preserve empirical inter-asset correlations.
""")

# Sidebar Input Parameters
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
time_horizon = st.sidebar.number_input("Time Horizon (Trading Days)", min_value=10, max_value=1260, value=252, step=10)
initial_investment = st.sidebar.number_input("Initial Portfolio Value ($)", min_value=1000, max_value=10000000, value=100000, step=5000)

st.sidebar.header("Asset Allocation & Parameters")
weights = np.array([0.60, 0.40])  # Equity / Bond split
expected_returns = np.array([0.09, 0.04])  # Annualized expected returns
volatilities = np.array([0.18, 0.06])       # Annualized volatilities
correlation_matrix = np.array([
    [1.00, -0.15],
    [-0.15, 1.00]
])

# Mathematical Engine Setup
dt = 1 / 252  # Time step (daily scale)
num_assets = len(weights)

# Calculate annual covariance matrix and transform to daily scale
cov_matrix_annual = np.outer(volatilities, volatilities) * correlation_matrix
cov_matrix_daily = cov_matrix_annual * dt

# Cholesky Decomposition for correlated random variables: L * L^T = Covariance
L = np.linalg.cholesky(cov_matrix_daily)

# Calculate daily drift vector (GBM Ito's Lemma correction factor: mu - 0.5 * sigma^2)
daily_drift = (expected_returns - 0.5 * (volatilities ** 2)) * dt

# Execution Button
if st.button("Run Institutional Simulation"):
    np.random.seed(42)  # For reproducible simulation paths

    # Generate uncorrelated standard normal shocks: shape = (days, assets, simulations)
    uncorrelated_shocks = np.random.normal(0, 1, size=(time_horizon, num_assets, num_simulations))

    # Correlate shocks across asset dimensions using Cholesky lower triangular matrix
    correlated_shocks = np.einsum('ij, tjs -> tis', L, uncorrelated_shocks)

    # Compute daily log returns for each asset path
    daily_log_returns = daily_drift[None, :, None] + correlated_shocks

    # Compound returns over the time horizon
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
    asset_trajectories = np.exp(cumulative_log_returns)

    # Aggregate individual asset trajectories weighted by portfolio allocation
    portfolio_growth_factors = np.tensordot(asset_trajectories, weights, axes=([1], [0]))

    # Add Day 0 starting value
    day_zero = np.ones((1, num_simulations))
    portfolio_trajectories = np.vstack([day_zero, portfolio_growth_factors]) * initial_investment

    # Extract Terminal Outcomes
    final_values = portfolio_trajectories[-1]

    # Calculate Statistical Risk Metrics
    mean_final = np.mean(final_values)
    median_final = np.median(final_values)
    var_95 = initial_investment - np.percentile(final_values, 5)  # Value at Risk (95%)
    cvar_95 = initial_investment - np.mean(final_values[final_values <= np.percentile(final_values, 5)]) # Conditional VaR

    # Metric Dashboard
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Mean Final Value", f"${mean_final:,.2f}")
    col2.metric("Median Final Value", f"${median_final:,.2f}")
    col3.metric("95% Value at Risk (VaR)", f"${var_95:,.2f}")
    col4.metric("95% Conditional VaR (CVaR)", f"${cvar_95:,.2f}")

    st.subheader("Simulated Portfolio Trajectories")
    
    # Render Streamlit-native Line Chart (displaying 100 sample paths for performance)
    display_paths = portfolio_trajectories[:, :100]
    chart_data = pd.DataFrame(display_paths, columns=[f"Path {i+1}" for i in range(100)])
    chart_data.index.name = "Trading Day"
    
    st.line_chart(chart_data)

    st.subheader("Distribution of Terminal Portfolio Outcomes")
    
    # Render Streamlit-native Histogram via DataFrame distribution
    hist_counts, bin_edges = np.histogram(final_values, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dist_df = pd.DataFrame({"Final Value ($)": bin_centers, "Frequency": hist_counts}).set_index("Final Value ($)")
    
    st.bar_chart(dist_df)
