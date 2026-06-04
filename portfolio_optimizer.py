import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Dashboard")

# Input
tickers_input = st.text_input("Tickers (space separated)", "RY.TO SU.TO MSFT.TO NKE.TO VFV.TO SHOP.TO WFG.TO")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

run = st.button("Run Analysis")

if run:

    # Data
    tickers = tickers_input.split()

    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    avg_returns = returns.mean()
    volatility = returns.std()
    cov_matrix = returns.cov()

    risk_free_rate = 0.0225 / 252

    # Overview
    st.subheader("Price Overview")
    st.line_chart(data)

    # Summary
    st.subheader("Returns Summary")
    metrics = pd.DataFrame({
        "Return": avg_returns,
        "Volatility": volatility,
        "Sharpe": (avg_returns - risk_free_rate) / volatility
    })
    st.dataframe(metrics)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    # Efficient Frontier
    st.subheader("Efficient Frontier")

    sims = 10000
    results = []
    weights_list = []

    for _ in range(sims):
        w = np.random.random(len(tickers))
        w = w / np.sum(w)

        r = np.dot(w, avg_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        s = (r - risk_free_rate) / v

        results.append([r, v, s])
        weights_list.append(w)

    results = np.array(results)
    weights_list = np.array(weights_list)

    # Best Sharpe
    best_idx = np.argmax(results[:, 2])
    best_w = weights_list[best_idx]

    # Min Volatility
    min_idx = np.argmin(results[:, 1])
    min_w = weights_list[min_idx]

    # Equal weight
    eq_w = np.array([1 / len(tickers)] * len(tickers))

    eq_r = np.dot(eq_w, avg_returns)
    eq_v = np.sqrt(np.dot(eq_w.T, np.dot(cov_matrix, eq_w)))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scipy Optimized Portfolio
    def neg_sharpe(w):
        r = np.dot(w, avg_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(r - risk_free_rate) / v

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * len(tickers)
    init = np.array([1 / len(tickers)] * len(tickers))

    res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    opt_w = res.x

    st.subheader("Optimal Portfolio (Max Sharpe)")
    opt_w = res.x
    opt_r = np.dot(opt_w, avg_returns)
    opt_v = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
    opt_s = (opt_r - risk_free_rate) / opt_v
    

    # All portfolios (efficient frontier cloud)
    scatter = ax.scatter(
        results[:, 1],
        results[:, 0],
        c=results[:, 2],
        cmap="viridis",
        s=10,
        alpha=0.6
    )

    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    # Best Sharpe (Monte Carlo)
    ax.scatter(
        results[best_idx, 1],
        results[best_idx, 0],
        color="red",
        s=250,
        marker="*",
        label="Best Sharpe"
    )

    # Min Volatility
    ax.scatter(
        results[min_idx, 1],
        results[min_idx, 0],
        color="blue",
        s=200,
        marker="o",
        label="Min Volatility"
    )

    # Equal weight
    ax.scatter(
        eq_v,
        eq_r,
        color="white",
        edgecolors="black",
        s=200,
        marker="o",
        label="Equal Weight"
    )
    # Best Sharpe (Optimized)
    ax.scatter(
    opt_v,
    opt_r,
    color="gold",
    s=300,
    marker="*",
    label="Max Sharpe (Optimized)"
    )


    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier (Monte Carlo Simulation)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


    # Risk
    st.subheader("Risk Insights (VaR + Probability)")

    threshold = -0.02

    for t in tickers:
        prob = (returns[t] < threshold).mean() * 100
        st.write(f"{t}: {prob:.2f}% chance of -2% daily drop")