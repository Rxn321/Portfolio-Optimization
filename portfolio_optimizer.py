import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Dashboard")

# example
tickers_input = st.text_input(
    "Tickers (space separated)",
    "RY.TO SU.TO MSFT.TO NKE.TO VFV.TO SHOP.TO WFG.TO"
)

# input
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

run = st.button("Run Analysis")

if run:

    # data
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
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Max Sharpe Portfolio
    def neg_sharpe(w):
        r = np.dot(w, avg_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(r - risk_free_rate) / v

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * len(tickers)
    init = np.array([1 / len(tickers)] * len(tickers))

    res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    opt_w = res.x

    opt_r = np.dot(opt_w, avg_returns)
    opt_v = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
    opt_sharpe = (opt_r - risk_free_rate) / opt_v

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

    best_idx = np.argmax(results[:, 2])
    min_idx = np.argmin(results[:, 1])

    eq_w = np.array([1 / len(tickers)] * len(tickers))
    eq_r = np.dot(eq_w, avg_returns)
    eq_v = np.sqrt(np.dot(eq_w.T, np.dot(cov_matrix, eq_w)))

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        results[:, 1],
        results[:, 0],
        c=results[:, 2],
        cmap="viridis",
        s=10,
        alpha=0.6
    )

    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ax.scatter(
        results[best_idx, 1],
        results[best_idx, 0],
        color="red",
        s=250,
        marker="*",
        label="Best Sharpe (MC)"
    )

    ax.scatter(
        results[min_idx, 1],
        results[min_idx, 0],
        color="blue",
        s=200,
        marker="o",
        label="Min Volatility"
    )

    ax.scatter(
        eq_v,
        eq_r,
        color="white",
        edgecolors="black",
        s=200,
        marker="o",
        label="Equal Weight"
    )

    ax.scatter(
        opt_v,
        opt_r,
        color="gold",
        s=300,
        marker="*",
        label="Max Sharpe (SciPy)"
    )

    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Max Sharpe Portfolio (SciPy)
    st.subheader("Max Sharpe Portfolio (SciPy)")

    for t, w in zip(tickers, opt_w):
        st.write(f"{t}: {w:.2%}")

    st.write({
        "Return": opt_r,
        "Volatility": opt_v,
        "Sharpe": opt_sharpe
    })

    # Example
    st.subheader("Risk Insights")

    threshold = -0.02

    for t in tickers:
        prob = (returns[t] < threshold).mean() * 100
        st.write(f"{t}: {prob:.2f}% chance of -2% drop")