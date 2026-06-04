import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Dashboard")

# Example
tickers_input = st.text_input("Tickers (space separated)", "RY.TO SU.TO MSFT.TO NKE.TO VFV.TO SHOP.TO WFG.TO")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

run = st.button("Run Analysis")

if run:

    # Example
    tickers = tickers_input.split()

    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    avg_returns = returns.mean()
    volatility = returns.std()
    cov_matrix = returns.cov()

    risk_free_rate = 0.0225 / 252

    # Example
    st.subheader("Price Overview")
    st.line_chart(data)

    # Example
    st.subheader("Returns Summary")
    metrics = pd.DataFrame({
        "Return": avg_returns,
        "Volatility": volatility,
        "Sharpe": (avg_returns - risk_free_rate) / volatility
    })
    st.dataframe(metrics)

    # Example
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Example
    weights = np.array([1 / len(tickers)] * len(tickers))

    port_return = np.dot(weights, avg_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    port_sharpe = (port_return - risk_free_rate) / port_vol

    st.subheader("Equal Weight Portfolio")
    st.write({
        "Return": port_return,
        "Volatility": port_vol,
        "Sharpe": port_sharpe
    })

    # Example
    st.subheader("Monte Carlo Simulation")

    sims = 5000
    results = []

    for _ in range(sims):
        w = np.random.random(len(tickers))
        w = w / np.sum(w)

        r = np.dot(w, avg_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        s = (r - risk_free_rate) / v

        results.append([r, v, s])

    results = np.array(results)

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap="viridis", s=10)
    plt.colorbar(scatter)
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Return")
    st.pyplot(fig2)

    # Example
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

    opt_r = np.dot(opt_w, avg_returns)
    opt_v = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
    opt_s = (opt_r - risk_free_rate) / opt_v

    for t, w in zip(tickers, opt_w):
        st.write(f"{t}: {w:.2%}")

    st.write({
        "Return": opt_r,
        "Volatility": opt_v,
        "Sharpe": opt_s
    })

    # Example
    st.subheader("Risk Insights (VaR + Probability)")

    threshold = -0.02

    for t in tickers:
        prob = (returns[t] < threshold).mean() * 100
        st.write(f"{t}: {prob:.2f}% chance of -2% daily drop")