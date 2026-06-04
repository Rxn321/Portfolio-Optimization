#Shoutout streamlit for saving the project and yfinance for providing the data. This is a simple portfolio optimization dashboard built with Streamlit 
#that allows users to input stock tickers, select a date range, and visualize the efficient frontier along with key metrics like returns, volatility, 
#and Sharpe ratio. The app also includes error handling for data loading and validation to ensure a smooth user experience.

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
    "Tickers\t (Space separated & Syntax: ticker.exhange suffix, e.g. AAPL for US stocks, RY.TO for Canadian stocks)",
    "RY.TO TD.TO SHOP.TO BCE.TO ENB.TO CNR.TO CAE.TO"
)

# input
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

run = st.button("Run Analysis")

if run:
    # error handling for data
    try:
        tickers = [t.strip().upper() for t in tickers_input.split() if t.strip()]

        if len(tickers) == 0:
            st.error("Please enter at least one ticker.")
            st.stop()

        data = yf.download(tickers, start=start_date, end=end_date)["Close"]

        if data.empty:
            st.error("No data returned. Check tickers or date range.")
            st.stop()

        returns = data.pct_change().dropna()

        if returns.empty:
            st.error("Not enough data to compute returns.")
            st.stop()

    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    tickers = list(returns.columns)

    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    avg_returns = returns.mean()
    volatility = returns.std()
    cov_matrix = returns.cov()

# risk-free rate (example: Canadian 3-month Treasury bill - i just googled this)
    risk_free_rate = 0.0229

    avg_returns = avg_returns * 252
    cov_matrix = cov_matrix * 252

    # Overview
    st.subheader("Price Overview")
    st.line_chart(data)

    # Summary
    st.subheader("Returns Summary")
    metrics = pd.DataFrame({
        "Return": avg_returns,
        "Volatility": volatility * np.sqrt(252),
        "Sharpe": (avg_returns - risk_free_rate) / (volatility * np.sqrt(252))
    })
    st.dataframe(metrics)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        returns.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,# force full correlation range
        vmax=1,
        center=0, 
        square=True,
        ax=ax_corr
    )

    st.pyplot(fig_corr)

    # Max Sharpe Portfolio
    def neg_sharpe(w):
        r = np.dot(w, avg_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        # ADDED penalty for concentration: prevents extreme 0% / 100% weights DIVERSIFICATIONNN

        penalty = 0.01 * np.sum(w**2)

        return -((r - risk_free_rate) / v) + penalty



    # ADDED diversification constraint: forces minimum exposure per asset DIVERSIFICATIONNN

    bounds = [(0.05, 0.4)] * len(tickers)

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    init = np.array([1 / len(tickers)] * len(tickers))

    res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    opt_w = res.x

    opt_r = np.dot(opt_w, avg_returns)
    opt_v = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
    opt_sharpe = (opt_r - risk_free_rate) / opt_v

    #Validation
    valid_tickers = returns.columns.tolist()

    if len(valid_tickers) == 0:
        st.error("No valid tickers with usable data.")
        st.stop()

    # Risk
    st.subheader("Risk Insights")

    threshold = -0.02

    probs = {}

    for t in valid_tickers:
        try:
            probs[t] = (returns[t] < threshold).mean() * 100
        except Exception:
            probs[t] = np.nan

    probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    for t, prob in probs_sorted:
        if np.isnan(prob):
            st.write(f"{t}: insufficient data")
        else:
            st.write(f"{t}: {prob:.2f}% chance of -2% drop")

    # Efficient Frontier
    st.subheader("Efficient Frontier")

    try:
        sims = 10000
        results = []
        weights_list = []

        for _ in range(sims):
            w = np.random.random(len(valid_tickers))
            w = w / np.sum(w)

            r = np.dot(w, avg_returns)
            v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

            if v == 0 or np.isnan(v):
                continue

            s = (r - risk_free_rate) / v

            results.append([r, v, s])
            weights_list.append(w)

        if len(results) == 0:
            st.error("Monte Carlo failed (check tickers/data quality).")
            st.stop()

        results = np.array(results)
        weights_list = np.array(weights_list)

    except Exception as e:
        st.error(f"Efficient frontier error: {e}")
        st.stop()

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


#Best Sharpe Portfolio (Monte Carlo) vs Max Sharpe Portfolio (SciPy)
    col1, col2 = st.columns(2)
    best_mc_w = weights_list[best_idx]
    
    with col1:
        st.subheader("Best Sharpe Portfolio (Monte Carlo)")

        mc_df = pd.DataFrame({
            "Weight": [f"{w:.2%}" for w in best_mc_w]
        }, index=valid_tickers)

        st.dataframe(mc_df, use_container_width=True)

        st.write({
            "Return": results[best_idx, 0],
            "Volatility": results[best_idx, 1],
            "Sharpe": results[best_idx, 2]
        })

    with col2:
        st.subheader("Max Sharpe Portfolio (SciPy)")

        scipy_df = pd.DataFrame({
            "Weight": [f"{w:.2%}" for w in opt_w]
        }, index=tickers)

        st.dataframe(scipy_df, use_container_width=True)

        st.caption("Minimum allocation floor: 5%")

        st.write({
            "Return": opt_r,
            "Volatility": opt_v,
            "Sharpe": opt_sharpe
        })

st.caption("Developed by Ryan Liu - 2026")