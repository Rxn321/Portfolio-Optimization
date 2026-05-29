import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  #need for matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import json
import io
import base64
#based on juptyer notebook, check the notebook for more details and graphs on sections

#convert plot to for react(idk)
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img

#data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()
    avg_returns = returns.mean()
    volatility = returns.std()
    return data, returns, avg_returns, volatility

#correlation
def compute_correlation(returns):
    return returns.corr()

#sharpe
def compute_sharpe(avg_returns, volatility, returns, risk_free_rate=0.035/252):
    sharpe_ratios = (avg_returns - risk_free_rate) / volatility
    weights = np.array([1/len(avg_returns)] * len(avg_returns))
    portfolio_return = np.dot(weights, avg_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratios, portfolio_return, portfolio_volatility, portfolio_sharpe, weights

#probability
def compute_distribution(returns, threshold=-0.02):
    tickers = returns.columns.tolist()
    results = {}

    for stock in tickers:
        stock_returns = returns[stock].dropna()
        mean = stock_returns.mean()
        std = stock_returns.std()

        if std == 0 or np.isnan(std):
            continue

        prob_below = (stock_returns < threshold).mean() * 100

        results[stock] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "prob_below_threshold": round(prob_below, 2),
            "threshold": threshold
        }

    return results

#VaR
def compute_var(returns, avg_returns, volatility, weights, portfolio_return, portfolio_volatility, confidence=0.95):
    alpha = 1 - confidence
    z_score = 1.645
    tickers = returns.columns.tolist()

    historical_var = returns.quantile(alpha)
    parametric_var = avg_returns - z_score * volatility

    cvar = {}
    for stock in tickers:
        below = returns[stock][returns[stock] <= historical_var[stock]]
        cvar[stock] = below.mean()
    cvar = pd.Series(cvar)

    portfolio_returns = returns.dot(weights)
    portfolio_hist_var = portfolio_returns.quantile(alpha)
    portfolio_param_var = portfolio_return - z_score * portfolio_volatility
    portfolio_cvar = portfolio_returns[portfolio_returns <= portfolio_hist_var].mean()

    var_table = pd.DataFrame({
        "Historical VaR (95%)": historical_var,
        "Parametric VaR (95%)": parametric_var,
        "Expected Shortfall": cvar
    }).round(4)
    var_table.loc["Portfolio"] = [portfolio_hist_var, portfolio_param_var, portfolio_cvar]

    return var_table

#monte carlo
def run_monte_carlo(returns, avg_returns, risk_free_rate=0.035/252, num_simulations=10000):
    tickers = returns.columns.tolist()
    mc_returns, mc_volatility, mc_sharpe, mc_weights = [], [], [], []

    for _ in range(num_simulations):
        w = np.random.random(len(tickers))
        w = w / np.sum(w)
        p_return = np.dot(w, avg_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))
        p_sharpe = (p_return - risk_free_rate) / p_vol
        mc_returns.append(p_return)
        mc_volatility.append(p_vol)
        mc_sharpe.append(p_sharpe)
        mc_weights.append(w)

    mc_returns = np.array(mc_returns)
    mc_volatility = np.array(mc_volatility)
    mc_sharpe = np.array(mc_sharpe)
    mc_weights = np.array(mc_weights)

    best_idx = np.argmax(mc_sharpe)
    min_vol_idx = np.argmin(mc_volatility)

    return mc_returns, mc_volatility, mc_sharpe, mc_weights, best_idx, min_vol_idx

#main api
def main(tickers, start_date, end_date):
    data, returns, avg_returns, volatility = fetch_data(tickers, start_date, end_date)
    corr = compute_correlation(returns)
    sharpe_ratios, portfolio_return, portfolio_volatility, portfolio_sharpe, weights = compute_sharpe(avg_returns, volatility, returns)
    var_table = compute_var(returns, avg_returns, volatility, weights, portfolio_return, portfolio_volatility)
    mc_returns, mc_volatility, mc_sharpe, mc_weights, best_idx, min_vol_idx = run_monte_carlo(returns, avg_returns)
    optimal_weights = {stock: f"{w*100:.1f}%" for stock, w in zip(tickers, mc_weights[best_idx])}
    distribution = compute_distribution(returns)

    return {
        "avg_returns": avg_returns.to_dict(),
        "volatility": volatility.to_dict(),
        "sharpe_ratios": sharpe_ratios.to_dict(),
        "portfolio_sharpe": portfolio_sharpe,
        "distribution": distribution,
        "var_table": var_table.to_dict(),
        "optimal_weights": optimal_weights,
    }

#──Test──
result = main(
    tickers=["AAPL", "TSLA", "MSFT", "NVDA"],
    start_date="2023-01-01",
    end_date="2025-05-11"
)

print("Average Returns:")
print(result["avg_returns"])

print("\nSharpe Ratios:")
print(result["sharpe_ratios"])

print("\nPortfolio Sharpe:")
print(result["portfolio_sharpe"])

print("\nOptimal Weights:")
print(result["optimal_weights"])

print("\nProbability of dropping below -2% in a day:")
for stock, data in result["distribution"].items():
    print(f"  {stock}: {data['prob_below_threshold']:.2f}%")

print("\nVaR Table:")
print(result["var_table"])
