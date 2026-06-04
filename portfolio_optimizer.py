import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


# USER INPUT

tickers_input = input("Enter tickers separated by space (e.g. AAPL MSFT RY.TO): ")
start_date = input("Start date (YYYY-MM-DD): ")
end_date = input("End date (YYYY-MM-DD): ")

tickers = tickers_input.split()


# DATA

print("\nDownloading data...")
data = yf.download(tickers, start=start_date, end=end_date)["Close"]


# RETURNS

returns = data.pct_change().dropna()

avg_returns = returns.mean()
volatility = returns.std()
cov_matrix = returns.cov()

risk_free_rate = 0.0225 / 252

print("\n=== AVERAGE RETURNS ===")
print(avg_returns)

print("\n=== VOLATILITY ===")
print(volatility)


# SHARPE

sharpe_ratios = (avg_returns - risk_free_rate) / volatility

print("\n=== SHARPE RATIOS ===")
print(sharpe_ratios.round(3))


# EQUAL WEIGHT PORTFOLIO

weights = np.array([1 / len(tickers)] * len(tickers))

portfolio_return = np.dot(weights, avg_returns)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

print("\n=== EQUAL WEIGHT PORTFOLIO ===")
print(f"Return: {portfolio_return:.4f}")
print(f"Volatility: {portfolio_volatility:.4f}")
print(f"Sharpe: {portfolio_sharpe:.3f}")


# MONTE CARLO

num_simulations = 5000

mc_returns = []
mc_vols = []
mc_sharpes = []
mc_weights = []

for _ in range(num_simulations):
    w = np.random.random(len(tickers))
    w = w / np.sum(w)

    p_ret = np.dot(w, avg_returns)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    p_sharpe = (p_ret - risk_free_rate) / p_vol

    mc_returns.append(p_ret)
    mc_vols.append(p_vol)
    mc_sharpes.append(p_sharpe)
    mc_weights.append(w)

mc_returns = np.array(mc_returns)
mc_vols = np.array(mc_vols)
mc_sharpes = np.array(mc_sharpes)
mc_weights = np.array(mc_weights)


# BEST PORTFOLIOS

best_idx = np.argmax(mc_sharpes)
min_vol_idx = np.argmin(mc_vols)

print("\n=== BEST SHARPE PORTFOLIO ===")
for t, w in zip(tickers, mc_weights[best_idx]):
    print(f"{t}: {w*100:.1f}%")
print("Sharpe:", mc_sharpes[best_idx])

print("\n=== MIN VOLATILITY PORTFOLIO ===")
for t, w in zip(tickers, mc_weights[min_vol_idx]):
    print(f"{t}: {w*100:.1f}%")
print("Volatility:", mc_vols[min_vol_idx])


# OPTIMIZATION

def neg_sharpe(w):
    p_ret = np.dot(w, avg_returns)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    return -(p_ret - risk_free_rate) / p_vol

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in tickers]
init = np.array([1/len(tickers)] * len(tickers))

result = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
opt_w = result.x

print("\n=== OPTIMIZED PORTFOLIO (SCI-PY) ===")
for t, w in zip(tickers, opt_w):
    print(f"{t}: {w*100:.1f}%")

opt_ret = np.dot(opt_w, avg_returns)
opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

print(f"Return: {opt_ret:.4f}")
print(f"Volatility: {opt_vol:.4f}")
print(f"Sharpe: {opt_sharpe:.3f}")


# CORRELATION HEATMAP

plt.figure(figsize=(8,6))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# EFFICIENT FRONTIER

plt.figure(figsize=(10,6))
plt.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap="viridis", s=10)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.show()