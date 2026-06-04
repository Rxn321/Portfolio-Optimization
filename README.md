## 📊 Portfolio Optimization Project

### A finance/data analysis project that builds a full portfolio optimization and risk analysis pipeline using Python, statistical modelling, and Monte Carlo simulation.
### Developed as a 2026 summer project.
---
## Features
- Historical stock data ingestion (Alpha Vantage / yFinance notebook version)
- Return, volatility, and correlation analysis
- Portfolio optimization (Monte Carlo + SciPy)
- Risk modeling (VaR + Expected Shortfall)
- Sharpe ratio evaluation (asset + portfolio level)
- Efficient Frontier visualization
---
## How to Use
1. Just click this: [Live Portfolio App](https://portfolio-optimization-ryanliu.streamlit.app/)
---
## Outputs 
(Very simple explanations for the outputs, check the Jupyter notebook for more details)

- Heat map: Values closer to 1 show stocks moving together, Values closer to -1 show stocks moving opposite
- Sharpe Ratio: The higher the better. Compares an investment's return with its risk. S&P 500 (long run)	~0.5 - 1.0
- Portfolio Sharpe Ratio: The portfolio volatility uses the covariance matrix (how stocks move together), thus a diversified portfolio can have lower risk
- VaR: On the worst (5%) days, the loss exceeds this
- Parametric VaR: VaR estimates using normal distribution
- Expected Shortfall: In the worst (5%) days, this is the avg loss
- Monte Carlo vs Scipy optimization: Monte Carlo assigns weights randomly 10k times and finds the max Sharpe (vs) Scipy calculates equal weights, finds the Sharpe, then moves the weight towards the direction with the higher Sharpe 
---

~~## ⚠️ Known Limitations~~
~~(This project was built as a learning-focused implementation using free-tier infrastructure and APIs, so many constraints exist.)~~
~~- Alpha Vantage Free Tier Restrictions~~
~~Only allows 25 requests/month~~
~~- Alpha Vantage Free Tier Restrictions~~
~~If the project is run on the site and not run in the Jupyter Notebook environment, the free Alpha Vantage API only provides access to approximately the ~~most recent 100 trading days of data. Historical data beyond this range requires a premium plan.~~
~~- Backend Cold Start Delay (Render Free tier)~~
~~When deployed on Render, the backend service may take approximately 20–30 seconds to spin up after periods of inactivity.~~
Streamlit came in and saved the day