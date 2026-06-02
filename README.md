## 📊 Portfolio Optimization Project

A finance/data analysis project that builds a full portfolio optimization and risk analysis pipeline using Python, statistical modelling, and Monte Carlo simulation.

Developed as a 2026 summer project. Note: 

## Features
-Historical stock data ingestion (Alpha Vantage / yFinance notebook version)
-Return, volatility, and correlation analysis
-Portfolio optimization (Monte Carlo + SciPy)
-Risk modeling (VaR + Expected Shortfall)
-Sharpe ratio evaluation (asset + portfolio level)
-Efficient Frontier visualization

## 📊 How to Use
1. Install dependencies
-pip install numpy pandas matplotlib seaborn requests scipy
2. Run the project
from your_file_name import main
result = main(
    tickers=["AAPL", "TSLA", "MSFT", "NVDA"],
    start_date="2023-01-01",
    end_date="2025-05-11"
3. Or just click this link (check limitations) >>  

## Outputs 
(Very simple explanations for the outputs, check the Jupyter notebook for more details)

-Heat map: Values closer to 1 show stocks moving together, Values closer to -1 show stocks moving opposite
-Sharpe Ratio: The higher the better. Compares an investment's return with its risk. S&P 500 (long run)	~0.5 - 1.0
-Portfolio Sharpe Ratio: The portfolio volatility uses the covariance matrix (how stocks move together), thus a diversified portfolio can have lower risk
-VaR: On the worst (5%) days, the loss exceeds this
-Parametric VaR: VaR estimates using normal distribution
-Expected Shortfall: In the worst (5%) days, this is the avg loss
-Monte Carlo vs Scipy optimization: Monte Carlo assigns weights randomly # of times and finds the max Sharpe (vs) Scipy calculates equal weights, finds the Sharpe, then moves the weight towards the direction with the higher Sharpe 

## ⚠️ Known Limitations
(This project was built as a learning-focused implementation using free-tier infrastructure and APIs, so certain constraints exist that reflect real-world deployment trade-offs.)
-Alpha Vantage Free Tier Restrictions
If the project is run in the site and not run in the Jupyter Notebook environment, the free Alpha Vantage API only provides access to approximately the most recent 100 trading days of data. Historical data beyond this range requires a premium plan.
-Backend Cold Start Delay (Render Free tier)
When deployed on Render, the backend service may take approximately 20–30 seconds to spin up after periods of inactivity.



## Backend Challenges & Solutions
-Yahoo Finance Rate Limiting
Cloud servers (Render.com) are frequently rate-limited by Yahoo Finance's API, 
returning YFRateLimitError on hosted environments. Resolved by migrating the 
data ingestion layer from yfinance to Alpha Vantage's REST API, which provides 
reliable cloud access with proper API key authentication.

-Alpha Vantage Premium Endpoints
Initial implementation used `TIME_SERIES_DAILY_ADJUSTED` and `outputsize=full`
which are premium features. Resolved by switching to last 100 trading days, which is available on the free tier.

-Alpha Vantage Rate Limiting
Free tier limits requests to 1 per second and 25 per day. Resolved by adding
a 1 second delay between each ticker request.

-Python Version Compatibility
Render defaulted to Python 3.14 which had no prebuilt wheels for scipy and numpy. 
Resolved by pinning Python 3.12.7 via environment variables and aligning all 
dependency versions to a compatible set.

-Numpy/Scipy Version Conflicts
scipy==1.15.0 required numpy<1.28.0 which conflicted with other dependencies. 
Resolved by downgrading to scipy==1.13.1 with numpy==1.26.4.
