# Portfolio Optimization Project

Mah 2026 summer project

Very simple explainations for the outputs, check jupyter notebook for more details

Heat map: Values closer to 1 show stocks moving together, Values closer to -1 show stocks moving opposite
Sharpe Ratio: Higher the better. Compares an investment's return with its risk. S&P 500 (long run)	~0.5 - 1.0
Portfolio Sharpe Ratio: The portfolio volatility uses the covariance matrix (how stocks moves tgr), thus diversified portfolio can have lower risk
VaR: On the worst (5%) days, the loss exceed this
Parametric VaR: VaR estimates using normal distribution
Expected Shortfall: In the worst (5%) days, this is the avg loss
Monte carlo vs Scipy optimization: monte carlo assign weights randomly # of times and finds the max sharpe (vs) Scipy calculates equal weights, finds sharpe then moves the weight towards the direction with the higher sharpe 

## Challenges & Solutions

-Yahoo Finance Rate Limiting
Cloud servers (Render.com) are frequently rate limited by Yahoo Finance's API, 
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
