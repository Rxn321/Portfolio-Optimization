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