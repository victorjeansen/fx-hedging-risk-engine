
# Corporate FX Hedging & Monte Carlo Risk Engine

Stochastic EUR/USD risk engine simulating 10,000 Geometric Brownian Motion trajectories to evaluate corporate hedging strategies (Unhedged Spot, FX Forward, and Zero-Cost Option Collar) against an operating profit (EBIT) bank covenant.

## Key Results ($100M USD Receivables vs. €75M EUR Costs)
- **Unhedged (Spot):** €16.89M expected EBIT | €4.61M 95% VaR | **27.4% probability** of breaching the €12.0M covenant.
- **FX Forward ($F_0 = 1.0963$):** €16.21M locked EBIT | €0.00M volatility | -€1.38M interest-rate carry cost.
- **Zero-Cost Collar (1.0745 – 1.1200):** €16.25M expected EBIT | **€14.29M hard EBIT floor** | €0.00M upfront option premium.

## Methodology
- **Monte Carlo Simulation:** 10,000 GBM paths calibrated on EUR/USD spot (1.0800), implied volatility (8.50%), and interest rate differential ($r_{USD}=4.0\%$, $r_{EUR}=2.5\%$).
- **Option Pricing:** Analytical Garman-Kohlhagen valuation paired with Brent's root-finding algorithm to calibrate the zero-cost strike floor.
- **3D Risk Surfaces:** Terminal EBIT probability density waterfall and 6-month pre-maturity Mark-to-Market / Vega stress surface.

## Repository Contents
- `fx_hedging_engine.py`: Core simulation, pricing, calibration, and 3D plotting script (`NumPy`, `SciPy`, `Matplotlib`).
- `FX_Hedging_Monte_Carlo_Note_FINAL.pdf`: Full quantitative treasury research note.
- `fx_hedging_distribution.jpg` & `fx_stress_payoff.jpg`: High-resolution 3D risk and sensitivity plots.
