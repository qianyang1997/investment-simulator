# Investment Simulator Notebook Documentation

## Overview
This lightweight repository allows you to simulate portfolio returns based on historical data and convex optimization techniques. The objective is to help construct optimal investment portfolios based on various objectives and constraints.

## Key Components

### 1. Data Retrieval
The system uses the `DataRetriever` class to fetch financial data:
- Historical price data for ETFs/stocks from Alpha Vantage API
- CPI (Consumer Price Index) & inflation data for inflation adjustments
- Data is cached locally to avoid repeated API calls

### 2. Investment Simulator
The `InvestmentSimulator` class is the main interface for portfolio optimization:
```python
sim = InvestmentSimulator(
    tickers=[...],  # List of ticker symbols
    input_date="YYYY-MM-DD",  # Start date for analysis
    output_date="YYYY-MM-DD"  # End date for analysis
)
```

### 3. Optimization Components

#### Objectives (set_objective)
Available optimization objectives include:
- `maximize_ema_return`: Maximize exponential moving average return
- `maximize_return`: Maximize nominal return
- `maximize_avg_return`: Maximize average historical return
- `minimize_maximal_drawdown`: Minimize maximum portfolio drawdown
- `minimize_classical_volatility`: Minimize portfolio variance

#### Constraints (set_constraints)
Common constraints include:
- `keep_long_positions_only`: Restrict to long-only positions
- `cap_maximal_dod_loss_at_threshold`: Limit daily losses
- `cap_maximal_drawdown_at_threshold`: Limit maximum drawdown

## Sample Usage

First, pull the project to your local environment and install the project.

Then, see notebooks/sample_notebook for sample usage. Note that you'll need a valid Alpha Vantage API key and set it to the environment variable ALPHA_VANTAGE_API_KEY.

For a comprehensive list of objectives, constraints, and metrics, browse the source code.

```python
# Initialize simulator with desired ETFs
sim = InvestmentSimulator(
    tickers=[
        "GLD",   # SPDR Gold Trust
        "DBC",   # Invesco DB Commodity Index
        "IVV",   # iShares Core S&P 500 ETF
        "QQQJ",  # Invesco NASDAQ Next Gen 100 ETF
        "FSTA",  # Fidelity MSCI Consumer Staples
        "ACWI",  # iShares MSCI ACWI ETF
        "JPST"   # JPMorgan Ultra-Short Income ETF
    ],
    input_date="2017-01-01",
    output_date="2025-01-01"
)

# Set optimization objective
so.maximize_ema_return(sim.data, sim.tickers, smoothing_window=100)

# Add constraints
sc.keep_long_positions_only()
# 10% max daily loss
sc.cap_maximal_dod_loss_at_threshold(sim.data, sim.tickers, 0.1)
# 20% max drawdown
sc.cap_maximal_drawdown_at_threshold(sim.data, sim.tickers, 0.2)

# Run optimization
sim.optimize()

# Generate report
sim.generate_report()

# Clear optimizer
sim.clear()

# View report
sim.report
```

## Output Report Structure
The optimization results include:
- Status: Optimization status (e.g., "optimal")
- Tickers: List of assets considered
- Value: Objective function value
- Allocation: Optimal portfolio weights
- Metrics: Various performance metrics
- Constraints: Applied constraints and their status

## Best Practices
1. Always verify the data range is sufficient for analysis
2. Consider transaction costs and rebalancing frequency
3. Test different constraint combinations
4. Monitor optimization status for solution quality
5. Review all metrics before implementing portfolio changes
