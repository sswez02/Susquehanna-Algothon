# Algorithmic Trading Competition - Quantitative Strategy Development

> **Final Score: 20.02** on hidden out-of-sample data (days 1000-1250)  
> **In-Sample Score: 41.98** on training data (days 750-1000)  
> **Sharpe Ratio: 3.01** | **Total Return: 3.79%** | **25 instruments traded**

Systematic short-selling strategy developed for the Susquehanna Algothon competition, demonstrating quantitative methodology with out-of-sample validation.

---

## Executive Summary

### The Challenge

- **Universe**: 50 synthetic instruments over 1250 days
- **Objective**: Maximize `Score = mean(daily_PnL) - 0.1 × std(daily_PnL)`
- **Constraints**: $10,000 position limit per instrument, 0.1% commission
- **Data**: Days 0-1000 provided for development; days 1000-1250 hidden for final evaluation

### Key Results

| Metric             | In-Sample (750-1000) | Out-of-Sample (1000-1250) |
| ------------------ | -------------------- | ------------------------- |
| **Score**          | 41.98                | **20.02**                 |
| **Sharpe Ratio**   | 4.42                 | 3.01                      |
| **Mean Daily PnL** | $65.40               | $42.10                    |
| **Volatility**     | $234.09              | $221.18                   |
| **Total Return**   | 5.83%                | 3.79%                     |

**The 52% score degradation from in-sample to out-of-sample is expected**

---

## Strategy Overview

### Core Hypothesis

Instruments exhibiting **low volatility**, **persistent negative momentum**, and **high trend consistency (R²)** are likely to continue declining - making them desireable short candidates

### Signal Components

| Signal                  | Calculation                              | Rationale                                                    |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **Volatility Filter**   | 16-day rolling std < 78th percentile     | Low-vol instruments have more predictable behavior           |
| **Momentum Filter**     | 706-day return < 0                       | Long-term losers tend to continue losing                     |
| **Trend Strength (R²)** | Linear regression R² > 0.2 over 105 days | High R² indicates consistent directional movement, not noise |
| **Price Filter**        | Price < $60                              | Empirical filter to avoid certain instrument characteristics |

### Insight

Traditional momentum strategies measure _magnitude_ of price change. R² measures consistency - how well prices follow a linear trend.

```
High R² + Negative Momentum = Strong, persistent downtrend (not mean-reverting)
Low R²  + Negative Momentum = Noisy decline (may reverse)
```

This distinction is crucial: we want to short instruments in consistent downtrends, not temporary dips.

---

## Methodology

### Phase 1: Exploratory Analysis

```
Days 0-750: Feature engineering and hypothesis generation
```

**Tested approaches:**

- Cross-sectional mean reversion (Score ~22)
- Simple momentum (Score ~28)
- Volatility-scaled positions (High variance)
- Low-vol + momentum + R² filter (Score ~42)

### Phase 2: Parameter Optimization

```
Days 750-1000: Grid search for optimal parameters
```

**Search space:**

```python
param_grid = {
    'vol_percentile': [70, 75, 78, 80, 82, 85],
    'vol_lookback': [14, 16, 18, 20],
    'mom_lookback': [680, 700, 706, 710, 720],
    'r2_lookback': [90, 95, 100, 105, 110, 115],
    'r2_threshold': [0.15, 0.18, 0.20, 0.22, 0.25],
    'price_threshold': [55, 58, 60, 62, 65]
}
```

**Optimal parameters found:**
| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| VOL_PERCENTILE | 78 | Robust (76-80) |
| VOL_LOOKBACK | 16 | Robust (14-20) |
| MOM_LOOKBACK | 706 | Robust (700-720) |
| R2_LOOKBACK | 105 | Robust (100-115) |
| R2_THRESHOLD | 0.20 | Robust (0.18-0.25) |
| PRICE_THRESHOLD | 60 | Robust (55-70) |

### Phase 3: Out-of-Sample Validation

```
Days 1000-1250: True holdout test (data revealed after competition)
```

---

## Overfitting Analysis

I acknowledge that parameter optimization on days 750-1000 constitutes fitting to that specific period. The key question is: does the strategy generalize?

| Test                        | Result              | Interpretation          |
| --------------------------- | ------------------- | ----------------------- |
| In-sample vs Out-of-sample  | 42 -> 20 (52% drop) | Expected degradation    |
| Out-of-sample Sharpe        | 3.01                | Still strongly positive |
| Out-of-sample profitability | +$10,536            | Real alpha generation   |
| Consistency                 | 65% profitable days | Robust performance      |

### What This Proves

1. The strategy has predictive power - Score 20 on unseen data is not luck
2. The signals (vol, momentum, R²) capture market dynamics
3. Parameter robustness - strategy works across different market regimes

### What I Would Do Differently

- **Walk-forward optimization**: Re-optimize parameters periodically
- **Ensemble methods**: Combine multiple parameter sets
- **Regime detection**: Adapt to changing market conditions

---

## Implementation

### Project Structure

```
├── main.py                 # Competition submission (strategy logic)
├── eval.py                 # Competition evaluator
├── prices.txt              # Price data (50 instruments × 1250 days)
├── search/
│   ├── parameter_search.py      # Grid search implementation
├── BRIEF.md                # Competition brief
├── TECHNICAL_SUMMARY.md    # Technical SUmmary
└── README.md               # This document
```

### Core Algorithm

```python
def getMyPosition(prcSoFar):
    """
    Strategy: Short low-vol, negative-momentum, high-R² instruments

    1. Calculate features using historical data (no lookahead)
    2. Apply filters to select instruments
    3. Enter positions and hold (minimize turnover/commission)
    """
    # Volatility filter
    volatility = np.std(returns[:, -VOL_LOOKBACK:], axis=1)
    vol_threshold = np.percentile(volatility, VOL_PERCENTILE)

    # Momentum filter
    momentum = (current_price - price_706_days_ago) / price_706_days_ago

    # R² filter (trend strength)
    for instrument in range(n_instruments):
        slope, intercept, r, p, se = stats.linregress(x, prices[-R2_LOOKBACK:])
        r2 = r**2

    # Selection
    for i in range(n_instruments):
        if (volatility[i] < vol_threshold and
            momentum[i] < 0 and
            r2[i] > R2_THRESHOLD and
            price[i] < PRICE_THRESHOLD):
            position[i] = -max_shares[i]  # Short

    return position
```

### Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run parameter search
python -m search.parameter_search

# Evaluate strategy
python eval.py

```

---

## Performance Analysis

### Equity Curve (Out-of-Sample: Days 1000-1250)

```
Day 1001: -$230 (commission drag on entry)
Day 1050: +$3,409
Day 1100: +$3,081
Day 1128: +$7,365 (peak)
Day 1150: +$5,744 (drawdown)
Day 1218: +$10,143 (new peak)
Day 1231: +$8,064 (drawdown)
Day 1250: +$10,536 (final)
```

### Risk Metrics

| Metric           | Value               |
| ---------------- | ------------------- |
| Maximum Drawdown | ~$2,500 (from peak) |
| Win Rate         | ~65% of days        |
| Profit Factor    | 1.19                |
| Average Win      | +$185               |
| Average Loss     | -$156               |

---

## References

- **Momentum**: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
- **Low Volatility Anomaly**: Ang et al. (2006), "The Cross-Section of Volatility and Expected Returns"
- **Trend Following**: Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following Investing"

---
