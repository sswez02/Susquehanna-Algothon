# Technical Summary

```
  IN-SAMPLE (750-1000)      │  OUT-OF-SAMPLE (1000-1250)         
  Score: 41.98              │  Score: 20.02                      
  Sharpe: 4.42              │  Sharpe: 3.01                      
  Return: 5.83%             │  Return: 3.79%                     
  (Used for parameter       │  (Hidden data)                     
   optimization)            │                                    
```

## Strategy

**Short instruments with low volatility + negative long-term momentum + high trend consistency (R²)**

## The Key Insight

R² from linear regression measures how _consistently_ price follows a trend:

- **High R² + falling price**: Persistent downtrend, keep shorting
- **Low R² + falling price**: Noisy, might reverse, avoid

## Parameters

```python
VOL_LOOKBACK = 16       # Days for volatility calculation
VOL_PERCENTILE = 78     # Short only low-vol instruments
MOM_LOOKBACK = 706      # 706 days momentum window
R2_LOOKBACK = 105       # 105 days for trend strength
R2_THRESHOLD = 0.20     # Minimum trend consistency
PRICE_THRESHOLD = 60    # Maximum price filter
```

## Why It Works

1. **Low volatility**: More predictable instruments
2. **Negative momentum**: Trend continuation effect
3. **High R²**: Distinguishes persistent trends from noise
4. **Buy-and-hold**: Minimizes commission drag (0.1% per trade)

## Validation Approach

```
Competition setup:
    Days 0-750:    Available for analysis
    Days 750-1000: Parameter optimization (in-sample)
    Days 1000-1250: Hidden test set (out-of-sample)

Parameters were frozen at day 1000
Score 20.02 achieved on data never seen during development
```

## Files

| File                         | Purpose                   |
| ---------------------------- | ------------------------- |
| `main.py`                    | Strategy implementation   |
| `search/parameter_search.py` | Grid search code          |

## Quick Start

```bash
# Run strategy evaluation
python eval.py

# Reproduce parameter search
python -m search.parameter_search
```