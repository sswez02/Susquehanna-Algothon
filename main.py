"""
Competition Submission - Best Strategy (1000 days)

Strategy: Low-Volatility + Negative Momentum + High R² (Trend Strength) Short

Key insight: R² measures how consistently price follows a trend.
High R² + negative momentum = strong, persistent downtrend likely to continue.

Parameters found via grid search:
- VOL_PERCENTILE = 78
- VOL_LOOKBACK = 16
- MOM_LOOKBACK = 706
- R2_LOOKBACK = 105
- R2_THRESHOLD = 0.2
- PRICE_THRESHOLD = 60

Score achieved: 41.98
Sharpe: 4.42
Instruments shorted: 25
"""

import numpy as np
from scipy import stats

nInst = 50
currentPos = np.zeros(nInst)

# Global state
nInst = 50
currentPos = np.zeros(nInst)
initialized = False


# Best parameters found through extensive search
VOL_LOOKBACK = 16
VOL_PERCENTILE = 78
MOM_LOOKBACK = 706
R2_LOOKBACK = 105
R2_THRESHOLD = 0.2
PRICE_THRESHOLD = 60
DLR_LIMIT = 10000

initialized = False


def getMyPosition(prcSoFar):
    """
    Main strategy function.

    Logic:
    1. Calculate volatility, momentum, R² using historical data only
    2. Select instruments with: vol < vol percentile + momentum < 0 + R² > R² threshold + price < price threshold
    3. Enter short positions on day 1 of trading, hold until end

    Args:
        prcSoFar: Price history (n_instruments x n_days)

    Returns:
        Position array
    """
    global currentPos, initialized

    nInst, n_days = prcSoFar.shape
    DLR_LIMIT = 10000

    # Wait for enough data
    if n_days < max(MOM_LOOKBACK, R2_LOOKBACK) + 1:
        return np.zeros(nInst)

    # Only initialize once (buy and hold to avoid commission)
    if not initialized:
        curPrices = prcSoFar[:, -1]
        returns = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]

        # Volatility filter
        volatility = np.std(returns[:, -VOL_LOOKBACK:], axis=1)
        vol_threshold = np.percentile(volatility, VOL_PERCENTILE)

        # Momentum filter
        past_prices = prcSoFar[:, -MOM_LOOKBACK]
        momentum = (curPrices - past_prices) / past_prices

        # R² filter
        x = np.arange(R2_LOOKBACK)
        r2_vals = []
        for i in range(nInst):
            y = prcSoFar[i, -R2_LOOKBACK:]
            slope, intercept, r, p, se = stats.linregress(x, y)
            r2_vals.append(r**2)
        r2_vals = np.array(r2_vals)

        # Position limits
        max_shares = np.array([int(DLR_LIMIT / curPrices[i]) for i in range(nInst)])

        # Select and short
        for i in range(nInst):
            if (
                volatility[i] < vol_threshold  # Low volatility
                and momentum[i] < 0  # Negative momentum
                and curPrices[i] < PRICE_THRESHOLD  # Low price
                and r2_vals[i] > R2_THRESHOLD
            ):  # Strong trend
                currentPos[i] = -1.0 * max_shares[i]

        initialized = True

    return currentPos.astype(int)


if __name__ == "__main__":
    """Test the strategy"""
    import pandas as pd
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "prices.txt"), sep=r"\s+", header=None)
    prices = df.values.T

    print(f"Testing: {prices.shape[0]} instruments, {prices.shape[1]} days")

    # Reset
    currentPos = np.zeros(50)
    initialized = False

    # Run
    for t in range(751, 752):
        pos = getMyPosition(prices[:, :t])
        n_short = np.sum(pos < 0)
        print(f"Day {t}: {n_short} instruments shorted")
        print(f"Instruments: {np.where(pos < 0)[0].tolist()}")
