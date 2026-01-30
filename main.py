"""
Competition Submission

This file contains the getMyPosition() function required by the competition.

The competition calls:
    getMyPosition(currentPrices, currentPosition) -> newPosition

Where:
    currentPrices: (n_instruments,) array of current prices
    currentPosition: (n_instruments,) array of current positions
    newPosition: (n_instruments,) array of target positions

Constraints:
    - Position limit: $10,000 per instrument (shares = 10000 / price)
    - Commission: 0.1% of traded value

Scoring:
    Score = mean(daily_PL) - 0.1 * std(daily_PL)
"""

import numpy as np
from typing import Optional


# Store historical data for feature computation
class State:
    def __init__(self):
        self.price_history = []
        self.day = 0
        self.n_instruments = 50


state = State()


# STRATEGY IMPLEMENTATION
def getMyPosition(currentPrices: np.ndarray, currentPosition: np.ndarray) -> np.ndarray:
    """
    Main entry point for the competition.

    Args:
        currentPrices: Current prices for all instruments
        currentPosition: Current positions

    Returns:
        New target positions
    """
    global state

    # Update state
    currentPrices = np.array(currentPrices)
    state.price_history.append(currentPrices.copy())
    state.day += 1
    state.n_instruments = len(currentPrices)

    # Convert history to array
    prices = np.array(state.price_history).T  # (n_instruments, n_days)
    n_days = prices.shape[1]

    # Parameters
    DLR_LIMIT = 10000
    LOOKBACK = 300
    SCALE = 0.3
    VOL_LOOKBACK = 60

    # Need enough history
    if n_days < LOOKBACK:
        return np.zeros(state.n_instruments)

    # STRATEGY: Volatility-Scaled Momentum

    # 1. Momentum signal (long-term trend)
    past_prices = prices[:, -LOOKBACK]
    momentum = (currentPrices - past_prices) / past_prices
    signal = np.sign(momentum)

    # 2. Volatility scaling (inverse volatility weighting)
    if n_days >= VOL_LOOKBACK:
        returns = (
            np.diff(prices[:, -VOL_LOOKBACK:], axis=1) / prices[:, -VOL_LOOKBACK:-1]
        )
        volatility = np.std(returns, axis=1) * np.sqrt(252)  # Annualized
        volatility = np.where(volatility < 0.01, 0.01, volatility)  # Floor

        # Scale by inverse volatility
        target_vol = 0.15
        vol_scale = target_vol / volatility
        vol_scale = np.clip(vol_scale, 0.3, 2.0)  # Limit extreme weights
    else:
        vol_scale = np.ones(state.n_instruments)

    # 3. Compute positions
    max_shares = DLR_LIMIT / currentPrices
    new_position = signal * vol_scale * SCALE * max_shares

    # 4. Apply position limits
    pos_limits = DLR_LIMIT / currentPrices
    new_position = np.clip(new_position, -pos_limits, pos_limits)

    return new_position.astype(int)


# TESTING
if __name__ == "__main__":
    print("Testing main.py")

    # Simulate a few days
    np.random.seed(42)
    n_inst = 50

    # Reset state
    state = State()

    position = np.zeros(n_inst)

    for day in range(350):
        prices = 100 + np.random.randn(n_inst) * 10 + day * 0.01
        position = getMyPosition(prices, position)

        if day % 100 == 0:
            n_long = np.sum(position > 0)
            n_short = np.sum(position < 0)
            print(f"Day {day}: Long={n_long}, Short={n_short}")
