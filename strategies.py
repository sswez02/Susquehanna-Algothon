"""
Trading Strategies

Implements various trading strategies as baselines:

1. MOMENTUM: Follow the trend
2. MEAN REVERSION: Fade the trend
3. PAIRS TRADING: Trade correlated pairs
4. VOLATILITY SCALED: Adjust for volatility
5. CROSS-SECTIONAL: Long winners, short losers

All strategies follow the same interface:
    strategy(current_prices, current_position, day) -> new_position
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuration for a strategy"""

    lookback: int = 60
    scale: float = 0.3
    threshold: float = 1.5
    n_long: int = 10
    n_short: int = 10


class StrategyFactory:
    """
    Factory for creating trading strategies.

    Stores the price data internally so strategies have access to history.

    Example:
    ```python
    factory = StrategyFactory(prices)
    strategy = factory.momentum(lookback=60, scale=0.3)

    # Use in backtest
    result = engine.run(strategy, start=751, end=1001)
    ```
    """

    def __init__(self, prices: np.ndarray, dlr_limit: float = 10000):
        """
        Args:
            prices: Price matrix (n_instruments x n_days)
            dlr_limit: Maximum dollar position per instrument
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape
        self.dlr_limit = dlr_limit

        # Precompute returns
        self.returns = np.zeros_like(prices)
        self.returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

    def momentum(self, lookback: int = 60, scale: float = 0.3) -> Callable:
        """
        Momentum strategy: follow the trend.

        Long instruments that went up, short instruments that went down.

        Args:
            lookback: Number of days to look back for trend
            scale: Position scale (0-1)
        """

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)

            # Calculate momentum
            past_prices = self.prices[:, day - lookback]
            momentum = (current_prices - past_prices) / past_prices

            # Convert to positions
            max_shares = self.dlr_limit / current_prices
            new_position = np.sign(momentum) * scale * max_shares

            return new_position.astype(int)

        return strategy

    def mean_reversion(
        self, lookback: int = 20, threshold: float = 2.0, scale: float = 0.3
    ) -> Callable:
        """
        Mean reversion strategy: fade the trend.

        Short instruments that are above their moving average,
        long instruments that are below.

        Args:
            lookback: Moving average window
            threshold: Z-score threshold for entry
            scale: Position scale
        """

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)

            # Calculate z-score
            window = self.prices[:, day - lookback : day]
            mean_price = np.mean(window, axis=1)
            std_price = np.std(window, axis=1)
            std_price = np.where(std_price < 1e-8, 1e-8, std_price)

            zscore = (current_prices - mean_price) / std_price

            # Trade when z-score exceeds threshold
            max_shares = self.dlr_limit / current_prices

            # Short when high, long when low
            signal = np.zeros(self.n_instruments)
            signal[zscore > threshold] = -1
            signal[zscore < -threshold] = 1

            new_position = signal * scale * max_shares

            return new_position.astype(int)

        return strategy

    def volatility_scaled(
        self,
        momentum_lookback: int = 300,
        vol_lookback: int = 60,
        scale: float = 0.3,
        target_vol: float = 0.10,
    ) -> Callable:
        """
        Volatility-scaled momentum strategy.

        Combines momentum with inverse volatility weighting.
        Higher volatility = smaller position.

        Args:
            momentum_lookback: Days for momentum calculation
            vol_lookback: Days for volatility calculation
            scale: Base position scale
            target_vol: Target annual volatility
        """

        def strategy(current_prices, position, day):
            if day < max(momentum_lookback, vol_lookback):
                return np.zeros(self.n_instruments)

            # Momentum signal
            past_prices = self.prices[:, day - momentum_lookback]
            momentum = (current_prices - past_prices) / past_prices
            signal = np.sign(momentum)

            # Volatility scaling
            vol_window = self.returns[:, day - vol_lookback : day]
            volatility = np.std(vol_window, axis=1) * np.sqrt(252)
            volatility = np.where(volatility < 0.01, 0.01, volatility)

            # Inverse volatility weights
            vol_scale = target_vol / volatility
            vol_scale = np.clip(vol_scale, 0.2, 2.0)  # Limit extreme weights

            # Positions
            max_shares = self.dlr_limit / current_prices
            new_position = signal * vol_scale * scale * max_shares

            return new_position.astype(int)

        return strategy

    def cross_sectional_momentum(
        self,
        lookback: int = 60,
        n_long: int = 10,
        n_short: int = 10,
        scale: float = 0.3,
    ) -> Callable:
        """
        Cross-sectional momentum: long winners, short losers.

        Ranks all instruments by past return.
        Goes long the top n_long, short the bottom n_short.

        Args:
            lookback: Days for ranking
            n_long: Number of instruments to go long
            n_short: Number of instruments to go short
            scale: Position scale
        """

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)

            # Calculate returns for ranking
            past_prices = self.prices[:, day - lookback]
            returns = (current_prices - past_prices) / past_prices

            # Rank instruments
            ranked = np.argsort(returns)

            # Long top performers, short bottom performers
            new_position = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            # Long top n_long
            for i in ranked[-n_long:]:
                new_position[i] = scale * max_shares[i]

            # Short bottom n_short
            for i in ranked[:n_short]:
                new_position[i] = -scale * max_shares[i]

            return new_position.astype(int)

        return strategy

    def buy_and_hold(self, scale: float = 0.3) -> Callable:
        """
        Buy and hold: equal weight long all instruments.

        Baseline strategy for comparison.
        """
        initialized = [False]
        stored_position = [None]

        def strategy(current_prices, position, day):
            if not initialized[0]:
                max_shares = self.dlr_limit / current_prices
                stored_position[0] = (scale * max_shares).astype(int)
                initialized[0] = True

            return stored_position[0]

        return strategy

    def combined(
        self,
        momentum_weight: float = 0.5,
        vol_weight: float = 0.3,
        cs_weight: float = 0.2,
        momentum_lookback: int = 300,
        vol_lookback: int = 60,
        cs_lookback: int = 60,
    ) -> Callable:
        """
        Combined strategy: blend multiple signals.

        Weights should sum to 1.0.

        Args:
            momentum_weight: Weight for momentum signal
            vol_weight: Weight for volatility-scaled momentum
            cs_weight: Weight for cross-sectional momentum
        """
        mom_strat = self.momentum(momentum_lookback, scale=1.0)
        vol_strat = self.volatility_scaled(momentum_lookback, vol_lookback, scale=1.0)
        cs_strat = self.cross_sectional_momentum(cs_lookback, scale=1.0)

        overall_scale = 0.2  # Reduce to avoid over-leveraging

        def strategy(current_prices, position, day):
            # Get signals from each strategy
            mom_pos = mom_strat(current_prices, position, day)
            vol_pos = vol_strat(current_prices, position, day)
            cs_pos = cs_strat(current_prices, position, day)

            # Blend signals
            combined = (
                momentum_weight * mom_pos + vol_weight * vol_pos + cs_weight * cs_pos
            )

            # Scale down and convert to int
            max_shares = self.dlr_limit / current_prices
            combined = np.clip(combined * overall_scale, -max_shares, max_shares)

            return combined.astype(int)

        return strategy

    def zero(self) -> Callable:
        """
        Zero strategy: hold no positions.

        Useful for comparison (should have 0 P&L and 0 score).
        """

        def strategy(current_prices, position, day):
            return np.zeros(self.n_instruments)

        return strategy


def get_all_strategies(prices: np.ndarray) -> dict:
    """
    Get dictionary of all baseline strategies.

    Args:
        prices: Price matrix

    Returns:
        Dictionary of name -> strategy function
    """
    factory = StrategyFactory(prices)

    return {
        "Zero": factory.zero(),
        "BuyHold": factory.buy_and_hold(scale=0.3),
        "Momentum_60d": factory.momentum(lookback=60, scale=0.3),
        "Momentum_120d": factory.momentum(lookback=120, scale=0.3),
        "Momentum_300d": factory.momentum(lookback=300, scale=0.3),
        "MeanRev_20d": factory.mean_reversion(lookback=20, threshold=2.0, scale=0.3),
        "MeanRev_50d": factory.mean_reversion(lookback=50, threshold=2.0, scale=0.3),
        "VolScaled_60d": factory.volatility_scaled(vol_lookback=60, scale=0.3),
        "CrossSectional_60d": factory.cross_sectional_momentum(lookback=60, scale=0.3),
        "Combined": factory.combined(
            momentum_weight=0.5, vol_weight=0.3, cs_weight=0.2
        ),
    }


if __name__ == "__main__":
    # Test strategies
    print("Testing Strategies")

    # Create dummy prices
    np.random.seed(42)
    n_inst, n_days = 50, 500
    returns = np.random.randn(n_inst, n_days) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=1)

    factory = StrategyFactory(prices)

    # Test each strategy
    strategies = {
        "Momentum": factory.momentum(60, 0.3),
        "MeanRev": factory.mean_reversion(20, 2.0, 0.3),
        "VolScaled": factory.volatility_scaled(300, 60, 0.3),
        "CrossSectional": factory.cross_sectional_momentum(60, 10, 10, 0.3),
    }

    # Quick test
    current_prices = prices[:, 350]
    position = np.zeros(n_inst)

    for name, strat in strategies.items():
        pos = strat(current_prices, position, 350)
        n_long = np.sum(pos > 0)
        n_short = np.sum(pos < 0)
        print(f"{name}: Long={n_long}, Short={n_short}")

    print("\nStrategies Ready")
