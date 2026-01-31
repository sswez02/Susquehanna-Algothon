"""
Trading strategies

Phase 3: Strategies designed for THIS specific market:
- BEARISH market (-10.27% avg return)
- NEAR-ZERO autocorrelation (returns unpredictable)
- HIGH correlations (0.66) - pairs/relative value works
"""

import numpy as np
from typing import Callable


class StrategyFactory:
    """
    Factory for creating trading strategies.

    Example:
    ```python
    factory = StrategyFactory(prices)
    strategy = factory.cross_sectional_mean_reversion()
    result = engine.run(strategy, start=751, end=1001)
    ```
    """

    def __init__(self, prices: np.ndarray, dlr_limit: float = 10000):
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape
        self.dlr_limit = dlr_limit
        self.returns = np.zeros_like(prices)
        self.returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

    def zero(self) -> Callable:
        """Zero strategy: hold no positions."""

        def strategy(current_prices, position, day):
            return np.zeros(self.n_instruments)

        return strategy

    def buy_and_hold(self, scale: float = 0.3) -> Callable:
        """Buy and hold: equal weight long all instruments."""
        stored = [None]

        def strategy(current_prices, position, day):
            if stored[0] is None:
                stored[0] = (scale * self.dlr_limit / current_prices).astype(int)
            return stored[0]

        return strategy

    def momentum(self, lookback: int = 60, scale: float = 0.3) -> Callable:
        """Momentum: follow the trend."""

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)
            past = self.prices[:, day - lookback]
            mom = (current_prices - past) / past
            max_shares = self.dlr_limit / current_prices
            return (np.sign(mom) * scale * max_shares).astype(int)

        return strategy

    def mean_reversion(
        self, lookback: int = 20, threshold: float = 2.0, scale: float = 0.3
    ) -> Callable:
        """Mean reversion: fade extremes."""

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)
            window = self.prices[:, day - lookback : day]
            mean_p = np.mean(window, axis=1)
            std_p = np.std(window, axis=1) + 1e-8
            zscore = (current_prices - mean_p) / std_p
            max_shares = self.dlr_limit / current_prices
            signal = np.where(
                zscore > threshold, -1, np.where(zscore < -threshold, 1, 0)
            )
            return (signal * scale * max_shares).astype(int)

        return strategy

    def volatility_scaled(
        self, momentum_lookback: int = 300, vol_lookback: int = 60, scale: float = 0.3
    ) -> Callable:
        """Volatility-scaled momentum."""

        def strategy(current_prices, position, day):
            if day < max(momentum_lookback, vol_lookback):
                return np.zeros(self.n_instruments)

            past = self.prices[:, day - momentum_lookback]
            mom = (current_prices - past) / past
            signal = np.sign(mom)

            vol = np.std(self.returns[:, day - vol_lookback : day], axis=1) * np.sqrt(
                252
            )
            vol = np.where(vol < 0.05, 0.05, vol)
            vol_scale = np.clip(0.15 / vol, 0.3, 2.0)

            max_shares = self.dlr_limit / current_prices
            return (signal * vol_scale * scale * max_shares).astype(int)

        return strategy

    def cross_sectional_momentum(
        self,
        lookback: int = 60,
        n_long: int = 10,
        n_short: int = 10,
        scale: float = 0.3,
    ) -> Callable:
        """Cross-sectional momentum: long winners, short losers."""

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)

            past = self.prices[:, day - lookback]
            returns = (current_prices - past) / past
            ranked = np.argsort(returns)

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            for i in ranked[-n_long:]:
                new_pos[i] = scale * max_shares[i]
            for i in ranked[:n_short]:
                new_pos[i] = -scale * max_shares[i]

            return new_pos.astype(int)

        return strategy

    def combined(self) -> Callable:
        """Combined: blend momentum signals."""
        mom = self.momentum(300, scale=1.0)
        vol = self.volatility_scaled(300, 60, scale=1.0)

        def strategy(current_prices, position, day):
            p1 = mom(current_prices, position, day)
            p2 = vol(current_prices, position, day)
            combined = 0.5 * p1 + 0.5 * p2
            max_shares = self.dlr_limit / current_prices
            return np.clip(combined * 0.2, -max_shares, max_shares).astype(int)

        return strategy

    def low_vol_short_bias(self, scale: float = 0.2, n_short: int = 15) -> Callable:
        """
        Low-Volatility Short Bias: COMBINES BEST STRATEGIES!
        """

        def strategy(current_prices, position, day):
            if day < 120:
                return np.zeros(self.n_instruments)

            vol = np.std(self.returns[:, day - 60 : day], axis=1) * np.sqrt(252)
            vol_ranked = np.argsort(vol)

            past = self.prices[:, day - 120]
            mom = (current_prices - past) / past

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            n_shorted = 0
            for i in vol_ranked:
                if n_shorted >= n_short:
                    break
                if mom[i] > 0.10:
                    continue
                new_pos[i] = -scale * max_shares[i]
                n_shorted += 1

            vol_scale = np.clip(0.10 / (vol + 0.01), 0.5, 1.5)
            new_pos = new_pos * vol_scale

            return new_pos.astype(int)

        return strategy

    def selective_short(self, scale: float = 0.2) -> Callable:
        """Only short instruments with negative momentum."""

        def strategy(current_prices, position, day):
            if day < 120:
                return np.zeros(self.n_instruments)

            past = self.prices[:, day - 120]
            mom = (current_prices - past) / past

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            for i in range(self.n_instruments):
                if mom[i] < 0:
                    new_pos[i] = -scale * max_shares[i]

            return new_pos.astype(int)

        return strategy

    def cross_sectional_mean_reversion(
        self, lookback: int = 10, n_long: int = 5, n_short: int = 5, scale: float = 0.15
    ) -> Callable:
        """
        Cross-sectional mean reversion: fade recent extremes.

        OPPOSITE of cross-sectional momentum:
        - Short recent WINNERS (expect reversion down)
        - Long recent LOSERS (expect reversion up)

        Works when autocorrelation is low.
        """

        def strategy(current_prices, position, day):
            if day < lookback + 10:
                return np.zeros(self.n_instruments)

            past = self.prices[:, day - lookback]
            returns = (current_prices - past) / past
            ranked = np.argsort(returns)

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            # Long bottom performers (expect reversion UP)
            for i in ranked[:n_long]:
                new_pos[i] = scale * max_shares[i]

            # Short top performers (expect reversion DOWN)
            for i in ranked[-n_short:]:
                new_pos[i] = -scale * max_shares[i]

            return new_pos.astype(int)

        return strategy

    def short_bias(self, scale: float = 0.1) -> Callable:
        """Short bias: net short for bearish market."""

        def strategy(current_prices, position, day):
            max_shares = self.dlr_limit / current_prices
            return (-scale * max_shares).astype(int)

        return strategy

    def pairs_trading(
        self,
        pair1: int = 27,
        pair2: int = 38,
        lookback: int = 20,
        threshold: float = 2.0,
        scale: float = 0.3,
    ) -> Callable:
        """
        Pairs trading: exploit high correlations.

        Instruments 27 and 38 have correlation 0.66.
        """

        def strategy(current_prices, position, day):
            if day < lookback:
                return np.zeros(self.n_instruments)

            p1 = self.prices[pair1, day - lookback : day + 1]
            p2 = self.prices[pair2, day - lookback : day + 1]
            spread = np.log(p1) - np.log(p2)

            spread_mean = np.mean(spread[:-1])
            spread_std = np.std(spread[:-1]) + 1e-8
            zscore = (spread[-1] - spread_mean) / spread_std

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            if zscore > threshold:
                new_pos[pair1] = -scale * max_shares[pair1]
                new_pos[pair2] = scale * max_shares[pair2]
            elif zscore < -threshold:
                new_pos[pair1] = scale * max_shares[pair1]
                new_pos[pair2] = -scale * max_shares[pair2]

            return new_pos.astype(int)

        return strategy

    def low_volatility(self, scale: float = 0.1) -> Callable:
        """
        Low volatility: only trade lowest volatility instruments.

        Score = mean - 0.1*std, so reducing std helps.
        """

        def strategy(current_prices, position, day):
            if day < 60:
                return np.zeros(self.n_instruments)

            # Calculate volatility for each instrument
            vol = np.std(self.returns[:, day - 60 : day], axis=1) * np.sqrt(252)

            # Only trade the 10 lowest volatility instruments
            ranked = np.argsort(vol)
            low_vol_idx = ranked[:10]

            # Long-term momentum for direction
            past = self.prices[:, day - 120] if day >= 120 else self.prices[:, 0]
            mom = (current_prices - past) / past

            new_pos = np.zeros(self.n_instruments)
            max_shares = self.dlr_limit / current_prices

            for i in low_vol_idx:
                new_pos[i] = np.sign(mom[i]) * scale * max_shares[i]

            return new_pos.astype(int)

        return strategy

    def conservative_momentum(self, scale: float = 0.05) -> Callable:
        """
        Conservative momentum: very small positions.

        Vol-Scaled made $2,097 but had high std.
        This uses same logic but much smaller scale.
        """

        def strategy(current_prices, position, day):
            if day < 300:
                return np.zeros(self.n_instruments)

            # Long-term momentum
            past = self.prices[:, day - 300]
            mom = (current_prices - past) / past
            signal = np.sign(mom)

            # Vol scaling
            vol = np.std(self.returns[:, day - 60 : day], axis=1) * np.sqrt(252)
            vol = np.where(vol < 0.05, 0.05, vol)
            vol_scale = np.clip(0.15 / vol, 0.3, 2.0)

            max_shares = self.dlr_limit / current_prices
            return (signal * vol_scale * scale * max_shares).astype(int)

        return strategy

    def combined_v2(self) -> Callable:
        """Combined V2: mean reversion focus for this market."""
        cs_mr = self.cross_sectional_mean_reversion(
            lookback=10, n_long=5, n_short=5, scale=0.1
        )
        cons_mom = self.conservative_momentum(scale=0.03)

        def strategy(current_prices, position, day):
            p1 = cs_mr(current_prices, position, day)
            p2 = cons_mom(current_prices, position, day)
            combined = 0.6 * p1 + 0.4 * p2
            max_shares = self.dlr_limit / current_prices
            return np.clip(combined, -max_shares, max_shares).astype(int)

        return strategy
