"""
Feature Engineering

Compute features from price data for ML models.

Features grouped by category:
1. MOMENTUM: Trend-following indicators
2. MEAN REVERSION: Z-scores, distance from mean
3. VOLATILITY: Risk measures
4. CROSS-SECTIONAL: Rankings relative to other instruments
5. TECHNICAL: Classic technical indicators
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """Container for features with names"""

    features: np.ndarray  # (n_samples, n_features)
    names: List[str]

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


class FeatureEngine:
    """
    Feature engineering for trading ML models.

    Computes features for each instrument at each time point.

    Example:
    ```python
    engine = FeatureEngine(prices)
    features = engine.compute_all(day=500)
    # features.shape = (n_instruments, n_features)
    ```
    """

    def __init__(self, prices: np.ndarray):
        """
        Args:
            prices: Price matrix (n_instruments x n_days)
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape

        # Precompute returns
        self.returns = np.zeros_like(prices)
        self.returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

        # Precompute log returns
        self.log_returns = np.zeros_like(prices)
        self.log_returns[:, 1:] = np.log(prices[:, 1:] / prices[:, :-1])

    def momentum_features(
        self, day: int, lookbacks: List[int] = [5, 10, 20, 60, 120, 250]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute momentum features.

        Returns price change over various lookback periods.
        """
        features = []
        names = []

        for lb in lookbacks:
            if day >= lb:
                past_prices = self.prices[:, day - lb]
                current_prices = self.prices[:, day]
                momentum = (current_prices - past_prices) / past_prices
            else:
                momentum = np.zeros(self.n_instruments)

            features.append(momentum)
            names.append(f"momentum_{lb}d")

        return np.column_stack(features), names

    def volatility_features(
        self, day: int, windows: List[int] = [10, 20, 60]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute volatility features.

        Returns rolling volatility measures.
        """
        features = []
        names = []

        for w in windows:
            if day >= w:
                window_returns = self.returns[:, day - w : day]
                vol = np.std(window_returns, axis=1)
            else:
                vol = np.zeros(self.n_instruments)

            features.append(vol)
            names.append(f"volatility_{w}d")

        # Volatility ratio (short/long)
        if day >= 60:
            vol_20 = np.std(self.returns[:, day - 20 : day], axis=1)
            vol_60 = np.std(self.returns[:, day - 60 : day], axis=1)
            vol_ratio = vol_20 / (vol_60 + 1e-8)
        else:
            vol_ratio = np.ones(self.n_instruments)

        features.append(vol_ratio)
        names.append("vol_ratio_20_60")

        return np.column_stack(features), names

    def mean_reversion_features(
        self, day: int, windows: List[int] = [10, 20, 50]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute mean reversion features.

        Z-scores measuring distance from rolling mean.
        """
        features = []
        names = []

        current_prices = self.prices[:, day]

        for w in windows:
            if day >= w:
                window_prices = self.prices[:, day - w : day]
                mean_price = np.mean(window_prices, axis=1)
                std_price = np.std(window_prices, axis=1)
                std_price = np.where(std_price < 1e-8, 1e-8, std_price)

                zscore = (current_prices - mean_price) / std_price
            else:
                zscore = np.zeros(self.n_instruments)

            features.append(zscore)
            names.append(f"zscore_{w}d")

        return np.column_stack(features), names

    def cross_sectional_features(
        self, day: int, lookbacks: List[int] = [5, 20, 60]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute cross-sectional features.

        Rankings relative to other instruments.
        """
        features = []
        names = []

        for lb in lookbacks:
            if day >= lb:
                # Return over lookback
                past_prices = self.prices[:, day - lb]
                current_prices = self.prices[:, day]
                returns = (current_prices - past_prices) / past_prices

                # Rank (0 = worst, 1 = best)
                rank = np.argsort(np.argsort(returns)) / (self.n_instruments - 1)
            else:
                rank = np.full(self.n_instruments, 0.5)

            features.append(rank)
            names.append(f"rank_{lb}d")

        return np.column_stack(features), names

    def technical_features(self, day: int) -> Tuple[np.ndarray, List[str]]:
        """
        Compute technical indicator features.
        """
        features = []
        names = []

        current_prices = self.prices[:, day]

        # RSI (Relative Strength Index)
        if day >= 14:
            window = self.returns[:, day - 14 : day]
            gains = np.where(window > 0, window, 0)
            losses = np.where(window < 0, -window, 0)

            avg_gain = np.mean(gains, axis=1)
            avg_loss = np.mean(losses, axis=1)

            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi = (rsi - 50) / 50  # Normalize to [-1, 1]
        else:
            rsi = np.zeros(self.n_instruments)

        features.append(rsi)
        names.append("rsi_14d")

        # MACD feature
        if day >= 26:
            ema_12 = np.mean(self.prices[:, day - 12 : day], axis=1)
            ema_26 = np.mean(self.prices[:, day - 26 : day], axis=1)
            macd = (ema_12 - ema_26) / ema_26
        else:
            macd = np.zeros(self.n_instruments)

        features.append(macd)
        names.append("macd")

        # Distance from 52-week high/low
        if day >= 252:
            high_252 = np.max(self.prices[:, day - 252 : day], axis=1)
            low_252 = np.min(self.prices[:, day - 252 : day], axis=1)

            dist_from_high = (current_prices - high_252) / high_252
            dist_from_low = (current_prices - low_252) / low_252

            pct_range = (current_prices - low_252) / (high_252 - low_252 + 1e-8)
        else:
            dist_from_high = np.zeros(self.n_instruments)
            dist_from_low = np.zeros(self.n_instruments)
            pct_range = np.full(self.n_instruments, 0.5)

        features.append(dist_from_high)
        names.append("dist_from_52w_high")
        features.append(pct_range)
        names.append("pct_52w_range")

        return np.column_stack(features), names

    def compute_all(self, day: int) -> FeatureSet:
        """
        Compute all features for a given day.

        Args:
            day: Day index to compute features for

        Returns:
            FeatureSet with features and names
        """
        all_features = []
        all_names = []

        # Momentum
        f, n = self.momentum_features(day)
        all_features.append(f)
        all_names.extend(n)

        # Volatility
        f, n = self.volatility_features(day)
        all_features.append(f)
        all_names.extend(n)

        # Mean reversion
        f, n = self.mean_reversion_features(day)
        all_features.append(f)
        all_names.extend(n)

        # Cross-sectional
        f, n = self.cross_sectional_features(day)
        all_features.append(f)
        all_names.extend(n)

        # Technical
        f, n = self.technical_features(day)
        all_features.append(f)
        all_names.extend(n)

        features = np.hstack(all_features)

        return FeatureSet(features=features, names=all_names)

    def compute_dataset(
        self, start_day: int, end_day: int, target_horizon: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute features and targets for training.

        Args:
            start_day: First day to include
            end_day: Last day to include (exclusive)
            target_horizon: Days forward for target returns

        Returns:
            (features, targets, feature_names)
            features: (n_samples, n_features) where n_samples = n_instruments * n_days
            targets: (n_samples,) future returns
        """
        all_features = []
        all_targets = []

        for day in range(start_day, end_day - target_horizon):
            feature_set = self.compute_all(day)

            # Target: mean return over next target_horizon days
            current_prices = self.prices[:, day]
            future_prices = self.prices[:, day + target_horizon]
            target_returns = (future_prices - current_prices) / current_prices

            all_features.append(feature_set.features)
            all_targets.append(target_returns)

        # Stack into arrays
        # features: (n_days, n_instruments, n_features) -> (n_samples, n_features)
        features = np.vstack(all_features)
        targets = np.concatenate(all_targets)

        return features, targets, feature_set.names  # type: ignore


def get_feature_names() -> List[str]:
    """Get list of all feature names (for documentation)"""
    np.random.seed(42)
    dummy_prices = np.random.randn(50, 300) * 10 + 100
    engine = FeatureEngine(dummy_prices)
    feature_set = engine.compute_all(299)
    return feature_set.names


if __name__ == "__main__":
    # Test feature computation
    print("Testing Feature Engineering")

    # Create dummy prices
    np.random.seed(42)
    n_inst, n_days = 50, 500
    returns = np.random.randn(n_inst, n_days) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=1)

    engine = FeatureEngine(prices)

    # Test single day features
    feature_set = engine.compute_all(day=400)
    print(f"\nFeatures for day 400:")
    print(f"  Shape: {feature_set.features.shape}")
    print(f"  Features: {feature_set.names}")

    # Test dataset creation
    X, y, names = engine.compute_dataset(start_day=260, end_day=400, target_horizon=10)
    print(f"\nDataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(names)}")

    print("\nFeature Engineering Ready")
