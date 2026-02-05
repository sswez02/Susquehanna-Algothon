"""Feature engineering"""

import numpy as np
from typing import List, Tuple


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

    def compute_all(self, day: int) -> Tuple[np.ndarray, List[str]]:
        """
        Compute features for a single day.

        Args:
            day: Day index (0 <= day < n_days).

        Returns:
            features: Array of shape (n_instruments, n_features).
            names: List of feature names aligned with columns in `features`.
        """
        features, names = [], []
        current = self.prices[:, day]

        # Momentum
        for lb in [5, 10, 20, 60, 120]:
            if day >= lb:
                mom = (current - self.prices[:, day - lb]) / self.prices[:, day - lb]
            else:
                mom = np.zeros(self.n_instruments)
            features.append(mom)
            names.append(f"mom_{lb}d")

        # Volatility
        for w in [10, 20, 60]:
            if day >= w:
                vol = np.std(self.returns[:, day - w : day], axis=1)
            else:
                vol = np.zeros(self.n_instruments)
            features.append(vol)
            names.append(f"vol_{w}d")

        # Z-score
        for w in [10, 20, 50]:
            if day >= w:
                mean = np.mean(self.prices[:, day - w : day], axis=1)
                std = np.std(self.prices[:, day - w : day], axis=1) + 1e-8
                zscore = (current - mean) / std
            else:
                zscore = np.zeros(self.n_instruments)
            features.append(zscore)
            names.append(f"zscore_{w}d")

        # Cross-sectional rank
        for lb in [5, 20, 60]:
            if day >= lb:
                rets = (current - self.prices[:, day - lb]) / self.prices[:, day - lb]
                rank = np.argsort(np.argsort(rets)) / (self.n_instruments - 1)
            else:
                rank = np.full(self.n_instruments, 0.5)
            features.append(rank)
            names.append(f"rank_{lb}d")

        return np.column_stack(features), names

    def compute_dataset(self, start: int, end: int, horizon: int = 10):
        """Build training dataset"""
        X_list, y_list = [], []

        for day in range(start, end - horizon):
            features, names = self.compute_all(day)
            target = (
                self.prices[:, day + horizon] - self.prices[:, day]
            ) / self.prices[:, day]
            X_list.append(features)
            y_list.append(target)

        return np.vstack(X_list), np.concatenate(y_list), names
