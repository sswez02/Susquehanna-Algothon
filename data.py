"""
Data Loading and Preprocessing

Handles loading price data and computing basic transformations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config import Config, get_config


class PriceData:
    """
    Container for price data with computed returns.

    Attributes:
        prices: Raw prices (n_instruments x n_days)
        returns: Daily returns (n_instruments x n_days)
        log_returns: Log returns (n_instruments x n_days)
        n_instruments: Number of instruments
        n_days: Number of trading days
    """

    def __init__(self, prices: np.ndarray):
        """
        Initialize with price matrix.

        Args:
            prices: Price matrix (n_instruments x n_days)
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape

        # Compute returns
        self.returns = np.zeros_like(prices)
        self.returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

        # Compute log returns
        self.log_returns = np.zeros_like(prices)
        self.log_returns[:, 1:] = np.log(prices[:, 1:] / prices[:, :-1])

    def get_prices(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Get prices for a date range"""
        return self.prices[:, start:end]

    def get_returns(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Get returns for a date range"""
        return self.returns[:, start:end]

    def summary(self) -> dict:
        """Get summary statistics"""
        total_returns = (self.prices[:, -1] - self.prices[:, 0]) / self.prices[:, 0]

        return {
            "n_instruments": self.n_instruments,
            "n_days": self.n_days,
            "winners": np.sum(total_returns > 0),
            "losers": np.sum(total_returns < 0),
            "avg_total_return": np.mean(total_returns),
            "avg_daily_return": np.mean(self.returns[:, 1:]),
            "avg_volatility": np.mean(np.std(self.returns[:, 1:], axis=1))
            * np.sqrt(252),
        }


def load_data(filepath: str = "prices.txt") -> PriceData:
    """
    Load price data from file.

    Args:
        filepath: Path to prices.txt

    Returns:
        PriceData object
    """
    # Load raw data
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    prices = df.values.T  # Transform to (n_instruments x n_days)

    return PriceData(prices)


def train_test_split(
    data: PriceData, config: Optional[Config] = None
) -> Tuple[PriceData, PriceData]:
    """
    Split data into training and test sets.

    Args:
        data: Full PriceData object
        config: Configuration (uses default if None)

    Returns:
        (train_data, test_data) tuple
    """
    if config is None:
        config = get_config()

    train_prices = data.prices[:, config.data.train_start : config.data.train_end]
    test_prices = data.prices[:, config.data.test_start : config.data.test_end]

    return PriceData(train_prices), PriceData(test_prices)


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading")

    try:
        data = load_data()
        print(f"\nData loaded successfully!")
        print(f"Shape: {data.n_instruments} instruments × {data.n_days} days")

        summary = data.summary()
        print(f"\nSummary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except FileNotFoundError:
        print("prices.txt not found")
