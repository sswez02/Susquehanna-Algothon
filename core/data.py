"""Data loading"""

import numpy as np
import pandas as pd


class PriceData:
    def __init__(self, prices: np.ndarray):
        """
        Container for price data with computed returns.

        Attributes:
            prices: Raw prices (n_instruments x n_days)
            returns: Daily returns (n_instruments x n_days)
            n_instruments: Number of instruments
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape
        self.returns = np.zeros_like(prices)
        self.returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]


def load_data(filepath: str = "prices.txt") -> PriceData:
    """
    Load price data from file.

    Args:
        filepath: Path to prices.txt

    Returns:
        PriceData object
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    return PriceData(df.values.T)
