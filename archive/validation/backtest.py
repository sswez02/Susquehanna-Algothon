"""Backtesting engine"""

import numpy as np
from typing import Callable
from dataclasses import dataclass
from core.metrics import compute_metrics, PerformanceMetrics


@dataclass
class BacktestResult:
    """Results from a backtest run"""

    daily_pnl: np.ndarray
    positions: np.ndarray
    metrics: PerformanceMetrics


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.

    This matches the competition's evaluation logic:
    1. Position limits of $10,000 per instrument
    2. 0.1% commission on traded value
    3. Score = mean(PL) - 0.1 * std(PL)

    Example usage:
    ```python
    engine = BacktestEngine(prices)

    def my_strategy(prices, position, day):
        # Return target positions
        return np.zeros(50)

    result = engine.run(my_strategy, start=751, end=1001)
    print(result.metrics)
    ```
    """

    def __init__(self, prices: np.ndarray, config=None):
        """
        Initialize backtester.

        Args:
            prices: Price matrix (n_instruments x n_days)
            config: Configuration object
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape
        self.dlr_limit = 10000 if config is None else config.trading.dlr_limit
        self.commission = 0.001 if config is None else config.trading.commission_rate

    def run(self, strategy: Callable, start: int, end: int) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Function(current_prices, current_position, day) -> new_position
            start: First trading day (inclusive)
            end: Last trading day (exclusive)
            verbose: Whether to print progress

        Returns:
            BacktestResult object
        """
        # Initialize state
        cash = 0.0
        position = np.zeros(self.n_instruments)
        # Track history
        daily_pnl, positions = [], []
        prev_value = 0.0

        for day in range(start, end):
            current = self.prices[:, day]

            # Get new positions from strategy
            new_pos = strategy(current, position.copy(), day)
            new_pos = np.array(new_pos, dtype=float)

            # Apply position limits
            limits = self.dlr_limit / current
            new_pos = np.clip(new_pos, -limits, limits).astype(int)

            # Calculate trading costs
            delta = new_pos - position
            turnover = np.sum(np.abs(delta) * current)
            commission = turnover * self.commission

            # Update cash (buy costs money, sell gives money)
            cash -= np.dot(current, delta) + commission

            # Update position
            position = new_pos.copy()

            # Calculate portfolio value
            value = cash + np.dot(position, current)

            # Record P&L (starting from second day)
            if day > start:
                daily_pnl.append(value - prev_value)
            prev_value = value
            positions.append(position.copy())

        return BacktestResult(
            daily_pnl=np.array(daily_pnl),
            positions=np.array(positions),
            metrics=compute_metrics(np.array(daily_pnl)),
        )
