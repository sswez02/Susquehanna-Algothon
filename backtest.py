"""
Backtesting Engine

Core backtesting functionality with proper handling of:
- Position limits
- Transaction costs
- Realistic execution

This is designed to match the competition's evaluation.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import Config, get_config
from metrics import PerformanceMetrics, compute_metrics


@dataclass
class BacktestResult:
    """Results from a backtest run"""

    daily_pnl: np.ndarray
    positions: np.ndarray
    turnover: np.ndarray
    metrics: PerformanceMetrics

    @property
    def cumulative_pnl(self) -> np.ndarray:
        return np.cumsum(self.daily_pnl)


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

    def __init__(self, prices: np.ndarray, config: Optional[Config] = None):
        """
        Initialize backtester.

        Args:
            prices: Price matrix (n_instruments x n_days)
            config: Configuration object
        """
        self.prices = prices
        self.n_instruments, self.n_days = prices.shape
        self.config = config or get_config()

        self.dlr_limit = self.config.trading.dlr_limit
        self.commission_rate = self.config.trading.commission_rate

    def run(
        self,
        strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        start: int,
        end: int,
        verbose: bool = False,
    ) -> BacktestResult:
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
        daily_pnl = []
        positions = []
        turnover = []
        prev_value = 0.0

        for day in range(start, end):
            current_prices = self.prices[:, day]

            # Get new positions from strategy
            new_position = strategy(current_prices, position.copy(), day)
            new_position = np.array(new_position, dtype=float)

            # Apply position limits
            pos_limits = self.dlr_limit / current_prices
            new_position = np.clip(new_position, -pos_limits, pos_limits)
            new_position = new_position.astype(int)

            # Calculate trading costs
            delta_position = new_position - position
            day_turnover = np.sum(np.abs(delta_position) * current_prices)
            commission = day_turnover * self.commission_rate

            # Update cash (buy costs money, sell gives money)
            cash -= np.dot(current_prices, delta_position)
            cash -= commission

            # Update position
            position = new_position.copy()

            # Calculate portfolio value
            portfolio_value = cash + np.dot(position, current_prices)

            # Record P&L (starting from second day)
            if day > start:
                daily_pnl.append(portfolio_value - prev_value)
                turnover.append(day_turnover)

            prev_value = portfolio_value
            positions.append(position.copy())

            if verbose and (day - start) % 50 == 0:
                print(
                    f"Day {day}: Value=${portfolio_value:.2f}, "
                    f"Turnover=${day_turnover:.2f}"
                )

        # Convert to arrays
        daily_pnl = np.array(daily_pnl)
        positions = np.array(positions)
        turnover = np.array(turnover) if turnover else np.array([])

        # Compute metrics
        metrics = compute_metrics(daily_pnl)

        return BacktestResult(
            daily_pnl=daily_pnl, positions=positions, turnover=turnover, metrics=metrics
        )

    def run_multiple(
        self, strategies: Dict[str, Callable], start: int, end: int
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for multiple strategies.

        Args:
            strategies: Dictionary of name -> strategy function
            start: First trading day
            end: Last trading day

        Returns:
            Dictionary of name -> BacktestResult
        """
        results = {}

        for name, strategy in strategies.items():
            print(f"Running {name}")
            results[name] = self.run(strategy, start, end)
            print(f"  Score: {results[name].metrics.score:.2f}")

        return results


def evaluate_submission(
    prices: np.ndarray, get_positions: Callable, start: int = 751, end: int = 1001
) -> PerformanceMetrics:
    """
    Evaluate a submission exactly as the competition does.

    This matches the competition's eval.py logic.

    Args:
        prices: Price matrix
        get_positions: Function matching competition interface
        start: First test day
        end: Last test day (exclusive)

    Returns:
        PerformanceMetrics
    """
    n_instruments = prices.shape[0]

    dlr_limit = 10000
    comm_rate = 0.001

    cash = 0.0
    position = np.zeros(n_instruments)
    daily_pnl = []
    prev_value = 0.0

    for day in range(start, end):
        current_prices = prices[:, day]

        # Get positions (competition interface)
        new_position = get_positions(current_prices, position.copy())

        # Apply limits
        pos_limits = dlr_limit / current_prices
        new_position = np.clip(new_position, -pos_limits, pos_limits).astype(int)

        # Transaction costs
        delta = new_position - position
        turnover = np.sum(np.abs(delta) * current_prices)
        commission = turnover * comm_rate

        # Update
        cash -= np.dot(current_prices, delta) + commission
        position = new_position.copy()

        value = cash + np.dot(position, current_prices)

        if day > start:
            daily_pnl.append(value - prev_value)
        prev_value = value

    return compute_metrics(np.array(daily_pnl))


if __name__ == "__main__":
    # Test backtester
    print("Testing BacktestEngine")

    # Create dummy prices
    np.random.seed(42)
    n_inst, n_days = 50, 1000
    returns = np.random.randn(n_inst, n_days) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=1)

    engine = BacktestEngine(prices)

    # Test with a simple strategy
    def momentum_strategy(current_prices, position, day):
        if day < 100:
            return np.zeros(50)

        past_prices = prices[:, day - 100]
        trend = (current_prices - past_prices) / past_prices

        max_shares = 10000 / current_prices
        return (np.sign(trend) * 0.3 * max_shares).astype(int)

    result = engine.run(momentum_strategy, start=100, end=500)
    print(result.metrics)
