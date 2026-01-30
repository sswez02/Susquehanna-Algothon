"""
Performance Metrics
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    Container for all performance metrics.

    The competition score is: mean(PL) - 0.1 * std(PL)
    """

    # Primary metrics
    score: float  # Competition score
    sharpe: float  # Annualized Sharpe ratio

    # Return metrics
    total_pnl: float  # Total P&L
    mean_pnl: float  # Mean daily P&L
    std_pnl: float  # Std of daily P&L

    # Risk metrics
    max_drawdown: float  # Maximum drawdown
    var_95: float  # 95% Value at Risk

    # Trade metrics
    win_rate: float  # Fraction of winning days
    profit_factor: float  # Gross profit / Gross loss

    # Other
    n_days: int  # Number of trading days

    def __str__(self) -> str:
        return f"""
Performance Metrics:
  Score:         {self.score:>10.2f}
  Sharpe:        {self.sharpe:>10.2f}
  Total P&L:     ${self.total_pnl:>9.2f}
  Mean P&L:      ${self.mean_pnl:>9.2f}
  Std P&L:       ${self.std_pnl:>9.2f}
  Max Drawdown:  ${self.max_drawdown:>9.2f}
  Win Rate:      {self.win_rate*100:>9.1f}%
  Profit Factor: {self.profit_factor:>10.2f}
  Trading Days:  {self.n_days:>10d}
"""


def compute_metrics(daily_pnl: np.ndarray) -> PerformanceMetrics:
    """
    Compute all performance metrics from daily P&L.

    Args:
        daily_pnl: Array of daily P&L values

    Returns:
        PerformanceMetrics object
    """
    if len(daily_pnl) == 0:
        return PerformanceMetrics(
            score=0,
            sharpe=0,
            total_pnl=0,
            mean_pnl=0,
            std_pnl=0,
            max_drawdown=0,
            var_95=0,
            win_rate=0,
            profit_factor=0,
            n_days=0,
        )

    # Basic statistics
    mean_pnl = np.mean(daily_pnl)
    std_pnl = np.std(daily_pnl)
    total_pnl = np.sum(daily_pnl)

    # Competition score
    score = mean_pnl - 0.1 * std_pnl

    # Sharpe ratio (annualized)
    sharpe = np.sqrt(252) * mean_pnl / std_pnl if std_pnl > 0 else 0

    # Maximum drawdown
    cumulative = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns)

    # Value at Risk (95%)
    var_95 = np.percentile(daily_pnl, 5)  # 5th percentile = 95% VaR

    # Win rate
    win_rate = np.mean(daily_pnl > 0)

    # Profit factor
    gains = daily_pnl[daily_pnl > 0].sum() if np.any(daily_pnl > 0) else 0
    losses = -daily_pnl[daily_pnl < 0].sum() if np.any(daily_pnl < 0) else 1e-10
    profit_factor = gains / losses

    return PerformanceMetrics(
        score=score,  # type: ignore
        sharpe=sharpe,
        total_pnl=total_pnl,
        mean_pnl=mean_pnl,  # type: ignore
        std_pnl=std_pnl,  # type: ignore
        max_drawdown=max_drawdown,
        var_95=var_95,  # type: ignore
        win_rate=win_rate,  # type: ignore
        profit_factor=profit_factor,
        n_days=len(daily_pnl),
    )


def compute_score(daily_pnl: np.ndarray) -> float:
    """
    Compute just the competition score.

    Score = mean(PL) - 0.1 * std(PL)
    """
    if len(daily_pnl) == 0:
        return 0.0
    return np.mean(daily_pnl) - 0.1 * np.std(daily_pnl)  # type: ignore


def compute_sharpe(daily_pnl: np.ndarray, annualize: bool = True) -> float:
    """
    Compute Sharpe ratio.

    Args:
        daily_pnl: Daily P&L values
        annualize: Whether to annualize (multiply by sqrt(252))
    """
    if len(daily_pnl) == 0:
        return 0.0

    mean_pnl = np.mean(daily_pnl)
    std_pnl = np.std(daily_pnl)

    if std_pnl == 0:
        return 0.0

    sharpe = mean_pnl / std_pnl

    if annualize:
        sharpe *= np.sqrt(252)

    return sharpe  # type: ignore


def compute_drawdown(daily_pnl: np.ndarray) -> np.ndarray:
    """
    Compute drawdown series.

    Returns:
        Array of drawdown values at each time point
    """
    cumulative = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative)
    return running_max - cumulative


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...")

    # Simulate some P&L data
    np.random.seed(42)
    daily_pnl = np.random.randn(250) * 50 + 20  # Mean=$20, Std=$50

    metrics = compute_metrics(daily_pnl)
    print(metrics)

    print(f"Quick score: {compute_score(daily_pnl):.2f}")
    print(f"Quick sharpe: {compute_sharpe(daily_pnl):.2f}")
