"""Performance metrics"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    Container for all performance metrics.

    The competition score is: mean(PL) - 0.1 * std(PL)
    """

    score: float
    sharpe: float
    total_pnl: float
    mean_pnl: float
    std_pnl: float
    max_drawdown: float
    win_rate: float
    n_days: int

    def __str__(self):
        return f"Score: {self.score:.2f} | Sharpe: {self.sharpe:.2f} | Total: ${self.total_pnl:.0f}"


def compute_score(daily_pnl: np.ndarray) -> float:
    """Score = mean - 0.1 * std"""
    return np.mean(daily_pnl) - 0.1 * np.std(daily_pnl) if len(daily_pnl) > 0 else 0.0


def compute_metrics(daily_pnl: np.ndarray) -> PerformanceMetrics:
    """
    Compute all performance metrics from daily P&L.

    Args:
        daily_pnl: Array of daily P&L values

    Returns:
        PerformanceMetrics object
    """
    if len(daily_pnl) == 0:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    mean_pnl = np.mean(daily_pnl)
    std_pnl = np.std(daily_pnl)
    cumulative = np.cumsum(daily_pnl)
    max_dd = np.max(np.maximum.accumulate(cumulative) - cumulative)

    return PerformanceMetrics(
        score=mean_pnl - 0.1 * std_pnl,
        sharpe=np.sqrt(252) * mean_pnl / std_pnl if std_pnl > 0 else 0,
        total_pnl=np.sum(daily_pnl),
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        max_drawdown=max_dd,
        win_rate=np.mean(daily_pnl > 0),
        n_days=len(daily_pnl),
    )
