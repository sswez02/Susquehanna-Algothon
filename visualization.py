"""
Visualization

Creates plots and charts for:
1. Performance analysis
2. Strategy comparison
3. Feature analysis
4. Risk metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os

from metrics import PerformanceMetrics

COLORS = {
    "primary": "#2563eb",
    "secondary": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "neutral": "#6b7280",
}


def set_style():
    """Set matplotlib style"""
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_equity_curve(
    daily_pnl: np.ndarray, title: str = "Equity Curve", save_path: Optional[str] = None
) -> plt.Figure:  # type: ignore
    """
    Plot cumulative P&L.

    Args:
        daily_pnl: Array of daily P&L values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    set_style()

    cumulative = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Equity curve
    ax.plot(cumulative, color=COLORS["primary"], linewidth=2, label="Cumulative P&L")

    # Running max
    ax.plot(
        running_max,
        color=COLORS["secondary"],
        linewidth=1,
        linestyle="--",
        alpha=0.7,
        label="Peak",
    )

    # Drawdown shading
    ax.fill_between(
        range(len(cumulative)),
        cumulative,
        running_max,
        color=COLORS["danger"],
        alpha=0.3,
        label="Drawdown",
    )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_returns_distribution(
    daily_pnl: np.ndarray,
    title: str = "P&L Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Plot histogram of daily returns.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(daily_pnl, bins=50, edgecolor="white", alpha=0.7)

    # Color positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):  # type: ignore
        if left_edge < 0:
            patch.set_facecolor(COLORS["danger"])
        else:
            patch.set_facecolor(COLORS["secondary"])

    # Mean line
    ax.axvline(
        np.mean(daily_pnl),  # type: ignore
        color=COLORS["primary"],
        linewidth=2,
        label=f"Mean: ${np.mean(daily_pnl):.2f}",
    )

    ax.set_xlabel("Daily P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rolling_sharpe(
    daily_pnl: np.ndarray,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio",
    save_path: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Plot rolling Sharpe ratio over time.
    """
    set_style()

    # Calculate rolling Sharpe
    rolling_sharpe = []
    for i in range(window, len(daily_pnl)):
        window_pnl = daily_pnl[i - window : i]
        sharpe = np.sqrt(252) * np.mean(window_pnl) / (np.std(window_pnl) + 1e-8)
        rolling_sharpe.append(sharpe)

    rolling_sharpe = np.array(rolling_sharpe)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot
    ax.plot(
        range(window, len(daily_pnl)),
        rolling_sharpe,
        color=COLORS["primary"],
        linewidth=1.5,
    )

    # Fill positive/negative
    ax.fill_between(
        range(window, len(daily_pnl)),
        0,
        rolling_sharpe,
        where=rolling_sharpe > 0,  # type: ignore
        color=COLORS["secondary"],
        alpha=0.3,
    )
    ax.fill_between(
        range(window, len(daily_pnl)),
        0,
        rolling_sharpe,
        where=rolling_sharpe < 0,  # type: ignore
        color=COLORS["danger"],
        alpha=0.3,
    )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(
        y=2,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=0.5,
        label="Target (2.0)",
    )

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Rolling Sharpe Ratio")
    ax.set_title(f"{title} ({window}-day window)", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_strategy_comparison(
    results: Dict[str, np.ndarray],
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Compare multiple strategies on same plot.

    Args:
        results: Dictionary of name -> daily_pnl array
    """
    set_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))  # type: ignore

    for (name, pnl), color in zip(results.items(), colors):
        cumulative = np.cumsum(pnl)
        ax.plot(cumulative, label=name, linewidth=2, color=color)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_monthly_returns(
    daily_pnl: np.ndarray,
    title: str = "Monthly Returns",
    save_path: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Plot monthly return bar chart.
    """
    set_style()

    # Group into months (21 trading days)
    n_months = len(daily_pnl) // 21
    monthly_pnl = []

    for i in range(n_months):
        start = i * 21
        end = (i + 1) * 21
        monthly_pnl.append(np.sum(daily_pnl[start:end]))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLORS["secondary"] if p > 0 else COLORS["danger"] for p in monthly_pnl]
    bars = ax.bar(range(len(monthly_pnl)), monthly_pnl, color=colors, alpha=0.8)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly P&L ($)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_performance_report(
    daily_pnl: np.ndarray,
    metrics: "PerformanceMetrics",
    output_dir: str = "outputs",
    name: str = "strategy",
) -> None:
    """
    Create full performance report with multiple plots.

    Args:
        daily_pnl: Daily P&L array
        metrics: PerformanceMetrics object
        output_dir: Directory to save plots
        name: Strategy name for filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating performance report for {name}")

    # 1. Equity curve
    plot_equity_curve(
        daily_pnl,
        title=f"{name} - Equity Curve",
        save_path=os.path.join(output_dir, f"{name}_equity.png"),
    )
    plt.close()

    # 2. Returns distribution
    plot_returns_distribution(
        daily_pnl,
        title=f"{name} - P&L Distribution",
        save_path=os.path.join(output_dir, f"{name}_distribution.png"),
    )
    plt.close()

    # 3. Rolling Sharpe
    if len(daily_pnl) > 60:
        plot_rolling_sharpe(
            daily_pnl,
            window=60,
            title=f"{name} - Rolling Sharpe",
            save_path=os.path.join(output_dir, f"{name}_sharpe.png"),
        )
        plt.close()

    # 4. Monthly returns
    if len(daily_pnl) > 42:
        plot_monthly_returns(
            daily_pnl,
            title=f"{name} - Monthly Returns",
            save_path=os.path.join(output_dir, f"{name}_monthly.png"),
        )
        plt.close()

    print(f"  Saved plots to {output_dir}/")
    print(f"\n  Metrics:")
    print(f"    Score: {metrics.score:.2f}")
    print(f"    Sharpe: {metrics.sharpe:.2f}")
    print(f"    Total P&L: ${metrics.total_pnl:.2f}")


if __name__ == "__main__":
    # Test visualization
    print("Testing Visualization")

    # Create dummy data
    np.random.seed(42)
    daily_pnl = np.random.randn(250) * 50 + 20

    # Test each plot
    fig = plot_equity_curve(daily_pnl, title="Test Equity Curve")
    plt.close()

    fig = plot_returns_distribution(daily_pnl)
    plt.close()

    fig = plot_rolling_sharpe(daily_pnl)
    plt.close()

    fig = plot_monthly_returns(daily_pnl)
    plt.close()

    # Test comparison
    results = {
        "Strategy A": np.random.randn(250) * 50 + 20,
        "Strategy B": np.random.randn(250) * 60 + 10,
        "Strategy C": np.random.randn(250) * 40 + 30,
    }
    fig = plot_strategy_comparison(results)
    plt.close()

    print("\nVisualization Ready")
