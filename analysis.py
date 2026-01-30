"""
Analysis and Diagnostics

Tools for understanding:
1. Data characteristics
2. Strategy performance
3. Overfitting detection
4. Feature importance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from backtest import BacktestResult
from data import PriceData
from metrics import compute_metrics, compute_score


@dataclass
class DataAnalysis:
    """Results from data analysis"""

    n_instruments: int
    n_days: int

    # Market direction
    n_winners: int
    n_losers: int
    avg_return: float

    # Volatility
    min_vol: float
    max_vol: float
    avg_vol: float

    # Autocorrelation
    lag1_autocorr: float
    lag5_autocorr: float

    # Cross-sectional
    avg_correlation: float
    max_correlation: float
    top_corr_pairs: List[Tuple[int, int, float]]


def analyze_data(data: PriceData) -> DataAnalysis:
    """
    Comprehensive analysis of price data.

    Args:
        data: PriceData object

    Returns:
        DataAnalysis object with all statistics
    """
    prices = data.prices
    returns = data.returns[:, 1:]  # Skip first day (0 returns)

    n_inst, n_days = data.n_instruments, data.n_days

    # Market direction
    total_returns = (prices[:, -1] - prices[:, 0]) / prices[:, 0]
    n_winners = np.sum(total_returns > 0)
    n_losers = np.sum(total_returns < 0)
    avg_return = np.mean(total_returns)

    # Volatility
    volatilities = np.std(returns, axis=1) * np.sqrt(252)
    min_vol = np.min(volatilities)
    max_vol = np.max(volatilities)
    avg_vol = np.mean(volatilities)

    # Autocorrelation
    lag1_autocorrs = []
    lag5_autocorrs = []
    for i in range(n_inst):
        r = returns[i]
        if len(r) > 5:
            lag1_autocorrs.append(np.corrcoef(r[:-1], r[1:])[0, 1])
            lag5_autocorrs.append(np.corrcoef(r[:-5], r[5:])[0, 1])

    lag1_autocorr = np.mean(lag1_autocorrs) if lag1_autocorrs else 0
    lag5_autocorr = np.mean(lag5_autocorrs) if lag5_autocorrs else 0

    # Cross-sectional correlations
    corr_matrix = np.corrcoef(returns)

    # Get top correlated pairs
    pairs = []
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            pairs.append((i, j, corr_matrix[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))
    top_corr_pairs = pairs[:5]

    # Average correlation (upper triangle, excluding diagonal)
    upper_triangle = corr_matrix[np.triu_indices(n_inst, k=1)]
    avg_correlation = np.mean(upper_triangle)
    max_correlation = np.max(np.abs(upper_triangle))

    return DataAnalysis(
        n_instruments=n_inst,
        n_days=n_days,
        n_winners=n_winners,  # type: ignore
        n_losers=n_losers,  # type: ignore
        avg_return=avg_return,  # type: ignore
        min_vol=min_vol,
        max_vol=max_vol,
        avg_vol=avg_vol,
        lag1_autocorr=lag1_autocorr,  # type: ignore
        lag5_autocorr=lag5_autocorr,  # type: ignore
        avg_correlation=avg_correlation,  # type: ignore
        max_correlation=max_correlation,
        top_corr_pairs=top_corr_pairs,
    )


def print_data_analysis(analysis: DataAnalysis):
    print("DATA ANALYSIS")

    print(f"\n BASIC INFO")
    print(f"  Instruments: {analysis.n_instruments}")
    print(f"  Trading Days: {analysis.n_days}")

    print(f"\n MARKET DIRECTION")
    print(f"  Winners: {analysis.n_winners}/{analysis.n_instruments}")
    print(f"  Losers: {analysis.n_losers}/{analysis.n_instruments}")
    print(f"  Avg Total Return: {analysis.avg_return*100:.2f}%")

    print(f"\n VOLATILITY (Annualized)")
    print(f"  Min: {analysis.min_vol*100:.2f}%")
    print(f"  Max: {analysis.max_vol*100:.2f}%")
    print(f"  Avg: {analysis.avg_vol*100:.2f}%")

    print(f"\n AUTOCORRELATION")
    print(f"  Lag-1: {analysis.lag1_autocorr:.4f}")
    print(f"  Lag-5: {analysis.lag5_autocorr:.4f}")

    print(f"\n CROSS-SECTIONAL")
    print(f"  Avg Correlation: {analysis.avg_correlation:.4f}")
    print(f"  Max Correlation: {analysis.max_correlation:.4f}")
    print(f"  Top Correlated Pairs:")
    for i, j, corr in analysis.top_corr_pairs[:3]:
        print(f"    ({i}, {j}): {corr:.4f}")


@dataclass
class OverfittingDiagnostics:
    """Results from overfitting analysis"""

    train_score: float
    test_score: float
    ratio: float

    train_scores_by_period: List[float]
    test_scores_by_period: List[float]

    diagnosis: str


def diagnose_overfitting(
    train_scores: List[float], test_scores: List[float]
) -> OverfittingDiagnostics:
    """
    Diagnose overfitting from train/test scores.

    Args:
        train_scores: Scores on training periods
        test_scores: Scores on test periods

    Returns:
        OverfittingDiagnostics object
    """
    avg_train = np.mean(train_scores)
    avg_test = np.mean(test_scores)

    ratio = avg_test / avg_train if avg_train != 0 else 0

    # Diagnosis
    if ratio >= 0.8:
        diagnosis = "Low overfitting"
    elif ratio >= 0.5:
        diagnosis = "Some overfitting"
    elif ratio >= 0.3:
        diagnosis = " Notable overfitting"
    else:
        diagnosis = "Heavy overfitting"

    return OverfittingDiagnostics(
        train_score=avg_train,  # type: ignore
        test_score=avg_test,  # type: ignore
        ratio=ratio,  # type: ignore
        train_scores_by_period=train_scores,
        test_scores_by_period=test_scores,
        diagnosis=diagnosis,
    )


def print_overfitting_diagnostics(diag: OverfittingDiagnostics):
    print("OVERFITTING DIAGNOSTICS")

    print(f"\n SCORES")
    print(f"  Train Score: {diag.train_score:.2f}")
    print(f"  Test Score: {diag.test_score:.2f}")
    print(f"  Ratio: {diag.ratio:.2f} ({diag.ratio*100:.0f}%)")

    print(f"\n BY PERIOD")
    print(f"  Train: {[f'{s:.1f}' for s in diag.train_scores_by_period]}")
    print(f"  Test: {[f'{s:.1f}' for s in diag.test_scores_by_period]}")

    print(f"\n DIAGNOSIS")
    print(f"  {diag.diagnosis}")


def compare_strategies(results: Dict[str, "BacktestResult"]) -> None:
    """
    Compare multiple strategy results.

    Args:
        results: Dictionary of name -> BacktestResult
    """
    print("STRATEGY COMPARISON")

    # Sort by score
    sorted_strategies = sorted(results.items(), key=lambda x: -x[1].metrics.score)

    print(f"\n{'Strategy':<25} {'Score':>10} {'Sharpe':>10} {'Win%':>8}")
    print("-" * 55)

    for name, result in sorted_strategies:
        m = result.metrics
        print(f"{name:<25} {m.score:>10.2f} {m.sharpe:>10.2f} {m.win_rate*100:>7.1f}%")

    # Best strategy
    best_name, best_result = sorted_strategies[0]
    print(f"\n Best: {best_name} (Score: {best_result.metrics.score:.2f})")


if __name__ == "__main__":
    # Test analysis tools
    print("Testing Analysis Tools")

    # Create dummy data
    from data import PriceData

    np.random.seed(42)
    n_inst, n_days = 50, 1000
    returns = np.random.randn(n_inst, n_days) * 0.02 - 0.0001  # Slight negative drift
    prices = 100 * np.cumprod(1 + returns, axis=1)

    data = PriceData(prices)

    # Test data analysis
    analysis = analyze_data(data)
    print_data_analysis(analysis)

    # Test overfitting diagnostics
    train_scores = [45.0, 52.0, 38.0]
    test_scores = [8.0, 11.0, 9.0]

    diag = diagnose_overfitting(train_scores, test_scores)
    print_overfitting_diagnostics(diag)

    print("\nAnalysis Tools Ready")
