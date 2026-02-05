"""
Analysis and Diagnostics

Tools for understanding:
1. Data characteristics
2. Strategy performance
3. Overfitting detection
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from core.data import PriceData


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
    returns = data.returns[:, 1:]  # Skip first day

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
    for i in range(n_inst):
        r = returns[i]
        if len(r) > 1:
            corr = np.corrcoef(r[:-1], r[1:])[0, 1]
            if not np.isnan(corr):
                lag1_autocorrs.append(corr)
    lag1_autocorr = np.mean(lag1_autocorrs) if lag1_autocorrs else 0

    # Cross-sectional correlations
    corr_matrix = np.corrcoef(returns)

    # Get top correlated pairs
    pairs = []
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            if not np.isnan(corr_matrix[i, j]):
                pairs.append((i, j, corr_matrix[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))
    top_corr_pairs = pairs[:5]

    # Average correlation
    upper_triangle = corr_matrix[np.triu_indices(n_inst, k=1)]
    upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
    avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
    max_correlation = np.max(np.abs(upper_triangle)) if len(upper_triangle) > 0 else 0

    return DataAnalysis(
        n_instruments=n_inst,
        n_days=n_days,
        n_winners=n_winners,
        n_losers=n_losers,
        avg_return=avg_return,
        min_vol=min_vol,
        max_vol=max_vol,
        avg_vol=avg_vol,
        lag1_autocorr=lag1_autocorr,
        avg_correlation=avg_correlation,
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

    if abs(analysis.lag1_autocorr) < 0.05:
        print(" Near-zero: Returns are unpredictable")

    print(f"\n CROSS-SECTIONAL")
    print(f"  Avg Correlation: {analysis.avg_correlation:.4f}")
    print(f"  Max Correlation: {analysis.max_correlation:.4f}")


@dataclass
class OverfittingDiagnostics:
    """Results from overfitting analysis"""

    train_score: float
    test_score: float
    ratio: float
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
        diagnosis = "GOOD: Low overfitting"
    elif ratio >= 0.5:
        diagnosis = "MODERATE: Some overfitting"
    elif ratio >= 0.3:
        diagnosis = "SIGNIFICANT: Notable overfitting"
    else:
        diagnosis = "SEVERE: Heavy overfitting"

    return OverfittingDiagnostics(
        train_score=avg_train,
        test_score=avg_test,
        ratio=ratio,
        diagnosis=diagnosis,
    )


def print_overfitting_diagnostics(diag: OverfittingDiagnostics):
    print("OVERFITTING DIAGNOSTICS")

    print(f"\n SCORES")
    print(f"  Train Score: {diag.train_score:.2f}")
    print(f"  Test Score: {diag.test_score:.2f}")
    print(f"  Ratio: {diag.ratio:.2f} ({diag.ratio*100:.0f}%)")

    print(f"\n DIAGNOSIS")
    print(f"  {diag.diagnosis}")
