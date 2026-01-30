"""
Entry point for running experiments.

Usage:
    python run.py explore      # Data exploration
    python run.py baseline     # Run baseline strategies
    python run.py cv           # Cross-validation analysis
    python run.py full         # Full experiment
    python run.py help         # Show help
"""

import sys
import numpy as np
import os

from config import Config, get_config
from data import load_data, PriceData
from backtest import BacktestEngine
from strategies import StrategyFactory, get_all_strategies
from validation import PurgedKFold, WalkForwardCV, cross_validate
from metrics import compute_metrics, compute_score
from analysis import (
    analyze_data,
    print_data_analysis,
    diagnose_overfitting,
    print_overfitting_diagnostics,
)


def run_exploration(data: PriceData, config: Config):
    """Run data exploration"""
    print("DATA EXPLORATION")

    analysis = analyze_data(data)
    print_data_analysis(analysis)

    # Additional insights
    print(" KEY INSIGHTS")

    insights = []

    if analysis.avg_return < 0:
        insights.append("Market is BEARISH overall")
    else:
        insights.append("Market is BULLISH overall")

    if abs(analysis.lag1_autocorr) < 0.05:
        insights.append("Near-zero autocorrelation")
        insights.append("Focus on relative (cross-sectional) patterns")

    if analysis.max_correlation > 0.5:
        insights.append(
            f"High correlations exist ({analysis.max_correlation:.2f}) - pairs trading viable"
        )

    for insight in insights:
        print(f"  {insight}")


def run_baseline(data: PriceData, config: Config):
    """Run baseline strategy comparison"""
    print(" BASELINE STRATEGIES")

    # Get all strategies
    factory = StrategyFactory(data.prices)
    strategies = {
        "Zero (no trades)": factory.zero(),
        "Buy & Hold": factory.buy_and_hold(scale=0.3),
        "Momentum 60d": factory.momentum(lookback=60, scale=0.3),
        "Momentum 120d": factory.momentum(lookback=120, scale=0.3),
        "Momentum 300d": factory.momentum(lookback=300, scale=0.3),
        "Mean Reversion 20d": factory.mean_reversion(lookback=20, scale=0.3),
        "Vol-Scaled": factory.volatility_scaled(momentum_lookback=300, scale=0.3),
        "Cross-Sectional": factory.cross_sectional_momentum(lookback=60, scale=0.3),
        "Combined": factory.combined(),
    }

    # Run backtests
    engine = BacktestEngine(data.prices, config)

    results = {}
    print(
        f"\nRunning backtests (days {config.data.test_start}-{config.data.test_end})..."
    )

    for name, strategy in strategies.items():
        result = engine.run(strategy, config.data.test_start, config.data.test_end)
        results[name] = result

        m = result.metrics
        print(
            f"{name:<25} Score: {m.score:>8.2f}  Sharpe: {m.sharpe:>6.2f}  "
            f"Total: ${m.total_pnl:>8.0f}"
        )

    # Best strategy
    best_name = max(results.keys(), key=lambda x: results[x].metrics.score)
    best_result = results[best_name]

    print(f"BEST STRATEGY: {best_name}")
    print(f"   Score: {best_result.metrics.score:.2f}")
    print(f"   Sharpe: {best_result.metrics.sharpe:.2f}")

    return results


def run_cv_analysis(data: PriceData, config: Config):
    """Run cross-validation analysis"""
    print("CROSS-VALIDATION ANALYSIS")

    factory = StrategyFactory(data.prices)
    engine = BacktestEngine(data.prices, config)

    # Test with momentum strategy
    strategy = factory.momentum(lookback=300, scale=0.3)

    # Purged K-Fold
    print("\n1. PURGED K-FOLD (5 splits)")

    cv = PurgedKFold(n_splits=5, purge_days=10, embargo_days=5)

    train_scores = []
    test_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(data.n_days)):
        # Test on this fold
        test_start = test_idx.min()
        test_end = test_idx.max() + 1

        if test_start >= 300:  # Need lookback for momentum
            result = engine.run(strategy, test_start, test_end)
            score = result.metrics.score
        else:
            score = 0

        print(f"  Fold {fold}: Days [{test_start}-{test_end}], Score: {score:.2f}")
        test_scores.append(score)

    # Walk-forward
    print("\n2. WALK-FORWARD CV (expanding)")

    cv_wf = WalkForwardCV(n_splits=5, train_size=500, test_size=100, expanding=True)

    wf_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv_wf.split(data.n_days)):
        test_start = test_idx.min()
        test_end = test_idx.max() + 1

        if test_start < data.n_days - 10:
            result = engine.run(strategy, test_start, min(test_end, data.n_days - 1))
            score = result.metrics.score
        else:
            score = 0

        print(f"  Fold {fold}: Test [{test_start}-{test_end}], Score: {score:.2f}")
        wf_scores.append(score)

    # Summary
    print("CV SUMMARY")
    print(
        f"  Purged K-Fold: Mean={np.mean(test_scores):.2f}, Std={np.std(test_scores):.2f}"
    )
    print(f"  Walk-Forward: Mean={np.mean(wf_scores):.2f}, Std={np.std(wf_scores):.2f}")


def run_full_experiment(data: PriceData, config: Config):
    """Run full experiment with all analyses"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "     FULL EXPERIMENT".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # 1. Data exploration
    run_exploration(data, config)

    # 2. Baseline strategies
    results = run_baseline(data, config)

    # 3. CV analysis
    run_cv_analysis(data, config)

    # 4. Overfitting check
    print("OVERFITTING ANALYSIS")

    # Simulate train vs test scores (using different periods)
    factory = StrategyFactory(data.prices)
    engine = BacktestEngine(data.prices, config)
    strategy = factory.momentum(lookback=300, scale=0.3)

    train_scores = []
    for start, end in [(300, 450), (450, 600), (600, 750)]:
        result = engine.run(strategy, start, end)
        train_scores.append(result.metrics.score)

    test_scores = []
    for start, end in [(751, 875), (875, 1001)]:
        if end <= data.n_days:
            result = engine.run(strategy, start, end)
            test_scores.append(result.metrics.score)

    if test_scores:
        diag = diagnose_overfitting(train_scores, test_scores)
        print_overfitting_diagnostics(diag)

    # Summary
    print(" EXPERIMENT SUMMARY")

    best_name = max(results.keys(), key=lambda x: results[x].metrics.score)
    best_score = results[best_name].metrics.score

    print(
        f"""
  Data: {data.n_instruments} instruments, {data.n_days} days
  
  Best Baseline Strategy: {best_name}
  Best Score: {best_score:.2f}
  
  Target Score: 50+
  Gap to Close: {50 - best_score:.2f}
  
  Next Steps:
  1. Implement ML model with proper regularization
  2. Use purged CV for model selection
  3. Focus on feature selection to reduce overfitting
"""
    )


def show_help():
    """Show help message"""
    print(
        """
Algorithmic Trading Competition - Runner

Usage:
    python run.py <command>

Commands:
    explore     Run data exploration
    baseline    Run baseline strategy comparison
    cv          Run cross-validation analysis
    full        Run full experiment
    help        Show this help message

Examples:
    python run.py explore
    python run.py baseline
    python run.py full
"""
    )


def main():
    """Main entry point"""
    # Parse command
    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1].lower()

    if command == "help":
        show_help()
        return

    # Load configuration
    config = get_config()

    # Load data
    print("Loading data...")
    try:
        data = load_data(config.data.data_file)
        print(f"Loaded: {data.n_instruments} instruments, {data.n_days} days")
    except FileNotFoundError:
        print(f"ERROR: {config.data.data_file} not found")
        print("Please ensure the data file is in the current directory.")
        return

    # Run command
    if command == "explore":
        run_exploration(data, config)
    elif command == "baseline":
        run_baseline(data, config)
    elif command == "cv":
        run_cv_analysis(data, config)
    elif command == "full":
        run_full_experiment(data, config)
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
