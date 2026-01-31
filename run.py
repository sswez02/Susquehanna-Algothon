"""
Entry point for running experiments.

Commands:
    python run.py explore     - Data exploration
    python run.py baseline    - ALL strategies comparison
    python run.py cv          - Cross-validation
    python run.py full        - Everything
"""

import sys
import numpy as np

from core import (
    Config,
    get_config,
    load_data,
    PriceData,
    compute_metrics,
    compute_score,
)
from validation.backtest import BacktestEngine
from strategies.strategies import StrategyFactory
from validation.cv import PurgedKFold, WalkForwardCV
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

    print("\n KEY INSIGHTS")
    insights = []

    if analysis.avg_return < 0:
        insights.append("Market is BEARISH overall")
    else:
        insights.append("Market is BULLISH overall")

    if abs(analysis.lag1_autocorr) < 0.05:
        insights.append("Near-zero autocorrelation -> momentum won't work!")
        insights.append("Focus on relative (cross-sectional) patterns")

    if analysis.max_correlation > 0.5:
        insights.append(
            f"High correlations ({analysis.max_correlation:.2f}) -> pairs trading viable"
        )

    for insight in insights:
        print(f"  - {insight}")


def run_baseline(data: PriceData, config: Config):
    """Run all strategy comparison"""
    print("ALL STRATEGIES COMPARISON")

    factory = StrategyFactory(data.prices)

    # Define all strategies
    strategies = {
        "Zero (no trades)": factory.zero(),
        "Buy & Hold": factory.buy_and_hold(scale=0.3),
        "Momentum 60d": factory.momentum(lookback=60, scale=0.3),
        "Momentum 300d": factory.momentum(lookback=300, scale=0.3),
        "Mean Reversion 20d": factory.mean_reversion(lookback=20, scale=0.3),
        "Vol-Scaled": factory.volatility_scaled(momentum_lookback=300, scale=0.3),
        "Cross-Sectional Mom": factory.cross_sectional_momentum(lookback=60, scale=0.3),
        "Combined": factory.combined(),
        "CS Mean Reversion": factory.cross_sectional_mean_reversion(
            lookback=10, n_long=5, n_short=5, scale=0.15
        ),
        "Short Bias": factory.short_bias(scale=0.1),
        "Pairs (27,38)": factory.pairs_trading(pair1=27, pair2=38, scale=0.3),
        "Low Volatility": factory.low_volatility(scale=0.1),
        "Conservative Mom": factory.conservative_momentum(scale=0.05),
        "Combined V2": factory.combined_v2(),
        "Short Bias (0.15)": factory.short_bias(scale=0.15),
        "Short Bias (0.20)": factory.short_bias(scale=0.20),
        "Low-Vol Short": factory.low_vol_short_bias(scale=0.20, n_short=15),
        "Selective Short": factory.selective_short(scale=0.2),
    }

    # Run backtests
    engine = BacktestEngine(data.prices, config)

    print(f"\nBacktest period: days {config.data.test_start}-{config.data.test_end}")
    print(f"\n{'Strategy':<22} {'Score':>8} {'Sharpe':>8} {'Total':>10} {'StdPL':>8}")
    print("-" * 62)

    results = {}
    for name, strategy in strategies.items():
        result = engine.run(strategy, config.data.test_start, config.data.test_end)
        results[name] = result
        m = result.metrics
        print(
            f"{name:<22} {m.score:>8.2f} {m.sharpe:>8.2f} ${m.total_pnl:>8.0f} {m.std_pnl:>8.1f}"
        )

    # Summary
    best_name = max(results.keys(), key=lambda x: results[x].metrics.score)
    best = results[best_name]

    print(f" BEST: {best_name}")
    print(f"   Score: {best.metrics.score:.2f}")
    print(f"   Sharpe: {best.metrics.sharpe:.2f}")
    print(f"   Total P&L: ${best.metrics.total_pnl:.0f}")

    # Show strategies that made money
    print("\n Strategies that made money:")
    for name, result in sorted(results.items(), key=lambda x: -x[1].metrics.total_pnl):
        if result.metrics.total_pnl > 0:
            print(f"   {name}: ${result.metrics.total_pnl:.0f}")

    return results


def run_tune(data: PriceData, config: Config):
    """Parameter tuning"""
    print("\n" + "=" * 70)
    print("PARAMETER TUNING - SHORT BIAS")
    print("=" * 70)

    factory = StrategyFactory(data.prices)
    engine = BacktestEngine(data.prices, config)

    print(f"\n{'Scale':<10} {'Score':>10} {'Sharpe':>10} {'Total':>12}")
    print("-" * 45)

    for scale in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        strategy = factory.short_bias(scale=scale)
        result = engine.run(strategy, config.data.test_start, config.data.test_end)
        m = result.metrics
        print(
            f"{scale:<10.2f} {m.score:>10.2f} {m.sharpe:>10.2f} ${m.total_pnl:>10.0f}"
        )


def run_cv_analysis(data: PriceData, config: Config):
    """Run cross-validation analysis"""
    print("CROSS-VALIDATION ANALYSIS")

    factory = StrategyFactory(data.prices)
    engine = BacktestEngine(data.prices, config)

    # Test the best new strategy
    strategy = factory.cross_sectional_mean_reversion(
        lookback=10, n_long=5, n_short=5, scale=0.15
    )

    print("\n1. PURGED K-FOLD (CS Mean Reversion)")
    cv = PurgedKFold(n_splits=5, purge_days=10, embargo_days=5)

    test_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(data.n_days)):
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        if test_start >= 30:
            result = engine.run(strategy, test_start, test_end)
            score = result.metrics.score
        else:
            score = 0
        print(f"  Fold {fold}: Days [{test_start}-{test_end}], Score: {score:.2f}")
        test_scores.append(score)

    print("\n2. WALK-FORWARD CV")
    cv_wf = WalkForwardCV(n_splits=5, train_size=500, test_size=100, expanding=True)

    wf_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv_wf.split(data.n_days)):
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        if test_start < data.n_days - 10:
            result = engine.run(strategy, test_start, min(test_end, data.n_days - 1))
            score = result.metrics.score
        else:
            score = 0
        print(f"  Fold {fold}: Test [{test_start}-{test_end}], Score: {score:.2f}")
        wf_scores.append(score)

    print("\nCV SUMMARY")
    print(
        f"  Purged K-Fold: Mean={np.mean(test_scores):.2f}, Std={np.std(test_scores):.2f}"
    )
    print(f"  Walk-Forward: Mean={np.mean(wf_scores):.2f}, Std={np.std(wf_scores):.2f}")


def run_full_experiment(data: PriceData, config: Config):
    """Run full experiment"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "     FULL EXPERIMENT".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    run_exploration(data, config)
    results = run_baseline(data, config)
    run_cv_analysis(data, config)

    # Overfitting check
    print("OVERFITTING ANALYSIS")

    factory = StrategyFactory(data.prices)
    engine = BacktestEngine(data.prices, config)
    strategy = factory.cross_sectional_mean_reversion(
        lookback=10, n_long=5, n_short=5, scale=0.15
    )

    train_scores = []
    for start, end in [(100, 300), (300, 500), (500, 750)]:
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
    best_name = max(results.keys(), key=lambda x: results[x].metrics.score)
    best_score = results[best_name].metrics.score

    print(" SUMMARY")
    print(
        f"""
  Best Strategy: {best_name}
  Best Score: {best_score:.2f}
  Target: 50+
  Gap: {50 - best_score:.2f}
"""
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <command>")
        print("Commands: explore, baseline, cv, full")
        return

    command = sys.argv[1].lower()
    config = get_config()

    print("Loading data...")
    try:
        data = load_data(config.data.data_file)
        print(f"Loaded: {data.n_instruments} instruments, {data.n_days} days")
    except FileNotFoundError:
        print(f"ERROR: {config.data.data_file} not found")
        return

    if command == "explore":
        run_exploration(data, config)
    elif command == "baseline":
        run_baseline(data, config)
    elif command == "cv":
        run_cv_analysis(data, config)
    elif command == "full":
        run_full_experiment(data, config)
    elif command == "tune":
        run_tune(data, config)
    else:
        print(f"Unknown: {command}")


if __name__ == "__main__":
    main()
