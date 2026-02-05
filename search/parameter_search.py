"""
Parameter Search

What this does:
1. Grid search over: vol_percentile, vol_lookback, mom_lookback, r2_lookback, r2_threshold, price_threshold
2. For each combination, calculate features using data up to day 750
3. Select instruments based on filters
4. Evaluate Score on days 751-1000
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple
import itertools
import json
import os
import sys

# Get project directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


@dataclass
class SearchResult:
    """Result from evaluating one parameter combination"""

    params: Dict
    score: float
    sharpe: float
    n_positions: int
    total_pnl: float


class ParameterSearcher:
    """
    Grid search over strategy parameters.

    The strategy:
    - Short instruments with: low volatility + negative momentum + high R² + low price
    - Enter on day 751, hold until day 1000
    """

    def __init__(self, prices_path: str = None):
        if prices_path is None:
            prices_path = os.path.join(PROJECT_DIR, "prices.txt")

        df = pd.read_csv(prices_path, sep=r"\s+", header=None)
        self.prices = df.values.T  # (n_instruments, n_days)
        self.n_inst, self.n_days = self.prices.shape

        self.DLR_LIMIT = 10000
        self.COMMISSION = 0.001

        print(f"Loaded {self.n_inst} instruments for {self.n_days} days")

    def calculate_features(self, t: int, params: Dict) -> Dict[str, np.ndarray]:
        """
        Calculate features at time t using data up to t.
        """
        prcSoFar = self.prices[:, :t]
        curPrices = prcSoFar[:, -1]
        rets = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]

        features = {}

        # Volatility
        vol_lb = params.get("vol_lookback", 16)
        features["volatility"] = np.std(rets[:, -vol_lb:], axis=1)

        # Momentum
        mom_lb = params.get("mom_lookback", 706)
        if prcSoFar.shape[1] >= mom_lb:
            past_p = prcSoFar[:, -mom_lb]
            features["momentum"] = (curPrices - past_p) / past_p
        else:
            features["momentum"] = np.zeros(self.n_inst)

        # R²
        r2_lb = params.get("r2_lookback", 105)
        if prcSoFar.shape[1] >= r2_lb:
            x = np.arange(r2_lb)
            r2_vals = []
            for i in range(self.n_inst):
                y = prcSoFar[i, -r2_lb:]
                slope, intercept, r, p, se = stats.linregress(x, y)
                r2_vals.append(r**2)
            features["r2"] = np.array(r2_vals)
        else:
            features["r2"] = np.zeros(self.n_inst)

        features["price"] = curPrices

        return features

    def select_instruments(self, features: Dict, params: Dict) -> List[int]:
        """Select instruments based on filter thresholds"""
        vol = features["volatility"]
        mom = features["momentum"]
        r2 = features["r2"]
        price = features["price"]

        vol_pct = params.get("vol_percentile", 78)
        vol_thresh = np.percentile(vol, vol_pct)
        r2_thresh = params.get("r2_threshold", 0.2)
        price_thresh = params.get("price_threshold", 60)
        use_r2 = params.get("use_r2", True)
        use_price = params.get("use_price", True)

        selected = []
        for i in range(self.n_inst):
            if vol[i] >= vol_thresh:
                continue
            if mom[i] >= 0:
                continue
            if use_r2 and r2[i] <= r2_thresh:
                continue
            if use_price and price[i] >= price_thresh:
                continue
            selected.append(i)

        return selected

    def evaluate_strategy(
        self, params: Dict, start: int = 751, end: int = 1001
    ) -> SearchResult:
        """
        Evaluate strategy on period [start, end).
        """
        # Calculate features using data up to start day
        features = self.calculate_features(start, params)
        selected = self.select_instruments(features, params)

        if len(selected) == 0:
            return SearchResult(
                params=params, score=0, sharpe=0, n_positions=0, total_pnl=0
            )

        # Simulate trading
        cash = 0.0
        curPos = np.zeros(self.n_inst)
        value = 0.0
        daily_pnl = []

        for t in range(start, end):
            curPrices = self.prices[:, t]

            # Enter positions on first day only
            if t == start:
                max_shares = np.array(
                    [int(self.DLR_LIMIT / curPrices[i]) for i in range(self.n_inst)]
                )
                newPos = np.zeros(self.n_inst)
                for i in selected:
                    newPos[i] = -max_shares[i]  # Short

                deltaPos = newPos - curPos
                turnover = np.sum(np.abs(deltaPos) * curPrices)
                comm = turnover * self.COMMISSION
                cash -= np.dot(curPrices, deltaPos) + comm
                curPos = newPos.copy()

            posValue = np.dot(curPos, curPrices)
            todayPL = cash + posValue - value
            daily_pnl.append(todayPL)
            value = cash + posValue

        pnl = np.array(daily_pnl)
        if len(pnl) == 0 or np.std(pnl) == 0:
            return SearchResult(
                params=params, score=0, sharpe=0, n_positions=len(selected), total_pnl=0
            )

        score = np.mean(pnl) - 0.1 * np.std(pnl)
        sharpe = np.sqrt(252) * np.mean(pnl) / np.std(pnl)

        return SearchResult(
            params=params,
            score=score,
            sharpe=sharpe,
            n_positions=len(selected),
            total_pnl=value,
        )

    def grid_search(
        self, param_grid: Dict, start: int = 751, end: int = 1001, verbose: bool = True
    ) -> List[SearchResult]:
        """Run grid search over all parameter combinations"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        total = len(combinations)

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            result = self.evaluate_strategy(params, start, end)
            results.append(result)

            if verbose and (idx + 1) % 100 == 0:
                print(f"Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")

        results.sort(key=lambda x: -x.score)
        return results

    def search(self) -> List[SearchResult]:
        """
        Search using paramter ranges
        """
        param_grid = {
            "vol_percentile": [70, 75, 78, 80, 82, 85],
            "vol_lookback": [14, 16, 18, 20],
            "mom_lookback": [680, 700, 706, 710, 720],
            "r2_lookback": [90, 95, 100, 105, 110, 115, 120],
            "r2_threshold": [0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
            "price_threshold": [55, 58, 60, 62, 65, 100],  # 100 = no filter
            "use_r2": [True],
            "use_price": [True],
        }

        print(
            f"\nSearching {np.prod([len(v) for v in param_grid.values()])} combinations"
        )
        results = self.grid_search(param_grid)

        return results


def main():
    print("PARAMETER SEARCH")

    searcher = ParameterSearcher()
    results = searcher.search()

    # Show top 10
    print(" TOP 10 PARAMETER COMBINATIONS")

    for i, r in enumerate(results[:10]):
        print(f"\n{i+1}. Score={r.score:.2f}, Sharpe={r.sharpe:.2f}, n={r.n_positions}")
        print(
            f"   vol_pct={r.params['vol_percentile']}, vol_lb={r.params['vol_lookback']}"
        )
        print(f"   mom_lb={r.params['mom_lookback']}")
        print(
            f"   r2_lb={r.params['r2_lookback']}, r2_thresh={r.params['r2_threshold']}"
        )
        print(f"   price_thresh={r.params['price_threshold']}")

    # Save best params
    output_dir = os.path.join(PROJECT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    best = results[0]
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(
            {
                "params": best.params,
                "score": best.score,
                "sharpe": best.sharpe,
                "n_positions": best.n_positions,
            },
            f,
            indent=2,
        )

    print(f"\n\nBest parameters saved to outputs/best_params.json")
    print(f"\n BEST SCORE: {best.score:.2f}")

    return results


if __name__ == "__main__":
    main()
