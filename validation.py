"""
Cross-Validation for Time Series

Implements proper cross-validation for financial time series:

1. PURGED K-FOLD: Removes samples too close to test set
2. WALK-FORWARD: Always train on past, test on future
3. COMBINATORIAL: Tests all combinations (for small datasets)

Why regular k-fold fails for time series:
- Assumes i.i.d. data (time series is NOT i.i.d.)
- Information leakage from future to past
- Overlapping labels cause optimistic estimates

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
"""

import numpy as np
from typing import Generator, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from config import Config, get_config


@dataclass
class CVFold:
    """Information about a CV fold"""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Key features:
    1. PURGE: Remove training samples within `purge_days` of test set
    2. EMBARGO: Skip `embargo_days` after test set before including in train

    This prevents information leakage from:
    - Overlapping labels (e.g., 10-day forward returns)
    - Serial correlation in features

    Example:
    ```python
    cv = PurgedKFold(n_splits=5, purge_days=10, embargo_days=5)

    for train_idx, test_idx in cv.split(1000):
        model.fit(X[train_idx], y[train_idx])
        score = model.evaluate(X[test_idx], y[test_idx])
    ```
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 10, embargo_days: int = 5):
        """
        Args:
            n_splits: Number of folds
            purge_days: Days to remove BEFORE test set
            embargo_days: Days to remove AFTER test set
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            n_samples: Total number of samples

        Yields:
            (train_indices, test_indices) tuples
        """
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            test_idx = indices[test_start:test_end]

            # Build training set (excluding purge and embargo zones)
            train_mask = np.ones(n_samples, dtype=bool)

            # Exclude test set
            train_mask[test_start:test_end] = False

            # Exclude purge zone (before test)
            purge_start = max(0, test_start - self.purge_days)
            train_mask[purge_start:test_start] = False

            # Exclude embargo zone (after test)
            embargo_end = min(n_samples, test_end + self.embargo_days)
            train_mask[test_end:embargo_end] = False

            train_idx = indices[train_mask]

            yield train_idx, test_idx

    def get_folds(self, n_samples: int) -> List[CVFold]:
        """Get list of CVFold objects"""
        folds = []
        for i, (train_idx, test_idx) in enumerate(self.split(n_samples)):
            folds.append(
                CVFold(fold_idx=i, train_indices=train_idx, test_indices=test_idx)
            )
        return folds


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation.

    - Always train on past data
    - Always test on future data
    - Simulates actual trading scenario

    Two modes:
    - EXPANDING: Training window grows (anchored start)
    - ROLLING: Training window is fixed size (rolls forward)

    Example:
    ```
    Expanding:
      Fold 1: Train [0:500],   Test [510:610]
      Fold 2: Train [0:610],   Test [620:720]
      Fold 3: Train [0:720],   Test [730:830]

    Rolling:
      Fold 1: Train [0:500],   Test [510:610]
      Fold 2: Train [100:600], Test [610:710]
      Fold 3: Train [200:700], Test [710:810]
    ```
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 500,
        test_size: int = 100,
        purge_days: int = 10,
        expanding: bool = True,
    ):
        """
        Args:
            n_splits: Number of folds
            train_size: Initial (or fixed) training window size
            test_size: Size of each test window
            purge_days: Gap between train and test
            expanding: If True, training expands; if False, it rolls
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_days = purge_days
        self.expanding = expanding

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices"""
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # Test window
            test_start = self.train_size + self.purge_days + i * self.test_size
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            # Training window
            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.purge_days - self.train_size
                train_start = max(0, train_start)

            train_end = test_start - self.purge_days

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_folds(self, n_samples: int) -> List[CVFold]:
        """Get list of CVFold objects"""
        folds = []
        for i, (train_idx, test_idx) in enumerate(self.split(n_samples)):
            folds.append(
                CVFold(fold_idx=i, train_indices=train_idx, test_indices=test_idx)
            )
        return folds


class TimeSeriesSplit:
    """
    Simple time series split (no purging).

    Just ensures training is always before testing.
    Use PurgedKFold or WalkForwardCV for production.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices"""
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = (i + 2) * fold_size if i < self.n_splits - 1 else n_samples

            yield indices[:train_end], indices[test_start:test_end]


def cross_validate(
    cv, evaluate_func, n_samples: int, verbose: bool = True
) -> Dict[str, Any]:
    """
    Run cross-validation with any CV splitter.

    Args:
        cv: CV splitter with split() method
        evaluate_func: Function(train_idx, test_idx) -> score
        n_samples: Total number of samples
        verbose: Whether to print progress

    Returns:
        Dictionary with scores and statistics
    """
    scores = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(n_samples)):
        score = evaluate_func(train_idx, test_idx)
        scores.append(score)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "score": score,
            }
        )

        if verbose:
            print(
                f"Fold {fold_idx}: Train={len(train_idx)}, "
                f"Test={len(test_idx)}, Score={score:.2f}"
            )

    scores = np.array(scores)

    results = {
        "fold_results": fold_results,
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
    }

    if verbose:
        print(f"\nCV Summary:")
        print(f"  Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
        print(f"  Range: [{results['min_score']:.2f}, {results['max_score']:.2f}]")

    return results


if __name__ == "__main__":
    # Test CV splitters
    print("Testing Cross-Validation Splitters")

    n_samples = 1000

    # Test PurgedKFold
    print("\n1. PURGED K-FOLD (5 splits, 10 purge, 5 embargo)")
    cv = PurgedKFold(n_splits=5, purge_days=10, embargo_days=5)

    for fold, (train, test) in enumerate(cv.split(n_samples)):
        gap = test.min() - train.max() if len(train) > 0 else 0
        print(
            f"  Fold {fold}: Train={len(train):4d}, Test={len(test):3d}, Gap={gap:2d}"
        )

    # Test WalkForwardCV
    print("\n2. WALK-FORWARD CV (expanding)")
    cv_wf = WalkForwardCV(n_splits=5, train_size=500, test_size=100, expanding=True)

    for fold, (train, test) in enumerate(cv_wf.split(n_samples)):
        print(
            f"  Fold {fold}: Train=[{train.min():3d}-{train.max():3d}], "
            f"Test=[{test.min():3d}-{test.max():3d}]"
        )

    # Test WalkForwardCV rolling
    print("\n3. WALK-FORWARD CV (rolling)")
    cv_wf_roll = WalkForwardCV(
        n_splits=5, train_size=300, test_size=100, expanding=False
    )

    for fold, (train, test) in enumerate(cv_wf_roll.split(n_samples)):
        print(
            f"  Fold {fold}: Train=[{train.min():3d}-{train.max():3d}], "
            f"Test=[{test.min():3d}-{test.max():3d}]"
        )

    print("CV Splitters Ready")
