"""
Cross-validation for time series
"""

import numpy as np
from typing import Generator, Tuple


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

            yield indices[train_mask], indices[test_start:test_end]


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation.

    The most realistic CV for trading:
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
                train_start = max(0, test_start - self.purge_days - self.train_size)

            train_end = test_start - self.purge_days

            yield indices[train_start:train_end], indices[test_start:test_end]


def cross_validate(cv, evaluate_func, n_samples: int, verbose: bool = True) -> dict:
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

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(n_samples)):
        score = evaluate_func(train_idx, test_idx)
        scores.append(score)

        if verbose:
            print(
                f"Fold {fold_idx}: Train={len(train_idx)}, "
                f"Test={len(test_idx)}, Score={score:.2f}"
            )

    scores = np.array(scores)

    return {
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
    }
