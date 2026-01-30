"""
Configuration and Constants

All configurable parameters for the trading system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Data-related configuration"""

    data_file: str = "prices.txt"
    n_instruments: int = 50
    n_days: int = 1250

    # Period splits
    train_start: int = 0
    train_end: int = 750
    test_start: int = 751
    test_end: int = 1001  # Exclusive


@dataclass
class TradingConfig:
    """Trading rules and constraints"""

    dlr_limit: float = 10000.0  # Max dollar position per instrument
    commission_rate: float = 0.001  # 0.1% commission


@dataclass
class ValidationConfig:
    """Cross-validation configuration"""

    n_splits: int = 5
    purge_days: int = 10  # Days to remove before test set
    embargo_days: int = 5  # Days to skip after test set
    min_train_size: int = 200  # Minimum training samples


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    lookback_windows: List[int] = field(
        default_factory=lambda: [5, 10, 20, 60, 120, 250]
    )
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 60])


@dataclass
class ModelConfig:
    """Model configuration"""

    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class Config:
    """Master configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Paths
    output_dir: str = "outputs"
    model_dir: str = "models"

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


# Global default configuration
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration"""
    return DEFAULT_CONFIG
