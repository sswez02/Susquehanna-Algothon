"""Configuration and constants"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration"""

    data_file: str = "prices.txt"
    n_instruments: int = 50
    n_days: int = 1250
    train_start: int = 0

    # Period splits
    train_end: int = 750
    test_start: int = 751
    test_end: int = 1001


@dataclass
class TradingConfig:
    """Trading rules and constraints"""

    dlr_limit: float = 10000.0  # Max dollar position per instrument
    commission_rate: float = 0.001  # 0.1% commission


@dataclass
class Config:
    """Master configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    seed: int = 42  # Random seed for reproducibility


_config = Config()


def get_config() -> Config:
    return _config
