"""
Hyperion Custom Exceptions
===========================
Typed exceptions for clear error handling throughout the system.
"""
from __future__ import annotations

from typing import Optional


class HyperionError(Exception):
    """Base exception for all Hyperion errors."""
    pass


class ConfigError(HyperionError):
    """Configuration validation errors."""
    pass


class DataError(HyperionError):
    """Data fetching/processing errors."""
    pass


class DataNotFoundError(DataError):
    """Data not found for a symbol/date range."""
    def __init__(self, symbol: str, start: str = "", end: str = ""):
        self.symbol = symbol
        self.start = start
        self.end = end
        super().__init__(f"Data not found: {symbol} [{start}:{end}]")


class InvalidFactorError(HyperionError):
    """Invalid factor specification."""
    pass


class ModelError(HyperionError):
    """Model training/inference errors."""
    pass


class ModelNotFittedError(ModelError):
    """Model used before fitting."""
    pass


class BacktestError(HyperionError):
    """Backtest execution errors."""
    pass


class OrderError(HyperionError):
    """Order placement/execution errors."""
    pass


class RiskLimitExceededError(HyperionError):
    """Risk management limit violation."""
    def __init__(self, limit_name: str, current: float, limit: float):
        self.limit_name = limit_name
        self.current = current
        self.limit = limit
        super().__init__(f"{limit_name}: {current:.4f} exceeds limit {limit:.4f}")


class DataPipelineError(HyperionError):
    """Data pipeline processing errors."""
    pass


class RetryableError(HyperionError):
    """Error that can be retried."""
    pass


class NetworkTimeoutError(RetryableError):
    """Network timeout — can be retried."""
    pass


class PortfolioError(HyperionError):
    """Portfolio optimization/construction errors."""
    pass