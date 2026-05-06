"""
Config Validator — Type-safe config validation with Pydantic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum


class ConfigValidator:
    """Base validator for Hyperion configs."""

    @classmethod
    def validate(cls, config: Any) -> List[str]:
        """Validate a config object, return list of errors."""
        errors = []
        if config is None:
            return ["config is None"]
        return errors

    @classmethod
    def check_positive(cls, value: float, name: str, errors: List[str]) -> None:
        if value <= 0:
            errors.append(f"{name} must be positive, got {value}")

    @classmethod
    def check_range(cls, value: float, name: str, low: float, high: float, errors: List[str]) -> None:
        if not (low <= value <= high):
            errors.append(f"{name} must be in [{low}, {high}], got {value}")

    @classmethod
    def check_not_empty(cls, value: Any, name: str, errors: List[str]) -> None:
        if value is None or (isinstance(value, (list, str, dict)) and len(value) == 0):
            errors.append(f"{name} must not be empty")


def validate_config(config: Any) -> None:
    """Validate entire HyperionConfig. Raises ConfigError on failure."""
    from hyperion.infra.exceptions import ConfigError

    errors = []

    # Data config
    if hasattr(config, 'data'):
        d = config.data
        ConfigValidator.check_not_empty(d.start_date, "data.start_date", errors)
        ConfigValidator.check_not_empty(d.end_date, "data.end_date", errors)
        ConfigValidator.check_not_empty(d.symbols, "data.symbols", errors)
        if d.purge_days < 0:
            errors.append(f"data.purge_days must be >= 0, got {d.purge_days}")

    # Engine config
    if hasattr(config, 'engine'):
        e = config.engine
        ConfigValidator.check_positive(e.initial_capital, "engine.initial_capital", errors)
        ConfigValidator.check_range(e.commission, "engine.commission", 0, 0.01, errors)
        ConfigValidator.check_range(e.slippage, "engine.slippage", 0, 0.05, errors)

    # Risk config
    if hasattr(config, 'risk'):
        r = config.risk
        ConfigValidator.check_range(r.max_position_pct, "risk.max_position_pct", 0, 1, errors)
        ConfigValidator.check_range(r.stop_loss_pct, "risk.stop_loss_pct", 0, 0.5, errors)

    # Strategy config
    if hasattr(config, 'strategy'):
        s = config.strategy
        ConfigValidator.check_positive(s.max_positions, "strategy.max_positions", errors)
        ConfigValidator.check_range(s.train_ratio, "strategy.train_ratio", 0.1, 0.95, errors)

    if errors:
        raise ConfigError("Config validation failed:\n  " + "\n  ".join(errors))