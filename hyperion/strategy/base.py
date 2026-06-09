"""
Hyperion Pro — 策略基类与 Alpha 框架
华尔街级别多策略框架：趋势跟踪、均值回归、动量突破、成交量异动、多因子Alpha
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class StrategySignal:
    timestamp: str
    code: str
    name: str
    signal_type: SignalType
    price: float
    stop_loss: float
    take_profit: float
    position_pct: float
    confidence: float
    strategy_name: str
    reason: str
    indicators: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp, "code": self.code, "name": self.name,
            "signal_type": self.signal_type.value, "price": self.price,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit,
            "position_pct": self.position_pct, "confidence": self.confidence,
            "strategy": self.strategy_name, "reason": self.reason,
        }


@dataclass
class StrategyResult:
    code: str
    strategy: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    max_drawdown_pct: float
    exit_reason: str


class BaseStrategy(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.signals: List[StrategySignal] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        ...

    @abstractmethod
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        ...


STRATEGY_REGISTRY: Dict[str, type] = {}

def register_strategy(cls):
    STRATEGY_REGISTRY[cls.__name__] = cls
    return cls

def get_strategy(name: str) -> Optional[BaseStrategy]:
    cls = STRATEGY_REGISTRY.get(name)
    return cls() if cls else None

def list_strategies() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())
