"""
Hyperion Pro — Strategy Layer
多策略 Alpha 框架：趋势跟踪、均值回归、动量突破、成交量异动、多因子Alpha
"""
from hyperion.strategy.base import BaseStrategy, StrategySignal, SignalType, register_strategy, list_strategies, get_strategy
from hyperion.strategy.strategies import (
    TrendFollowingStrategy, MeanReversionStrategy,
    MomentumBreakoutStrategy, VolumeAnomalyStrategy, MultiFactorAlphaStrategy
)

__all__ = ["BaseStrategy", "StrategySignal", "SignalType", "register_strategy", "list_strategies", "get_strategy",
           "TrendFollowingStrategy", "MeanReversionStrategy", "MomentumBreakoutStrategy",
           "VolumeAnomalyStrategy", "MultiFactorAlphaStrategy"]
