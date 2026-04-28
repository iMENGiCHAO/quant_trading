"""
Hyperion Strategy Layer (Layer 3)
==================================
融合:
  VnPy CTA策略模板 → 回测/实盘统一API
  Jesse 极简设计 → 最少代码写策略
  FinRL → Gym TradingEnv
  Qlib → ML预测策略
"""
from hyperion.strategy.base import BaseStrategy
from hyperion.strategy.ml_strategy import MLMultiFactorStrategy

__all__ = ["BaseStrategy", "MLMultiFactorStrategy"]
