"""
风控管理器 (VnPy RiskManager 设计)
====================================
- 单标的最大仓位
- 行业集中度限制
- 日内最大亏损
- 止损/止盈
- 订单流控
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """风控参数"""
    max_position_pct: float = 0.10      # 单一标的最大仓位10%
    max_sector_pct: float = 0.30        # 单一行业最大仓位30%
    max_total_exposure: float = 0.95    # 最大总仓位95%
    stop_loss_pct: float = 0.05         # 单笔止损5%
    daily_loss_limit: float = 0.03      # 日内最大亏损3%
    max_daily_trades: int = 50          # 日最大交易次数
    min_holding_days: int = 1           # 最短持有天数(T+1)


class RiskManager:
    """风控管理器
    
    Usage:
        rm = RiskManager(RiskLimits())
        if rm.check_order(order):
            execute(order)
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._initial_capital = 0.0
    
    def set_capital(self, capital: float):
        self._initial_capital = capital
    
    def check_order(self, symbol: str, direction: str,
                    quantity: int, price: float,
                    current_positions: Dict[str, dict],
                    sector_map: Optional[Dict[str, str]] = None) -> tuple:
        """检查订单是否通过风控
        
        Returns:
            (approved: bool, reason: str)
        """
        # 1. 日内亏损限制
        if abs(self._daily_pnl) > self.limits.daily_loss_limit * self._initial_capital:
            return False, f"每日亏损限制: {abs(self._daily_pnl):,.0f}"
        
        # 2. 日交易次数
        if self._daily_trades >= self.limits.max_daily_trades:
            return False, "日交易次数达到上限"
        
        # 3. 单标仓位检查
        order_value = quantity * price
        if order_value > self.limits.max_position_pct * self._initial_capital:
            return False, f"超过单标的最大仓位{self.limits.max_position_pct*100:.0f}%"
        
        # 4. 总仓位检查
        total_position = sum(p.get("market_value", 0) for p in current_positions.values())
        if direction == "BUY":
            new_exposure = (total_position + order_value) / self._initial_capital
            if new_exposure > self.limits.max_total_exposure:
                return False, f"总仓位超过限制{self.limits.max_total_exposure*100:.0f}%"
        
        return True, "OK"
    
    def update_daily_pnl(self, pnl: float):
        self._daily_pnl += pnl
        self._daily_trades += 1
    
    def reset_daily(self):
        """每日重置"""
        self._daily_pnl = 0.0
        self._daily_trades = 0
    
    def check_stop_loss(self, symbol: str, entry_price: float,
                        current_price: float, position: dict) -> tuple:
        """止损检查
        
        Returns:
            (triggered: bool, reason: str)
        """
        loss_pct = (current_price - entry_price) / entry_price
        if loss_pct <= -self.limits.stop_loss_pct:
            return True, f"止损触发: {loss_pct*100:.1f}%"
        return False, ""
