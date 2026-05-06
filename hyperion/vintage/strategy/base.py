"""
策略基类 (VnPy CTA + Jesse 融合)
=================================
回测和实盘使用同一套策略代码 (VnPy哲学)
极简API (Jesse哲学)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    direction: str          # "BUY", "SELL", "HOLD"
    strength: float = 1.0   # 0-1 信号强度
    target_weight: float = 0.0  # 目标仓位权重
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BaseStrategy(ABC):
    """交易策略抽象基类
    
    所有策略必须实现:
    - on_bar(): 每根K线调用, 返回交易信号
    - on_init(): 策略初始化
    
    可选实现:
    - on_tick(): Tick数据回调
    - on_order(): 订单状态回调
    - on_trade(): 成交回报
    """
    
    def __init__(self, name: str = "base_strategy",
                 symbols: Optional[List[str]] = None,
                 params: Optional[Dict] = None):
        self.name = name
        self.symbols = symbols or []
        self.params = params or {}
        self._initialized = False
        
        # 内部状态
        self.positions: Dict[str, Position] = {}
        self.capital: float = 0.0
        self.equity: float = 0.0
        self.current_date: Optional[str] = None
        
        # 数据缓存
        self._data: Dict[str, pd.DataFrame] = {}
        self._bar_index: Dict[str, int] = {}
    
    @abstractmethod
    def on_bar(self, bar_data: Dict[str, pd.Series]) -> List[Signal]:
        """K线回调
        
        Args:
            bar_data: {symbol: Series with OHLCV}
            
        Returns:
            List of Signal
        """
        ...
    
    def on_init(self) -> None:
        """策略初始化 (加载模型、预热等)"""
        self._initialized = True
    
    def on_tick(self, tick_data: Dict[str, Dict]) -> List[Signal]:
        """Tick回调 (可选)"""
        return []
    
    def on_order(self, order: Dict) -> None:
        """订单状态回调"""
        pass
    
    def on_trade(self, trade: Dict) -> None:
        """成交回报"""
        pass
    
    def on_stop(self) -> None:
        """策略停止"""
        pass
    
    def update_position(self, symbol: str, position: Position):
        """更新持仓"""
        self.positions[symbol] = position
    
    def update_data(self, symbol: str, data: pd.DataFrame):
        """更新数据缓存"""
        self._data[symbol] = data
        self._bar_index[symbol] = 0
    
    def get_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """获取历史数据"""
        if symbol not in self._data:
            return pd.DataFrame()
        idx = self._bar_index.get(symbol, len(self._data[symbol]))
        end = min(idx + 1, len(self._data[symbol]))
        start = max(0, end - lookback)
        return self._data[symbol].iloc[start:end]
    
    def get_available_capital(self) -> float:
        """获取可用资金"""
        used = sum(p.market_value for p in self.positions.values())
        return max(0, self.capital - used)
    
    def get_exposure(self) -> float:
        """获取当前仓位暴露"""
        if self.capital <= 0:
            return 0.0
        used = sum(p.market_value for p in self.positions.values())
        return used / self.capital
