"""
Broker 抽象层 (Hummingbot connector + VnPy gateway)
=====================================================
所有交易执行通过此接口, 支持:
- 模拟交易 (Paper Trading)
- 实盘交易 (CTP/IB/...)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    direction: str  # BUY/SELL
    order_type: str  # LIMIT/MARKET
    quantity: int
    price: float
    status: str = "PENDING"  # PENDING/FILLED/CANCELLED/REJECTED
    filled_qty: int = 0
    filled_price: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict = None


@dataclass
class Account:
    """账户信息"""
    total_asset: float = 0.0
    available_cash: float = 0.0
    frozen_cash: float = 0.0
    market_value: float = 0.0
    positions: Dict[str, Dict] = None


class Broker(ABC):
    """经纪商抽象基类"""
    
    @abstractmethod
    def connect(self) -> bool:
        """连接经纪商"""
        ...
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        ...
    
    @abstractmethod
    def submit_order(self, symbol: str, direction: str,
                     quantity: int, price: float = 0.0,
                     order_type: str = "LIMIT") -> Order:
        """提交订单"""
        ...
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        ...
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        ...
    
    @abstractmethod
    def get_account(self) -> Account:
        """查询账户"""
        ...
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict]:
        """查询持仓"""
        ...
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否连接"""
        ...
