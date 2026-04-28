"""
模拟交易Broker (Paper Trading)
===============================
VnPy paper_account + Freqtrade dry-run 融合。

本地模拟撮合, 无真实资金风险。
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

from hyperion.execution.broker import Broker, Order, Account

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """模拟交易 (Paper Trading)
    
    纯本地撮合, 用于策略验证。
    """
    
    def __init__(self, initial_capital: float = 1_000_000.0,
                 commission_rate: float = 0.0003,
                 stamp_duty: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        
        self._connected = False
        self._cash = initial_capital
        self._frozen_cash = 0.0
        self._positions: Dict[str, Dict] = {}
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        
        # 当前行情 (外部写入)
        self._quotes: Dict[str, float] = {}
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        self._connected = True
        logger.info("PaperBroker connected")
        return True
    
    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected")
    
    def update_quote(self, symbol: str, price: float):
        """更新最新价格"""
        self._quotes[symbol] = price
    
    def submit_order(self, symbol: str, direction: str,
                     quantity: int, price: float = 0.0,
                     order_type: str = "LIMIT") -> Order:
        """提交模拟订单"""
        order_id = str(uuid.uuid4())[:8]
        
        # 使用市价 (如果未指定)
        if price <= 0 and symbol in self._quotes:
            price = self._quotes[symbol]
        
        if price <= 0:
            return Order(
                order_id=order_id, symbol=symbol, direction=direction,
                order_type=order_type, quantity=quantity, price=0,
                status="REJECTED", timestamp=datetime.now()
            )
        
        order = Order(
            order_id=order_id, symbol=symbol, direction=direction,
            order_type=order_type, quantity=quantity, price=price,
            status="PENDING", timestamp=datetime.now()
        )
        
        # 模拟成交
        self._fill_order(order)
        
        self._orders[order_id] = order
        self._order_history.append(order)
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == "PENDING":
                order.status = "CANCELLED"
                return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    def get_account(self) -> Account:
        """计算模拟账户"""
        market_value = 0.0
        for sym, pos in self._positions.items():
            price = self._quotes.get(sym, pos.get("avg_cost", 0))
            market_value += pos.get("quantity", 0) * price
        
        return Account(
            total_asset=self._cash + market_value,
            available_cash=self._cash,
            frozen_cash=self._frozen_cash,
            market_value=market_value,
            positions=self._positions.copy()
        )
    
    def get_positions(self) -> Dict[str, Dict]:
        return self._positions.copy()
    
    def _fill_order(self, order: Order):
        """模拟成交"""
        exec_price = order.price
        symbol = order.symbol
        quantity = order.quantity
        
        trade_value = quantity * exec_price
        
        if order.direction == "BUY":
            cost = trade_value * (1 + self.commission_rate)
            if cost > self._cash:
                order.status = "REJECTED"
                return
            
            self._cash -= cost
            
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["quantity"] + quantity
                total_cost = pos["avg_cost"] * pos["quantity"] + trade_value
                pos["quantity"] = total_qty
                pos["avg_cost"] = total_cost / total_qty
            else:
                self._positions[symbol] = {
                    "quantity": quantity,
                    "avg_cost": exec_price,
                    "symbol": symbol
                }
        else:  # SELL
            if symbol not in self._positions:
                order.status = "REJECTED"
                return
            
            pos = self._positions[symbol]
            if pos["quantity"] < quantity:
                quantity = pos["quantity"]
            
            # 印花税 (卖出)
            tax = trade_value * self.stamp_duty
            commission = trade_value * self.commission_rate
            
            self._cash += trade_value - tax - commission
            
            pos["quantity"] -= quantity
            if pos["quantity"] <= 0:
                del self._positions[symbol]
        
        order.status = "FILLED"
        order.filled_qty = quantity
        order.filled_price = exec_price
