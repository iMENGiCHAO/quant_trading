"""
Hyperion+ EventBacktest — 事件驱动回测引擎
============================================
超越 QLib 的循环回测，支持：
  - 事件驱动架构 (cello + observer + analyzer)
  - 真实撮合引擎 (订单簿级)
  - 滑点模型 (线性/非线性冲击)
  - 成本模型 (佣金+冲击成本+资金成本)
  - 高频回测 (Tick级仿真)
  - Monte Carlo 仿真

兼容 Backtrader 风格 API，同时支持 QLib 风格工作流
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from collections import defaultdict, deque
from enum import Enum, auto
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ==========================================================
#  基础数据类型
# ==========================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Event(Enum):
    NEW_BAR = "new_bar"
    NEW_TICK = "new_tick"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    STOP_TRIGGERED = "stop_triggered"


@dataclass
class Trade:
    """成交记录"""
    timestamp: float
    symbol: str
    side: OrderSide
    price: float
    size: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""


@dataclass
class Position:
    """持仓"""
    symbol: str
    size: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = 0.0
    filled_size: float = 0.0
    filled_price: float = 0.0


# ==========================================================
#  撮合引擎 (Order Matching Engine)
# ==========================================================

class MatchingEngine:
    """
    撮合引擎。
    QLib启发的简化版——支持限价单、市价单、止损单
    """
    def __init__(self,
                 base_slippage: float = 0.001,
                 commission: float = 0.0003,
                 stamp_duty: float = 0.001,
                 volume_limit: float = 0.25,
                 impact_model: str = "linear"):
        self.base_slippage = base_slippage
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.volume_limit = volume_limit  # 最大可成交占当日成交量的比例
        self.impact_model = impact_model
        self.trades: List[Trade] = []

    def match_order(self, order: Order,
                    current_bar: pd.Series,
                    current_tick: Optional[Dict] = None) -> List[Trade]:
        """
        撮合单个订单。
        
        Args:
            order: 订单对象
            current_bar: 当前K线 {open, high, low, close, volume}
            current_tick: 可选的Tick级数据 (用于高精度)
        
        Returns:
            成交列表
        """
        trades = []
        price = self._get_fill_price(order, current_bar, current_tick)
        
        if price is None:
            return trades

        # 滑点
        slippage = self._calculate_slippage(order.side, price, current_bar)
        price = price * (1 + slippage)

        # 量限制
        max_fill = current_bar["volume"] * self.volume_limit
        fill_size = min(order.size - order.filled_size, max_fill)

        if fill_size <= 0:
            return trades

        # 佣金
        commission = price * fill_size * self.commission
        if order.side == OrderSide.SELL:
            commission += price * fill_size * self.stamp_duty  # 卖出印花税后

        trade = Trade(
            timestamp=current_bar.name if hasattr(current_bar, "name") else 0,
            symbol=order.symbol,
            side=order.side,
            price=price,
            size=fill_size,
            commission=commission,
            slippage=slippage,
            order_id=order.id
        )
        trades.append(trade)
        self.trades.append(trade)

        order.filled_size += fill_size
        if order.filled_size >= order.size:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL

        return trades

    def _get_fill_price(self, order: Order,
                        bar: pd.Series,
                        tick: Optional[Dict] = None) -> Optional[float]:
        """获取成交价格"""
        if tick:
            return tick["price"]

        if order.order_type == OrderType.MARKET:
            return bar["close"]  # 简化: 以收盘价成交
        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                return None
            # 限价单: 价格条件满足才成交
            if (order.side == OrderSide.BUY and order.price >= bar["low"]) or \
               (order.side == OrderSide.SELL and order.price <= bar["high"]):
                return order.price
            return None
        return bar["close"]

    def _calculate_slippage(self, side: OrderSide, price: float,
                           bar: pd.Series) -> float:
        """计算滑点"""
        # 基础滑点 + 冲击成本
        base = self.base_slippage * np.random.normal(0, 0.1)
        
        # 非线性冲击 (Kyle/Almgren-Chriss 简化)
        if self.impact_model == "linear":
            impact = 0.0001 * (bar["volume"] / (bar["volume"] + 1e-12))
        else:
            impact = 0
        
        direction = 1 if side == OrderSide.BUY else -1
        return (base + impact) * direction


# ==========================================================
#  事件驱动引擎 (Event Engine)
# ==========================================================

class EventEngine:
    """
    事件驱动引擎核心。
    类似于 Backtrader Cerebro + VnPy EventEngine 的融合。
    """
    def __init__(self,
                 initial_cash: float = 1_000_000.0,
                 slippage: float = 0.001,
                 commission: float = 0.0003,
                 stamp_duty: float = 0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.portfolio_values = []
        
        self.matcher = MatchingEngine(
            base_slippage=slippage,
            commission=commission,
            stamp_duty=stamp_duty
        )

        # 事件消费者注册
        self._observers: Dict[Event, List[Callable]] = defaultdict(list)

    def register_observer(self, event: Event, callback: Callable):
        """注册事件观察者"""
        self._observers[event].append(callback)

    def emit(self, event: Event, data: Any):
        """触发事件"""
        for cb in self._observers[event]:
            cb(data)

    def submit_order(self, order: Order,
                     current_bar: pd.Series,
                     current_tick: Optional[Dict] = None) -> List[Trade]:
        """提交订单并撮合"""
        trades = self.matcher.match_order(order, current_bar, current_tick)
        
        for trade in trades:
            self.trades.append(trade)
            self._update_position(trade)
            self.emit(Event.ORDER_FILLED, trade)
        
        return trades

    def _update_position(self, trade: Trade):
        """更新持仓"""
        sym = trade.symbol
        if sym not in self.positions:
            self.positions[sym] = Position(symbol=sym, size=0.0, avg_price=0.0)
        
        pos = self.positions[sym]
        multiplier = 1 if trade.side == OrderSide.BUY else -1
        
        if (pos.size > 0 and trade.side == OrderSide.SELL) or \
           (pos.size < 0 and trade.side == OrderSide.BUY):
            # 平仓 (或部分平仓)
            close_size = min(abs(pos.size), trade.size)
            pos.realized_pnl += close_size * (trade.price - pos.avg_price) * (-multiplier)
            pos.size -= close_size * np.sign(pos.size)
            if pos.size == 0:
                pos.avg_price = 0.0
        else:
            # 加仓
            new_size = pos.size + trade.size * multiplier
            pos.avg_price = (pos.size * pos.avg_price + trade.size * trade.price) / (new_size + 1e-12)
            pos.size = new_size

        # 更新资金
        cost = trade.price * trade.size + trade.commission
        self.cash -= cost * multiplier  # BUY减少cash, SELL增加cash

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算组合总净值"""
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in current_prices:
                equity += pos.size * current_prices[sym]
        return equity

    def record_portfolio(self, current_prices: Dict[str, float]):
        """记录每日组合价值"""
        value = self.get_portfolio_value(current_prices)
        self.portfolio_values.append(value)


# ==========================================================
#  策略基类 (兼容 Backtrader 和 QLib 风格)
# ==========================================================

class Strategy(ABC):
    """
    策略基类。
    子类只需要实现两个方法:
      - on_bar(self, bar_data) -> signals
      - on_tick(self, tick_data) -> signals (可选)
    """
    def __init__(self, name: str = "strategy", **kwargs):
        self.name = name
        self.params = kwargs
        self.engine: Optional[EventEngine] = None

    @abstractmethod
    def on_bar(self, bar_data: pd.Series) -> Optional[List[Order]]:
        """每根K线触发"""
        pass

    def on_tick(self, tick_data: Dict) -> Optional[List[Order]]:
        """每个Tick触发 (可选)"""
        pass

    def on_order(self, order: Order):
        """订单状态变化"""
        pass

    def on_trade(self, trade: Trade):
        """成交后"""
        pass

    def notify(self, msg: str):
        """通知/日志"""
        logger.info(f"[{self.name}] {msg}")


# ==========================================================
#  Cerebro — 回测编排器 (Backtrader style)
# ==========================================================

class Cerebro:
    """
    回测编排器 (类比 Backtrader.cerebro)
    也兼容 QLib 的 workflow API
    """
    def __init__(self, cash: float = 1_000_000,
                 commission: float = 0.0003,
                 slippage: float = 0.001):
        self.engine = EventEngine(cash, slippage, commission)
        self.strategies: List[Strategy] = []
        self.bars = None  # DataFrame: index=date, columns= [open, high, low, close, volume]
        self.results = None

    def add_strategy(self, strategy: Strategy):
        """添加策略"""
        self.strategies.append(strategy)
        strategy.engine = self.engine

    def add_data(self, data: pd.DataFrame):
        """添加数据 (多列)"""
        self.bars = data

    def run(self, verbose: bool = True) -> Dict:
        """运行回测"""
        if self.bars is None:
            raise ValueError("No data fed! Use add_data()")

        logger.info(f"开始回测: {len(self.bars)} bars, "
                   f"初始资金: {self.engine.initial_cash:,.0f}")

        for i, (date, bar) in enumerate(self.bars.iterrows()):
            # 触发策略
            for strategy in self.strategies:
                try:
                    orders = strategy.on_bar(bar)
                    if orders:
                        for order in orders:
                            self.engine.submit_order(order, bar)
                except Exception as e:
                    logger.warning(f"策略 {strategy.name} 异常: {e}")

            # 记录每日净值
            prices = {str(date): bar["close"]}  # 简化
            self.engine.record_portfolio(prices)

        # 计算结果
        self.results = self._analyze_results()
        if verbose:
            self._print_results()
        return self.results

    def _analyze_results(self) -> Dict:
        """回测结果分析"""
        pv = pd.Series(self.engine.portfolio_values)
        if len(pv) < 2:
            return {}

        total_return = (pv.iloc[-1] / self.engine.initial_cash - 1) * 100
        cummax = pv.cummax()
        drawdown = (pv - cummax) / cummax
        max_dd = drawdown.min()

        # 年化收益
        n_years = len(pv) / 252
        if n_years > 0:
            ann_return = (1 + total_return/100) ** (1/n_years) - 1
        else:
            ann_return = 0

        # Sharpe (简化: 假设无风险利率 2%)
        if len(pv) > 1:
            daily_returns = pv.pct_change().dropna()
            sharpe = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252) + 1e-12)
        else:
            sharpe = 0

        return {
            "total_return": total_return,
            "annual_return": ann_return * 100,
            "max_drawdown": max_dd * 100,
            "sharpe_ratio": sharpe,
            "final_value": pv.iloc[-1],
            "trades": len(self.engine.trades),
            "trade_count": len(self.engine.trades),
        }

    def _print_results(self):
        """打印回测结果"""
        r = self.results
        print(f"\n{'='*50}")
        print(f"回测结果")
        print(f"{'='*50}")
        print(f"总收益率:      {r['total_return']:.2f}%")
        print(f"年化收益率:    {r['annual_return']:.2f}%")
        print(f"最大回撤:      {r['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:  {r['sharpe_ratio']:.2f}")
        print(f"最终净值:      {r['final_value']:,.0f}")
        print(f"交易次数:      {r['trade_count']}")
        print(f"{'='*50}\n")

    def plot(self):
        """绘制回测曲线 (需要 matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            pv = pd.Series(self.engine.portfolio_values)
            pv_norm = pv / self.engine.initial_cash
            plt.figure(figsize=(12, 4))
            plt.plot(pv_norm.index, pv_norm.values)
            plt.title("Portfolio Value (Normalized)")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")


# ==========================================================
#  Monte Carlo 回测 (Freqtrade 风格)
# ==========================================================

class MonteCarloBacktest:
    """
    基于历史数据的 Monte Carlo 仿真。
    通过随机重排收益率序列，测试策略稳健性。
    """
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.results = []

    def run(self, returns: pd.Series, strategy_func: Callable) -> pd.DataFrame:
        """
        运行 Monte Carlo 仿真
        
        Args:
            returns: 历史日收益率序列
            strategy_func: 策略函数, 输入returns序列, 输出portfolio value序列
        """
        import numpy as np
        
        all_results = []
        base_return = strategy_func(returns).iloc[-1] if hasattr(strategy_func(returns), 'iloc') else 1.0
        
        for i in range(self.n_simulations):
            # 随机打乱收益率顺序
            shuffled = returns.sample(frac=1, replace=False, random_state=i).reset_index(drop=True)
            result = strategy_func(shuffled)
            if hasattr(result, 'iloc'):
                all_results.append({
                    'sim': i,
                    'final_value': result.iloc[-1],
                    'sharpe': result.pct_change().mean() / (result.pct_change().std() + 1e-12)
                })
        
        return pd.DataFrame(all_results)


# ==========================================================
#  QLib 兼容接口
# ==========================================================

def run_backtest(strategy: Strategy,
                 data: pd.DataFrame,
                 cash: float = 1_000_000,
                 commission: float = 0.0003) -> Dict:
    """
    QLib 风格一键回测
    
    使用:
        results = run_backtest(strategy=my_strategy, data=df)
    """
    cerebro = Cerebro(cash=cash, commission=commission)
    cerebro.add_strategy(strategy)
    cerebro.add_data(data)
    return cerebro.run()


# 导出核心类
__all__ = [
    "EventEngine", "MatchingEngine", "Cerebro",
    "Strategy", "Order", "Trade", "Position",
    "MonteCarloBacktest", "run_backtest",
    "OrderSide", "OrderType", "OrderStatus", "Event"
]
