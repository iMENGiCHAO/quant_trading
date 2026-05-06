"""
Hyperion+ HFUltra — 高频微观结构因子引擎
============================================
基于 Alpha Hunter HFT 核心能力，从Tick/分钟/逐笔数据中提取
微观结构 Alpha。

核心组件：
  1. 订单簿重构 (OrderBook Reconstruction)
  2. 冰山检测 (Iceberg Detection)
  3. 狙击引擎 (Sniper Engine)
  4. 拆单算法 (Execution Algorithms)
  5. Tick处理器 (Tick Processor)
  6. 微观结构信号生成器

输入: Tick级数据 (毫秒精度)
输出: 可交易微观结构信号
"""

from __future__ import annotations

import logging
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque, Callable
from enum import Enum, auto

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
#  数据类型定义
# ==========================================================

class Side(Enum):
    BID = 1
    ASK = -1


@dataclass
class Tick:
    """单个Tick/逐笔成交"""
    timestamp: float  # Unix timestamp
    price: float
    volume: float
    side: Side  # 主动买/主动卖
    trade_id: str = ""


@dataclass
class OrderBookLevel:
    """订单簿单档"""
    price: float
    volume: float
    order_count: int = 0


@dataclass
class SliceData:
    """切片数据 (适配现有回测系统)"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0


# ==========================================================
#  订单簿重构引擎 (OrderBook Reconstruction)
# ==========================================================

class OrderBook:
    """
    基于逐笔/快照数据重构订单簿。
    支持 L2 深度重建。
    """
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.bids: List[OrderBookLevel] = []  # highest price first
        self.asks: List[OrderBookLevel] = []  # lowest price first
        self.last_update_timestamp: Optional[float] = None
        self.trade_history: Deque[Tick] = deque(maxlen=10000)

    def update_lob(self, timestamp: float,
                   bids: List[Tuple[float, float, int]],
                   asks: List[Tuple[float, float, int]]) -> None:
        """
        更新订单簿 (price, volume, count)
        bids/asks: [(price, volume, order_count), ...]
        """
        self.last_update_timestamp = timestamp
        self.bids = [OrderBookLevel(p, v, c) for p, v, c in sorted(bids, reverse=True)[:self.max_depth]]
        self.asks = [OrderBookLevel(p, v, c) for p, v, c in sorted(asks)[:self.max_depth]]

    def add_trade(self, tick: Tick) -> None:
        """记录成交"""
        self.trade_history.append(tick)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask - self.best_bid

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2

    @property
    def depth_imbalance(self) -> float:
        """买卖深度不平衡 (-1 to +1, + = buy heavy)"""
        bid_depth = sum(l.volume for l in self.bids)
        ask_depth = sum(l.volume for l in self.asks)
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / (total + 1e-12)

    def get_cumulative_depth(self, levels: int = 5) -> Tuple[float, float]:
        """返回买卖深度 (累计到第几档)"""
        bid_depth = sum(l.volume for l in self.bids[:levels])
        ask_depth = sum(l.volume for l in self.asks[:levels])
        return bid_depth, ask_depth

    def to_snapshot(self) -> Dict:
        """快照输出"""
        return {
            "timestamp": self.last_update_timestamp,
            "bid_price": self.best_bid,
            "ask_price": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "depth_imbalance": self.depth_imbalance,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
        }


class OrderBookAggregator:
    """将高频 Tick 聚合到 K线/Snapshots"""

    def __init__(self, interval: str = "1min"):
        self.interval = interval
        self.current_slice = None
        self.slices = []
        self._init_slice()

    def _init_slice(self):
        self.current_slice = {
            "open": None, "high": -np.inf, "low": np.inf,
            "close": None, "volume": 0, "vwap_sum": 0,
        }

    def on_tick(self, tick: Tick):
        """处理单个Tick"""
        cs = self.current_slice
        p, v = tick.price, tick.volume

        if cs["open"] is None:
            cs["open"] = p
        cs["high"] = max(cs["high"], p)
        cs["low"] = min(cs["low"], p)
        cs["close"] = p
        cs["volume"] += v
        cs["vwap_sum"] += p * v

    def snapshot(self) -> Optional[SliceData]:
        """获取当前切片"""
        cs = self.current_slice
        if cs["open"] is None:
            return None
        return SliceData(
            timestamp=0,  # placeholder
            open=cs["open"], high=cs["high"], low=cs["low"],
            close=cs["close"], volume=cs["volume"],
            vwap=cs["vwap_sum"] / (cs["volume"] + 1e-12)
        )


# ==========================================================
#  冰山检测算法 (Iceberg Detection)
# ==========================================================

class IcebergDetector:
    """
    冰山委托单检测
    原理: 通过分析大单频发的模式、订单簿变化模式，
          检测隐藏在多个小单背后的大单意图。
    """
    def __init__(self, window: int = 20, sensitivity: float = 3.0):
        self.window = window
        self.sensitivity = sensitivity
        self.trade_buffer: Deque[Tick] = deque(maxlen=1000)
        self.iceberg_signals = []

    def on_trade(self, tick: Tick):
        self.trade_buffer.append(tick)

    def detect(self) -> List[Dict]:
        """返回检测到的冰山单信号"""
        if len(self.trade_buffer) < self.window:
            return []

        # 计算成交量异常值
        volumes = [t.volume for t in self.trade_buffer]
        vol_mean = np.mean(volumes)
        vol_std = np.std(volumes)

        signals = []
        for i, t in enumerate(list(self.trade_buffer)[-self.window:]):
            # CUSUM-like detection
            z_score = (t.volume - vol_mean) / (vol_std + 1e-12)
            if z_score > self.sensitivity:
                signals.append({
                    "timestamp": t.timestamp,
                    "price": t.price,
                    "volume": t.volume,
                    "z_score": z_score,
                    "type": "possible_iceberg",
                    "side": "buy" if t.side == Side.BID else "sell"
                })

        return signals


# ==========================================================
#  狙击引擎 (Sniper Engine)
# ==========================================================

class SniperEngine:
    """
    在高频数据中捕捉瞬间价格错配信号。
    
    策略逻辑:
      1. 检测订单簿缺口 (spread > 正常阈值)
      2. 检测到方向性大单流
      3. 结合微观结构信号 (tick delta, depth imbalance)
      4. 在几毫秒内执行
    """
    def __init__(self,
                 spread_threshold: float = 0.001,
                 depth_threshold: float = 0.3,
                 hold_time_ms: int = 500):
        self.spread_threshold = spread_threshold
        self.depth_threshold = depth_threshold
        self.hold_time_ms = hold_time_ms
        self.positions = []

    def evaluate(self, ob: OrderBook, tick: Tick) -> Optional[Dict]:
        """
        评估是否触发狙击条件。
        
        Return: {'side': 'buy'/'sell', 'size': float, 'reason': str} or None
        """
        spread = ob.spread
        if spread <= 0:
            return None

        spread_bps = spread / ob.mid_price
        if spread_bps < self.spread_threshold:
            return None

        # 深度不平衡判断
        imbalance = ob.depth_imbalance
        if abs(imbalance) < self.depth_threshold:
            return None

        # 方向判断
        if imbalance > 0:  # 买方深度占优 → 卖方较弱 → 做空
            return {
                "side": "buy",
                "size": 100,  # shares/pct
                "reason": f"spread={spread_bps:.4f}, imbalance={imbalance:.2f}",
                "target": ob.asks[0].price if ob.asks else ob.mid_price
            }
        else:  # 卖方深度占优 → 买方较弱 → 做多
            return {
                "side": "sell",
                "size": 100,
                "reason": f"spread={spread_bps:.4f}, imbalance={imbalance:.2f}",
                "target": ob.bids[0].price if ob.bids else ob.mid_price
            }
        return None


# ==========================================================
#  微观结构 Alpha 信号生成器
# ==========================================================

class MicrostructureAlpha:
    """
    从订单簿+逐笔数据生成微观结构Alpha因子。
    这些因子可以直接进入 AlphaUltra 引擎。
    """
    def __init__(self, lookback: int = 1000):
        self.lookback = lookback
        self.order_book = OrderBook(max_depth=10)
        self.iceberg_detector = IcebergDetector()
        self.sniper = SniperEngine()
        self.aggregator = OrderBookAggregator(interval="1min")
        self.trade_history: Deque[Tick] = deque(maxlen=10000)

    def process_tick(self, tick: Tick,
                    bid_levels: Optional[List] = None,
                    ask_levels: Optional[List] = None) -> Dict:
        """
        处理单个Tick并返回微观结构特征。
        
        Returns: dict of microstructure features
        """
        self.trade_history.append(tick)
        self.aggregator.on_tick(tick)

        if bid_levels and ask_levels:
            self.order_book.update_lob(tick.timestamp, bid_levels, ask_levels)
            self.order_book.add_trade(tick)

        self.iceberg_detector.on_trade(tick)

        # 生成特征
        features = {}
        features.update(self._book_features(tick))
        features.update(self._flow_features(tick))
        features.update(self._impact_features(tick))
        features.update(self._snipe_signals(tick))

        return features

    def _book_features(self, tick: Tick) -> Dict:
        """订单簿特征"""
        f = {}
        ob = self.order_book
        if not ob.bids or not ob.asks:
            return f

        spread = ob.spread
        mid = ob.mid_price
        f["ms_spread_bps"] = spread / mid * 10000 if mid > 0 else 0
        f["ms_depth_imb"] = ob.depth_imbalance

        # 深度比率
        bid_d, ask_d = ob.get_cumulative_depth(levels=5)
        f["ms_depth_ratio"] = bid_d / (ask_d + 1e-12)
        f["ms_total_depth"] = bid_d + ask_d

        return f

    def _flow_features(self, tick: Tick) -> Dict:
        """资金流特征"""
        f = {}
        if len(self.trade_history) < 20:
            return f

        recent = list(self.trade_history)[-20:]
        buy_vol = sum(t.volume for t in recent if t.side == Side.BID)
        sell_vol = sum(t.volume for t in recent if t.side == Side.ASK)
        total_vol = buy_vol + sell_vol

        f["ms_buy_ratio"] = buy_vol / (total_vol + 1e-12)
        f["ms_flow_imb"] = (buy_vol - sell_vol) / (total_vol + 1e-12)
        f["ms_flow_speed"] = total_vol / 20  # volume per tick

        return f

    def _impact_features(self, tick: Tick) -> Dict:
        """价格冲击特征"""
        f = {}
        if len(self.trade_history) < 10:
            return f

        recent = list(self.trade_history)[-10:]
        prices = [t.price for t in recent]
        volumes = [t.volume for t in recent]

        # Kyle's Lambda proxy (订单流对价格的冲击)
        if len(prices) > 1:
            price_change = abs(prices[-1] - prices[0])
            total_flow = sum(volumes)
            f["ms_kyle_lambda"] = price_change / (total_flow + 1e-12)

        # 成交量的价格弹性
        if len(volumes) > 1:
            f["ms_price_elasticity"] = abs(prices[-1] - prices[0]) / (sum(volumes) + 1e-12)

        return f

    def _snipe_signals(self, tick: Tick) -> Dict:
        """狙击信号检测"""
        f = {}
        if not self.order_book.bids or not self.order_book.asks:
            return f

        signal = self.sniper.evaluate(self.order_book, tick)
        f["ms_snipe_signal"] = 1 if signal else 0
        f["ms_snipe_side"] = 1 if signal and signal.get("side") == "buy" else (-1 if signal else 0)

        # 冰山检测
        iceberg_signals = self.iceberg_detector.detect()
        f["ms_iceberg_count"] = len(iceberg_signals)

        return f

    def get_slice(self) -> Optional[SliceData]:
        """获取当前1分钟切片"""
        return self.aggregator.snapshot()


# ==========================================================
#  算法交易执行层
# ==========================================================

class ExecutionAlgorithm:
    """
    订单执行算法 (经过 QLib 回测的驱动层)
    """
    def __init__(self, strategy: str = "twap"):
        self.strategy = strategy

    def twap(self, total_qty: float, start_time: float, end_time: float,
             num_slices: int = 10) -> List[Dict]:
        """时间加权平均价格执行"""
        slice_qty = total_qty / num_slices
        interval = (end_time - start_time) / num_slices
        orders = []
        for i in range(num_slices):
            orders.append({
                "time": start_time + i * interval,
                "qty": slice_qty,
                "type": "twap_slice",
                "slice_num": i
            })
        return orders

    def vwap(self, total_qty: float, start_time: float, end_time: float,
             volume_profile: Optional[List[float]] = None) -> List[Dict]:
        """成交量加权平均价格执行"""
        if volume_profile is None:
            # 默认均匀分布
            num_slices = 10
            volume_profile = [1.0 / num_slices] * num_slices

        volume_profile = np.array(volume_profile)
        volume_profile = volume_profile / volume_profile.sum()

        qtys = total_qty * volume_profile
        interval = (end_time - start_time) / len(volume_profile)

        orders = []
        for i, qty in enumerate(qtys):
            orders.append({
                "time": start_time + i * interval,
                "qty": qty,
                "type": "vwap_slice",
                "profile_weight": volume_profile[i]
            })
        return orders

    def sniper(self, total_qty: float, target_price: float,
               slippage: float = 0.001) -> Dict:
        """狙击单: 仅在特定价格立即成交"""
        return {
            "qty": total_qty,
            "target": target_price,
            "slippage": slippage,
            "type": "sniper",
            "method": "immediate_or_cancel",
            "strategy": "fill_within_slippage"
        }

    def iceberg(self, total_qty: float,
                visible_qty: float,
                interval_ms: int = 500) -> List[Dict]:
        """冰山单: 分批隐藏真实意图"""
        orders = []
        remaining = total_qty
        while remaining > 0:
            slice_qty = min(visible_qty, remaining)
            orders.append({
                "qty": slice_qty,
                "type": "iceberg_slice",
                "visible": visible_qty,
                "interval_ms": interval_ms,
                "method": "passive_post"
            })
            remaining -= slice_qty
        return orders


# ==========================================================
#  便捷接口
# ==========================================================

def create_microstructure_alpha(tick_df: pd.DataFrame) -> pd.DataFrame:
    """从Tick DataFrame生成微观结构因子"""
    engine = MicrostructureAlpha()
    features_list = []

    for _, row in tick_df.iterrows():
        tick = Tick(timestamp=row.get("timestamp", 0),
                    price=row["price"],
                    volume=row["volume"],
                    side=Side.BID if row.get("side", 1) > 0 else Side.ASK)
        features = engine.process_tick(tick)
        features_list.append(features)

    return pd.DataFrame(features_list, index=tick_df.index)


# 兼容 QLib 接口风格
MicroAlpha = MicrostructureAlpha
OrderBookReconstructor = OrderBook
IcebergDetect = IcebergDetector
Sniper = SniperEngine
Exec = ExecutionAlgorithm
