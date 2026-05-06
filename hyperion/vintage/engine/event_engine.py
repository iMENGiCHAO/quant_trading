"""
事件驱动引擎 (VnPy核心)
=========================
高性能事件驱动框架, 所有模块通过事件总线通信。

事件类型:
- TICK: Tick行情
- BAR: K线行情
- ORDER: 订单状态
- TRADE: 成交回报
- TIMER: 定时器
- SIGNAL: 交易信号
"""
from __future__ import annotations

import logging
import time
import threading
from queue import Queue, Empty
from enum import Enum
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    TICK = "tick"
    BAR = "bar"
    ORDER = "order"
    TRADE = "trade"
    TIMER = "timer"
    SIGNAL = "signal"
    RISK_CHECK = "risk_check"
    LOG = "log"


@dataclass
class Event:
    """事件对象"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    
    def __repr__(self):
        return f"Event({self.type.value}, data_keys={list(self.data.keys())})"


HandlerType = Callable[[Event], None]


class EventEngine:
    """事件驱动引擎
    
    特点:
    - 多生产者/消费者
    - 线程安全
    - 优先级队列
    - 支持同步/异步处理
    
    Usage:
        engine = EventEngine()
        engine.register(EventType.BAR, my_handler)
        engine.start()
        engine.put(Event(EventType.BAR, data={"symbol": "000001.SZ"}))
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._handlers: Dict[EventType, List[HandlerType]] = {
            et: [] for et in EventType
        }
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._timer_interval = 1  # 定时器间隔(秒)
        self._timer_count = 0
        
        # 统计
        self._event_count: Dict[EventType, int] = {et: 0 for et in EventType}
        
        # 注册默认日志处理器
        self.register(EventType.LOG, self._log_handler)
    
    # ============================================================
    #  Handler Management
    # ============================================================
    
    def register(self, event_type: EventType, handler: HandlerType):
        """注册事件处理器"""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Registered handler for {event_type.value}: {handler.__name__}")
    
    def unregister(self, event_type: EventType, handler: HandlerType):
        """注销事件处理器"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
    
    def clear_handlers(self, event_type: Optional[EventType] = None):
        """清空处理器"""
        if event_type:
            self._handlers[event_type].clear()
        else:
            for et in EventType:
                self._handlers[et].clear()
    
    # ============================================================
    #  Event Loop
    # ============================================================
    
    def start(self):
        """启动事件引擎"""
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("EventEngine started")
    
    def stop(self):
        """停止事件引擎"""
        self._active = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("EventEngine stopped")
    
    def _run(self):
        """主事件循环"""
        last_timer = time.time()
        
        while self._active:
            try:
                # 非堵塞获取事件
                event = self._queue.get(timeout=0.1)
                self._process(event)
            except Empty:
                pass
            
            # 定时器
            now = time.time()
            if now - last_timer >= self._timer_interval:
                self._timer_count += 1
                timer_event = Event(
                    type=EventType.TIMER,
                    data={"count": self._timer_count},
                    timestamp=datetime.now()
                )
                self._process(timer_event)
                last_timer = now
    
    def _process(self, event: Event):
        """处理单个事件"""
        self._event_count[event.type] += 1
        
        for handler in self._handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.type.value}: {e}", exc_info=True)
    
    # ============================================================
    #  Event Producers
    # ============================================================
    
    def put(self, event: Event):
        """放入事件"""
        if not self._active:
            self._process(event)  # 同步处理
        else:
            try:
                self._queue.put(event, timeout=1)
            except Exception:
                logger.warning("Event queue full, dropping event")
    
    def put_bar(self, symbol: str, bar_data: dict):
        """放入K线事件"""
        self.put(Event(
            type=EventType.BAR,
            data={"symbol": symbol, **bar_data},
            timestamp=datetime.now()
        ))
    
    def put_signal(self, symbol: str, direction: str, strength: float = 1.0):
        """放入交易信号"""
        self.put(Event(
            type=EventType.SIGNAL,
            data={"symbol": symbol, "direction": direction, "strength": strength}
        ))
    
    # ============================================================
    #  Utilities
    # ============================================================
    
    @property
    def stats(self) -> dict:
        return {
            "active": self._active,
            "queue_size": self._queue.qsize(),
            "event_counts": {et.value: c for et, c in self._event_count.items()},
            "handlers": {et.value: len(h) for et, h in self._handlers.items()},
            "timer_count": self._timer_count
        }
    
    @staticmethod
    def _log_handler(event: Event):
        """默认日志处理器"""
        level = event.data.get("level", "info")
        msg = event.data.get("message", "")
        getattr(logger, level)(msg)
