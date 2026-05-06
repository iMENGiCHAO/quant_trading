"""
高性能A股数据服务器 (Qlib风格)
================================
融合设计:
- Qlib: 列式缓存 + 高性能数据访问
- VnPy: 多数据源适配
- Freqtrade: 数据下载/转换/缓存管理

核心特性:
- SQLite存储原始数据 (兼容现有hyperion数据)
- Parquet列式缓存加速读取 (10x faster than CSV)
- Point-in-Time数据无前视偏差
- 多字段批量查询
- 交易日历管理
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from hyperion.data.cache import DataCache

logger = logging.getLogger(__name__)

# A股默认字段
DEFAULT_FIELDS = ["open", "high", "low", "close", "volume", "amount",
                  "turnover", "change_pct", "amplitude", "money_flow"]


class DataServer:
    """A股高性能数据服务器
    
    Usage:
        server = DataServer()
        df = server.fetch("000001.SZ", "2024-01-01", "2024-12-31")
        batch = server.fetch_multi(["000001.SZ", "000002.SZ"], "2024-01-01", "2024-12-31")
    """
    
    def __init__(self, db_path: str = "~/.quant_trading/ashare.db",
                 cache_dir: str = "~/.quant_trading/cache"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = DataCache(disk_cache_dir=cache_dir)
        self._conn: Optional[sqlite3.Connection] = None
        self._calendar: Optional[pd.DatetimeIndex] = None
        self._symbols: Optional[List[str]] = None
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA cache_size=-64000")
            self._init_tables()
        return self._conn
    
    def _init_tables(self):
        """初始化数据表 (兼容hyperion结构)"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL, turnover REAL,
                change_pct REAL, amplitude REAL, money_flow REAL,
                PRIMARY KEY (symbol, date)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date)
        """)
        self.conn.commit()
    
    def fetch(self, symbol: str, start: str, end: str,
              fields: Optional[List[str]] = None) -> pd.DataFrame:
        """查询单只股票数据
        
        Args:
            symbol: 股票代码 '000001.SZ', '000300.SH'
            start: 起始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            fields: 字段列表
            
        Returns:
            DataFrame with DatetimeIndex
        """
        fields = fields or DEFAULT_FIELDS
        
        # 先尝试缓存
        cache_key = f"{symbol}_{start}_{end}_{'_'.join(fields)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        fields_str = ", ".join(fields)
        sql = f"""SELECT date, {fields_str} FROM daily 
                  WHERE symbol = ? AND date BETWEEN ? AND ?
                  ORDER BY date"""
        
        try:
            df = pd.read_sql(sql, self.conn, params=(symbol, start, end))
        except Exception as e:
            logger.error(f"Query failed for {symbol}: {e}")
            return pd.DataFrame(columns=["date"] + fields)
        
        if df.empty:
            return df
        
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # 缓存结果
        self.cache.set(cache_key, df)
        
        return df
    
    def fetch_multi(self, symbols: List[str], start: str, end: str,
                    fields: Optional[List[str]] = None) -> pd.DataFrame:
        """批量查询多只股票 (MultiIndex DataFrame)
        
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        frames = []
        for sym in symbols:
            df = self.fetch(sym, start, end, fields)
            if not df.empty:
                df["symbol"] = sym
                df.set_index("symbol", append=True, inplace=True)
                frames.append(df)
        
        if not frames:
            return pd.DataFrame()
        
        result = pd.concat(frames)
        return result.swaplevel().sort_index()  # (symbol, date) → (date, symbol) after
    
    def store(self, symbol: str, data: pd.DataFrame) -> int:
        """存储单只股票数据
        
        Args:
            symbol: 股票代码
            data: OHLCV DataFrame (DatetimeIndex)
            
        Returns:
            写入行数
        """
        if data.empty:
            return 0
        
        df = data.copy()
        df.index.name = "date"
        df = df.reset_index()
        df["symbol"] = symbol
        
        # 确保必要列存在
        cols = ["symbol", "date"]
        avail = [c for c in DEFAULT_FIELDS if c in df.columns]
        cols.extend(avail)
        
        rows = 0
        for _, row in df[cols].iterrows():
            try:
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO daily ({', '.join(cols)}) VALUES ({placeholders})"
                self.conn.execute(sql, tuple(row[cols]))
                rows += 1
            except Exception as e:
                logger.debug(f"Insert error for {symbol}: {e}")
        
        self.conn.commit()
        
        # 失效相关缓存
        self.cache.invalidate(symbol)
        
        return rows
    
    def store_batch(self, data_dict: Dict[str, pd.DataFrame]) -> int:
        """批量存储数据"""
        total = 0
        for symbol, df in data_dict.items():
            total += self.store(symbol, df)
        return total
    
    @property
    def symbols(self) -> List[str]:
        """获取所有股票列表"""
        if self._symbols is None:
            cur = self.conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol")
            self._symbols = [r[0] for r in cur.fetchall()]
        return self._symbols
    
    @property
    def calendar(self) -> pd.DatetimeIndex:
        """获取交易日历"""
        if self._calendar is None:
            cur = self.conn.execute("SELECT DISTINCT date FROM daily ORDER BY date")
            self._calendar = pd.to_datetime([r[0] for r in cur.fetchall()])
        return self._calendar
    
    def date_range(self) -> Tuple[str, str]:
        """数据覆盖的日期范围"""
        if len(self.calendar) == 0:
            return ("N/A", "N/A")
        return (str(self.calendar[0].date()), str(self.calendar[-1].date()))
    
    def stats(self) -> dict:
        """数据统计"""
        return {
            "db_path": str(self.db_path),
            "symbols_count": len(self.symbols),
            "date_range": self.date_range(),
            "cache": self.cache.stats()
        }
    
    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
