"""
高性能数据缓存层
================
Freqtrade-style data caching with:
- LRU内存缓存
- Parquet磁盘持久化 (Qlib style)
- 线程安全
- 自动过期
"""
from __future__ import annotations

import logging
import time
import threading
from pathlib import Path
from typing import Any, Optional, List
import pickle

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """线程安全的LRU数据缓存
    
    两层缓存:
      1. 内存 LRU (快速访问)
      2. Parquet磁盘 (持久化, 列式存储)
    """
    
    def __init__(self, max_memory_items: int = 500,
                 default_ttl: int = 3600,
                 disk_cache_dir: str = "~/.quant_trading/cache"):
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        self.disk_cache_dir = Path(disk_cache_dir).expanduser()
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: dict = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        with self._lock:
            # 1. 内存缓存
            if key in self._cache:
                value, expire_time = self._cache[key]
                if time.time() <= expire_time:
                    self._touch(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            
            # 2. 磁盘缓存
            disk_val = self._read_disk(key)
            if disk_val is not None:
                self._put_memory(key, disk_val)
                self._hits += 1
                return disk_val
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: pd.DataFrame,
            ttl: Optional[int] = None, persist: bool = True):
        """存入缓存"""
        with self._lock:
            ttl = ttl or self.default_ttl
            self._put_memory(key, value, ttl)
            if persist:
                self._write_disk(key, value)
    
    def invalidate(self, pattern: str = ""):
        """清除缓存(支持部分匹配)"""
        with self._lock:
            if not pattern:
                self._cache.clear()
                self._access_order.clear()
            else:
                keys = [k for k in self._cache if pattern in k]
                for k in keys:
                    del self._cache[k]
                    if k in self._access_order:
                        self._access_order.remove(k)
    
    def stats(self) -> dict:
        """缓存统计"""
        return {
            "memory_items": len(self._cache),
            "max_memory": self.max_memory_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "disk_dir": str(self.disk_cache_dir)
        }
    
    # === Private ===
    
    def _touch(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _put_memory(self, key: str, value: Any, ttl: int):
        self._cache[key] = (value, time.time() + ttl)
        self._touch(key)
        # LRU eviction
        while len(self._cache) > self.max_memory_items:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
    
    def _read_disk(self, key: str) -> Optional[pd.DataFrame]:
        path = self.disk_cache_dir / f"{key}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Failed to read parquet cache: {e}")
        return None
    
    def _write_disk(self, key: str, value: pd.DataFrame):
        path = self.disk_cache_dir / f"{key}.parquet"
        try:
            value.to_parquet(path, index=True)
        except Exception as e:
            logger.warning(f"Failed to write parquet cache: {e}")
