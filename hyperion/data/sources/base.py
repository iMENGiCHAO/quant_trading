"""
数据源基类 (VnPy gateway + Hummingbot connector 融合设计)
==========================================================
所有数据源必须实现此接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataSource(ABC):
    """数据源抽象基类
    
    所有数据源必须实现:
    - download_daily(): 下载日线数据
    - download_minute(): 下载分钟线数据
    - name: 数据源名称
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        ...
    
    @abstractmethod
    def download_daily(self, symbols: List[str],
                       start: str, end: str,
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """下载日线数据
        
        Args:
            symbols: 股票代码列表
            start: 起始日期 YYYY-MM-DD
            end: 结束日期 YYYY-MM-DD
            fields: 字段列表
            
        Returns:
            MultiIndex DataFrame (date, symbol)
        """
        ...
    
    def download_minute(self, symbols: List[str],
                        start: str, end: str,
                        freq: str = "1min") -> pd.DataFrame:
        """下载分钟线数据 (可选实现)
        
        Args:
            symbols: 股票代码列表
            start: 起始datetime
            end: 结束datetime
            freq: 频率
            
        Returns:
            MultiIndex DataFrame
        """
        raise NotImplementedError(f"Minute data not supported by {self.name}")
    
    def download_tick(self, symbols: List[str],
                      date: str) -> pd.DataFrame:
        """下载Tick数据 (可选)"""
        raise NotImplementedError(f"Tick data not supported by {self.name}")
    
    def list_symbols(self, market: str = "A") -> List[str]:
        """获取股票列表"""
        raise NotImplementedError
    
    def get_trade_calendar(self, start: str, end: str) -> List[str]:
        """获取交易日历"""
        raise NotImplementedError
