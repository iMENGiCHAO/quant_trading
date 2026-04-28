"""
AkShare 数据源适配器
====================
免费A股数据源，无需API Key。
支持日线、分钟线、股票列表、交易日历。

融合设计:
- Freqtrade: download-data命令模式
- Qlib: 数据格式标准化
"""
from __future__ import annotations

import logging
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import numpy as np

from hyperion.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

# AkShare字段映射到标准OHLCV
COLUMN_MAP = {
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "change_pct",
    "振幅": "amplitude",
}


class AkShareSource(BaseDataSource):
    """AkShare免费A股数据源
    
    数据来源: https://github.com/akfamily/akshare
    无需注册，直接使用。
    
    Usage:
        source = AkShareSource()
        df = source.download_daily(["000001", "000002"], "2024-01-01", "2024-12-31")
    """
    
    @property
    def name(self) -> str:
        return "akshare"
    
    def download_daily(self, symbols: List[str],
                       start: str, end: str,
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """从AkShare下载A股日线数据"""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError(
                "akshare is required for AkShareSource. "
                "Install with: pip install akshare"
            )
        
        all_frames = []
        
        for sym in symbols:
            try:
                # 处理代码格式: 自动添加后缀
                code = self._normalize_symbol(sym)
                
                # 下载个股历史数据
                df = ak.stock_zh_a_hist(
                    symbol=code.replace(".SH", "").replace(".SZ", ""),
                    period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust="qfq"  # 前复权
                )
                
                if df is None or df.empty:
                    logger.warning(f"No data for {sym}")
                    continue
                
                # 重命名列
                df.rename(columns=COLUMN_MAP, inplace=True)
                
                # 确保日期列
                df["日期"] = pd.to_datetime(df.get("日期", df.index))
                df.set_index("日期", inplace=True)
                
                # 标准化字段
                std_df = pd.DataFrame(index=df.index)
                for ak_col, std_col in COLUMN_MAP.items():
                    if ak_col in df.columns:
                        std_df[std_col] = pd.to_numeric(df[ak_col], errors="coerce")
                
                # 计算衍生字段
                if "close" in std_df.columns and "open" in std_df.columns:
                    std_df["change_pct"] = std_df.get("change_pct", 
                        std_df["close"].pct_change() * 100)
                
                if "high" in std_df.columns and "low" in std_df.columns:
                    std_df["amplitude"] = std_df.get("amplitude",
                        (std_df["high"] - std_df["low"]) / std_df["close"] * 100)
                
                std_df["symbol"] = sym
                std_df.set_index("symbol", append=True, inplace=True)
                
                all_frames.append(std_df)
                
            except Exception as e:
                logger.error(f"Failed to download {sym}: {e}")
        
        if not all_frames:
            return pd.DataFrame()
        
        result = pd.concat(all_frames)
        result = result.swaplevel().sort_index()  # (date, symbol)
        return result
    
    def list_symbols(self, market: str = "A") -> List[str]:
        """获取A股全部股票列表"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            symbols = []
            for _, row in df.iterrows():
                code = row["代码"]
                market_code = row.get("市场编号", "")
                if market_code == "SH":
                    symbols.append(f"{code}.SH")
                else:
                    symbols.append(f"{code}.SZ")
            return symbols
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []
    
    @staticmethod
    def _normalize_symbol(sym: str) -> str:
        """标准化股票代码
        
        Examples:
            '000001' → '000001.SZ'
            '600000' → '600000.SH'
            '000300.SH' → '000300.SH'
        """
        sym = sym.strip().upper()
        
        if "." in sym:
            # Already has market suffix
            return sym
        
        # 判断市场
        if sym.startswith(("6", "9")):
            return f"{sym}.SH"
        elif sym.startswith(("0", "3", "2")):
            return f"{sym}.SZ"
        else:
            return f"{sym}.SZ"  # default
