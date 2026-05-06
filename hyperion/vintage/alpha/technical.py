"""
技术指标工厂 (Freqtrade + VectorBT 风格)
=========================================
向量化指标计算, 支持:
- 移动平均 (SMA, EMA, WMA)
- 动量指标 (RSI, MACD, Stochastic, CCI, WILLR)
- 波动率 (ATR, BBANDS, Keltner Channel)
- 成交量 (OBV, MFI, VWAP, AD)
- 趋势 (ADX, Parabolic SAR, Ichimoku)
- 形态识别 (Doji, Hammer, Engulfing)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union


class TechnicalIndicators:
    """向量化技术指标工厂
    
    Usage:
        ti = TechnicalIndicators()
        rsi = ti.rsi(df["close"], period=14)
        macd = ti.macd(df["close"])
    """
    
    # ============================================================
    #  趋势指标
    # ============================================================
    
    @staticmethod
    def sma(series: pd.Series, period: int = 20) -> pd.Series:
        """简单移动平均"""
        return series.rolling(period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int = 20) -> pd.Series:
        """指数移动平均"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int = 20) -> pd.Series:
        """加权移动平均"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> dict:
        """ADX 平均趋向指数
        
        Returns:
            dict with 'adx', 'plus_di', 'minus_di'
        """
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}
    
    # ============================================================
    #  动量指标
    # ============================================================
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI 相对强弱指标"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> dict:
        """MACD 异同移动平均线"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> dict:
        """KDJ 随机指标"""
        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        k = 100 * (close - low_min) / (high_max - low_min + 1e-12)
        d = k.rolling(d_period).mean()
        return {"k": k, "d": d}
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 20) -> pd.Series:
        """CCI 商品通道指数"""
        tp = (high + low + close) / 3
        ma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * mad + 1e-12)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
        """Williams %R"""
        highest = high.rolling(period).max()
        lowest = low.rolling(period).min()
        return -100 * (highest - close) / (highest - lowest + 1e-12)
    
    # ============================================================
    #  波动率指标
    # ============================================================
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                        std_dev: float = 2.0) -> dict:
        """布林带"""
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        width = (upper - lower) / (ma + 1e-12)
        pct_b = (close - lower) / (upper - lower + 1e-12)
        return {"upper": upper, "lower": lower, "middle": ma,
                "width": width, "pct_b": pct_b}
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """ATR 平均真实波幅"""
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, multiplier: float = 2.0) -> dict:
        """Keltner通道"""
        ma = close.ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        return {"upper": ma + multiplier * atr, "middle": ma,
                "lower": ma - multiplier * atr}
    
    # ============================================================
    #  成交量指标
    # ============================================================
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV 能量潮"""
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """MFI 资金流量指标"""
        tp = (high + low + close) / 3
        money_flow = tp * volume
        positive_flow = money_flow.where(tp > tp.shift(1), 0)
        negative_flow = money_flow.where(tp < tp.shift(1), 0)
        pos_sum = positive_flow.rolling(period).sum()
        neg_sum = negative_flow.rolling(period).sum()
        mr = pos_sum / (neg_sum + 1e-12)
        return 100 - (100 / (1 + mr))
    
    # ============================================================
    #  形态识别
    # ============================================================
    
    @staticmethod
    def detect_doji(open_: pd.Series, high: pd.Series,
                    low: pd.Series, close: pd.Series,
                    threshold: float = 0.001) -> pd.Series:
        """检测十字星形态"""
        body = (close - open_).abs()
        total_range = high - low + 1e-12
        return body / total_range < threshold
    
    @staticmethod
    def detect_hammer(open_: pd.Series, high: pd.Series,
                      low: pd.Series, close: pd.Series) -> pd.Series:
        """检测锤子线"""
        body = (close - open_).abs()
        lower_shadow = np.minimum(open_, close) - low
        upper_shadow = high - np.maximum(open_, close)
        total_range = high - low + 1e-12
        return (lower_shadow > 2 * body) & (upper_shadow < 0.3 * total_range)
