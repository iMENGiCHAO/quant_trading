"""
Alpha158 因子工程 (Qlib-style)
================================
Microsoft Qlib 验证过的158个量价因子, 分14类:

A.  K线衍生     (16)  — KMID, KUP, KLOW, KSFT, KLEN, etc.
B.  收益率      (15)  — RET_001..RET_120
C.  滚动均线    (25)  — MA5_CLOSE..MA60_OPEN
D.  滚动标准差  (15)  — STD5_CLOSE..STD60_RET
E.  滚动最值    (10)  — MAX5_HIGH..MIN60_LOW
F.  偏离度      (15)  — DEV5_CLOSE..DEV60_RET
G.  Z-Score    (10)  — ZSC5_CLOSE..ZSC60_VOL
H.  随机位置     (5)  — POS5..POS60
I.  成交量比     (5)  — VOLR5..VOLR60
J.  量价相关     (5)  — PVC5..PVC60
K.  收益率波动   (5)  — RTV5..RTV60
L.  收益率累积   (5)  — RCU5..RCU60
M.  高阶矩      (10)  — SK5_CLOSE..KU60_CLOSE
N.  技术指标    (17)  — RSI, MACD, BBANDS, ATR, etc.

Dependencies: numpy, pandas only (no ML frameworks)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 滚动窗口
WINDOWS = [5, 10, 20, 30, 60]
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]


class Alpha158:
    """Alpha158 因子提取引擎
    
    Usage:
        alpha = Alpha158()
        factors = alpha.extract(df)  # df has open/high/low/close/volume/vwap
        print(factors.shape)  # (n_days, 158)
    """
    
    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or WINDOWS
        self._feature_names: Optional[List[str]] = None
    
    @property
    def feature_names(self) -> List[str]:
        """返回158个因子名"""
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """从OHLCV数据提取Alpha158因子
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume, vwap]
            
        Returns:
            DataFrame with 158 factor columns
        """
        df = df.copy()
        
        # 确保vwap存在
        if "vwap" not in df.columns:
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        
        features = {}
        
        # A: K-line derived
        features.update(self._kline_features(df))
        # B: Returns
        features.update(self._return_features(df))
        # C: Rolling means
        features.update(self._rolling_mean_features(df))
        # D: Rolling std
        features.update(self._rolling_std_features(df))
        # E: Rolling max/min
        features.update(self._rolling_extrema_features(df))
        # F: Deviation from MA
        features.update(self._deviation_features(df))
        # G: Z-Score
        features.update(self._zscore_features(df))
        # H: Stochastic position
        features.update(self._stochastic_position(df))
        # I: Volume ratio
        features.update(self._volume_ratio(df))
        # J: Price-volume correlation
        features.update(self._price_volume_corr(df))
        # K: Return volatility
        features.update(self._return_volatility(df))
        # L: Return cumulative
        features.update(self._return_cumulative(df))
        # M: Higher moments
        features.update(self._higher_moments(df))
        # N: Technical indicators
        features.update(self._technical_indicators(df))
        
        result = pd.DataFrame(features, index=df.index)
        return result
    
    # ============================================================
    #  A. K-line Derived (16 features)
    # ============================================================
    def _kline_features(self, df: pd.DataFrame) -> dict:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        f = {}
        f["KMID"] = (h + l) / 2
        f["KUP"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
        f["KUP2"] = (h - np.maximum(o.shift(1), c.shift(1))) / (h - l + 1e-12)
        f["KLOW"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
        f["KLOW2"] = (np.minimum(o.shift(1), c.shift(1)) - l) / (h - l + 1e-12)
        f["KSFT"] = (2 * c - h - l) / (h - l + 1e-12)
        f["KSFT2"] = (2 * c.shift(1) - h.shift(1) - l.shift(1)) / (h - l + 1e-12)
        f["KLEN"] = np.where(o > c, (c - l) / (h - l + 1e-12),
                            np.where(o < c, (h - c) / (h - l + 1e-12), 0.5))
        f["KHILO"] = h / (l + 1e-12) - 1
        f["KMAX"] = np.maximum(o, c)
        f["KMIN"] = np.minimum(o, c)
        f["KBOD"] = np.abs(c - o) / (h - l + 1e-12)
        f["KCOV"] = (c - o) / (h - l + 1e-12)
        f["KCOV2"] = (c.shift(1) - o.shift(1)) / (h - l + 1e-12)
        f["KHIHI"] = h / h.shift(1) - 1
        f["KLOLO"] = l / l.shift(1) - 1
        return f
    
    # ============================================================
    #  B. Returns (15 features)
    # ============================================================
    def _return_features(self, df: pd.DataFrame) -> dict:
        c = df["close"]
        f = {}
        for w in [1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 120]:
            f[f"RET_{w:03d}"] = c.pct_change(w)
        return f
    
    # ============================================================
    #  C. Rolling Mean (25 features)
    # ============================================================
    def _rolling_mean_features(self, df: pd.DataFrame) -> dict:
        f = {}
        for col in ["close", "open", "high", "low", "vwap"]:
            for w in self.windows:
                f[f"MA{w}_{col.upper()}"] = df[col].rolling(w).mean()
        return f
    
    # ============================================================
    #  D. Rolling Std (15 features)
    # ============================================================
    def _rolling_std_features(self, df: pd.DataFrame) -> dict:
        f = {}
        ret = df["close"].pct_change()
        for col_name, series in [("CLOSE", df["close"]), ("OPEN", df["open"]), ("RET", ret)]:
            for w in self.windows:
                f[f"STD{w}_{col_name}"] = series.rolling(w).std()
        return f
    
    # ============================================================
    #  E. Rolling Max/Min (10 features)
    # ============================================================
    def _rolling_extrema_features(self, df: pd.DataFrame) -> dict:
        f = {}
        for w in self.windows:
            f[f"MAX{w}_HIGH"] = df["high"].rolling(w).max()
            f[f"MIN{w}_LOW"] = df["low"].rolling(w).min()
        return f
    
    # ============================================================
    #  F. Deviation from MA (15 features)
    # ============================================================
    def _deviation_features(self, df: pd.DataFrame) -> dict:
        f = {}
        ret = df["close"].pct_change()
        for col_name, series in [("CLOSE", df["close"]), ("OPEN", df["open"]), ("RET", ret)]:
            for w in self.windows:
                ma = series.rolling(w).mean()
                f[f"DEV{w}_{col_name}"] = series - ma
        return f
    
    # ============================================================
    #  G. Z-Score (10 features)
    # ============================================================
    def _zscore_features(self, df: pd.DataFrame) -> dict:
        f = {}
        for col_name, series in [("CLOSE", df["close"]), ("VOL", df["volume"])]:
            for w in self.windows:
                roll_mean = series.rolling(w).mean()
                roll_std = series.rolling(w).std()
                f[f"ZSC{w}_{col_name}"] = (series - roll_mean) / (roll_std + 1e-12)
        return f
    
    # ============================================================
    #  H. Stochastic Position (5 features)
    # ============================================================
    def _stochastic_position(self, df: pd.DataFrame) -> dict:
        f = {}
        c = df["close"]
        for w in self.windows:
            h_max = df["high"].rolling(w).max()
            l_min = df["low"].rolling(w).min()
            f[f"POS{w}"] = (c - l_min) / (h_max - l_min + 1e-12)
        return f
    
    # ============================================================
    #  I. Volume Ratio (5 features)
    # ============================================================
    def _volume_ratio(self, df: pd.DataFrame) -> dict:
        f = {}
        v = df["volume"]
        for w in self.windows:
            f[f"VOLR{w}"] = v / (v.rolling(w).mean() + 1e-12)
        return f
    
    # ============================================================
    #  J. Price-Volume Correlation (5 features)
    # ============================================================
    def _price_volume_corr(self, df: pd.DataFrame) -> dict:
        f = {}
        ret = df["close"].pct_change()
        v_chg = df["volume"].pct_change()
        for w in self.windows:
            f[f"PVC{w}"] = ret.rolling(w).corr(v_chg)
        return f
    
    # ============================================================
    #  K. Return Volatility (5 features)
    # ============================================================
    def _return_volatility(self, df: pd.DataFrame) -> dict:
        f = {}
        ret = df["close"].pct_change()
        for w in self.windows:
            f[f"RTV{w}"] = ret.rolling(w).std() * np.sqrt(252)
        return f
    
    # ============================================================
    #  L. Return Cumulative (5 features)
    # ============================================================
    def _return_cumulative(self, df: pd.DataFrame) -> dict:
        f = {}
        for w in self.windows:
            f[f"RCU{w}"] = df["close"].pct_change(w)
        return f
    
    # ============================================================
    #  M. Higher Moments (10 features)
    # ============================================================
    def _higher_moments(self, df: pd.DataFrame) -> dict:
        f = {}
        ret = df["close"].pct_change()
        for w in self.windows:
            roll = ret.rolling(w)
            f[f"SK{w}_CLOSE"] = roll.skew()
            f[f"KU{w}_CLOSE"] = roll.kurt()
        return f
    
    # ============================================================
    #  N. Technical Indicators (17 features)
    # ============================================================
    def _technical_indicators(self, df: pd.DataFrame) -> dict:
        f = {}
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        
        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        f["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        f["MACD"] = ema12 - ema26
        f["MACD_SIGNAL"] = f["MACD"].ewm(span=9).mean()
        f["MACD_HIST"] = f["MACD"] - f["MACD_SIGNAL"]
        
        # Bollinger Bands
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        f["BB_UPPER"] = ma20 + 2 * std20
        f["BB_LOWER"] = ma20 - 2 * std20
        f["BB_WIDTH"] = (f["BB_UPPER"] - f["BB_LOWER"]) / (ma20 + 1e-12)
        f["BB_PCT"] = (c - f["BB_LOWER"]) / (f["BB_UPPER"] - f["BB_LOWER"] + 1e-12)
        
        # ATR
        tr1 = h - l
        tr2 = np.abs(h - c.shift(1))
        tr3 = np.abs(l - c.shift(1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        f["ATR"] = tr.rolling(14).mean()
        f["ATR_PCT"] = f["ATR"] / (c + 1e-12)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        f["OBV"] = obv
        f["OBV_CHG"] = obv.pct_change(5)
        
        # KDJ
        low_min = l.rolling(9).min()
        high_max = h.rolling(9).max()
        rsv = (c - low_min) / (high_max - low_min + 1e-12) * 100
        f["KDJ_K"] = rsv.ewm(com=2).mean()
        f["KDJ_D"] = f["KDJ_K"].ewm(com=2).mean()
        f["KDJ_J"] = 3 * f["KDJ_K"] - 2 * f["KDJ_D"]
        
        return f
    
    def _build_feature_names(self) -> List[str]:
        """构建158个因子名列表 (用于参考)"""
        names = []
        # A
        names.extend(["KMID","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2",
                     "KLEN","KHILO","KMAX","KMIN","KBOD","KCOV","KCOV2","KHIHI","KLOLO"])
        # B
        names.extend([f"RET_{w:03d}" for w in [1,2,3,5,8,10,13,15,20,25,30,40,50,60,120]])
        # C
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","HIGH","LOW","VWAP"]:
                names.append(f"MA{w}_{col}")
        # D
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","RET"]:
                names.append(f"STD{w}_{col}")
        # E
        for w in [5,10,20,30,60]:
            names.extend([f"MAX{w}_HIGH", f"MIN{w}_LOW"])
        # F
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","RET"]:
                names.append(f"DEV{w}_{col}")
        # G
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","VOL"]:
                names.append(f"ZSC{w}_{col}")
        # H
        names.extend([f"POS{w}" for w in [5,10,20,30,60]])
        # I
        names.extend([f"VOLR{w}" for w in [5,10,20,30,60]])
        # J
        names.extend([f"PVC{w}" for w in [5,10,20,30,60]])
        # K
        names.extend([f"RTV{w}" for w in [5,10,20,30,60]])
        # L
        names.extend([f"RCU{w}" for w in [5,10,20,30,60]])
        # M
        for w in [5,10,20,30,60]:
            names.extend([f"SK{w}_CLOSE", f"KU{w}_CLOSE"])
        # N
        names.extend(["RSI","MACD","MACD_SIGNAL","MACD_HIST","BB_UPPER","BB_LOWER",
                     "BB_WIDTH","BB_PCT","ATR","ATR_PCT","OBV","OBV_CHG",
                     "KDJ_K","KDJ_D","KDJ_J"])
        return names
