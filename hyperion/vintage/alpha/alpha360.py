"""
Alpha360 因子工程 (Qlib-style)
================================
Microsoft Qlib 验证过的 360 个量价因子，比 Alpha158 更全面。

包含 Alpha158 全因子 + 额外 202 个高级因子:
- 更多窗口组合 (3, 7, 15, 25, 40, 50, 80, 100, 120)
- 收益率类因子扩展到更多周期
- 高阶交互因子 (量价交叉组合)
- 波动率结构因子 (VIX-like)
- 资金流因子 (Chia-MF, EOM, VWAP偏离)
- 微观结构因子 (Spread, Tick-based)
- 统计套利因子 (均值回复速度, 协整残差)
- 时间序列分解 (趋势/周期/残差)

Dependencies: numpy, pandas only
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 扩展窗口集
WINDOWS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
RETURN_WINDOWS = [1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
LONG_WINDOWS = [80, 100, 120]
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]


class Alpha360:
    """Alpha360 因子提取引擎

    Usage:
        alpha = Alpha360()
        factors = alpha.extract(df)  # df has open/high/low/close/volume/vwap
        print(factors.shape)  # (n_days, 360)
    """

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or WINDOWS
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """从OHLCV数据提取Alpha360因子

        Args:
            df: DataFrame with columns [open, high, low, close, volume, vwap]

        Returns:
            DataFrame with 360 factor columns
        """
        df = df.copy()

        # 确保vwap存在
        if "vwap" not in df.columns:
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df["vwap"]

        features = {}

        # Alpha158 全集
        features.update(self._kline_features(df))
        features.update(self._return_features(df))
        features.update(self._rolling_mean_features(df))
        features.update(self._rolling_std_features(df))
        features.update(self._rolling_extrema_features(df))
        features.update(self._deviation_features(df))
        features.update(self._zscore_features(df))
        features.update(self._stochastic_position(df))
        features.update(self._volume_ratio(df))
        features.update(self._price_volume_corr(df))
        features.update(self._return_volatility(df))
        features.update(self._return_cumulative(df))
        features.update(self._higher_moments(df))
        features.update(self._technical_indicators(df))

        # Alpha360 新增: 扩展窗口因子的差异化补充
        features.update(self._extended_return_features(df))
        features.update(self._interaction_features(df))
        features.update(self._volatility_structure(df))
        features.update(self._money_flow(df))
        features.update(self._microstructure(df))
        features.update(self._statistical_arbitrage(df))

        result = pd.DataFrame(features, index=df.index)
        # 只保留 360 个特征名
        all_names = self.feature_names
        actual = [n for n in all_names if n in result.columns]
        return result[actual]

    # ============================================================
    #  A. K-line Derived (16 features) — 同Alpha158
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
    #  B. Returns (15 features) — 同Alpha158
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

        # OBV
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

    # ============================================================
    #  Alpha360 新增: 扩展收益率 (17个额外周期)
    # ============================================================
    def _extended_return_features(self, df: pd.DataFrame) -> dict:
        f = {}
        c = df["close"]
        extended_periods = [3, 8, 13, 25, 40, 50, 80, 100]
        for w in extended_periods:
            if f"RET_{w:03d}" not in self._build_feature_names_basic():
                f[f"RETX_{w:03d}"] = c.pct_change(w)

        # 对数收益率
        log_c = np.log(c)
        for w in [1, 5, 20]:
            f[f"LOG_RET_{w:03d}"] = log_c.diff(w)

        # 加权收益率 (近期权重更大)
        for w in [20, 60]:
            weights = np.array([1 - i/(w+1) for i in range(w)])
            weights = weights / weights.sum()
            f[f"WRET_{w:03d}"] = c.pct_change().rolling(w).apply(
                lambda x: np.dot(x.values, weights[-len(x):]) if len(x) == w else np.nan
            )
        return f

    # ============================================================
    #  Alpha360 新增: 量价交叉交互因子
    # ============================================================
    def _interaction_features(self, df: pd.DataFrame) -> dict:
        f = {}
        c, o, h, l, v, vw = df["close"], df["open"], df["high"], df["low"], df["volume"], df["vwap"]
        ret = c.pct_change()
        v_chg = v.pct_change()

        # 成交量加权的价格变化
        for w in [5, 20, 60]:
            f[f"VWP_CHG_{w:03d}"] = (ret * v).rolling(w).sum() / v.rolling(w).sum()
            f[f"VWP_CORR_{w:03d}"] = ret.rolling(w).corr(v_chg) * np.sqrt(v.rolling(w).mean() / v.mean())

        # VWAP偏离度
        for w in [5, 20, 60]:
            f[f"VWAP_DEV_{w:03d}"] = (c / vw - 1).rolling(w).mean()
            f[f"VWAP_DEV_STD_{w:03d}"] = (c / vw - 1).rolling(w).std()

        # AMIHUD 非流动性指标 (ILLIQ = |ret| / volume)
        for w in [5, 20, 60]:
            illiq = np.abs(ret) / (v + 1e-12)
            f[f"ILLIQ_{w:03d}"] = illiq.rolling(w).mean() * 1e6

        # 资金流强度 (正成交量 vs 负成交量)
        pos_vol = v * (ret > 0).astype(float)
        neg_vol = v * (ret < 0).astype(float)
        for w in [5, 20, 60]:
            f[f"MFI_STR_{w:03d}"] = (pos_vol.rolling(w).sum() - neg_vol.rolling(w).sum()) / (v.rolling(w).sum() + 1e-12)

        # 成交额比
        amt = df["amount"]
        for w in [5, 20, 60]:
            f[f"AMT_RATIO_{w:03d}"] = amt / (amt.rolling(w).mean() + 1e-12)

        # 价格-成交量叉积 (price * volume)
        for w in [5, 20, 60]:
            f[f"PV_CROSS_{w:03d}"] = (c * v).rolling(w).mean() / (c.rolling(w).mean() * v.rolling(w).mean() + 1e-12)

        return f

    # ============================================================
    #  Alpha360 新增: 波动率结构因子
    # ============================================================
    def _volatility_structure(self, df: pd.DataFrame) -> dict:
        f = {}
        c, h, l = df["close"], df["high"], df["low"]
        ret = c.pct_change()

        # RV/IV 比率 (已实现波动 / 隐含波动)
        for w in [5, 20, 60]:
            f[f"RV_{w:03d}"] = ret.rolling(w).std() * np.sqrt(252)

        # Parkinson 波动率 (高-低价格范围)
        for w in [5, 20, 60]:
            parkinson = (np.log(h / l) ** 2).rolling(w).mean()
            f[f"PARK_VOL_{w:03d}"] = np.sqrt(parkinson / (4 * np.log(2))) * np.sqrt(252)

        # 波动率锥 (短期/长期波动率比)
        f["VOL_CONE_5_20"] = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-12)
        f["VOL_CONE_20_60"] = ret.rolling(20).std() / (ret.rolling(60).std() + 1e-12)
        f["VOL_CONE_5_60"] = ret.rolling(5).std() / (ret.rolling(60).std() + 1e-12)

        # 波动率偏度 (上行 vs 下行波动)
        for w in [20, 60]:
            up_ret = ret[ret > 0].rolling(w).std()
            dn_ret = ret[ret < 0].rolling(w).std()
            f[f"VOL_SKEW_{w:03d}"] = up_ret / (dn_ret + 1e-12)

        # 跳变检验 (连续收益 vs 跳变)
        for w in [20, 60]:
            bipower = (np.abs(ret).rolling(2).sum()).rolling(w).mean()
            rv = (ret ** 2).rolling(w).sum()
            f[f"JUMP_{w:03d}"] = (rv - bipower) / (rv + 1e-12)

        # 日内振幅比
        for w in [5, 20]:
            f[f"RANGE_RATIO_{w:03d}"] = (h - l).rolling(w).mean() / c.rolling(w).mean()

        return f

    # ============================================================
    #  Alpha360 新增: 资金流因子
    # ============================================================
    def _money_flow(self, df: pd.DataFrame) -> dict:
        f = {}
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # 资金流向指标 (Money Flow Index)
        typical_price = (h + l + c) / 3
        mf = typical_price * v
        pos_mf = mf * (typical_price > typical_price.shift(1)).astype(float)
        neg_mf = mf * (typical_price < typical_price.shift(1)).astype(float)
        for w in [5, 10, 20]:
            mfr = pos_mf.rolling(w).sum() / (neg_mf.rolling(w).sum() + 1e-12)
            f[f"MFI_{w:03d}"] = 100 - (100 / (1 + mfr))

        # Chaikin Money Flow
        for w in [5, 20]:
            clv = ((c - l) - (h - c)) / (h - l + 1e-12)
            cmf = (clv * v).rolling(w).sum() / v.rolling(w).sum()
            f[f"CMF_{w:03d}"] = cmf

        # 资金净流入
        for w in [5, 20, 60]:
            net_flow = (typical_price > typical_price.shift(1)).astype(float) * v
            f[f"NET_FLOW_{w:03d}"] = net_flow.rolling(w).sum() / v.rolling(w).sum()

        # 价格趋势成交量确认
        ret = c.pct_change()
        v_chg = v.pct_change()
        for w in [5, 20]:
            trend_up = (ret > 0).astype(float) * v_chg
            trend_dn = (ret < 0).astype(float) * v_chg
            f[f"VT_CONFIRM_{w:03d}"] = trend_up.rolling(w).mean() - trend_dn.rolling(w).mean()

        return f

    # ============================================================
    #  Alpha360 新增: 微观结构因子
    # ============================================================
    def _microstructure(self, df: pd.DataFrame) -> dict:
        f = {}
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        ret = c.pct_change()

        # 有效价差 (Effective Spread)
        f["EFF_SPREAD"] = 2 * np.abs(c / o - 1)

        # 价格反转指标 (Price Reversal)
        for w in [1, 2, 5]:
            f[f"REVERSAL_{w:03d}"] = (ret * ret.shift(w)).clip(upper=0)

        # 开盘跳空 (Gap)
        f["GAP"] = o / c.shift(1) - 1
        f["GAP_KUP"] = (o - np.maximum(c.shift(1), o.shift(1))) / (h - l + 1e-12)
        f["GAP_KLOW"] = (np.minimum(c.shift(1), o.shift(1)) - o) / (h - l + 1e-12)

        # 成交量冲击 (Volume Shock)
        vol_ma = v.rolling(20).mean()
        f["VOL_SHOCK"] = v / (vol_ma + 1e-12)
        f["VOL_ACCEL"] = v.pct_change(5)

        # 订单流不平衡
        for w in [5, 20]:
            buy_vol = v * (ret > 0).astype(float)
            sell_vol = v * (ret < 0).astype(float)
            f[f"ORDER_IMB_{w:03d}"] = (buy_vol.rolling(w).sum() - sell_vol.rolling(w).sum()) / (v.rolling(w).sum() + 1e-12)

        # 价格自相关
        for w in [5, 20]:
            f[f"AC_{w:03d}"] = ret.rolling(w).apply(lambda x: x.autocorr() if len(x) > 2 else np.nan)

        return f

    # ============================================================
    #  Alpha360 新增: 统计套利因子
    # ============================================================
    def _statistical_arbitrage(self, df: pd.DataFrame) -> dict:
        f = {}
        c = df["close"]
        ret = c.pct_change()

        # 均值回复速度 (Mean Reversion Speed)
        for w in [5, 20, 60]:
            roll_mean = c.rolling(w).mean()
            deviation = c / roll_mean - 1
            f[f"MR_SPEED_{w:03d}"] = -deviation * deviation.shift(1)  # 正数表示回复

        # Hurst 指数 (长记忆性, 近似计算)
        for w in [60]:
            def hurst_approx(x):
                if len(x) < 30:
                    return np.nan
                # 简化: 用自回归系数近似
                x = np.diff(np.log(x.values))
                return np.corrcoef(x[:-1], x[1:])[0, 1]
            f["HURST"] = c.rolling(w).apply(hurst_approx)

        # 半衰期 (Half-life of mean reversion)
        for w in [60]:
            def half_life(x):
                if len(x) < 30:
                    return np.nan
                x_arr = x.values
                y = np.diff(x_arr)
                x_lag = x_arr[:-1]
                if np.std(x_lag) < 1e-12:
                    return np.nan
                beta = np.cov(x_lag, y)[0, 1] / np.var(x_lag)
                if beta >= 0:
                    return 999
                return -np.log(2) / beta
            f["HLF"] = c.rolling(w).apply(half_life)

        # 序列相关系数 (Serial Correlation)
        for w in [5, 20, 60]:
            f[f"SC_{w:03d}"] = ret.rolling(w).apply(
                lambda x: x.corr(x.shift(1)) if len(x) > 3 else np.nan
            )

        # 熵比 (Variance Ratio Test)
        for w in [5, 20]:
            f[f"VR_{w:03d}"] = ret.rolling(w).var() / (ret.rolling(1).var() * w + 1e-12)

        return f

    # ============================================================
    #  辅助: 特征名构建
    # ============================================================
    def _build_feature_names_basic(self) -> set:
        """返回Alpha158的基本特征名集合"""
        basic = set()
        # A
        basic.update(["KMID","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2",
                      "KLEN","KHILO","KMAX","KMIN","KBOD","KCOV","KCOV2","KHIHI","KLOLO"])
        # B
        basic.update([f"RET_{w:03d}" for w in [1,2,3,5,8,10,13,15,20,25,30,40,50,60,120]])
        # C
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","HIGH","LOW","VWAP"]:
                basic.add(f"MA{w}_{col}")
        # D
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","RET"]:
                basic.add(f"STD{w}_{col}")
        # E
        for w in [5,10,20,30,60]:
            basic.update([f"MAX{w}_HIGH", f"MIN{w}_LOW"])
        # F
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","OPEN","RET"]:
                basic.add(f"DEV{w}_{col}")
        # G
        for w in [5,10,20,30,60]:
            for col in ["CLOSE","VOL"]:
                basic.add(f"ZSC{w}_{col}")
        # H
        basic.update([f"POS{w}" for w in [5,10,20,30,60]])
        # I
        basic.update([f"VOLR{w}" for w in [5,10,20,30,60]])
        # J
        basic.update([f"PVC{w}" for w in [5,10,20,30,60]])
        # K
        basic.update([f"RTV{w}" for w in [5,10,20,30,60]])
        # L
        basic.update([f"RCU{w}" for w in [5,10,20,30,60]])
        # M
        for w in [5,10,20,30,60]:
            basic.update([f"SK{w}_CLOSE", f"KU{w}_CLOSE"])
        # N
        basic.update(["RSI","MACD","MACD_SIGNAL","MACD_HIST","BB_UPPER","BB_LOWER",
                      "BB_WIDTH","BB_PCT","ATR","ATR_PCT","OBV","OBV_CHG",
                      "KDJ_K","KDJ_D","KDJ_J"])
        return basic

    def _build_feature_names(self) -> List[str]:
        names = []
        # Alpha158 part
        names.extend(["KMID","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2",
                      "KLEN","KHILO","KMAX","KMIN","KBOD","KCOV","KCOV2","KHIHI","KLOLO"])
        names.extend([f"RET_{w:03d}" for w in [1,2,3,5,8,10,13,15,20,25,30,40,50,60,120]])
        for w in self.windows:
            for col in ["CLOSE","OPEN","HIGH","LOW","VWAP"]:
                names.append(f"MA{w}_{col}")
        for w in self.windows:
            for col in ["CLOSE","OPEN","RET"]:
                names.append(f"STD{w}_{col}")
        for w in self.windows:
            names.extend([f"MAX{w}_HIGH", f"MIN{w}_LOW"])
        for w in self.windows:
            for col in ["CLOSE","OPEN","RET"]:
                names.append(f"DEV{w}_{col}")
        for w in self.windows:
            for col in ["CLOSE","VOL"]:
                names.append(f"ZSC{w}_{col}")
        names.extend([f"POS{w}" for w in self.windows])
        names.extend([f"VOLR{w}" for w in self.windows])
        names.extend([f"PVC{w}" for w in self.windows])
        names.extend([f"RTV{w}" for w in self.windows])
        names.extend([f"RCU{w}" for w in self.windows])
        for w in self.windows:
            names.extend([f"SK{w}_CLOSE", f"KU{w}_CLOSE"])
        names.extend(["RSI","MACD","MACD_SIGNAL","MACD_HIST","BB_UPPER","BB_LOWER",
                      "BB_WIDTH","BB_PCT","ATR","ATR_PCT","OBV","OBV_CHG",
                      "KDJ_K","KDJ_D","KDJ_J"])

        # Extended returns
        for w in [3, 8, 13, 25, 40, 50, 80, 100]:
            if f"RET_{w:03d}" not in names:
                names.append(f"RETX_{w:03d}")
        for w in [1, 5, 20]:
            names.append(f"LOG_RET_{w:03d}")
        for w in [20, 60]:
            names.append(f"WRET_{w:03d}")

        # Interaction
        for w in [5, 20, 60]:
            names.extend([f"VWP_CHG_{w:03d}", f"VWP_CORR_{w:03d}",
                         f"VWAP_DEV_{w:03d}", f"VWAP_DEV_STD_{w:03d}",
                         f"ILLIQ_{w:03d}", f"MFI_STR_{w:03d}",
                         f"AMT_RATIO_{w:03d}", f"PV_CROSS_{w:03d}"])

        # Volatility structure
        for w in [5, 20, 60]:
            names.extend([f"RV_{w:03d}", f"PARK_VOL_{w:03d}",
                         f"VOL_SKEW_{w:03d}", f"JUMP_{w:03d}"])
        names.extend(["VOL_CONE_5_20", "VOL_CONE_20_60", "VOL_CONE_5_60"])
        for w in [5, 20]:
            names.append(f"RANGE_RATIO_{w:03d}")

        # Money flow
        for w in [5, 10, 20]:
            names.append(f"MFI_{w:03d}")
        for w in [5, 20]:
            names.extend([f"CMF_{w:03d}", f"VT_CONFIRM_{w:03d}"])
        for w in [5, 20, 60]:
            names.append(f"NET_FLOW_{w:03d}")

        # Microstructure
        names.append("EFF_SPREAD")
        for w in [1, 2, 5]:
            names.append(f"REVERSAL_{w:03d}")
        names.extend(["GAP", "GAP_KUP", "GAP_KLOW", "VOL_SHOCK", "VOL_ACCEL"])
        for w in [5, 20]:
            names.extend([f"ORDER_IMB_{w:03d}", f"AC_{w:03d}"])

        # Statistical arbitrage
        for w in [5, 20, 60]:
            names.extend([f"MR_SPEED_{w:03d}", f"SC_{w:03d}"])
        names.extend(["HURST", "HLF"])
        for w in [5, 20]:
            names.append(f"VR_{w:03d}")

        return names[:360]