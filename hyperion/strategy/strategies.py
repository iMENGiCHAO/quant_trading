"""
Hyperion Pro — 具体策略实现
============================
华尔街级别多策略 Alpha 框架：
  1. TrendFollowingStrategy  — 双均线 + MACD 趋势跟踪
  2. MeanReversionStrategy   — 布林带 + RSI 超卖反弹
  3. MomentumBreakoutStrategy — 突破新高 + 量能确认
  4. VolumeAnomalyStrategy    — 放量异动检测
  5. MultiFactorAlphaStrategy — 综合评分选股
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from .base import (
    BaseStrategy, StrategySignal, SignalType,
    register_strategy
)


# ==========================================================
#  趋势跟踪策略
# ==========================================================
@register_strategy
class TrendFollowingStrategy(BaseStrategy):
    """
    双均线 + MACD 趋势跟踪
    
    入场条件:
      - MA10 > MA30 (金叉确认)
      - 价格回调至 MA10 附近
      - MACD DIF > DEA 且在零轴上方
    
    出场条件:
      - MA10 < MA30 (死叉)
      - 价格跌破 MA30 超过 2%
      - 或达到目标收益
    """
    
    def __init__(self):
        super().__init__("TrendFollowing", "双均线+MACD趋势跟踪")
    
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        close = df["close"].values
        if len(close) < 60:
            return []
        
        ma10 = pd.Series(close).rolling(10).mean().values
        ma30 = pd.Series(close).rolling(30).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        
        signals = []
        
        # 只在最近20个bar中找信号
        for i in range(max(60, len(close) - 20), len(close)):
            # 金叉确认：MA10 > MA30 且 前一日 MA10 <= MA30
            golden_cross = ma10[i] > ma30[i] and ma10[i-1] <= ma30[i-1]
            
            # 回调买入：MA10 > MA30，价格在 MA10 上方但不超过 3%
            pullback_buy = (ma10[i] > ma30[i] and 
                           close[i] > ma10[i] and 
                           close[i] < ma10[i] * 1.03 and
                           dif[i] > dea[i] and dif[i] > 0)
            
            if golden_cross or pullback_buy:
                # ATR 止损
                atr = self._calc_atr(df, i)
                stop_loss = close[i] - 2.0 * atr
                take_profit = close[i] + 3.0 * atr
                
                # 趋势强度
                trend_strength = min(1.0, (ma10[i] / ma30[i] - 1) * 20)
                confidence = 0.5 + 0.3 * trend_strength + 0.2 * (1 if dif[i] > 0 else 0)
                
                signals.append(StrategySignal(
                    timestamp=str(df["date"].iloc[i]),
                    code=code, name=name,
                    signal_type=SignalType.BUY,
                    price=float(close[i]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    position_pct=float(min(0.15, max(0.03, 0.08 * trend_strength))),
                    confidence=float(np.clip(confidence, 0.3, 0.9)),
                    strategy_name=self.name,
                    reason="金叉突破" if golden_cross else "回调买入",
                    indicators={"ma10": float(ma10[i]), "ma30": float(ma30[i]),
                               "dif": float(dif[i]), "dea": float(dea[i])}
                ))
        
        return signals
    
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        close = df["close"].values
        i = len(close) - 1
        
        if close[i] <= stop_loss:
            return True, "stop_loss"
        if close[i] >= take_profit:
            return True, "take_profit"
        
        if i > entry_date + 5:
            ma10 = pd.Series(close).rolling(10).mean().values
            ma30 = pd.Series(close).rolling(30).mean().values
            if ma10[i] < ma30[i] and i > 10:
                return True, "死叉信号"
        
        if i > entry_date + 60:
            if close[i] < entry_price:
                return True, "超时未盈利"
        
        return False, ""
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
        high = df["high"].values[max(0,i-period):i+1]
        low = df["low"].values[max(0,i-period):i+1]
        close = df["close"].values[max(0,i-period):i+1]
        tr = np.maximum(high[-len(close):] - low[-len(close):],
                       np.abs(high[-len(close):] - np.roll(close, 1)[-len(close):]))
        return float(pd.Series(tr).rolling(period).mean().iloc[-1]) if len(tr) >= period else 0.02 * close[-1]


# ==========================================================
#  均值回归策略
# ==========================================================
@register_strategy
class MeanReversionStrategy(BaseStrategy):
    """
    布林带 + RSI 均值回归
    
    入场条件:
      - 价格触及布林带下轨
      - RSI < 35 (超卖)
      - 成交量放大 (>1.5x 均值)
    
    出场条件:
      - 价格回归布林带中轨
      - RSI > 60
      - 或达到止损
    """
    
    def __init__(self):
        super().__init__("MeanReversion", "布林带+RSI均值回归")
    
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        close = df["close"].values
        volume = df["volume"].values
        if len(close) < 30:
            return []
        
        # 布林带
        ma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        # RSI
        rsi = self._calc_rsi(close, 14)
        
        # 成交量均值
        vol_ma20 = pd.Series(volume).rolling(20).mean().values
        
        signals = []
        
        for i in range(max(30, len(close) - 20), len(close)):
            touch_lower = close[i] <= lower[i] * 1.01
            rsi_oversold = rsi[i] < 35
            vol_surge = volume[i] > vol_ma20[i] * 1.5
            rsi_turning = rsi[i] > rsi[i-2] and rsi[i-2] < rsi[i-4]
            
            conditions = [touch_lower, rsi_oversold, rsi_turning]
            score = sum(conditions)
            
            if score >= 2:
                atr = self._calc_atr(df, i)
                stop_loss = close[i] - 1.5 * atr
                take_profit = ma20[i]
                
                confidence = 0.4 + 0.15 * score
                if vol_surge:
                    confidence += 0.15
                
                signals.append(StrategySignal(
                    timestamp=str(df["date"].iloc[i]),
                    code=code, name=name,
                    signal_type=SignalType.BUY,
                    price=float(close[i]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    position_pct=float(min(0.12, max(0.03, 0.05 * score))),
                    confidence=float(np.clip(confidence, 0.3, 0.85)),
                    strategy_name=self.name,
                    reason=f"超卖反弹(R={rsi[i]:.0f}, 触下轨)" if touch_lower else f"RSI超卖({rsi[i]:.0f})",
                    indicators={"rsi": float(rsi[i]), "boll_lower": float(lower[i]),
                               "boll_mid": float(ma20[i])}
                ))
        
        return signals
    
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        close = df["close"].values
        i = len(close) - 1
        
        if close[i] <= stop_loss:
            return True, "stop_loss"
        if close[i] >= take_profit:
            return True, "take_profit"
        
        rsi = self._calc_rsi(close, 14)
        if rsi[i] > 65 and i > entry_date + 3:
            return True, "RSI超买回归"
        
        if i > entry_date + 15:
            return True, "持有超时"
        
        return False, ""
    
    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
        high = df["high"].values[max(0,i-period):i+1]
        low = df["low"].values[max(0,i-period):i+1]
        close = df["close"].values[max(0,i-period):i+1]
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        return float(pd.Series(tr).rolling(period).mean().iloc[-1]) if len(tr) >= period else 0.02 * close[-1]


# ==========================================================
#  动量突破策略
# ==========================================================
@register_strategy
class MomentumBreakoutStrategy(BaseStrategy):
    """
    动量突破策略
    
    入场条件:
      - 价格突破 20 日高点
      - 成交量 > 1.5x 20日均量
      - MACD 柱状图在扩大
    
    出场条件:
      - 价格回调超过 ATR * 2
      - 或达到目标
    """
    
    def __init__(self):
        super().__init__("MomentumBreakout", "放量突破新高")
    
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        close = df["close"].values
        high = df["high"].values
        volume = df["volume"].values
        if len(close) < 30:
            return []
        
        # 20日最高价
        high_20 = pd.Series(high).rolling(20).max().values
        
        # 成交量均值
        vol_ma20 = pd.Series(volume).rolling(20).mean().values
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        macd_hist = 2 * (dif - dea)
        
        signals = []
        
        for i in range(max(30, len(close) - 20), len(close)):
            breakout = close[i] > high_20[i-1] * 1.005
            vol_confirm = volume[i] > vol_ma20[i] * 1.5
            macd_expanding = macd_hist[i] > macd_hist[i-1] and macd_hist[i] > 0
            
            if breakout and (vol_confirm or macd_expanding):
                atr = self._calc_atr(df, i)
                stop_loss = close[i] - 2.0 * atr
                take_profit = close[i] + 3.0 * atr
                
                confidence = 0.5
                if vol_confirm:
                    confidence += 0.15
                if macd_expanding:
                    confidence += 0.15
                if dif[i] > 0:
                    confidence += 0.1
                
                signals.append(StrategySignal(
                    timestamp=str(df["date"].iloc[i]),
                    code=code, name=name,
                    signal_type=SignalType.BUY,
                    price=float(close[i]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    position_pct=float(min(0.12, max(0.03, 0.06 * confidence))),
                    confidence=float(np.clip(confidence, 0.3, 0.9)),
                    strategy_name=self.name,
                    reason="放量突破20日高" if vol_confirm else "动量突破",
                    indicators={"high_20": float(high_20[i]), "macd_hist": float(macd_hist[i]),
                               "vol_ratio": float(volume[i] / max(vol_ma20[i], 1))}
                ))
        
        return signals
    
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        close = df["close"].values
        i = len(close) - 1
        
        if close[i] <= stop_loss:
            return True, "stop_loss"
        if close[i] >= take_profit:
            return True, "take_profit"
        
        # 追踪止损：从最高点回撤超过 2*ATR
        if i > entry_date + 3:
            peak = df["high"].values[entry_date:i+1].max()
            atr = self._calc_atr(df, i)
            if close[i] < peak - 2 * atr:
                return True, "追踪止损"
        
        if i > entry_date + 40:
            return True, "持有超时"
        
        return False, ""
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
        high = df["high"].values[max(0,i-period):i+1]
        low = df["low"].values[max(0,i-period):i+1]
        close = df["close"].values[max(0,i-period):i+1]
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        return float(pd.Series(tr).rolling(period).mean().iloc[-1]) if len(tr) >= period else 0.02 * close[-1]


# ==========================================================
#  成交量异动策略
# ==========================================================
@register_strategy
class VolumeAnomalyStrategy(BaseStrategy):
    """
    成交量异动检测
    
    入场条件:
      - 成交量突然放大至 5日均量的 2.5x 以上
      - 价格同步上涨 (排除放量下跌)
      - 之前10日处于缩量盘整
    
    出场条件:
      - 成交量回落至正常水平
      - 价格回落超过 2%
      - 持有 3 天
    """
    
    def __init__(self):
        super().__init__("VolumeAnomaly", "放量异动突破")
    
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        close = df["close"].values
        volume = df["volume"].values
        if len(close) < 30:
            return []
        
        vol_ma5 = pd.Series(volume).rolling(5).mean().values
        vol_ma20 = pd.Series(volume).rolling(20).mean().values
        
        signals = []
        
        for i in range(max(30, len(close) - 20), len(close)):
            vol_ratio_5 = volume[i] / max(vol_ma5[i], 1)
            vol_ratio_20 = volume[i] / max(vol_ma20[i], 1)
            price_up = close[i] > close[i-1]
            
            # 前10日是否缩量
            prev_vol = volume[i-10:i]
            prev_vol_mean = np.mean(prev_vol)
            was_quiet = prev_vol_mean < vol_ma20[i] * 0.8
            
            if vol_ratio_5 > 2.5 and price_up and was_quiet:
                stop_loss = close[i] * 0.97
                take_profit = close[i] * 1.06
                
                confidence = min(0.85, 0.4 + (vol_ratio_5 - 2.5) * 0.15)
                
                signals.append(StrategySignal(
                    timestamp=str(df["date"].iloc[i]),
                    code=code, name=name,
                    signal_type=SignalType.BUY,
                    price=float(close[i]),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    position_pct=float(min(0.10, 0.05)),
                    confidence=float(confidence),
                    strategy_name=self.name,
                    reason=f"放量异动(量比{vol_ratio_5:.1f}x)",
                    indicators={"vol_ratio_5": float(vol_ratio_5),
                               "vol_ratio_20": float(vol_ratio_20)}
                ))
        
        return signals
    
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        close = df["close"].values
        volume = df["volume"].values
        i = len(close) - 1
        
        if close[i] <= stop_loss:
            return True, "stop_loss"
        if close[i] >= take_profit:
            return True, "take_profit"
        
        # 量能回落
        if i > entry_date + 2:
            vol_ma5 = np.mean(volume[i-4:i+1])
            entry_vol = volume[entry_date]
            if vol_ma5 < entry_vol * 0.4:
                return True, "量能衰竭"
        
        if i > entry_date + 5:
            return True, "持有超时(短线)"
        
        return False, ""


# ==========================================================
#  多因子 Alpha 策略
# ==========================================================
@register_strategy
class MultiFactorAlphaStrategy(BaseStrategy):
    """
    多因子综合 Alpha 策略
    
    因子:
      - 动量因子: 20日收益 / 波动率
      - 趋势因子: 均线多头排列 + MACD
      - 量价因子: 量比 + 价量配合
      - 反转因子: RSI 超卖 + 布林下轨
    
    综合评分 > 阈值 → 买入信号
    """
    
    def __init__(self):
        super().__init__("MultiFactorAlpha", "多因子综合Alpha选股")
    
    def generate_signals(self, df: pd.DataFrame, code: str, name: str) -> List[StrategySignal]:
        close = df["close"].values
        volume = df["volume"].values
        if len(close) < 60:
            return []
        
        n = len(close)
        
        # 动量因子 (满分25)
        returns_20 = (close[-1] / close[-20] - 1) if n >= 21 else 0
        vol_20 = np.std(pd.Series(close).pct_change().dropna().tail(20)) * np.sqrt(252)
        mom_score = min(25, max(0, 12.5 + (returns_20 / max(vol_20, 0.01)) * 5))
        
        # 趋势因子 (满分30)
        ma10 = pd.Series(close).rolling(10).mean().values
        ma30 = pd.Series(close).rolling(30).mean().values
        trend_score = 15.0
        if ma10[-1] > ma30[-1]:
            trend_score += 10
        if close[-1] > ma30[-1]:
            trend_score += 5
        trend_score = min(30, max(0, trend_score))
        
        # 量价因子 (满分20)
        vol_ma20 = np.mean(volume[-20:]) if n >= 20 else volume[-1]
        vol_ratio = volume[-1] / max(vol_ma20, 1)
        vol_score = min(20, max(0, 10 + (vol_ratio - 1) * 5))
        
        # 反转因子 (满分15)
        rsi = self._calc_rsi(close, 14)
        reversal_score = 0
        if rsi[-1] < 35:
            reversal_score = 15
        elif rsi[-1] < 45:
            reversal_score = 10
        elif rsi[-1] < 55:
            reversal_score = 5
        reversal_score = min(15, reversal_score)
        
        # 综合评分
        total_score = mom_score + trend_score + vol_score + reversal_score
        
        if total_score > 55:
            atr = self._calc_atr(df, n - 1)
            stop_loss = close[-1] - 2.0 * atr
            take_profit = close[-1] + 2.5 * atr
            confidence = min(0.9, 0.4 + (total_score - 55) / 100)
            
            return [StrategySignal(
                timestamp=str(df["date"].iloc[-1]),
                code=code, name=name,
                signal_type=SignalType.BUY,
                price=float(close[-1]),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                position_pct=float(min(0.12, max(0.03, total_score / 800))),
                confidence=float(confidence),
                strategy_name=self.name,
                reason=f"多因子综合评分{total_score:.0f}",
                indicators={"momentum": mom_score, "trend": trend_score,
                           "volume": vol_score, "reversal": reversal_score}
            )]
        
        return []
    
    def should_exit(self, df: pd.DataFrame, entry_price: float,
                    entry_date: int, stop_loss: float,
                    take_profit: float) -> Tuple[bool, str]:
        close = df["close"].values
        i = len(close) - 1
        
        if close[i] <= stop_loss:
            return True, "stop_loss"
        if close[i] >= take_profit:
            return True, "take_profit"
        
        # 重算因子分数
        if i > entry_date + 10:
            returns_20 = (close[i] / close[max(0,i-20)] - 1)
            ma10 = pd.Series(close).rolling(10).mean().values[i]
            ma30 = pd.Series(close).rolling(30).mean().values[i]
            rsi = self._calc_rsi(close, 14)
            
            trend_broken = ma10 < ma30
            momentum_lost = returns_20 < -0.05
            rsi_weak = rsi[i] < 30
            
            if (trend_broken and momentum_lost) or rsi_weak:
                return True, "多因子恶化"
        
        if i > entry_date + 45:
            return True, "持有超时"
        
        return False, ""
    
    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calc_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
        high = df["high"].values[max(0,i-period):i+1]
        low = df["low"].values[max(0,i-period):i+1]
        close = df["close"].values[max(0,i-period):i+1]
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        return float(pd.Series(tr).rolling(period).mean().iloc[-1]) if len(tr) >= period else 0.02 * close[-1]
