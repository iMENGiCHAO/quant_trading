"""
Hyperion Pro — 技术分析引擎
=============================
涵盖：
  - 传统技术指标 (MA, MACD, RSI, KDJ, BOLL)
  - 量价关系分析
  - 形态识别 (支撑/阻力/突破)
  - 量能分析
  - 波动率分析

每个分析输出明确的交易信号和操作建议
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class TechnicalAnalyzer:
    """技术分析器"""
    
    @staticmethod
    def ma_analysis(close: np.ndarray, date_idx=...) -> dict:
        """均线系统分析"""
        if len(close) < 60:
            close = np.asarray(close)
        else:
            close = np.asarray(close)
        
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        
        current = close[-1]
        c5, c10, c20, c60 = ma5[-1], ma10[-1], ma20[-1], ma60[-1]
        
        # 均线排列
        if c5 > c10 > c20 > c60:
            alignment = "多头排列"
            signal = "strong_buy"
            desc = "均线多头排列，上升趋势强劲"
        elif c5 < c10 < c20 < c60:
            alignment = "空头排列"
            signal = "strong_sell"
            desc = "均线空头排列，下降趋势明显"
        elif c5 > c10 and c10 < c20:
            alignment = "黄金交叉形成中"
            signal = "buy"
            desc = "短期均线上穿，可能形成金叉"
        elif c5 < c10 and c10 > c20:
            alignment = "死亡交叉形成中"
            signal = "sell"
            desc = "短期均线下穿，可能形成死叉"
        else:
            alignment = "交叉缠绕"
            signal = "hold"
            desc = "均线相互缠绕，方向不明确"
        
        # 价格相对于均线位置
        above_ma5 = "是" if current > c5 else "否"
        above_ma20 = "是" if current > c20 else "否"
        above_ma60 = "是" if current > c60 else "否"
        
        # 乖离率
        bias_ma5 = (current / c5 - 1) * 100 if c5 > 0 else 0
        bias_ma20 = (current / c20 - 1) * 100 if c20 > 0 else 0
        
        return {
            "ma5": round(c5, 2),
            "ma10": round(c10, 2),
            "ma20": round(c20, 2),
            "ma60": round(c60, 2),
            "alignment": alignment,
            "signal": signal,
            "description": desc,
            "above_ma5": above_ma5,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
            "bias_ma5": round(bias_ma5, 2),
            "bias_ma20": round(bias_ma20, 2),
        }
    
    @staticmethod
    def macd_analysis(close: np.ndarray, fast=12, slow=26, signal=9) -> dict:
        """MACD 分析"""
        close = np.asarray(close, dtype=float)
        
        ema_fast = pd.Series(close).ewm(span=fast).mean().values
        ema_slow = pd.Series(close).ewm(span=slow).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=signal).mean().values
        macd_hist = 2 * (dif - dea)
        
        curr_dif, curr_dea, curr_hist = dif[-1], dea[-1], macd_hist[-1]
        prev_hist = macd_hist[-2] if len(macd_hist) > 1 else 0
        
        # 信号判断
        if curr_dif > 0 and curr_hist > 0 and curr_hist > prev_hist:
            signal = "buy"
            desc = "MACD在零轴上方金叉，强势上涨信号"
        elif curr_dif < 0 and curr_hist < 0 and curr_hist < prev_hist:
            signal = "sell"
            desc = "MACD在零轴下方死叉，弱势下跌信号"
        elif curr_dif > 0 and curr_hist < 0:
            signal = "weak_buy"
            desc = "MACD在零轴上方但红柱缩短，上涨动能减弱"
        elif curr_dif < 0 and curr_hist > 0:
            signal = "weak_sell"
            desc = "MACD在零轴下方但绿柱缩短，下跌动能减弱"
        else:
            signal = "hold"
            desc = "MACD方向不明，观望为宜"
        
        # 背离判断 (简化)
        divergence = "无"
        if len(close) > 20:
            price_low = close[-5:].min()
            price_prev_low = close[-15:-5].min()
            dif_low = dif[-5:].min()
            dif_prev_low = dif[-15:-5].min()
            
            if price_low < price_prev_low and dif_low > dif_prev_low:
                divergence = "底背离"
                signal = "strong_buy"
                desc = "价格新低但MACD未创新低，底背离，强烈买入信号"
            elif price_low > price_prev_low and dif_low < dif_prev_low:
                divergence = "顶背离"
                signal = "strong_sell"
                desc = "价格新高但MACD未创新高，顶背离，强烈卖出信号"
        
        return {
            "dif": round(curr_dif, 4),
            "dea": round(curr_dea, 4),
            "histogram": round(curr_hist, 4),
            "signal": signal,
            "description": desc,
            "divergence": divergence,
            "above_zero": "是" if curr_dif > 0 else "否",
        }
    
    @staticmethod
    def rsi_analysis(close: np.ndarray, period=14) -> dict:
        """RSI 分析"""
        close = np.asarray(close, dtype=float)
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        
        curr_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        # 信号
        if curr_rsi > 80:
            signal = "oversold"
            desc = f"RSI={curr_rsi:.1f}，超买区域，注意回调风险"
            action = "减仓或卖出"
        elif curr_rsi > 70:
            signal = "strong_bull"
            desc = f"RSI={curr_rsi:.1f}，强势区域，持有但需警惕"
            action = "持有，可考虑部分获利了结"
        elif curr_rsi > 50:
            signal = "bull"
            desc = f"RSI={curr_rsi:.1f}，偏强区域，趋势向好"
            action = "持有为主"
        elif curr_rsi > 30:
            signal = "bear"
            desc = f"RSI={curr_rsi:.1f}，偏弱区域，观望为宜"
            action = "观望或轻仓"
        elif curr_rsi > 20:
            signal = "strong_bear"
            desc = f"RSI={curr_rsi:.1f}，弱势区域，注意止损"
            action = "减仓控制风险"
        else:
            signal = "overbought"
            desc = f"RSI={curr_rsi:.1f}，超卖区域，关注反弹机会"
            action = "关注买入机会"
        
        # RSI趋势
        if len(rsi) >= 3:
            rsi_trend = "向上" if rsi[-1] > rsi[-3] else ("向下" if rsi[-1] < rsi[-3] else "震荡")
        else:
            rsi_trend = "震荡"
        
        return {
            "rsi": round(curr_rsi, 1),
            "signal": signal,
            "description": desc,
            "action": action,
            "trend": rsi_trend if len(rsi) >= 3 else "震荡",
        }
    
    @staticmethod
    def kdj_analysis(close: np.ndarray, high: np.ndarray, low: np.ndarray, n=9) -> dict:
        """KDJ 分析"""
        c, h, l = np.asarray(close), np.asarray(high), np.asarray(low)
        
        low_min = pd.Series(l).rolling(n).min().values
        high_max = pd.Series(h).rolling(n).max().values
        
        rsv = (c - low_min) / (high_max - low_min + 1e-12) * 100
        
        k = pd.Series(rsv).ewm(com=2).mean().values
        d = pd.Series(k).ewm(com=2).mean().values
        j = 3 * k - 2 * d
        
        curr_k, curr_d, curr_j = k[-1], d[-1], j[-1]
        
        # 信号
        signals = []
        if curr_k > 80 and curr_d > 80:
            signals.append("超买")
        if curr_k < 20 and curr_d < 20:
            signals.append("超卖")
        
        if len(k) > 1 and k[-1] > k[-2] and d[-1] > d[-2]:
            signals.append("向上")
        elif len(k) > 1 and k[-1] < k[-2] and d[-1] < d[-2]:
            signals.append("向下")
        
        if len(k) > 2 and k[-1] > d[-1] and k[-2] < d[-2]:
            signals.append("金叉")
        elif len(k) > 2 and k[-1] < d[-1] and k[-2] > d[-2]:
            signals.append("死叉")
        
        signal_str = "，".join(signals) if signals else "中性"
        action = ""
        if "超卖" in signal_str and "金叉" in signal_str:
            action = "强烈买入信号"
        elif "超买" in signal_str and "死叉" in signal_str:
            action = "强烈卖出信号"
        elif "超卖" in signal_str:
            action = "关注买入机会"
        elif "超买" in signal_str:
            action = "注意回调风险"
        elif "金叉" in signal_str:
            action = "买入信号"
        elif "死叉" in signal_str:
            action = "卖出信号"
        else:
            action = "观望"
        
        return {
            "k": round(curr_k, 1),
            "d": round(curr_d, 1),
            "j": round(curr_j, 1),
            "signal": signal_str,
            "action": action,
        }
    
    @staticmethod
    def boll_analysis(close: np.ndarray, period=20, std=2) -> dict:
        """布林带分析"""
        c = np.asarray(close, dtype=float)
        
        ma = pd.Series(c).rolling(period).mean().values
        std_val = pd.Series(c).rolling(period).std().values
        
        upper = ma + std * std_val
        lower = ma - std * std_val
        bandwidth = (upper - lower) / (ma + 1e-12)
        position = (c - lower) / (upper - lower + 1e-12)
        
        curr_c, curr_up, curr_low = c[-1], upper[-1], lower[-1]
        curr_bw, curr_pos = bandwidth[-1], position[-1]
        
        # 信号
        if curr_c > curr_up:
            signal = "突破上轨"
            action = "超买，注意回调风险"
        elif curr_c < curr_low:
            signal = "跌破下轨"
            action = "超卖，关注反弹机会"
        elif curr_c > ma[-1]:
            signal = "中轨上方"
            action = "偏强，持有"
        else:
            signal = "中轨下方"
            action = "偏弱，观望"
        
        # 带宽分析 (开口/收窄)
        bw_trend = "开口扩大" if curr_bw > 0.1 else "收窄" if curr_bw < 0.05 else "正常"
        bw_signal = ""
        if bw_trend == "开口扩大":
            bw_signal = "波动加大，趋势可能加速"
        elif bw_trend == "收窄":
            bw_signal = "波动收窄，变盘在即"
        
        return {
            "upper": round(curr_up, 2),
            "middle": round(ma[-1], 2),
            "lower": round(curr_low, 2),
            "bandwidth": round(curr_bw * 100, 2),
            "position": round(curr_pos * 100, 1),
            "signal": signal,
            "action": action,
            "bandwidth_trend": bw_trend,
            "bandwidth_signal": bw_signal,
        }
    
    @staticmethod
    def volume_analysis(volume: np.ndarray, close: np.ndarray) -> dict:
        """成交量分析"""
        v = np.asarray(volume, dtype=float)
        c = np.asarray(close, dtype=float)
        
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        
        curr_v = v[-1]
        curr_v5 = vol_ma5[-1]
        curr_v20 = vol_ma20[-1]
        
        vol_ratio_5 = curr_v / (curr_v5 + 1e-12)
        vol_ratio_20 = curr_v / (curr_v20 + 1e-12)
        
        ret = c[-1] / c[-2] - 1 if len(c) > 1 else 0
        
        # 量价关系
        if vol_ratio_20 > 2 and ret > 0.02:
            vp_signal = "放量上涨"
            action = "量价齐升，强势信号"
        elif vol_ratio_20 > 2 and ret < -0.02:
            vp_signal = "放量下跌"
            action = "放量下跌，注意风险"
        elif vol_ratio_20 < 0.6 and ret > 0.01:
            vp_signal = "缩量上涨"
            action = "缩量上涨，上涨动力不足"
        elif vol_ratio_20 < 0.6 and ret < -0.01:
            vp_signal = "缩量下跌"
            action = "缩量下跌，抛压减轻"
        elif vol_ratio_5 > 1.5:
            vp_signal = "温和放量"
            action = "近期成交量放大，关注趋势变化"
        else:
            vp_signal = "正常"
            action = "量能正常"
        
        return {
            "current_volume": int(curr_v),
            "ma5_volume": int(curr_v5),
            "ma20_volume": int(curr_v20),
            "vol_ratio_5d": round(vol_ratio_5, 2),
            "vol_ratio_20d": round(vol_ratio_20, 2),
            "vp_relation": vp_signal,
            "action": action,
        }
    
    @staticmethod
    def support_resistance(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict:
        """支撑阻力位分析"""
        c, h, l = np.asarray(close), np.asarray(high), np.asarray(low)
        
        # 前高/前低
        if len(c) >= 20:
            recent_high = np.max(h[-20:])
            recent_low = np.min(l[-20:])
        else:
            recent_high = np.max(h)
            recent_low = np.min(l)
        
        # 均线支撑/阻力
        ma20 = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
        ma60 = np.mean(c[-60:]) if len(c) >= 60 else c[-1]
        
        # 整数关口
        current = c[-1]
        int_level = round(current / 10) * 10 if current > 100 else round(current)
        
        support_levels = [recent_low, ma20, ma60, int_level * 0.95]
        resistance_levels = [recent_high, ma20 * 1.05, ma60 * 1.1, int_level * 1.05]
        
        # 当前价格距离支撑/阻力的百分比
        nearest_support = min(support_levels, key=lambda x: abs(x - current))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current))
        
        dist_to_support = (current / nearest_support - 1) * 100
        dist_to_resistance = (nearest_resistance / current - 1) * 100
        
        # 突破判断
        if len(c) > 5 and c[-1] > recent_high * 0.99:
            break_signal = "接近突破前高"
        elif len(c) > 5 and c[-1] < recent_low * 1.01:
            break_signal = "接近跌破前低"
        else:
            break_signal = "区间运行"
        
        return {
            "current_price": round(current, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "nearest_support": round(nearest_support, 2),
            "nearest_resistance": round(nearest_resistance, 2),
            "dist_to_support": round(dist_to_support, 2),
            "dist_to_resistance": round(dist_to_resistance, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "break_signal": break_signal,
            "break_action": "突破可加仓" if "突破" in break_signal else "跌破需止损" if "跌破" in break_signal else "高抛低吸",
        }
    
    @staticmethod
    def comprehensive_analysis(ohlcv: pd.DataFrame) -> dict:
        """
        综合分析：融合所有技术指标
        
        Args:
            ohlcv: DataFrame with [open, high, low, close, volume]
            
        Returns:
            dict: 综合评分+操作建议
        """
        close = ohlcv["close"].values
        high = ohlcv["high"].values
        low = ohlcv["low"].values
        volume = ohlcv.get("volume", pd.Series([0]*len(close))).values
        
        results = {}
        signals_map = {"strong_sell": -2, "sell": -1, "hold": 0, "buy": 1, "strong_buy": 2}
        
        # 各指标分析
        results["ma"] = TechnicalAnalyzer.ma_analysis(close)
        results["macd"] = TechnicalAnalyzer.macd_analysis(close)
        results["rsi"] = TechnicalAnalyzer.rsi_analysis(close)
        results["kdj"] = TechnicalAnalyzer.kdj_analysis(close, high, low)
        results["boll"] = TechnicalAnalyzer.boll_analysis(close)
        results["volume"] = TechnicalAnalyzer.volume_analysis(volume, close)
        results["sr"] = TechnicalAnalyzer.support_resistance(close, high, low)
        
        # 综合评分
        scores = []
        
        # MA评分
        ma_sig = results["ma"].get("signal", "hold")
        scores.append(signals_map.get(ma_sig, 0) * 0.25)
        
        # MACD评分
        macd_sig = results["macd"].get("signal", "hold")
        scores.append(signals_map.get(macd_sig, 0) * 0.25)
        
        # RSI评分
        rsi_val = results["rsi"].get("rsi", 50)
        rsi_score = (rsi_val - 50) / 50
        scores.append(rsi_score * 0.2)
        
        # KDJ
        kdj_action = results["kdj"].get("action", "")
        if "强烈买入" in kdj_action:
            scores.append(1.0 * 0.1)
        elif "强烈卖出" in kdj_action:
            scores.append(-1.0 * 0.1)
        elif "买入" in kdj_action:
            scores.append(0.5 * 0.1)
        elif "卖出" in kdj_action:
            scores.append(-0.5 * 0.1)
        
        # 量价
        vp_action = results["volume"].get("vp_relation", "")
        if "放量上涨" in vp_action:
            scores.append(0.8 * 0.1)
        elif "放量下跌" in vp_action:
            scores.append(-0.8 * 0.1)
        
        # 布林带位置
        boll_pos = results["boll"].get("position", 50)
        boll_score = (boll_pos - 50) / 50
        scores.append(-boll_score * 0.1)  # 高位负分，低位正分
        
        total_score = sum(scores) * 100  # 转为百分制
        
        if total_score > 30:
            final_signal = "强烈买入"
            advice = "多项技术指标共振向好，建议积极布局"
        elif total_score > 10:
            final_signal = "买入"
            advice = "技术面偏多，可适度建仓"
        elif total_score > -10:
            final_signal = "观望"
            advice = "多空分歧较大，建议等待明确信号"
        elif total_score > -30:
            final_signal = "卖出"
            advice = "技术面偏空，建议减仓"
        else:
            final_signal = "强烈卖出"
            advice = "多项技术指标共振走弱，建议清仓观望"
        
        results["comprehensive"] = {
            "total_score": round(total_score, 1),
            "signal": final_signal,
            "advice": advice,
            "ma_score": round(scores[0] * 100, 1) if scores else 0,
            "macd_score": round(scores[1] * 100, 1) if len(scores) > 1 else 0,
            "rsi_score": round(rsi_score * 20, 1) if 'rsi_score' in dir() else 0,
        }
        
        return results
