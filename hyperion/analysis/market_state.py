"""
Hyperion Pro — 市场状态分析
=============================
核心功能：
  1. 大盘整体态势判断 (通过上证指数Sina API获取)
  2. 行业轮动分析
  3. 市场情绪指标
  4. 风险预警
  5. 综合市场展望

每个分析模块都输出可操作的判断结论
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.market import (
    fetch_realtime_quotes, fetch_index_quotes, 
    fetch_history, sector_quotes, money_flow,
    CORE_STOCKS, INDICES, get_stock_industry
)

logger = logging.getLogger(__name__)


class MarketStateAnalyzer:
    """市场状态综合分析器——输出市场阶段+操作策略"""
    
    @staticmethod
    def analyze_overall() -> dict:
        """
        大盘整体分析
        
        Returns:
            dict: {
                "market_state": "牛市/熊市/震荡/反弹/回调",
                "confidence": 0.0-1.0,
                "key_metrics": {...},
                "advice": "操作建议"
            }
        """
        now = datetime.now()
        
        # 获取主要指数数据
        index_quotes = fetch_index_quotes()
        history_sh = _fetch_index_kline("sh000001", days=120)
        
        result = {
            "timestamp": now.isoformat(),
            "index_status": {},
            "volume_analysis": {},
            "breadth_analysis": {},
            "market_state": "不明朗",
            "confidence": 0.3,
            "advice": "市场数据不足，建议谨慎操作",
            "risk_level": "中",
        }
        
        # 1. 指数技术分析
        if history_sh is not None and not history_sh.empty:
            close = history_sh["close"].values
            ret = history_sh["change_pct"].values / 100.0  # convert percentage to decimal
            
            ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
            ma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
            ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else close[-1]
            current = close[-1]
            
            short_trend = "向上" if ma5 > ma20 else "向下"
            mid_trend = "向上" if ma20 > ma60 else "向下"
            
            if ma5 > ma10 > ma20 > ma60:
                ma_alignment = "多头排列"
            elif ma5 < ma10 < ma20 < ma60:
                ma_alignment = "空头排列"
            else:
                ma_alignment = "交叉缠绕"
            
            vol_20d = np.std(ret[-20:]) if len(ret) >= 20 else 0
            vol_60d = np.std(ret[-60:]) if len(ret) >= 60 else vol_20d
            vol_ratio = vol_20d / (vol_60d + 1e-12)
            
            vol_series = history_sh.get("volume", pd.Series([0]*len(close)))
            if hasattr(vol_series, "iloc"):
                vol_current = vol_series.iloc[-1]
            else:
                vol_current = vol_series[-1]
            vol_ma20 = np.mean(vol_series[-20:]) if len(vol_series) >= 20 else 1
            vol_status = "放量" if vol_current > vol_ma20 * 1.5 else ("缩量" if vol_current < vol_ma20 * 0.6 else "正常")
            
            quotes = fetch_realtime_quotes()
            if "change_pct" in quotes.columns:
                up_count = int((quotes["change_pct"] > 0).sum())
                down_count = int((quotes["change_pct"] < 0).sum())
                total_count = len(quotes)
                up_ratio = up_count / (total_count + 1e-12)
            else:
                up_count, down_count, total_count, up_ratio = 0, 0, 0, 0.5
            
            recent_ret = np.mean(ret[-5:]) if len(ret) >= 5 else 0
            monthly_ret = np.mean(ret[-20:]) if len(ret) >= 20 else 0
            quarterly_ret = np.mean(ret[-60:]) if len(ret) >= 60 else 0
            
            if quarterly_ret > 0.001 and monthly_ret > 0.001 and short_trend == "向上":
                market_state = "牛市"
                confidence = min(0.9, 0.5 + abs(monthly_ret) * 100)
                risk_level = "低"
                advice = "市场处于上升趋势，建议维持多头仓位，逢低加仓优质标的"
            elif quarterly_ret < -0.001 and monthly_ret < -0.001 and short_trend == "向下":
                market_state = "熊市"
                confidence = min(0.9, 0.5 + abs(monthly_ret) * 100)
                risk_level = "高"
                advice = "市场处于下行趋势，建议降低仓位、控制风险，等待企稳信号"
            elif abs(monthly_ret) < 0.0005 and vol_ratio < 1.3:
                market_state = "震荡"
                confidence = 0.6
                risk_level = "中"
                advice = "市场区间震荡，建议高抛低吸，注意仓位管理"
            elif recent_ret > 0.002 and monthly_ret > -0.001:
                market_state = "反弹"
                confidence = 0.5
                risk_level = "中"
                advice = "市场出现反弹迹象，可适度参与，但需警惕反弹力度不足"
            elif recent_ret < -0.002 and monthly_ret < 0.001:
                market_state = "回调"
                confidence = 0.5
                risk_level = "中偏高"
                advice = "市场短期回调，关注支撑位，等待企稳后再加仓"
            else:
                market_state = "不明朗"
                confidence = 0.3
                risk_level = "中"
                advice = "市场方向不明，建议观望或轻仓参与"
            
            result["index_status"] = {
                "current": round(current, 2),
                "ma5": round(ma5, 2),
                "ma10": round(ma10, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "short_trend": short_trend,
                "mid_trend": mid_trend,
                "ma_alignment": ma_alignment,
                "short_return": round(recent_ret * 100, 2),
                "monthly_return": round(monthly_ret * 100, 2),
                "quarterly_return": round(quarterly_ret * 100, 2),
                "volatility_20d": round(vol_20d * 100, 2),
            }
            result["volume_analysis"] = {
                "current_volume": int(vol_current),
                "ma20_volume": int(vol_ma20),
                "vol_status": vol_status,
                "vol_ratio": round(vol_ratio, 2),
            }
            result["breadth_analysis"] = {
                "up_stocks": up_count,
                "down_stocks": down_count,
                "total": total_count,
                "up_ratio": round(up_ratio * 100, 1),
                "breadth_status": "强势" if up_ratio > 0.6 else ("弱势" if up_ratio < 0.4 else "均衡"),
            }
            result["market_state"] = market_state
            result["confidence"] = round(confidence, 2)
            result["advice"] = advice
            result["risk_level"] = risk_level
        
        return result
    
    @staticmethod
    def analyze_sectors() -> pd.DataFrame:
        """行业板块轮动分析"""
        sectors = sector_quotes()
        if sectors.empty:
            return pd.DataFrame()
        
        sectors["strength"] = sectors["avg_change"].apply(
            lambda x: "强势" if x > 1 else ("中性" if x > -1 else "弱势")
        )
        
        top3 = sectors.head(3).index.tolist() if len(sectors) >= 3 else []
        bottom3 = sectors.tail(3).index.tolist() if len(sectors) >= 3 else []
        
        sectors["is_hot"] = sectors.index.isin(top3)
        sectors["is_cold"] = sectors.index.isin(bottom3)
        sectors["rotation_signal"] = sectors["avg_change"].apply(
            lambda x: "关注" if x > 1.5 else ("回避" if x < -1.5 else "中性")
        )
        
        return sectors
    
    @staticmethod
    def analyze_emotion() -> dict:
        """市场情绪分析"""
        quotes = fetch_realtime_quotes()
        
        if quotes.empty or "change_pct" not in quotes.columns:
            return {"emotion": "未知", "score": 0.5}
        
        up = int((quotes["change_pct"] > 0).sum())
        down = int((quotes["change_pct"] < 0).sum())
        total = len(quotes)
        
        up_ratio = up / (total + 1e-12)
        limit_up = int((quotes["change_pct"] > 9.5).sum())
        limit_down = int((quotes["change_pct"] < -9.5).sum())
        median_change = float(quotes["change_pct"].median())
        
        emotion_score = 0.5 * up_ratio + 0.3 * (median_change / 5 + 0.5) + 0.2 * (1 - limit_down / max(total, 1))
        emotion_score = np.clip(emotion_score, 0, 1)
        
        if emotion_score > 0.7:
            emotion = "乐观"
        elif emotion_score > 0.5:
            emotion = "中性偏乐观"
        elif emotion_score > 0.3:
            emotion = "中性偏悲观"
        else:
            emotion = "悲观"
        
        return {
            "emotion": emotion,
            "score": round(float(emotion_score), 2),
            "up_ratio": round(up_ratio * 100, 1),
            "limit_up": limit_up,
            "limit_down": limit_down,
            "median_change": round(median_change, 2),
            "up_stocks": up,
            "down_stocks": down,
            "total_stocks": total,
        }
    
    @staticmethod
    def generate_outlook() -> dict:
        """生成综合市场展望"""
        overall = MarketStateAnalyzer.analyze_overall()
        sectors = MarketStateAnalyzer.analyze_sectors()
        emotion = MarketStateAnalyzer.analyze_emotion()
        
        risk_scores = {
            "牛市": 2, "反弹": 4, "震荡": 5, "回调": 7, "熊市": 8, "不明朗": 6
        }
        risk_score = risk_scores.get(overall.get("market_state", "不明朗"), 5)
        
        position_map = {
            "牛市": "70%-90%",
            "反弹": "50%-70%",
            "震荡": "40%-60%",
            "回调": "20%-40%",
            "熊市": "10%-30%",
            "不明朗": "30%-50%",
        }
        recommended_position = position_map.get(overall.get("market_state", "不明朗"), "30%-50%")
        
        hot_sectors = []
        cold_sectors = []
        if not sectors.empty:
            hot_sectors = sectors[sectors["is_hot"]].index.tolist()
            cold_sectors = sectors[sectors["is_cold"]].index.tolist()
        
        action = overall.get("advice", "")
        state = overall.get("market_state", "")
        
        if state in ("牛市", "反弹"):
            action += "。重点关注强势行业龙头，顺势而为"
        elif state == "震荡":
            action += "。注意高抛低吸节奏，不要追涨杀跌"
        elif state in ("熊市", "回调"):
            action += "。防御性配置，关注低估值蓝筹和红利资产"
        
        if hot_sectors:
            action += f"。当前热点板块：{'、'.join(hot_sectors[:3])}"
        if cold_sectors:
            action += f"。回避板块：{'、'.join(cold_sectors[:3])}"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_state": state,
            "confidence": overall.get("confidence", 0),
            "emotion": emotion.get("emotion", "未知"),
            "emotion_score": emotion.get("score", 0.5),
            "risk_level": overall.get("risk_level", "中"),
            "risk_score": risk_score,
            "recommended_position": recommended_position,
            "hot_sectors": hot_sectors,
            "cold_sectors": cold_sectors,
            "key_metrics": {
                "up_ratio": emotion.get("up_ratio", 0),
                "limit_up": emotion.get("limit_up", 0),
                "limit_down": emotion.get("limit_down", 0),
                "median_change": emotion.get("median_change", 0),
                "trend": overall.get("index_status", {}).get("short_trend", "未知"),
            },
            "action_advice": action,
        }


def _fetch_index_kline(sina_code: str, days: int = 120) -> Optional[pd.DataFrame]:
    """通过 Sina API 获取指数K线数据"""
    import requests
    import json
    
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_code}&scale=240&ma=no&datalen={days+10}"
    )
    r = requests.get(
        url,
        headers={"Referer": "https://finance.sina.com.cn"},
        timeout=8
    )
    if r.status_code == 200 and r.text.strip():
        data = json.loads(r.text)
        if isinstance(data, list) and len(data) > 0:
            rows = []
            for bar in data:
                try:
                    o = float(bar.get("open", 0))
                    h = float(bar.get("high", 0))
                    l = float(bar.get("low", 0))
                    c = float(bar.get("close", 0))
                    v = float(bar.get("volume", 0))
                    rows.append({
                        "date": pd.Timestamp(bar["day"]),
                        "open": o, "high": h, "low": l, "close": c,
                        "volume": int(v), "amount": c * v * 100,
                    })
                except (ValueError, KeyError):
                    continue
            if rows:
                df = pd.DataFrame(rows)
                df.sort_values("date", inplace=True)
                df["change_pct"] = df["close"].pct_change() * 100
                return df
    return None
