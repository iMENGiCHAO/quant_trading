"""
Hyperion Pro — 交易信号生成器
===============================
将市场状态 + 技术分析 + 基本面 + 资金面
综合生成可执行的交易信号

核心输出:
  - 每只股票的买卖评级 (强烈买入/买入/持有/卖出/强烈卖出)
  - 目标价/止损价
  - 置信度评分
  - 具体的操作建议
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from ..data.market import (
    fetch_realtime_quotes, fetch_history,
    fetch_index_quotes, CORE_STOCKS, 
    get_stock_name, get_stock_industry
)
from .technical import TechnicalAnalyzer
from .market_state import MarketStateAnalyzer


@dataclass
class Signal:
    """交易信号"""
    code: str              # 股票代码
    name: str              # 股票名称
    industry: str          # 行业
    signal: str            # 强烈买入/买入/持有/卖出/强烈卖出
    score: float           # 综合评分 (-100 ~ 100)
    confidence: float      # 置信度 (0-1)
    current_price: float   # 当前价格
    target_price: float    # 目标价
    stop_loss: float       # 止损价
    upside_potential: float # 上涨空间百分比
    downside_risk: float   # 下跌风险百分比
    reward_risk_ratio: float # 盈亏比
    reasons: List[str]     # 推荐理由
    advice: str            # 操作建议
    timestamp: str         # 时间戳
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "industry": self.industry,
            "signal": self.signal,
            "score": self.score,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "upside_potential": self.upside_potential,
            "downside_risk": self.downside_risk,
            "reward_risk_ratio": self.reward_risk_ratio,
            "reasons": self.reasons,
            "advice": self.advice,
        }


class SignalGenerator:
    """
    交易信号生成器
    
    生成过程：
    1. 获取所有股票的技术分析
    2. 结合市场状态调整
    3. 计算综合评分
    4. 生成评级和操作建议
    """
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
    
    def analyze_stock(self, code: str, days: int = 120) -> Optional[Signal]:
        """
        对单只股票进行全面分析
        
        Returns:
            Signal 或 None (数据不足时)
        """
        history = fetch_history(code, days=days)
        if history.empty or len(history) < 30:
            return None
        
        name = get_stock_name(code)
        industry = get_stock_industry(code)
        
        # 技术分析
        tech = self.analyzer.comprehensive_analysis(history)
        
        # 获取当前价格
        current_price = history["close"].values[-1]
        
        # 均线目标价
        ma20 = tech["ma"]["ma20"]
        ma60 = tech["ma"]["ma60"]
        
        # 目标价 = 均线系统上轨
        target_price = max(tech["sr"]["recent_high"], ma20 * 1.1, current_price * 1.08)
        stop_loss = min(tech["sr"]["recent_low"], ma60 * 0.95, current_price * 0.95)
        
        # 上涨/下跌空间
        upside = (target_price / current_price - 1) * 100
        downside = (1 - stop_loss / current_price) * 100
        
        # 盈亏比
        rr = upside / (downside + 0.01)
        
        # 综合评分
        score = tech["comprehensive"]["total_score"]
        
        # 置信度 (多指标一致时置信度高)
        n_indicators = 0
        agreement = 0
        for key in ["ma", "macd", "rsi", "kdj", "boll"]:
            if key in tech:
                n_indicators += 1
                item = tech[key]
                sig = item.get("signal", "").lower()
                if "buy" in sig or "oversold" in sig or "strong_bull" in sig:
                    agreement += 1 if score > 0 else 0
                elif "sell" in sig or "overbought" in sig or "strong_bear" in sig:
                    agreement += 1 if score < 0 else 0
        
        confidence = agreement / max(n_indicators, 1)
        confidence = min(0.95, max(0.2, confidence))
        
        # 评级的理由
        reasons = []
        if "买入" in tech["comprehensive"]["signal"]:
            reasons.append(tech["ma"].get("description", ""))
            reasons.append(tech["macd"].get("description", ""))
        if tech.get("macd", {}).get("divergence", "无") != "无":
            reasons.append(f"{tech['macd']['divergence']}信号")
        if tech["boll"].get("bandwidth_signal", ""):
            reasons.append(tech["boll"]["bandwidth_signal"])
        
        reasons = [r for r in reasons if r][:3]
        if not reasons:
            reasons = [tech["comprehensive"]["advice"]]
        
        # 综合信号
        if score > 30:
            signal_label = "强烈买入"
        elif score > 10:
            signal_label = "买入"
        elif score > -10:
            signal_label = "持有"
        elif score > -30:
            signal_label = "卖出"
        else:
            signal_label = "强烈卖出"
        
        # 操作建议
        if signal_label in ("强烈买入", "买入"):
            advice = f"建议买入，目标价{target_price:.2f}，止损价{stop_loss:.2f}"
        elif signal_label == "持有":
            advice = f"建议持有，关注{stop_loss:.2f}支撑位"
        elif signal_label in ("卖出", "强烈卖出"):
            advice = f"建议卖出，控制风险"
        else:
            advice = "观望"
        
        # 加入市场状态
        market = MarketStateAnalyzer.analyze_overall()
        market_state = market.get("market_state", "不明朗")
        
        if market_state in ("熊市", "回调") and signal_label in ("强烈买入", "买入"):
            signal_label = "买入(谨慎)"
            advice += "。注意大盘偏弱，控制仓位"
            confidence *= 0.8
        
        return Signal(
            code=code,
            name=name,
            industry=industry,
            signal=signal_label,
            score=round(score, 1),
            confidence=round(confidence, 2),
            current_price=round(current_price, 2),
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            upside_potential=round(upside, 1),
            downside_risk=round(downside, 1),
            reward_risk_ratio=round(rr, 2),
            reasons=reasons,
            advice=advice,
            timestamp=datetime.now().isoformat(),
        )
    
    def scan_all_stocks(self, 
                        codes: Optional[List[str]] = None,
                        min_score: float = 0,
                        sort_by: str = "score",
                        top_n: int = 50) -> List[Signal]:
        """
        全市场扫描
        
        Args:
            codes: 指定股票列表 (默认全核心池)
            min_score: 最低评分过滤
            sort_by: 排序字段
            top_n: 返回前N只
            
        Returns:
            信号列表
        """
        if codes is None:
            codes = [c for c, _, _ in CORE_STOCKS]
        
        signals = []
        for code in codes:
            sig = self.analyze_stock(code)
            if sig and sig.score >= min_score:
                signals.append(sig)
        
        # 排序
        reverse = True if sort_by in ("score", "upside_potential", "reward_risk_ratio") else False
        signals.sort(key=lambda s: getattr(s, sort_by, 0), reverse=reverse)
        
        return signals[:top_n]
    
    def top_buy_signals(self, n: int = 10) -> List[Signal]:
        """获取最看好的买入信号"""
        return self.scan_all_stocks(min_score=10, top_n=n)
    
    def top_sell_signals(self, n: int = 10) -> List[Signal]:
        """获取最看空的卖出信号"""
        signals = self.scan_all_stocks(top_n=n)
        signals = [s for s in signals if s.score < -10]
        return sorted(signals, key=lambda s: s.score)[:n]
    
    def sector_recommendations(self) -> Dict[str, dict]:
        """
        行业配置建议
        
        Returns:
            dict: {行业: {建议, 推荐标的, ...}}
        """
        all_signals = self.scan_all_stocks(top_n=200)
        
        # 按行业分组
        sector_signals: Dict[str, List[Signal]] = {}
        for sig in all_signals:
            sector_signals.setdefault(sig.industry, []).append(sig)
        
        # 计算行业平均得分
        recommendations = {}
        for industry, sigs in sector_signals.items():
            avg_score = np.mean([s.score for s in sigs])
            buy_count = sum(1 for s in sigs if s.score > 10)
            sell_count = sum(1 for s in sigs if s.score < -10)
            top_stocks = sorted(sigs, key=lambda s: s.score, reverse=True)[:3]
            
            if avg_score > 20:
                rec = "强烈推荐"
            elif avg_score > 10:
                rec = "推荐"
            elif avg_score > -5:
                rec = "中性"
            elif avg_score > -20:
                rec = "低配"
            else:
                rec = "回避"
            
            recommendations[industry] = {
                "industry": industry,
                "avg_score": round(avg_score, 1),
                "recommendation": rec,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total": len(sigs),
                "top_stocks": [{"code": s.code, "name": s.name, "score": s.score} for s in top_stocks],
            }
        
        return dict(sorted(recommendations.items(), key=lambda x: x[1]["avg_score"], reverse=True))
