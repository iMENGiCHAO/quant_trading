"""
Hyperion Pro — 投资决策引擎
=============================
华尔街级别的投资决策框架：
  1. 综合评分系统（多因子）
  2. 风险收益比计算
  3. 仓位配置建议（凯利公式优化）
  4. 止盈止损策略
  5. 情景分析（乐观/基准/悲观）
  6. 市场择时信号

核心输出：每个交易标的的可执行投资决策
—— 不是"可买可卖"，而是"买多少、什么价买、什么价出、什么情况下止损"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..data.market import (
    fetch_realtime_quotes, fetch_history, fetch_index_quotes,
    CORE_STOCKS, INDICES, get_stock_name, get_stock_industry
)
from .technical import TechnicalAnalyzer
from .market_state import MarketStateAnalyzer
from .signals import Signal, SignalGenerator

from ..engine.backtest import BacktestEngine
from ..strategy.base import list_strategies, get_strategy

# ── 策略信号权重配置 ────────────────────────────────────────

STRATEGY_SIGNAL_WEIGHTS = {
    "TrendFollowingStrategy": 0.25,
    "MeanReversionStrategy": 0.20,
    "MomentumBreakoutStrategy": 0.20,
    "VolumeAnomalyStrategy": 0.15,
    "MultiFactorAlphaStrategy": 0.20,
}


class DecisionType(Enum):
    """投资决策类型"""
    STRONG_BUY = "强烈买入"       # 得分 > 70
    BUY = "买入"                  # 得分 > 50
    ACCUMULATE = "逢低建仓"       # 得分 30-50
    HOLD = "持有"                 # 得分 -30 ~ 30
    REDUCE = "减仓"               # 得分 -50 ~ -30
    SELL = "卖出"                 # 得分 -70 ~ -50
    STRONG_SELL = "清仓"          # 得分 < -70


@dataclass
class InvestmentDecision:
    """投资决策 — 完整的可执行投资指令"""
    # === 基本信息 ===
    code: str
    name: str
    industry: str
    timestamp: str
    
    # === 评分体系 ===
    decision: str                     # 决策类型
    composite_score: float            # 综合评分 (-100 ~ 100)
    confidence: float                 # 置信度 (0-1)
    
    # === 技术面评分 ===
    technical_score: float            # 技术面评分
    trend_score: float                # 趋势强度评分
    momentum_score: float             # 动量评分
    volatility_score: float           # 波动率评分（高波动不加分）
    
    # === 基本面/估值 ===
    fundamental_score: float          # 基本面评分
    
    # === 资金面 ===
    capital_flow_score: float         # 资金流向评分
    
    # === 市场环境 ===
    market_adjustment: float          # 市场环境调整因子
    
    # === 策略回测 ===
    strategy_score: float             # 策略回测综合评分
    strategy_consensus: str           # 策略共识方向
    
    # === 价格与目标 ===
    current_price: float
    fair_value: float                 # 公允价值估算
    target_price_optimistic: float    # 乐观目标价
    target_price_base: float          # 基准目标价
    target_price_pessimistic: float   # 悲观目标价
    
    # === 风险控制 ===
    stop_loss: float                  # 硬止损价
    trailing_stop_pct: float          # 移动止损百分比
    max_position_pct: float           # 最大仓位比例（凯利公式优化）
    risk_per_trade: float             # 单笔风险暴露（占总资金%）
    reward_risk_ratio: float          # 盈亏比
    
    # === 操作指令 ===
    entry_price: float                # 建议入场价
    entry_zone_low: float             # 入场区间下沿
    entry_zone_high: float            # 入场区间上沿
    take_profit_1: float              # 第一止盈位（50%仓位）
    take_profit_2: float              # 第二止盈位（剩余仓位）
    
    # === 持有期限 ===
    holding_period: str               # 短/中/长线
    expected_return: float            # 预期收益率(%)
    expected_holding_days: int        # 预期持有天数
    
    # === 风险提示 ===
    risk_factors: List[str] = field(default_factory=list)
    key_catalysts: List[str] = field(default_factory=list)
    scenario_analysis: dict = field(default_factory=dict)
    
    # === 操作建议文本 ===
    action_plan: str = ""
    summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "industry": self.industry,
            "decision": self.decision, "composite_score": self.composite_score,
            "confidence": self.confidence,
            "technical_score": self.technical_score,
            "trend_score": self.trend_score, "momentum_score": self.momentum_score,
            "volatility_score": self.volatility_score,
            "fundamental_score": self.fundamental_score,
            "capital_flow_score": self.capital_flow_score,
            "market_adjustment": self.market_adjustment,
            "strategy_score": self.strategy_score,
            "strategy_consensus": self.strategy_consensus,
            "current_price": self.current_price,
            "fair_value": self.fair_value,
            "target_price_base": self.target_price_base,
            "target_price_optimistic": self.target_price_optimistic,
            "target_price_pessimistic": self.target_price_pessimistic,
            "stop_loss": self.stop_loss,
            "max_position_pct": self.max_position_pct,
            "reward_risk_ratio": self.reward_risk_ratio,
            "entry_price": self.entry_price,
            "entry_zone_low": self.entry_zone_low,
            "entry_zone_high": self.entry_zone_high,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "holding_period": self.holding_period,
            "expected_return": self.expected_return,
            "expected_holding_days": self.expected_holding_days,
            "risk_factors": self.risk_factors,
            "key_catalysts": self.key_catalysts,
            "action_plan": self.action_plan,
            "summary": self.summary,
        }


class InvestmentDecisionEngine:
    """
    投资决策引擎
    
    这是 Hyperion Pro 的核心——将原始的技术指标和市场数据
    转化为具体、可执行的投资决策。
    
    设计理念：
    - 每个输出都是"操作指令"，不是"分析结论"
    - 包含具体的买卖价格区间、仓位大小、止损条件
    - 多情景分析让使用者了解上行/下行风险
    - 综合评分透明可追溯
    """
    
    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.tech_analyzer = TechnicalAnalyzer()
    
    # ==========================================================
    #  主入口：生成单个标的的完整投资决策
    # ==========================================================
    
    def analyze(self, code: str, days: int = 250) -> Optional[InvestmentDecision]:
        """
        对单个股票生成完整的投资决策
        
        Args:
            code: 股票代码
            days: 回溯分析天数
            
        Returns:
            InvestmentDecision 或 None（数据不足时）
        """
        # 获取数据
        history = fetch_history(code, days=days)
        if history.empty or len(history) < 60:
            return None
        
        name = get_stock_name(code)
        industry = get_stock_industry(code)
        
        close = history["close"].values
        high = history["high"].values
        low = history["low"].values
        volume = history["volume"].values
        
        current_price = close[-1]
        
        # === 1. 技术分析 ===
        tech = self.tech_analyzer.comprehensive_analysis(history)
        
        # === 2. 趋势评分 (0-30) ===
        trend_score = self._score_trend(tech, close)
        
        # === 3. 动量评分 (0-25) ===
        momentum_score = self._score_momentum(tech, close)
        
        # === 4. 波动率评分 (0-15) ===
        volatility_score = self._score_volatility(close)
        
        # === 5. 技术综合评分 ===
        technical_score = trend_score + momentum_score + volatility_score
        
        # === 6. 基本面评分 (0-20) ===
        fundamental_score = self._score_fundamental(code)
        
        # === 7. 资金面评分 (0-15) ===
        capital_score = self._score_capital_flow(code)
        
        # === 8. 市场环境调整 ===
        market = MarketStateAnalyzer.analyze_overall()
        market_adjustment = self._calc_market_adjustment(market)
        
        # === 9. 策略回测信号评分 ===
        strategy_score, strategy_consensus = self._score_strategy_signals(code, history)
        
        # === 10. 综合评分 ===
        # 技术(70) + 基本面(20) + 资金面(15) + 策略信号(15) = 满分120
        composite_raw = (technical_score + fundamental_score + 
                        capital_score + float(strategy_score))
        composite_score = composite_raw * market_adjustment
        
        # 归一化到 -100 ~ 100
        # 总分120分制映射到-100~100: 60分(及格)→0, 120分(满分)→100, 0分→-100
        composite_score = np.clip((composite_score / 120) * 200 - 100, -100, 100)
        
        # === 10. 确定决策类型 ===
        decision = self._classify_decision(composite_score)
        
        # === 11. 目标价估算 ===
        targets = self._estimate_targets(close, high, low, tech, market)
        
        # === 12. 止损计算 ===
        stop_loss = self._calc_stop_loss(close, low, tech, composite_score)
        
        # === 13. 仓位计算（凯利公式优化）===
        max_position, risk_per_trade = self._calc_position_size(
            composite_score, current_price, stop_loss, market
        )
        
        # === 14. 盈亏比 ===
        upside_pct = (targets["base"] / current_price - 1) * 100
        downside_pct = max(3.0, (1 - stop_loss / current_price) * 100)  # 最少3%下行
        rr_ratio = upside_pct / downside_pct
        
        # === 15. 持有期限 ===
        holding_period, expected_days = self._determine_holding_period(
            composite_score, tech, industry
        )
        
        # === 16. 入场区间 ===
        entry_price = round(float(current_price), 2)
        entry_low = round(float(current_price) * 0.98, 2)
        entry_high = round(float(current_price) * 1.02, 2)
        
        # 如果是强烈买入，在当前价附近入场
        if "买入" in decision.value:
            entry_low = round(float(current_price) * 0.97, 2)
            entry_high = round(float(current_price) * 1.01, 2)
        
        # === 17. 止盈位 ===
        tp1 = round(current_price * (1 + upside_pct * 0.5 / 100), 2)  # 50%收益时止盈一半
        tp2 = round(targets["base"], 2)
        
        # === 18. 置信度 ===
        confidence = self._calc_confidence(composite_score, tech, market, len(close))
        
        # === 19. 风险因素 ===
        risk_factors = self._identify_risks(tech, market, industry)
        
        # === 20. 催化剂 ===
        catalysts = self._identify_catalysts(tech, industry)
        
        # === 21. 情景分析 ===
        scenarios = self._scenario_analysis(close, targets, stop_loss, current_price)
        
        # === 22. 生成操作计划 ===
        action_plan = self._generate_action_plan(
            decision, current_price, entry_low, entry_high,
            stop_loss, tp1, tp2, max_position, holding_period
        )
        
        # === 23. 生成摘要 ===
        summary = self._generate_summary(
            decision, composite_score, current_price, targets,
            stop_loss, rr_ratio, holding_period, market
        )
        
        return InvestmentDecision(
            code=code,
            name=name,
            industry=industry,
            timestamp=datetime.now().isoformat(),
            decision=decision.value,
            composite_score=round(composite_score, 1),
            confidence=round(confidence, 2),
            technical_score=round(technical_score, 1),
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            volatility_score=round(volatility_score, 1),
            fundamental_score=round(fundamental_score, 1),
            capital_flow_score=round(capital_score, 1),
            market_adjustment=round(market_adjustment, 2),
            strategy_score=round(strategy_score, 1),
            strategy_consensus=strategy_consensus,
            current_price=round(current_price, 2),
            fair_value=round(targets.get("fair_value", current_price), 2),
            target_price_optimistic=round(targets.get("optimistic", current_price * 1.2), 2),
            target_price_base=round(targets["base"], 2),
            target_price_pessimistic=round(targets.get("pessimistic", current_price * 0.9), 2),
            stop_loss=round(stop_loss, 2),
            trailing_stop_pct=8.0 if holding_period == "中线" else (5.0 if holding_period == "短线" else 12.0),
            max_position_pct=round(max_position, 1),
            risk_per_trade=round(risk_per_trade, 2),
            reward_risk_ratio=round(rr_ratio, 2),
            entry_price=round(entry_price, 2),
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            take_profit_1=tp1,
            take_profit_2=tp2,
            holding_period=holding_period,
            expected_return=round(upside_pct, 1),
            expected_holding_days=expected_days,
            risk_factors=risk_factors,
            key_catalysts=catalysts,
            scenario_analysis=scenarios,
            action_plan=action_plan,
            summary=summary,
        )
    
    # ==========================================================
    #  评分子模块
    # ==========================================================
    
    def _score_strategy_signals(self, code: str, history: pd.DataFrame) -> Tuple[float, str]:
        """基于回测引擎运行所有策略，检查当前是否有买入信号"""
        if history.empty or len(history) < 60:
            return 7.5, "数据不足"
        
        score = 0.0
        signal_count = 0
        buy_count = 0
        
        for strategy_name in list_strategies():
            weight = STRATEGY_SIGNAL_WEIGHTS.get(strategy_name, 0.15)
            strategy = get_strategy(strategy_name)
            
            if strategy is None:
                continue
            
            try:
                signals = strategy.generate_signals(history, code, "")
                if signals:
                    latest = signals[-1]
                    signal_count += 1
                    
                    if latest.signal_type.value == "BUY":
                        buy_count += 1
                        score += weight * 15.0 * latest.confidence
            except Exception:
                continue
        
        if signal_count == 0:
            return 7.5, "无信号"
        
        buy_ratio = buy_count / max(signal_count, 1)
        
        # 需要至少 2 个策略发出信号才判断方向
        if signal_count < 2:
            consensus = "信号不足"
        elif buy_ratio >= 0.8:
            consensus = "强烈看多"
        elif buy_ratio >= 0.6:
            consensus = "偏多"
        elif buy_ratio >= 0.4:
            consensus = "分歧"
        elif buy_ratio >= 0.2:
            consensus = "偏空"
        else:
            consensus = "看空"
        
        score = np.clip(score, 0, 15)
        return round(float(score), 1), consensus

    def _score_trend(self, tech: dict, close: np.ndarray) -> float:
        """趋势强度评分 (满分30)"""
        score = 15.0  # 中性起点
        
        # 均线排列 (10分)
        ma = tech.get("ma", {})
        if ma.get("alignment") == "多头排列":
            score += 10
        elif ma.get("alignment") == "空头排列":
            score -= 10
        elif ma.get("alignment") == "黄金交叉形成中":
            score += 5
        elif ma.get("alignment") == "死亡交叉形成中":
            score -= 5
        
        # 价格 vs 均线 (10分)
        above_ma20 = ma.get("above_ma20") == "是"
        above_ma60 = ma.get("above_ma60") == "是"
        if above_ma20 and above_ma60:
            score += 10
        elif above_ma20 and not above_ma60:
            score += 5
        elif not above_ma20 and above_ma60:
            score += 0
        else:
            score -= 10
        
        # 均线斜率 (10分)
        if len(close) >= 20:
            ma20_vals = pd.Series(close).rolling(20).mean().values
            if ma20_vals[-1] > ma20_vals[-5]:
                score += 10
            elif ma20_vals[-1] < ma20_vals[-5]:
                score -= 5
        
        return np.clip(score, 0, 30)
    
    def _score_momentum(self, tech: dict, close: np.ndarray) -> float:
        """动量评分 (满分25)"""
        score = 12.5
        
        # MACD (8分)
        macd = tech.get("macd", {})
        macd_sig = macd.get("signal", "hold")
        if macd_sig == "strong_buy":
            score += 8
        elif macd_sig == "buy":
            score += 5
        elif macd_sig == "strong_sell":
            score -= 8
        elif macd_sig == "sell":
            score -= 5
        
        # MACD背离 (5分)
        if macd.get("divergence") == "底背离":
            score += 5
        elif macd.get("divergence") == "顶背离":
            score -= 5
        
        # RSI (7分)
        rsi = tech.get("rsi", {})
        rsi_val = rsi.get("rsi", 50)
        if 40 <= rsi_val <= 60:
            score += 3  # 中性区间，趋势可能延续
        elif 30 <= rsi_val < 40:
            score += 5  # 超卖反弹机会
        elif 60 < rsi_val <= 70:
            score += 1  # 偏强但未超买
        elif rsi_val > 70:
            score -= 3  # 超买谨慎
        elif rsi_val < 30:
            score += 7  # 极度超卖
        elif rsi_val > 80:
            score -= 5  # 严重超买
        
        # KDJ (5分)
        kdj = tech.get("kdj", {})
        kdj_act = kdj.get("action", "")
        if "强烈买入" in kdj_act:
            score += 5
        elif "买入" in kdj_act:
            score += 3
        elif "强烈卖出" in kdj_act:
            score -= 5
        elif "卖出" in kdj_act:
            score -= 3
        
        return np.clip(score, 0, 25)
    
    def _score_volatility(self, close: np.ndarray) -> float:
        """波动率评分 (满分15)——低波动更稳健"""
        if len(close) < 20:
            return 7.5
        
        returns = pd.Series(close).pct_change().dropna()
        vol_20d = returns.tail(20).std() * np.sqrt(252)  # 年化波动率
        
        # 波动率越低越稳健
        if vol_20d < 0.2:
            score = 15.0  # 低波动，稳健
        elif vol_20d < 0.3:
            score = 12.0
        elif vol_20d < 0.4:
            score = 9.0
        elif vol_20d < 0.5:
            score = 6.0
        elif vol_20d < 0.6:
            score = 3.0
        else:
            score = 0.0  # 高波动，风险大
        
        return score
    
    def _score_fundamental(self, code: str) -> float:
        """基本面评分 (满分20)——基于估值、市值等"""
        score = 10.0
        
        try:
            quotes = fetch_realtime_quotes([code])
            if not quotes.empty:
                row = quotes.iloc[0]
                
                # PE评分 (8分)
                pe = row.get("pe_ttm", 999)
                if pd.notna(pe) and pe > 0:
                    if pe < 15:
                        score += 8  # 低估值
                    elif pe < 25:
                        score += 5
                    elif pe < 40:
                        score += 2
                    elif pe < 60:
                        score += 0
                    else:
                        score -= 3  # 高估值
                
                # PB评分 (5分)
                pb = row.get("pb", 999)
                if pd.notna(pb) and pb > 0:
                    if pb < 1.5:
                        score += 5
                    elif pb < 3:
                        score += 3
                    elif pb < 5:
                        score += 1
                    else:
                        score -= 2
                
                # 市值评分 (7分)——偏好中大市值，流动性好
                total_mv = row.get("total_mv", 0)
                if pd.notna(total_mv):
                    mv_billion = total_mv / 1e8
                    if mv_billion > 1000:
                        score += 7  # 千亿以上蓝筹
                    elif mv_billion > 500:
                        score += 5
                    elif mv_billion > 100:
                        score += 3
                    elif mv_billion > 50:
                        score += 0
                    else:
                        score -= 3  # 小盘风险
        except Exception:
            pass
        
        return np.clip(score, 0, 20)
    
    def _score_capital_flow(self, code: str) -> float:
        """资金面评分 (满分15)"""
        from ..data.market import money_flow
        mf = money_flow(code)
        
        score = 7.5
        main_ratio = mf.get("main_force_ratio", 0)
        
        if main_ratio > 5:
            score += 7
        elif main_ratio > 2:
            score += 5
        elif main_ratio > 0:
            score += 3
        elif main_ratio > -2:
            score += 0
        elif main_ratio > -5:
            score -= 3
        else:
            score -= 7
        
        return np.clip(score, 0, 15)
    
    def _calc_market_adjustment(self, market: dict) -> float:
        """市场环境调整因子(0.6-1.2)"""
        state = market.get("market_state", "不明朗")
        
        factors = {
            "牛市": 1.15, "反弹": 1.05,
            "震荡": 1.0, "回调": 0.85,
            "熊市": 0.70, "不明朗": 0.85,
        }
        return factors.get(state, 0.85)
    
    # ==========================================================
    #  决策分类
    # ==========================================================
    
    def _classify_decision(self, score: float) -> DecisionType:
        if score > 70:
            return DecisionType.STRONG_BUY
        elif score > 50:
            return DecisionType.BUY
        elif score > 30:
            return DecisionType.ACCUMULATE
        elif score > -30:
            return DecisionType.HOLD
        elif score > -50:
            return DecisionType.REDUCE
        elif score > -70:
            return DecisionType.SELL
        else:
            return DecisionType.STRONG_SELL
    
    # ==========================================================
    #  目标价估算
    # ==========================================================
    
    def _estimate_targets(self, close, high, low, tech, market) -> dict:
        """估算目标价 (乐观/基准/悲观)"""
        current = close[-1]
        
        # 基准：均线系统估值
        ma = tech.get("ma", {})
        ma20 = ma.get("ma20", current)
        ma60 = ma.get("ma60", current)
        
        # 使用布林带上轨作为乐观目标
        boll = tech.get("boll", {})
        upper = boll.get("upper", current * 1.1)
        
        base_target = max(ma20, ma60, current * 1.05)
        optimistic = max(upper, current * 1.15)
        pessimistic = min(ma20, ma60, current * 0.95)
        
        # 根据市场状态调整
        state = market.get("market_state", "不明朗")
        if state == "牛市":
            optimistic *= 1.1
            base_target *= 1.05
        elif state == "熊市":
            optimistic *= 0.9
            pessimistic *= 0.95
        
        return {
            "base": round(base_target, 2),
            "optimistic": round(optimistic, 2),
            "pessimistic": round(pessimistic, 2),
            "fair_value": round(current, 2),
        }
    
    # ==========================================================
    #  止损计算
    # ==========================================================
    
    def _calc_stop_loss(self, close, low, tech, score) -> float:
        """计算止损价 —— 综合ATR和支撑位"""
        current = float(np.asarray(close[-1]).flat[0])
        
        # ATR止损 (14日)
        if len(close) >= 14:
            atr = float(np.asarray(pd.Series(close).diff().abs().tail(14).mean()).flat[0])
            atr_stop = current - max(2.0 * atr, current * 0.03)
        else:
            atr_stop = current * 0.93
        
        # 支撑位止损
        sr = tech.get("sr", {})
        support_val = float(np.asarray(sr.get("nearest_support", current * 0.93)).flat[0])
        support_stop = min(support_val * 0.98, current * 0.95)
        
        # 技术止损: 近期最低价
        recent_low = float(np.asarray(low[-20:].min()).flat[0]) if len(low) >= 20 else current * 0.90
        tech_stop = min(recent_low * 0.99, current * 0.95)
        
        # 取三者中最紧的，但确保至少3%距离
        candidate = max(atr_stop, support_stop, tech_stop)
        min_distance = current * 0.93  # 至少7%距离（对于低波动股票）
        
        # 评分越低，止损越紧
        if score < -30:
            min_distance = current * 0.96  # 4%距离
        elif score > 50:
            min_distance = current * 0.90  # 10%距离，给趋势更多空间
        
        stop = min(candidate, min_distance)
        
        return round(float(stop), 2)
    
    # ==========================================================
    #  仓位计算（凯利公式）
    # ==========================================================
    
    def _calc_position_size(self, score, current_price, stop_loss, market) -> Tuple[float, float]:
        """
        凯利公式优化仓位
        """
        current_price = float(np.asarray(current_price).flat[0])
        stop_loss = float(np.asarray(stop_loss).flat[0])
        score = float(score)
        
        # Win rate estimation from composite score (capped 35%-75% — realistic range)
        win_rate = max(0.25, min(0.85, 0.5 + score / 200.0))
        
        # Upside / downside ratio
        upside = 0.05   # conservative 5% target return
        downside = max(0.01, 1.0 - stop_loss / current_price)
        rr = upside / downside
        
        # Full Kelly fraction
        kelly = win_rate - (1.0 - win_rate) / max(rr, 0.5)
        kelly = max(0.0, float(kelly))
        
        # Half-Kelly is safer — cap at 15% of total portfolio per single position
        half_kelly = kelly * 0.5
        half_kelly = min(half_kelly, 0.15)
        
        # Market-state multiplier (conservative calibration)
        state = market.get("market_state", "不明朗")
        risk_mult = {
            "牛市": 1.0,
            "反弹": 0.7,
            "震荡": 0.5,
            "回调": 0.35,
            "熊市": 0.2,
            "不明朗": 0.35,
        }
        mult = risk_mult.get(state, 0.4)
        
        # Position as percentage of portfolio (1%–20%)
        position_pct = half_kelly * 100.0 * mult
        max_position = min(20.0, max(1.0, position_pct))
        
        # Risk per trade = position * downside * safety factor
        risk_per_trade = min(3.0, max(0.1, max_position * downside * 0.2))
        
        return max_position, risk_per_trade

    def _build_portfolio_advice(self, top_picks, market) -> dict:
        """构建组合配置建议"""
        state = market.get("market_state", "不明朗")
        
        # 推荐组合
        if state in ("牛市", "反弹"):
            # 成长型配置
            allocation = {
                "进攻型(科技/新能源)": "40%",
                "成长型(消费/医药)": "35%",
                "防御型(银行/公用事业)": "15%",
                "现金": "10%",
            }
        elif state == "震荡":
            allocation = {
                "成长型(消费/医药)": "30%",
                "防御型(银行/公用事业)": "30%",
                "进攻型(科技/新能源)": "15%",
                "现金": "25%",
            }
        else:
            allocation = {
                "防御型(银行/公用事业)": "30%",
                "红利型(高股息)": "20%",
                "成长型(消费/医药)": "10%",
                "现金": "40%",
            }
        
        return {
            "strategy": "成长进攻" if state in ("牛市", "反弹") else ("防御为主" if state in ("熊市", "回调") else "均衡配置"),
            "allocation": allocation,
            "suggested_stocks": [d.name for d in top_picks[:6]],
        }

    # ==========================================================
    #  持有期限
    # ==========================================================

    def _determine_holding_period(self, score, tech, industry) -> Tuple[str, int]:
        """确定持有期限"""
        if score > 50:
            return "中长线", 60
        elif score > 30:
            return "中线", 30
        elif score > -30:
            return "短线", 10
        else:
            return "不持有", 0

    # ==========================================================
    #  置信度
    # ==========================================================

    def _calc_confidence(self, score, tech, market, data_len) -> float:
        """计算综合置信度"""
        conf = 0.5
        if data_len >= 200:
            conf += 0.2
        elif data_len >= 100:
            conf += 0.1
        signals = []
        for key in ["ma", "macd", "rsi"]:
            sig = tech.get(key, {}).get("signal", "")
            if "buy" in sig.lower():
                signals.append(1)
            elif "sell" in sig.lower():
                signals.append(-1)
        if signals:
            agreement = abs(sum(signals)) / len(signals)
            conf += agreement * 0.2
        if market.get("market_state") == "震荡":
            conf -= 0.1
        return float(np.clip(conf, 0.2, 0.95))

    # ==========================================================
    #  风险因素和催化剂
    # ==========================================================

    def _identify_risks(self, tech, market, industry) -> List[str]:
        """识别风险因素"""
        risks = []
        macd = tech.get("macd", {})
        if macd.get("divergence") == "顶背离":
            risks.append("MACD顶背离——价格新高但动能减弱")
        rsi = tech.get("rsi", {})
        rsi_val = rsi.get("rsi", 50)
        if rsi_val > 75:
            risks.append(f"RSI超买({rsi_val:.1f})——短期回调风险加大")
        boll = tech.get("boll", {})
        if boll.get("signal") == "突破上轨":
            risks.append("价格触及布林带上轨——超买风险")
        state = market.get("market_state", "")
        if state in ("熊市", "回调"):
            risks.append(f"大盘处于{state}——系统性风险较高")
        vol = tech.get("volume", {})
        if vol.get("vp_relation") == "放量下跌":
            risks.append("放量下跌——资金出逃迹象")
        if industry in ("房地产", "钢铁", "煤炭"):
            risks.append(f"{industry}行业面临政策不确定性")
        return risks[:5]

    def _identify_catalysts(self, tech, industry) -> List[str]:
        """识别上涨催化剂"""
        catalysts = []
        macd = tech.get("macd", {})
        if macd.get("divergence") == "底背离":
            catalysts.append("MACD底背离——反弹信号")
        rsi = tech.get("rsi", {})
        rsi_val = rsi.get("rsi", 50)
        if rsi_val < 30:
            catalysts.append(f"RSI超卖({rsi_val:.1f})——技术反弹需求")
        ma = tech.get("ma", {})
        if ma.get("alignment") == "黄金交叉形成中":
            catalysts.append("均线金叉形成中——趋势转多")
        boll = tech.get("boll", {})
        if boll.get("signal") == "跌破下轨":
            catalysts.append("触及布林带下轨——超跌反弹机会")
        if ma.get("above_ma60") == "是":
            catalysts.append("站稳60日均线——中期趋势支撑")
        return catalysts[:4]

    # ==========================================================
    #  情景分析
    # ==========================================================

    def _scenario_analysis(self, close, targets, stop_loss, current) -> dict:
        """三情景分析"""
        return {
            "optimistic": {
                "probability": "30%",
                "price": targets.get("optimistic", current * 1.2),
                "return": f"{round((targets.get('optimistic', current * 1.2)/current - 1) * 100, 1)}%",
                "trigger": "大盘走强 + 行业利好 + 放量突破前高",
            },
            "base": {
                "probability": "50%",
                "price": targets["base"],
                "return": f"{round((targets['base']/current - 1) * 100, 1)}%",
                "trigger": "大盘平稳 + 个股技术面正常推进",
            },
            "pessimistic": {
                "probability": "20%",
                "price": stop_loss,
                "return": f"{round((stop_loss/current - 1) * 100, 1)}%",
                "trigger": "大盘走弱 + 个股跌破关键支撑",
            },
        }

    # ==========================================================
    #  操作计划生成
    # ==========================================================

    def _generate_action_plan(self, decision, price, entry_low, entry_high,
                              stop_loss, tp1, tp2, max_pos, holding):
        """生成详细的操作计划"""
        lines = []
        if decision in (DecisionType.STRONG_BUY, DecisionType.BUY):
            lines = [
                "📋 **操作计划**", "",
                f"1️⃣ **入场策略**: ¥{entry_low:.2f} - ¥{entry_high:.2f} 区间分批建仓",
                f"   - 第一笔(40%仓位)在 ¥{entry_high:.2f} 附近",
                f"   - 第二笔(30%仓位)若回踩 ¥{entry_low:.2f} 加仓",
                "   - 剩余30%仓位等待放量突破确认后追加", "",
                f"2️⃣ **仓位管理**: 总仓位不超过总资金的 {max_pos:.0f}%", "",
                f"3️⃣ **止损纪律**: ¥{stop_loss:.2f} (硬止损，触及立即离场)",
                "   - 移动止损: 浮盈超 5% 后上移至成本，浮盈超 10% 后上移至 ¥{:,.2f}".format(stop_loss * 1.03),
                f"   - 时间止损: {holding}后无论盈亏均重新评估", "",
                "4️⃣ **阶梯止盈**:",
                f"   - 第一目标 ¥{tp1:.2f} → 平仓 50%，剩余止损上移至成本",
                f"   - 第二目标 ¥{tp2:.2f} → 清仓离场",
                "   - 若价格加速突破第二目标，可持有至趋势转弱", "",
                "5️⃣ **关键观察**: 量比 > 1.5、MACD柱状线放大、板块联动",
            ]
        elif decision == DecisionType.ACCUMULATE:
            lines = [
                "📋 **操作计划**", "",
                f"1️⃣ **试探建仓**: ¥{entry_low:.2f} 附近轻仓(≤{max_pos:.0f}%)进入",
                f"2️⃣ **硬止损**: ¥{stop_loss:.2f}，触及即出，不恋战",
                "3️⃣ **加仓条件**: 放量突破前高 + MACD金叉确认 → 加仓至标准仓位",
                "4️⃣ **观察窗口**: 5个交易日，若未启动则减仓观望",
                f"5️⃣ **首道止盈**: ¥{tp1:.2f}",
            ]
        elif decision in (DecisionType.REDUCE, DecisionType.SELL, DecisionType.STRONG_SELL):
            lines = [
                "📋 **操作计划**", "",
                "1️⃣ **立即行动**:",
                f"   - 减仓/清仓: ¥{price:.2f} 附近执行",
                "2️⃣ **最后防线**:",
                f"   - 硬止损 ¥{stop_loss:.2f}，跌破无条件清仓",
                "3️⃣ **反手条件**: 不急于抄底，等待以下信号共振:",
                "   - 放量长阳 + MACD金叉 + 站上MA20",
                "4️⃣ **资金去处**: 转入低β防御标的或货币基金，保留子弹",
            ]
        else:
            lines = [
                "📋 **操作计划**", "",
                f"1️⃣ **持有观望**: 当前价 ¥{price:.2f}，无明确操作信号",
                f"2️⃣ **上沿观察**: 放量突破 ¥{entry_high:.2f} → 可考虑加仓",
                f"3️⃣ **下沿保护**: 有效跌破 ¥{stop_loss:.2f} → 减仓至半仓以下",
                "4️⃣ **时间窗口**: 若连续3日缩量震荡 → 警惕变盘风险",
                f"5️⃣ **关键指标**:",
                "   - MACD: 关注DIF与DEA的收敛/发散方向",
                "   - 量能: 量比>1.2为有效突破信号",
                "   - 板块: 同行业至少2只以上联动才算确认",
            ]
        return "\n".join(lines)

    def _generate_summary(self, decision, score, price, targets, stop_loss, rr, holding, market):
        """生成一句话摘要"""
        if decision in (DecisionType.STRONG_BUY, DecisionType.BUY):
            return (
                f"【{decision.value}】综合评分 {score:+.0f} 分，当前价 ¥{price:.2f}，"
                f"目标价 ¥{targets['base']:.2f}，止损价 ¥{stop_loss:.2f}，"
                f"盈亏比 {rr:.1f}:1。{holding}持有。"
            )
        elif decision == DecisionType.ACCUMULATE:
            return (f"【{decision.value}】综合评分 {score:+.0f} 分，可轻仓试错。")
        elif decision == DecisionType.HOLD:
            return (f"【{decision.value}】综合评分 {score:+.0f} 分，不宜操作。")
        else:
            return (f"【{decision.value}】综合评分 {score:+.0f} 分，建议减仓/清仓。")

    # ==========================================================
    #  批量分析
    # ==========================================================

    def analyze_portfolio(self, codes: List[str] = None, top_n: int = 50) -> List["InvestmentDecision"]:
        """批量分析投资组合"""
        if codes is None:
            from ..data.market import CORE_STOCKS
            codes = [c for c, _, _ in CORE_STOCKS]
        decisions = []
        for code in codes:
            try:
                dec = self.analyze(code)
                if dec:
                    decisions.append(dec)
            except Exception:
                continue
        decisions.sort(key=lambda d: d.composite_score, reverse=True)
        return decisions[:top_n]

    def top_picks(self, n: int = 10) -> List["InvestmentDecision"]:
        """获取最佳投资标的"""
        all_decisions = self.analyze_portfolio()
        return [d for d in all_decisions if d.composite_score > 50][:n]

    def risk_warnings(self, n: int = 10) -> List["InvestmentDecision"]:
        """获取风险预警标的"""
        all_decisions = self.analyze_portfolio()
        return [d for d in all_decisions if d.composite_score < -30][:n]

    def market_outlook_report(self) -> dict:
        """市场展望 + 操作指南"""
        market = MarketStateAnalyzer.generate_outlook()
        top = self.top_picks(10)
        warnings = self.risk_warnings(5)
        industries = {}
        for d in top:
            industries.setdefault(d.industry, []).append(d.name)
        portfolio_advice = self._build_portfolio_advice(top, market)
        
        # Build a human-readable summary
        state = market.get("market_state", "不明朗")
        emotion = market.get("emotion", "未知")
        risk = market.get("risk_level", "中")
        position = market.get("recommended_position", "30%-50%")
        
        summary_lines = [
            f"市场状态: {state}",
            f"市场情绪: {emotion}",
            f"风险等级: {risk}",
            f"推荐仓位: {position}",
            f"建议策略: {portfolio_advice.get("strategy", "观望")}",
        ]
        if top:
            top_names = [d.name for d in top[:5]]
            summary_lines.append(f"重点关注: {', '.join(top_names)}")
        if warnings:
            warn_names = [d.name for d in warnings[:3]]
            summary_lines.append(f"风险提示: {', '.join(warn_names)}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_state": market,
            "top_picks": [d.to_dict() for d in top],
            "risk_warnings": [d.to_dict() for d in warnings],
            "industry_allocation": industries,
            "portfolio_advice": portfolio_advice,
            "summary": "\n".join(summary_lines),
        }
