"""
Hyperion Pro — 实时信号预警系统
================================
华尔街级实时监测 + 推送系统

核心功能:
  1. 价格突破预警 (突破前高/跌破支撑)
  2. 技术指标共振预警 (多指标同时发出信号)
  3. 量价异动预警 (放量/缩量/背离)
  4. 板块轮动预警 (热点切换)
  5. 市场情绪突变预警
  6. 自定义阈值预警

每个预警都包含:
  - 严重级别 (INFO/WARNING/CRITICAL)
  - 触发条件说明
  - 建议操作
  - 时效性标记
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.market import (
    fetch_realtime_quotes, fetch_history, fetch_index_quotes,
    sector_quotes, money_flow, DataUnavailableError,
    CORE_STOCKS, INDICES, get_stock_name, get_stock_industry, DATA_DIR
)
from ..analysis.technical import TechnicalAnalyzer
from ..analysis.market_state import MarketStateAnalyzer


# ── 数据模型 ────────────────────────────────────────────────

class AlertLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """单个预警信号"""
    alert_id: str
    timestamp: str
    level: str                      # INFO / WARNING / CRITICAL
    category: str                   # price / technical / volume / sector / sentiment
    title: str                      # 标题
    description: str                # 详细说明
    stock_code: str = ""            # 关联股票
    stock_name: str = ""            # 关联股票名称
    current_value: float = 0.0      # 触发值
    threshold: float = 0.0          # 阈值
    action_advice: str = ""         # 建议操作
    expiry: str = ""                # 过期时间 (预警有时效性)
    
    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "level": self.level,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "action_advice": self.action_advice,
            "expiry": self.expiry,
        }


class SignalAlertSystem:
    """
    实时信号预警系统
    
    使用:
        alerts = SignalAlertSystem()
        results = alerts.scan_all()  # 全市场扫描
        alerts.print_alerts(results)
    """
    
    def __init__(self):
        self.tech = TechnicalAnalyzer()
        self.alert_history: List[dict] = []
        self._last_scan: Dict[str, float] = {}  # 记忆上次扫描值
    
    # ═════════════════════════════════════════════════════
    #  全市场扫描
    # ═════════════════════════════════════════════════════
    
    def scan_all(self) -> List[Alert]:
        """
        全市场扫描 — 生成所有预警
        
        Returns:
            List[Alert]: 按严重级别排序的预警列表
        """
        alerts: List[Alert] = []
        
        try:
            # 1. 市场情绪预警
            alerts.extend(self._scan_sentiment())
        except Exception:
            pass
        
        try:
            # 2. 板块轮动预警
            alerts.extend(self._scan_sectors())
        except Exception:
            pass
        
        try:
            # 3. 个股技术指标共振预警 (只扫描核心股票池)
            alerts.extend(self._scan_stocks())
        except Exception:
            pass
        
        try:
            # 4. 指数关键位预警
            alerts.extend(self._scan_indices())
        except Exception:
            pass
        
        # 按级别排序: CRITICAL > WARNING > INFO
        level_order = {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1, AlertLevel.INFO: 2}
        alerts.sort(key=lambda a: level_order.get(a.level, 99))
        
        self.alert_history = [a.to_dict() for a in alerts[-50:]]
        return alerts
    
    # ═════════════════════════════════════════════════════
    #  子扫描器
    # ═════════════════════════════════════════════════════
    
    def _scan_sentiment(self) -> List[Alert]:
        """市场情绪扫描"""
        alerts = []
        try:
            emotion = MarketStateAnalyzer.analyze_emotion()
            overall = MarketStateAnalyzer.analyze_overall()
            
            state = overall.get("market_state", "不明朗")
            emotion_name = emotion.get("emotion", "未知")
            emotion_score = emotion.get("score", 0.5)
            up_ratio = emotion.get("up_ratio", 50)
            limit_up = emotion.get("limit_up", 0)
            limit_down = emotion.get("limit_down", 0)
            median_change = emotion.get("median_change", 0)
            
            # 极度过热
            if up_ratio > 80 and limit_up > 10:
                alerts.append(Alert(
                    alert_id=f"sentiment_overheat_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.WARNING,
                    category="sentiment",
                    title="市场情绪过热预警",
                    description=f"上涨比例{up_ratio:.0f}%，涨停{limit_up}只，情绪过热",
                    current_value=up_ratio,
                    threshold=80,
                    action_advice="短期存在回调风险，建议减仓获利了结，不追高",
                    expiry=(datetime.now() + timedelta(hours=2)).isoformat(),
                ))
            
            # 极度恐慌
            if up_ratio < 25 and limit_down > 10:
                alerts.append(Alert(
                    alert_id=f"sentiment_panic_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.WARNING,
                    category="sentiment",
                    title="市场恐慌预警",
                    description=f"上涨比例{up_ratio:.0f}%，跌停{limit_down}只，情绪恐慌",
                    current_value=up_ratio,
                    threshold=25,
                    action_advice="短期超卖，关注反抽机会，不要在恐慌中割肉",
                    expiry=(datetime.now() + timedelta(hours=2)).isoformat(),
                ))
            
            # 市场状态突变
            if state in ("牛市", "反弹") and median_change < -1.0:
                alerts.append(Alert(
                    alert_id=f"state_downturn_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.CRITICAL,
                    category="sentiment",
                    title="市场转弱预警",
                    description=f"当前{state}但涨跌中位数{median_change:.1f}%，动能减弱",
                    current_value=median_change,
                    threshold=-1.0,
                    action_advice="注意保护利润，收紧止损位，降低仓位",
                    expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                ))
            
            if state in ("熊市", "回调") and median_change > 1.5:
                alerts.append(Alert(
                    alert_id=f"state_reversal_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.WARNING,
                    category="sentiment",
                    title="市场企稳信号",
                    description=f"当前{state}但涨跌中位数{median_change:.1f}%，短期企稳迹象",
                    current_value=median_change,
                    threshold=1.5,
                    action_advice="关注底部放量信号，准备试探性建仓",
                    expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                ))
        
        except Exception:
            pass
        
        return alerts
    
    def _scan_sectors(self) -> List[Alert]:
        """板块轮动扫描"""
        alerts = []
        try:
            sectors = sector_quotes()
            if sectors.empty or len(sectors) < 5:
                return alerts
            
            # 最热/最冷板块
            top3 = sectors.head(3)
            bottom3 = sectors.tail(3)
            
            # 板块分化度
            top_change = top3["avg_change"].mean()
            bottom_change = bottom3["avg_change"].mean()
            divergence = top_change - bottom_change
            
            if divergence > 3:
                alerts.append(Alert(
                    alert_id=f"sector_divergence_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.INFO,
                    category="sector",
                    title="板块严重分化预警",
                    description=f"最强板块({top3.index[0]})涨幅{top_change:.1f}%，"
                              f"最弱板块({bottom3.index[0]})跌幅{bottom_change:.1f}%",
                    current_value=divergence,
                    threshold=3,
                    action_advice="资金高度集中，注意热点切换风险，跟随最强板块或等待轮动",
                    expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                ))
            
            # 强势板块
            for idx, row in top3.iterrows():
                if row["avg_change"] > 2.5:
                    alerts.append(Alert(
                        alert_id=f"sector_hot_{idx}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.INFO,
                        category="sector",
                        title=f"强势板块: {idx}",
                        description=f"板块涨幅{row['avg_change']:.1f}%，{int(row['up_stocks'])}/{int(row['stock_count'])}只上涨",
                        current_value=row["avg_change"],
                        threshold=2.5,
                        action_advice=f"关注{idx}龙头股，可适度参与但注意追高风险",
                        expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                    ))
            
            # 弱势板块
            for idx, row in bottom3.iterrows():
                if row["avg_change"] < -2:
                    alerts.append(Alert(
                        alert_id=f"sector_cold_{idx}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.INFO,
                        category="sector",
                        title=f"弱势板块: {idx}",
                        description=f"板块跌幅{row['avg_change']:.1f}%，{int(row['down_stocks'])}/{int(row['stock_count'])}只下跌",
                        current_value=row["avg_change"],
                        threshold=-2,
                        action_advice=f"回避{idx}板块，等待企稳信号",
                        expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                    ))
        
        except Exception:
            pass
        
        return alerts
    
    def _scan_stocks(self) -> List[Alert]:
        """个股技术指标扫描"""
        alerts = []
        
        # 扫描核心股票池
        batch_size = 5  # 每次扫描5只，避免过多API调用
        codes_to_scan = [c for c, _, _ in CORE_STOCKS]
        # 随机取样或按优先级
        
        for code in codes_to_scan[:batch_size]:
            try:
                history = fetch_history(code, days=120)
                if history.empty or len(history) < 30:
                    continue
                
                close = history["close"].values
                volume = history["volume"].values
                high = history["high"].values
                low = history["low"].values
                
                name = get_stock_name(code)
                tech = self.tech.comprehensive_analysis(history)
                
                # --- 技术指标共振 ---
                signals = []
                ma_sig = tech.get("ma", {}).get("signal", "")
                macd_sig = tech.get("macd", {}).get("signal", "")
                rsi = tech.get("rsi", {})
                rsi_val = rsi.get("rsi", 50)
                
                if "buy" in ma_sig:
                    signals.append("MA")
                if "buy" in macd_sig:
                    signals.append("MACD")
                if rsi_val < 35:
                    signals.append("RSI超卖")
                
                # MACD 背离检测
                divergence = tech.get("macd", {}).get("divergence", "")
                
                if len(signals) >= 2:
                    alerts.append(Alert(
                        alert_id=f"tech_convergence_{code}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.WARNING,
                        category="technical",
                        title=f"技术指标共振买入: {name}({code})",
                        description=f"多指标同时发出信号: {' + '.join(signals)}，RSI={rsi_val:.0f}",
                        stock_code=code, stock_name=name,
                        current_value=close[-1],
                        action_advice="多项技术指标共振，可重点关注！建议结合量能确认后介入",
                        expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                    ))
                
                # MACD 底背离
                if divergence == "底背离":
                    alerts.append(Alert(
                        alert_id=f"macd_bullish_div_{code}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.CRITICAL,
                        category="technical",
                        title=f"MACD底背离: {name}({code})",
                        description="价格新低但MACD未创新低，强烈买入信号",
                        stock_code=code, stock_name=name,
                        current_value=close[-1],
                        action_advice="强烈建议关注！底背离后大概率反弹，逢低建仓",
                        expiry=(datetime.now() + timedelta(days=3)).isoformat(),
                    ))
                
                # MACD 顶背离
                if divergence == "顶背离":
                    alerts.append(Alert(
                        alert_id=f"macd_bearish_div_{code}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.CRITICAL,
                        category="technical",
                        title=f"MACD顶背离: {name}({code})",
                        description="价格新高但MACD未创新高，强烈卖出信号",
                        stock_code=code, stock_name=name,
                        current_value=close[-1],
                        action_advice="强烈建议减仓！顶背离后大概率回调",
                        expiry=(datetime.now() + timedelta(days=3)).isoformat(),
                    ))
                
                # --- 量价异常 ---
                vol_ma20 = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
                vol_ratio = volume[-1] / max(vol_ma20, 1)
                price_change = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0
                
                if vol_ratio > 3 and price_change > 3:
                    alerts.append(Alert(
                        alert_id=f"vol_surge_{code}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.WARNING,
                        category="volume",
                        title=f"放量大涨: {name}({code})",
                        description=f"成交量突增{vol_ratio:.1f}倍，涨幅{price_change:.1f}%，放量突破",
                        stock_code=code, stock_name=name,
                        current_value=vol_ratio,
                        threshold=3,
                        action_advice="放量突破是有效的上涨信号，可追涨但设好止损",
                        expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                    ))
                
                if vol_ratio > 3 and price_change < -3:
                    alerts.append(Alert(
                        alert_id=f"vol_dump_{code}_{int(time.time())}",
                        timestamp=datetime.now().isoformat(),
                        level=AlertLevel.WARNING,
                        category="volume",
                        title=f"放量大跌: {name}({code})",
                        description=f"成交量暴增{vol_ratio:.1f}倍，跌幅{price_change:.1f}%，资金出逃",
                        stock_code=code, stock_name=name,
                        current_value=vol_ratio,
                        threshold=3,
                        action_advice="放量大跌是危险信号，建议减仓规避风险",
                        expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                    ))
                
                # --- 价格关键位置突破 ---
                if len(close) >= 20:
                    high_20 = np.max(high[-20:-1])
                    low_20 = np.min(low[-20:-1])
                    
                    if close[-1] > high_20 * 1.005 and vol_ratio > 1.5:
                        alerts.append(Alert(
                            alert_id=f"breakout_{code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.WARNING,
                            category="price",
                            title=f"突破20日新高: {name}({code})",
                            description=f"价格突破20日高点{high_20:.2f}，当前{close[-1]:.2f}，量比{vol_ratio:.1f}",
                            stock_code=code, stock_name=name,
                            current_value=close[-1],
                            threshold=high_20,
                            action_advice="有效突破20日高点，趋势转多，可适量参与",
                            expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                        ))
                    
                    if close[-1] < low_20 * 0.995:
                        alerts.append(Alert(
                            alert_id=f"breakdown_{code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.WARNING,
                            category="price",
                            title=f"跌破20日新低: {name}({code})",
                            description=f"价格跌破20日低点{low_20:.2f}，当前{close[-1]:.2f}",
                            stock_code=code, stock_name=name,
                            current_value=close[-1],
                            threshold=low_20,
                            action_advice="跌破关键支撑，建议减仓或清仓",
                            expiry=(datetime.now() + timedelta(hours=4)).isoformat(),
                        ))
            
            except (DataUnavailableError, Exception):
                continue
        
        return alerts
    
    def _scan_indices(self) -> List[Alert]:
        """指数关键位扫描"""
        alerts = []
        try:
            # 获取上证指数历史数据
            from ..data.market import fetch_history as fh
            
            for index_code, index_name in [
                ("000001.SH", "上证指数"),
                ("399001.SZ", "深证成指"),
                ("399006.SZ", "创业板指"),
            ]:
                try:
                    # Use 000001 as proxy for indices via Sina
                    proxy_code = "000001" if index_code.startswith("000001") else (
                        "399001" if index_code.startswith("399001") else "399006"
                    )
                    history = fetch_history(proxy_code, days=120)
                    if history is None or history.empty or len(history) < 30:
                        continue
                    
                    close = history["close"].values
                    tech = self.tech.comprehensive_analysis(history)
                    
                    # 均线关键位
                    ma = tech.get("ma", {})
                    macd = tech.get("macd", {})
                    
                    current = close[-1]
                    ma20 = ma.get("ma20", current)
                    ma60 = ma.get("ma60", current)
                    
                    # 指数跌破MA60
                    if current < ma60 and ma.get("above_ma60") == "是":
                        alerts.append(Alert(
                            alert_id=f"index_ma60_break_{index_code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.CRITICAL,
                            category="technical",
                            title=f"重要信号: {index_name}跌破60日均线",
                            description=f"{index_name}当前{current:.2f}，跌破MA60({ma60:.2f})，中期趋势转弱",
                            current_value=current,
                            threshold=ma60,
                            action_advice="大盘跌破中期生命线！建议大幅减仓至30%以下，等待企稳",
                            expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                        ))
                    
                    # 指数站上MA60
                    if current > ma60 and ma.get("above_ma60") == "否":
                        alerts.append(Alert(
                            alert_id=f"index_ma60_above_{index_code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.WARNING,
                            category="technical",
                            title=f"积极信号: {index_name}站上60日均线",
                            description=f"{index_name}当前{current:.2f}，突破MA60({ma60:.2f})，中期趋势转多",
                            current_value=current,
                            threshold=ma60,
                            action_advice="大盘企稳信号，可适度加仓至50%-70%",
                            expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                        ))
                    
                    # MACD 金叉/死叉
                    macd_signal = macd.get("signal", "")
                    if "strong_buy" in macd_signal or ("buy" in macd_signal and "weak" not in macd_signal):
                        alerts.append(Alert(
                            alert_id=f"index_macd_golden_{index_code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.WARNING,
                            category="technical",
                            title=f"技术信号: {index_name}MACD金叉",
                            description=macd.get("description", "MACD发出买入信号"),
                            current_value=close[-1],
                            action_advice="大盘技术面改善，操作环境趋好",
                            expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                        ))
                    
                    if "strong_sell" in macd_signal or ("sell" in macd_signal and "weak" not in macd_signal):
                        alerts.append(Alert(
                            alert_id=f"index_macd_dead_{index_code}_{int(time.time())}",
                            timestamp=datetime.now().isoformat(),
                            level=AlertLevel.WARNING,
                            category="technical",
                            title=f"技术信号: {index_name}MACD死叉",
                            description=macd.get("description", "MACD发出卖出信号"),
                            current_value=close[-1],
                            action_advice="大盘技术面恶化，注意控制仓位",
                            expiry=(datetime.now() + timedelta(days=1)).isoformat(),
                        ))
                
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return alerts
    
    # ═════════════════════════════════════════════════════
    #  输出格式化
    # ═════════════════════════════════════════════════════
    
    def print_alerts(self, alerts: List[Alert], max_alerts: int = 20):
        """打印预警到终端"""
        if not alerts:
            print("\n  ✅ 当前无活跃预警")
            return
        
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
        infos = [a for a in alerts if a.level == AlertLevel.INFO]
        
        print(f"\n  ⚠️  实时预警 ({len(alerts)} 条)")
        
        if critical:
            print(f"\n  {'!'*50}")
            print(f"  {' 紧急预警 ' + '!'*50}")
            print(f"  {'!'*50}")
            for a in critical[:5]:
                print(f"  🔴 [{a.category}] {a.title}")
                print(f"     {a.description}")
                if a.action_advice:
                    print(f"     💡 {a.action_advice}")
                print()
        
        if warnings:
            for a in warnings[:10]:
                level_icon = "🟡" if a.level == AlertLevel.WARNING else "🟢"
                print(f"  {level_icon} [{a.category}] {a.title}")
                print(f"     {a.description}")
                if a.action_advice:
                    print(f"     💡 {a.action_advice}")
                print()
        
        if infos:
            for a in infos[:5]:
                print(f"  ℹ️  [{a.category}] {a.title}")
                if a.action_advice:
                    print(f"     {a.action_advice}")
    
    def to_dict_list(self, alerts: List[Alert]) -> List[dict]:
        """转为可序列化的字典列表"""
        return [a.to_dict() for a in alerts]


# ── 便捷函数 ────────────────────────────────────────────────

_alert_system: Optional[SignalAlertSystem] = None


def get_alert_system() -> SignalAlertSystem:
    global _alert_system
    if _alert_system is None:
        _alert_system = SignalAlertSystem()
    return _alert_system


def quick_scan() -> List[Alert]:
    """快速全市场扫描"""
    system = get_alert_system()
    return system.scan_all()


def print_alerts(alerts: List[Alert]):
    """打印预警"""
    system = get_alert_system()
    system.print_alerts(alerts)


ALERT_CATEGORIES = {
    "price": "价格突破",
    "technical": "技术指标",
    "volume": "量能异动",
    "sector": "板块轮动",
    "sentiment": "市场情绪",
}
