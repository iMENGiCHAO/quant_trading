"""
Hyperion Pro — 信号绩效追踪引擎
=================================
华尔街级别绩效追踪 —— 这是系统从"分析工具"到"交易系统"的关键跃迁

核心功能:
  1. 记录每一次投资建议 (信号日志)
  2. 定期回溯验证已发出信号的准确性
  3. 计算胜率、盈亏比、期望值
  4. 策略归因分析 — 哪些策略赚了钱、哪些在亏钱
  5. 绩效报告 — 告诉使用者系统靠不靠谱

没有绩效追踪的量化系统 = 算命先生
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.market import fetch_history, get_stock_name, get_stock_industry, DATA_DIR

# 信号日志存储路径
SIGNAL_LOG_DIR = DATA_DIR / "signal_logs"
SIGNAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_LOG_FILE = SIGNAL_LOG_DIR / "signals.jsonl"
PERFORMANCE_FILE = SIGNAL_LOG_DIR / "performance.json"


@dataclass
class SignalRecord:
    """单次投资建议记录"""
    signal_id: str                # 唯一标识
    timestamp: str                # 发出时间
    code: str                     # 股票代码
    name: str                     # 股票名
    industry: str                 # 行业
    
    decision: str                 # 决策类型 (强烈买入/买入/卖出等)
    composite_score: float        # 综合评分
    
    signal_price: float           # 建议时的价格
    target_price: float           # 目标价
    stop_loss: float              # 止损价
    holding_period: str           # 推荐持有期
    
    strategy_sources: List[str]   # 触发信号的策略源
    
    # 事后验证字段 (后期填写)
    verified: bool = False
    actual_price_5d: Optional[float] = None    # 5日后价格
    actual_price_20d: Optional[float] = None   # 20日后价格
    actual_price_60d: Optional[float] = None   # 60日后价格
    hit_target: Optional[bool] = None          # 是否达到目标价
    hit_stop: Optional[bool] = None            # 是否触发止损
    realized_return: Optional[float] = None    # 实际收益率
    verification_date: Optional[str] = None    # 验证日期
    
    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "code": self.code,
            "name": self.name,
            "industry": self.industry,
            "decision": self.decision,
            "composite_score": self.composite_score,
            "signal_price": self.signal_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "holding_period": self.holding_period,
            "strategy_sources": self.strategy_sources,
            "verified": self.verified,
            "actual_price_5d": self.actual_price_5d,
            "actual_price_20d": self.actual_price_20d,
            "actual_price_60d": self.actual_price_60d,
            "hit_target": self.hit_target,
            "hit_stop": self.hit_stop,
            "realized_return": self.realized_return,
            "verification_date": self.verification_date,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SignalRecord":
        return cls(**d)


class PerformanceTracker:
    """
    信号绩效追踪器
    
    用法:
        tracker = PerformanceTracker()
        
        # 记录信号
        tracker.log_signal(decision)  # InvestmentDecision
        
        # 回溯验证已发出的信号
        report = tracker.verify_past_signals()
        
        # 获取绩效摘要
        summary = tracker.get_performance_summary()
    """
    
    def __init__(self):
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        if not SIGNAL_LOG_FILE.exists():
            SIGNAL_LOG_FILE.touch()
    
    def log_signal(self, decision) -> str:
        """
        记录一条投资建议
        
        Args:
            decision: InvestmentDecision 对象
            
        Returns:
            signal_id
        """
        signal_id = f"{decision.code}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(decision.code) % 10000:04d}"
        
        record = SignalRecord(
            signal_id=signal_id,
            timestamp=decision.timestamp,
            code=decision.code,
            name=decision.name,
            industry=decision.industry,
            decision=decision.decision,
            composite_score=decision.composite_score,
            signal_price=decision.current_price,
            target_price=decision.target_price_base,
            stop_loss=decision.stop_loss,
            holding_period=decision.holding_period,
            strategy_sources=[decision.strategy_consensus],
        )
        
        with open(SIGNAL_LOG_FILE, "a") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        
        return signal_id
    
    def get_unverified_signals(self, days: int = 5) -> List[SignalRecord]:
        """获取尚未验证的信号 (T+days天前的)"""
        cutoff = datetime.now() - timedelta(days=days)
        signals = []
        
        if not SIGNAL_LOG_FILE.exists():
            return signals
        
        with open(SIGNAL_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = SignalRecord.from_dict(json.loads(line))
                    sig_time = datetime.fromisoformat(record.timestamp)
                    if sig_time < cutoff and not record.verified:
                        signals.append(record)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return signals
    
    def verify_past_signals(self, min_days: int = 5) -> dict:
        """
        回溯验证所有已过期的信号
        对比信号发出时的价格和实际最新价格
        
        Returns:
            dict: 验证结果统计
        """
        unverified = self.get_unverified_signals(min_days)
        
        if not unverified:
            return {"verified": 0, "total": 0, "message": "没有待验证的信号"}
        
        # 按股票代码去重，减少API调用
        codes = list(set(s.code for s in unverified))
        price_cache = {}
        
        for code in codes:
            try:
                df = fetch_history(code, days=250)
                if not df.empty:
                    close = df["close"].values
                    price_cache[code] = {
                        "latest": float(close[-1]),
                        "max_since": float(np.max(close[-60:])) if len(close) >= 60 else float(close[-1]),
                        "min_since": float(np.min(close[-60:])) if len(close) >= 60 else float(close[-1]),
                    }
            except Exception:
                continue
        
        # 验证每条信号
        verified = []
        for signal in unverified:
            prices = price_cache.get(signal.code)
            if prices is None:
                continue
            
            current_price = prices["latest"]
            signal.actual_price_20d = current_price
            signal.actual_price_60d = prices["max_since"]
            signal.verified = True
            signal.verification_date = datetime.now().isoformat()
            
            # 判断是否达到目标
            signal.hit_target = prices["max_since"] >= signal.target_price
            signal.hit_stop = prices["min_since"] <= signal.stop_loss
            
            # 计算实际收益率 (按建议价到当前价)
            signal.realized_return = round((current_price / signal.signal_price - 1) * 100, 2)
            
            verified.append(signal)
        
        # 写回日志 (更新验证状态)
        self._update_signals(verified)
        
        # 计算统计
        return self._compute_verification_stats(verified)
    
    def _update_signals(self, verified_signals: List[SignalRecord]):
        """更新信号日志中的验证状态"""
        all_signals = []
        verified_ids = {s.signal_id: s for s in verified_signals}
        
        with open(SIGNAL_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record["signal_id"] in verified_ids:
                        updated = verified_ids[record["signal_id"]]
                        record.update(updated.to_dict())
                    all_signals.append(record)
                except (json.JSONDecodeError, KeyError):
                    all_signals.append({})
        
        with open(SIGNAL_LOG_FILE, "w") as f:
            for record in all_signals:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def _compute_verification_stats(self, verified: List[SignalRecord]) -> dict:
        """计算验证统计"""
        if not verified:
            return {"total": 0, "message": "无已验证信号"}
        
        buy_signals = [s for s in verified if "买入" in s.decision]
        sell_signals = [s for s in verified if "卖出" in s.decision]
        
        # 买入信号统计
        buy_correct = [s for s in buy_signals if s.realized_return and s.realized_return > 0]
        buy_wrong = [s for s in buy_signals if s.realized_return and s.realized_return <= 0]
        buy_hit_rate = len(buy_correct) / max(len(buy_signals), 1) * 100
        
        avg_buy_return = np.mean([s.realized_return for s in buy_signals if s.realized_return is not None]) if buy_signals else 0
        avg_sell_return = np.mean([s.realized_return for s in sell_signals if s.realized_return is not None]) if sell_signals else 0
        
        target_hit_rate = len([s for s in buy_signals if s.hit_target]) / max(len(buy_signals), 1) * 100
        stop_hit_rate = len([s for s in buy_signals if s.hit_stop]) / max(len(buy_signals), 1) * 100
        
        # 按策略分组
        by_strategy = defaultdict(lambda: {"total": 0, "correct": 0, "returns": []})
        for s in buy_signals:
            for src in s.strategy_sources:
                by_strategy[src]["total"] += 1
                if s.realized_return and s.realized_return > 0:
                    by_strategy[src]["correct"] += 1
                if s.realized_return is not None:
                    by_strategy[src]["returns"].append(s.realized_return)
        
        strategy_perf = {}
        for name, data in by_strategy.items():
            strategy_perf[name] = {
                "trades": data["total"],
                "win_rate": round(data["correct"] / max(data["total"], 1) * 100, 1),
                "avg_return": round(np.mean(data["returns"]), 2) if data["returns"] else 0,
            }
        
        # 按行业分组
        by_industry = defaultdict(lambda: {"total": 0, "correct": 0, "returns": []})
        for s in buy_signals:
            by_industry[s.industry]["total"] += 1
            if s.realized_return and s.realized_return > 0:
                by_industry[s.industry]["correct"] += 1
            if s.realized_return is not None:
                by_industry[s.industry]["returns"].append(s.realized_return)
        
        industry_perf = {}
        for ind, data in by_industry.items():
            industry_perf[ind] = {
                "trades": data["total"],
                "win_rate": round(data["correct"] / max(data["total"], 1) * 100, 1),
                "avg_return": round(np.mean(data["returns"]), 2) if data["returns"] else 0,
            }
        
        return {
            "verified_at": datetime.now().isoformat(),
            "total_verified": len(verified),
            "buy_signals_total": len(buy_signals),
            "buy_signals_correct": len(buy_correct),
            "buy_signals_wrong": len(buy_wrong),
            "buy_win_rate": round(buy_hit_rate, 1),
            "avg_buy_return": round(avg_buy_return, 2),
            "avg_sell_return": round(avg_sell_return, 2),
            "target_hit_rate": round(target_hit_rate, 1),
            "stop_hit_rate": round(stop_hit_rate, 1),
            "strategy_performance": strategy_perf,
            "industry_performance": industry_perf,
            "recent_signals": [s.to_dict() for s in verified[-15:]],
        }
    
    def get_performance_summary(self) -> dict:
        """获取完整绩效摘要"""
        # 尝试加载缓存的绩效数据
        if PERFORMANCE_FILE.exists():
            try:
                with open(PERFORMANCE_FILE, "r") as f:
                    cached = json.load(f)
                    age = (datetime.now() - datetime.fromisoformat(cached.get("verified_at", "2000-01-01"))).total_seconds()
                    if age < 3600:  # 1小时内缓存
                        return cached
            except (json.JSONDecodeError, KeyError):
                pass
        
        # 运行验证
        stats = self.verify_past_signals(min_days=5)
        
        # 加载所有历史信号计算累计指标
        all_signals = self._load_all_signals()
        verified_count = len([s for s in all_signals if s.get("verified")])
        total_count = len(all_signals)
        
        # 计算累计的胜率等
        if verified_count > 0:
            verified_list = [s for s in all_signals if s.get("verified") and s.get("realized_return") is not None]
            buy_verified = [s for s in verified_list if "买入" in s.get("decision", "")]
            cumulative_win_rate = len([s for s in buy_verified if s.get("realized_return", 0) > 0]) / max(len(buy_verified), 1) * 100
            cumulative_avg_return = np.mean([s["realized_return"] for s in buy_verified]) if buy_verified else 0
        else:
            cumulative_win_rate = 0
            cumulative_avg_return = 0
        
        summary = {
            **stats,
            "total_signals_ever": total_count,
            "verified_signals_ever": verified_count,
            "cumulative_win_rate": round(cumulative_win_rate, 1),
            "cumulative_avg_return": round(cumulative_avg_return, 2),
            "data_freshness": "实时数据" if total_count > 0 else "无历史记录",
        }
        
        # 缓存
        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary
    
    def _load_all_signals(self) -> List[dict]:
        """加载所有历史信号"""
        signals = []
        if not SIGNAL_LOG_FILE.exists():
            return signals
        
        with open(SIGNAL_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return signals
    
    def get_track_record_text(self) -> str:
        """生成可读的绩效追踪报告"""
        summary = self.get_performance_summary()
        
        total = summary.get("buy_signals_total", 0)
        correct = summary.get("buy_signals_correct", 0)
        win_rate = summary.get("buy_win_rate", 0)
        avg_ret = summary.get("avg_buy_return", 0)
        
        lines = [
            "=" * 50,
            "  信号绩效追踪报告",
            "=" * 50,
            "",
            f"  总买入信号: {total} 次",
            f"  盈利信号: {correct} 次",
            f"  胜率: {win_rate:.1f}%",
            f"  平均收益率: {avg_ret:+.2f}%",
            f"  累计信号数: {summary.get('total_signals_ever', 0)}",
            f"  已验证信号: {summary.get('verified_signals_ever', 0)}",
            "",
        ]
        
        # 按策略排名
        strategy = summary.get("strategy_performance", {})
        if strategy:
            ranked = sorted(strategy.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)
            lines.append("策略绩效排名:")
            for name, perf in ranked:
                lines.append(f"  {name}: 胜率 {perf.get('win_rate', 0):.1f}%, 均收益 {perf.get('avg_return', 0):+.2f}%")
        
        # 最近信号
        recent = summary.get("recent_signals", [])
        if recent:
            lines.append("")
            lines.append("最近验证信号:")
            for s in recent[-5:]:
                ret = s.get("realized_return", 0) or 0
                emoji = "✅" if ret > 0 else "❌"
                lines.append(f"  {emoji} {s.get('name', '')}({s.get('code', '')}) "
                           f"建议价{s.get('signal_price', 0):.2f} → 现价{s.get('actual_price_20d', 0):.2f} "
                           f"({ret:+.2f}%)")
        
        lines.append("")
        lines.append(f"  报告生成时间: {summary.get('verified_at', 'N/A')}")
        lines.append("")
        
        return "\n".join(lines)


# ==========================================================
#  便捷函数
# ==========================================================

_tracker_instance: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PerformanceTracker()
    return _tracker_instance


def log_decision(decision) -> str:
    """便捷函数: 记录投资决策"""
    return get_tracker().log_signal(decision)


def verify_and_report() -> str:
    """便捷函数: 验证并返回绩效报告"""
    return get_tracker().get_track_record_text()
