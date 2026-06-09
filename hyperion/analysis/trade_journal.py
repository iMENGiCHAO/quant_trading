"""
Hyperion Pro — 交易日志与绩效归因系统
======================================
华尔街级交易日记系统 — 记录每笔交易, 分析盈亏原因

核心功能:
  1. 用户可记录自己的实盘交易
  2. 自动对比系统建议 vs 实际操作
  3. 盈亏归因分析 (技术面/基本面/情绪/运气)
  4. 行为金融学分析 (追涨杀跌/过早止盈/扛单)
  5. 月度/季度绩效报告
  6. 交易心理评估

价值:
  - 没有交易日志的量化交易 = 闭着眼睛开车
  - 知道自己赚在哪、亏在哪，才能持续进步
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.market import DATA_DIR, CORE_STOCKS, get_stock_name, get_stock_industry
from ..data.market import fetch_history, fetch_realtime_quotes

# 交易日志存储
JOURNAL_DIR = DATA_DIR / "trade_journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
TRADES_FILE = JOURNAL_DIR / "trades.jsonl"
JOURNAL_CONFIG = JOURNAL_DIR / "config.json"


# ── 交易方向 ────────────────────────────────────────────────

class TradeDirection:
    BUY = "买入"
    SELL = "卖出"
    SHORT = "融券卖出"
    COVER = "买券还券"


class TradeStatus:
    OPEN = "持仓中"
    CLOSED = "已平仓"
    CANCELLED = "已取消"


class ExitReason:
    TAKE_PROFIT = "止盈"
    STOP_LOSS = "止损"
    TRAILING_STOP = "移动止损"
    TIME_STOP = "时间止损"
    SIGNAL_REVERSAL = "信号反转"
    MANUAL = "手动平仓"
    OTHER = "其他"


# ── 交易记录 ────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: str
    code: str
    name: str
    industry: str
    
    direction: str              # 买入/卖出
    entry_date: str             # 入场日期
    entry_price: float          # 入场价格
    quantity: int               # 数量
    entry_reason: str           # 入场理由
    
    # 出场
    exit_date: str = ""         # 出场日期
    exit_price: float = 0.0     # 出场价格
    exit_reason: str = ""       # 出场理由
    status: str = "持仓中"      # 状态
    
    # 盈亏
    pnl: float = 0.0            # 盈亏金额
    pnl_pct: float = 0.0        # 盈亏百分比
    holding_days: int = 0       # 持有天数
    
    # 对比系统建议
    system_signal: str = ""     # 系统当时的信号
    signal_price: float = 0.0   # 系统信号价
    signal_score: float = 0.0   # 系统评分
    followed_signal: bool = False  # 是否跟随了系统建议
    
    # 归因分析
    attribution: str = ""       # 盈亏归因
    lesson_learned: str = ""    # 经验教训
    emotion_state: str = ""     # 交易时的情绪状态
    
    # 标签
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "code": self.code,
            "name": self.name,
            "industry": self.industry,
            "direction": self.direction,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_reason": self.entry_reason,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "status": self.status,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "holding_days": self.holding_days,
            "system_signal": self.system_signal,
            "signal_price": self.signal_price,
            "signal_score": self.signal_score,
            "followed_signal": self.followed_signal,
            "attribution": self.attribution,
            "lesson_learned": self.lesson_learned,
            "emotion_state": self.emotion_state,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        return cls(**d)


# ── 交易日志管理器 ──────────────────────────────────────────

class TradeJournal:
    """
    交易日志管理器
    
    用法:
        journal = TradeJournal()
        
        # 记录一笔买入
        journal.record_entry(
            code="600519", entry_price=1850.00, quantity=100,
            entry_reason="MACD金叉+站上MA60", system_signal="买入",
            signal_score=72.5
        )
        
        # 记录卖出
        journal.record_exit(
            trade_id="...", exit_price=1980.00,
            exit_reason="止盈", lesson="趋势保持，但提前止盈"
        )
        
        # 生成月度报告
        report = journal.monthly_report(2026, 6)
    """
    
    def __init__(self):
        self._ensure_files()
    
    def _ensure_files(self):
        if not TRADES_FILE.exists():
            TRADES_FILE.touch()
        if not JOURNAL_CONFIG.exists():
            with open(JOURNAL_CONFIG, "w") as f:
                json.dump({
                    "created_at": datetime.now().isoformat(),
                    "total_trades": 0,
                    "version": "1.0",
                }, f, ensure_ascii=False, indent=2)
    
    # ═════════════════════════════════════════════════════
    #  记录交易
    # ═════════════════════════════════════════════════════
    
    def record_entry(self, code: str, entry_price: float, quantity: int,
                     entry_reason: str = "",
                     system_signal: str = "",
                     signal_price: float = 0.0,
                     signal_score: float = 0.0,
                     emotion_state: str = "",
                     tags: List[str] = None) -> str:
        """记录一笔买入"""
        name = get_stock_name(code)
        industry = get_stock_industry(code)
        trade_id = f"{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = TradeRecord(
            trade_id=trade_id,
            code=code,
            name=name,
            industry=industry,
            direction=TradeDirection.BUY,
            entry_date=datetime.now().isoformat(),
            entry_price=entry_price,
            quantity=quantity,
            entry_reason=entry_reason,
            status=TradeStatus.OPEN,
            system_signal=system_signal,
            signal_price=signal_price,
            signal_score=signal_score,
            emotion_state=emotion_state,
            tags=tags or [],
        )
        
        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        
        # 更新计数
        config = json.loads(TRADES_FILE.read_text() if TRADES_FILE.stat().st_size > 0 else "{}")
        
        print(f"\n  📝 交易记录已保存: {name}({code})")
        print(f"     方向: 买入 | 价格: ¥{entry_price:.2f} | 数量: {quantity}")
        print(f"     理由: {entry_reason or '未记录'}")
        print(f"     系统信号: {system_signal or '未参考'}")
        print(f"     ID: {trade_id}")
        
        return trade_id
    
    def record_exit(self, trade_id: str, exit_price: float,
                    exit_reason: str = ExitReason.MANUAL,
                    lesson_learned: str = "",
                    emotion_state: str = ""):
        """记录卖出平仓"""
        trades = self._load_all_trades()
        
        for t in trades:
            if t["trade_id"] == trade_id:
                t["exit_date"] = datetime.now().isoformat()
                t["exit_price"] = exit_price
                t["exit_reason"] = exit_reason
                t["status"] = TradeStatus.CLOSED
                
                # 计算盈亏
                entry_price = t["entry_price"]
                quantity = t["quantity"]
                t["pnl"] = (exit_price - entry_price) * quantity
                t["pnl_pct"] = (exit_price / entry_price - 1) * 100
                
                entry_dt = datetime.fromisoformat(t["entry_date"])
                exit_dt = datetime.now()
                t["holding_days"] = (exit_dt - entry_dt).days
                
                if lesson_learned:
                    t["lesson_learned"] = lesson_learned
                if emotion_state:
                    t["emotion_state"] = emotion_state
                
                # write back
                self._rewrite_all_trades(trades)
                
                profit_emoji = "✅" if t["pnl"] > 0 else "❌"
                print(f"\n  {profit_emoji} 平仓记录已保存")
                print(f"     {t['name']}({t['code']})")
                print(f"     入场: ¥{entry_price:.2f} | 出场: ¥{exit_price:.2f}")
                print(f"     盈亏: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")
                print(f"     持有: {t['holding_days']}天")
                print(f"     理由: {exit_reason}")
                if lesson_learned:
                    print(f"     心得: {lesson_learned}")
                return t
        
        print(f"  ⚠️ 未找到交易 ID: {trade_id}")
        return None
    
    # ═════════════════════════════════════════════════════
    #  分析与报告
    # ═════════════════════════════════════════════════════
    
    def monthly_report(self, year: int, month: int) -> dict:
        """生成月度交易报告"""
        trades = self._load_all_trades()
        
        # 筛选月度数据
        filtered = []
        for t in trades:
            exit_date = t.get("exit_date", "")
            if exit_date:
                dt = datetime.fromisoformat(exit_date)
                if dt.year == year and dt.month == month:
                    filtered.append(t)
        
        if not filtered:
            return {
                "year": year,
                "month": month,
                "total_trades": 0,
                "message": "本月无平仓交易",
            }
        
        closed = [t for t in filtered if t["status"] == "已平仓"]
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] < 0]
        
        total_pnl = sum(t["pnl"] for t in closed)
        win_rate = len(wins) / max(len(closed), 1) * 100
        
        # 按策略归因
        by_reason = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for t in closed:
            reason = t.get("exit_reason", "未知")
            by_reason[reason]["count"] += 1
            by_reason[reason]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                by_reason[reason]["wins"] += 1
        
        # 情绪分析
        emotions = defaultdict(int)
        for t in closed:
            emo = t.get("emotion_state", "")
            if emo:
                emotions[emo] += 1
        
        # 最佳/最差交易
        best = max(closed, key=lambda t: t["pnl_pct"]) if closed else None
        worst = min(closed, key=lambda t: t["pnl_pct"]) if closed else None
        
        report = {
            "year": year,
            "month": month,
            "period": f"{year}年{month}月",
            "total_trades": len(closed),
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / max(len(closed), 1), 2),
            "max_win": round(best["pnl_pct"], 2) if best else 0,
            "max_loss": round(worst["pnl_pct"], 2) if worst else 0,
            "pnl_by_exit_reason": dict(by_reason),
            "emotion_distribution": dict(emotions),
            "best_trade": {
                "code": best["code"],
                "name": best["name"],
                "pnl_pct": round(best["pnl_pct"], 2),
                "entry_reason": best.get("entry_reason", ""),
            } if best else None,
            "worst_trade": {
                "code": worst["code"],
                "name": worst["name"],
                "pnl_pct": round(worst["pnl_pct"], 2),
                "exit_reason": worst.get("exit_reason", ""),
            } if worst else None,
            "recommendation": self._generate_recommendation(win_rate, by_reason, emotions),
        }
        
        return report
    
    def _generate_recommendation(self, win_rate: float,
                                 by_reason: dict,
                                 emotions: dict) -> str:
        """基于数据生成改进建议"""
        lines = []
        
        if win_rate > 60:
            lines.append("胜率优秀，继续保持纪律性操作")
        elif win_rate > 50:
            lines.append("胜率适中，建议优化入场时机，减少亏损交易")
        elif win_rate > 40:
            lines.append("胜率偏低，建议检查入场条件是否过于激进")
        else:
            lines.append("胜率较低，建议暂停交易，回顾系统信号，重建信心")
        
        # 止损执行分析
        stop_loss_pnl = by_reason.get("止损", {}).get("pnl", 0)
        take_profit_pnl = by_reason.get("止盈", {}).get("pnl", 0)
        
        if stop_loss_pnl < -1000:
            lines.append(f"止损亏损{stop_loss_pnl:.0f}元，建议收紧止损位或降低仓位")
        if take_profit_pnl > 0:
            lines.append(f"止盈盈利{take_profit_pnl:.0f}元，纪律性止盈值得坚持")
        
        # 情绪分析
        if emotions.get("焦虑", 0) > 2:
            lines.append("交易中焦虑情绪较多，建议降低仓位或缩小交易频率")
        if emotions.get("冲动", 0) > 0:
            lines.append("存在冲动交易痕迹，建议严格执行系统信号，不随意开仓")
        
        return "；".join(lines) if lines else "继续坚持纪律性交易"
    
    def performance_overview(self) -> dict:
        """总体绩效概览"""
        trades = self._load_all_trades()
        closed = [t for t in trades if t["status"] == "已平仓"]
        open_trades = [t for t in trades if t["status"] == "持仓中"]
        
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] < 0]
        
        total_pnl = sum(t["pnl"] for t in closed)
        win_rate = len(wins) / max(len(closed), 1) * 100
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
        
        # 跟随系统 vs 不跟随
        followed = [t for t in closed if t.get("followed_signal")]
        unfollowed = [t for t in closed if not t.get("followed_signal")]
        followed_win_rate = len([t for t in followed if t["pnl"] > 0]) / max(len(followed), 1) * 100
        unfollowed_win_rate = len([t for t in unfollowed if t["pnl"] > 0]) / max(len(unfollowed), 1) * 100
        
        return {
            "total_trades": len(trades),
            "closed_trades": len(closed),
            "open_trades": len(open_trades),
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.01), 2),
            "followed_signal_win_rate": round(followed_win_rate, 1),
            "unfollowed_signal_win_rate": round(unfollowed_win_rate, 1),
        }
    
    def print_performance_overview(self):
        """打印绩效概览"""
        p = self.performance_overview()
        
        print(f"\n{'='*55}")
        print(f"  交易绩效概览")
        print(f"{'='*55}")
        print(f"  总交易: {p['total_trades']} | 已平仓: {p['closed_trades']} | 持仓中: {p['open_trades']}")
        print(f"  胜率: {p['win_rate']:.1f}% | 总盈亏: {p['total_pnl']:+.2f}")
        print(f"  平均盈利: {p['avg_win_pct']:+.2f}% | 平均亏损: {p['avg_loss_pct']:.2f}%")
        print(f"  盈亏比: {p['profit_factor']:.2f}")
        print()
        print(f"  系统信号跟踪分析:")
        print(f"    跟随系统: 胜率 {p['followed_signal_win_rate']:.1f}%")
        print(f"    未跟随系统: 胜率 {p['unfollowed_signal_win_rate']:.1f}%")
        print()
        
        if p['followed_signal_win_rate'] > p['unfollowed_signal_win_rate']:
            print(f"  💡 数据表明: 跟随系统信号的胜率更高，建议信任系统!")
        else:
            print(f"  💡 数据表明: 自主决策效果更好，建议优化系统信号或结合自主判断")
    
    # ═════════════════════════════════════════════════════
    #  数据管理
    # ═════════════════════════════════════════════════════
    
    def _load_all_trades(self) -> List[dict]:
        """加载所有交易记录"""
        trades = []
        if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0:
            return trades
        with open(TRADES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return trades
    
    def _rewrite_all_trades(self, trades: List[dict]):
        """重写所有交易记录"""
        with open(TRADES_FILE, "w") as f:
            for t in trades:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    def get_open_trades(self) -> List[dict]:
        """获取当前持仓"""
        return [t for t in self._load_all_trades() if t["status"] == TradeStatus.OPEN]
    
    def get_closed_trades(self) -> List[dict]:
        """获取已平仓"""
        return [t for t in self._load_all_trades() if t["status"] == TradeStatus.CLOSED]


# ── 便捷函数 ────────────────────────────────────────────────

_journal_instance: Optional[TradeJournal] = None


def get_journal() -> TradeJournal:
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = TradeJournal()
    return _journal_instance


def record_trade(code: str, entry_price: float, quantity: int,
                 entry_reason: str = "",
                 system_signal: str = "",
                 signal_score: float = 0.0) -> str:
    """便捷记录买入"""
    return get_journal().record_entry(
        code=code, entry_price=entry_price, quantity=quantity,
        entry_reason=entry_reason, system_signal=system_signal,
        signal_score=signal_score
    )


def close_trade(trade_id: str, exit_price: float,
                exit_reason: str = "手动平仓",
                lesson: str = "") -> dict:
    """便捷记录卖出"""
    return get_journal().record_exit(
        trade_id=trade_id, exit_price=exit_price,
        exit_reason=exit_reason, lesson_learned=lesson
    )


def monthly_summary(year: int, month: int) -> dict:
    """月度汇总"""
    return get_journal().monthly_report(year, month)
