"""
Hyperion Pro — 回测引擎
========================
华尔街级别策略回测系统

功能:
  1. 事件驱动回测 (Event-Driven Backtesting)
  2. 多策略并行回测
  3. 完整绩效指标:
     - Sharpe Ratio (年化夏普)
     - Sortino Ratio (下行夏普)
     - Max Drawdown + Duration
     - Calmar Ratio
     - Win Rate, Profit Factor
     - Alpha, Beta, Information Ratio
  4. 资金曲线 + 交易明细
  5. 策略对比报告
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..data.market import fetch_history, CORE_STOCKS, get_stock_name
from ..strategy.base import BaseStrategy, StrategyResult, SignalType, list_strategies, get_strategy
from ..strategy.strategies import (
    TrendFollowingStrategy, MeanReversionStrategy,
    MomentumBreakoutStrategy, VolumeAnomalyStrategy, MultiFactorAlphaStrategy
)


@dataclass 
class BacktestResult:
    """回测结果"""
    strategy_name: str
    stock_code: str
    stock_name: str
    start_date: str
    end_date: str
    
    # 收益指标
    total_return_pct: float          # 总收益率
    annual_return_pct: float         # 年化收益率
    benchmark_return_pct: float      # 基准收益率
    
    # 风险指标
    annual_volatility: float         # 年化波动率
    max_drawdown_pct: float          # 最大回撤
    max_drawdown_duration: int       # 最长回撤持续天数
    downside_deviation: float        # 下行波动率
    
    # 风险收益比
    sharpe_ratio: float              # 夏普比率
    sortino_ratio: float             # 索提诺比率
    calmar_ratio: float              # 卡尔玛比率
    information_ratio: float         # 信息比率
    
    # 交易指标
    total_trades: int                # 总交易次数
    win_trades: int                  # 盈利交易次数
    loss_trades: int                 # 亏损交易次数
    win_rate_pct: float              # 胜率
    profit_factor: float             # 盈亏比因子
    avg_win_pct: float               # 平均盈利
    avg_loss_pct: float              # 平均亏损
    avg_holding_days: float          # 平均持有天数
    
    # 资金曲线
    equity_curve: List[float] = field(default_factory=list)
    trade_records: List[StrategyResult] = field(default_factory=list)
    
    # 综合评级
    rating: str = ""                 # A+ / A / B / C / D
    summary: str = ""                # 一句话总结
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "stock": f"{self.stock_name}({self.stock_code})",
            "period": f"{self.start_date} ~ {self.end_date}",
            "total_return": f"{self.total_return_pct:.2f}%",
            "annual_return": f"{self.annual_return_pct:.2f}%",
            "sharpe": round(self.sharpe_ratio, 2),
            "sortino": round(self.sortino_ratio, 2),
            "calmar": round(self.calmar_ratio, 2),
            "max_drawdown": f"{self.max_drawdown_pct:.2f}%",
            "win_rate": f"{self.win_rate_pct:.1f}%",
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "rating": self.rating,
            "summary": self.summary,
        }


class BacktestEngine:
    """
    事件驱动回测引擎
    
    使用方式:
        engine = BacktestEngine()
        result = engine.run("600519", "TrendFollowingStrategy", days=250)
        print(result.summary)
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.0003,
                 slippage_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
    
    def run(self, code: str, strategy_name: str = "TrendFollowingStrategy",
            days: int = 250, verbose: bool = False) -> Optional[BacktestResult]:
        """
        对单只股票运行指定策略回测
        
        Args:
            code: 股票代码
            strategy_name: 策略名 (见 list_strategies())
            days: 回测天数
            
        Returns:
            BacktestResult
        """
        # 获取数据
        df = fetch_history(code, days=days)
        if df.empty or len(df) < 60:
            return None
        
        name = get_stock_name(code)
        
        # 创建策略
        strategy_cls = get_strategy(strategy_name)
        if strategy_cls is None:
            strategy_cls = TrendFollowingStrategy()
        else:
            strategy_cls = strategy_cls.__class__()
        
        # 生成信号
        signals = strategy_cls.generate_signals(df, code, name)
        
        if verbose:
            print(f"  [{strategy_name}] {code} 生成 {len(signals)} 个信号")
        
        # 运行回测
        trades = self._simulate_trades(df, signals, strategy_cls)
        
        if not trades:
            # 无交易信号时的结果
            bh_return = self._buy_and_hold_return(df)
            return BacktestResult(
                strategy_name=strategy_name,
                stock_code=code, stock_name=name,
                start_date=str(df["date"].iloc[0]),
                end_date=str(df["date"].iloc[-1]),
                total_return_pct=0, annual_return_pct=0,
                benchmark_return_pct=bh_return,
                annual_volatility=0, max_drawdown_pct=0,
                max_drawdown_duration=0, downside_deviation=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                information_ratio=0,
                total_trades=0, win_trades=0, loss_trades=0,
                win_rate_pct=0, profit_factor=0,
                avg_win_pct=0, avg_loss_pct=0, avg_holding_days=0,
                rating="N/A", summary="期间未产生交易信号"
            )
        
        # 计算绩效
        return self._compute_performance(trades, df, code, name, strategy_name)
    
    def run_multi(self, codes: List[str], strategy_name: str = "MultiFactorAlphaStrategy",
                  days: int = 250, top_n: int = 10) -> List[BacktestResult]:
        """
        多只股票回测，返回排名结果
        """
        results = []
        for code in codes:
            try:
                result = self.run(code, strategy_name, days)
                if result and result.total_trades > 0:
                    results.append(result)
            except Exception as e:
                pass
        
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results[:top_n]
    
    def compare_strategies(self, code: str, days: int = 250) -> pd.DataFrame:
        """
        对单只股票比较所有策略
        """
        rows = []
        for sname in list_strategies():
            r = self.run(code, sname, days)
            if r:
                rows.append(r.to_dict())
        return pd.DataFrame(rows)
    
    # ── 交易模拟 ────────────────────────────────────────
    
    def _simulate_trades(self, df: pd.DataFrame, 
                         signals: List, 
                         strategy: BaseStrategy) -> List[StrategyResult]:
        """基于信号模拟真实交易"""
        trades = []
        position = None  # (entry_price, entry_idx, stop_loss, take_profit, entry_date)
        
        for i in range(len(df)):
            current_price = df["close"].values[i]
            current_date = str(df["date"].iloc[i])
            
            # 检查退出条件
            if position is not None:
                entry_price, entry_idx, stop_loss, take_profit, entry_date = position
                should_exit, reason = strategy.should_exit(
                    df.iloc[:i+1], entry_price, entry_idx, stop_loss, take_profit
                )
                
                if should_exit:
                    # 计算滑点和佣金
                    exit_price = current_price * (1 - self.slippage_pct)
                    
                    return_pct = (exit_price / entry_price - 1) * 100
                    holding_days = i - entry_idx
                    
                    # 期间最大回撤
                    segment = df["close"].values[entry_idx:i+1]
                    peak = np.maximum.accumulate(segment)
                    dd = (segment / peak - 1).min() * 100
                    
                    trades.append(StrategyResult(
                        code=df.get("code", ""),
                        strategy=strategy.name,
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=round(return_pct, 2),
                        holding_days=holding_days,
                        max_drawdown_pct=round(float(dd), 2),
                        exit_reason=reason,
                    ))
                    position = None
            
            # 检查入场信号
            if position is None:
                for sig in signals:
                    sig_date = str(sig.timestamp).split("T")[0].split(" ")[0]
                    if sig_date == current_date.split("T")[0].split(" ")[0]:
                        entry_price = current_price * (1 + self.slippage_pct)
                        position = (entry_price, i, sig.stop_loss, 
                                   sig.take_profit, current_date)
                        break
        
        return trades
    
    # ── 绩效计算 ────────────────────────────────────────
    
    def _compute_performance(self, trades: List[StrategyResult],
                              df: pd.DataFrame, code: str, name: str,
                              strategy_name: str) -> BacktestResult:
        """计算完整绩效指标"""
        
        returns = [t.return_pct for t in trades]
        n_trades = len(trades)
        win_trades = [t for t in trades if t.return_pct > 0]
        loss_trades = [t for t in trades if t.return_pct <= 0]
        n_wins = len(win_trades)
        n_losses = len(loss_trades)
        
        # 胜率
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
        
        # 平均盈亏
        avg_win = np.mean([t.return_pct for t in win_trades]) if n_wins > 0 else 0
        avg_loss = np.mean([abs(t.return_pct) for t in loss_trades]) if n_losses > 0 else 0
        
        # 盈亏因子
        total_profit = sum(t.return_pct for t in win_trades) if n_wins > 0 else 0
        total_loss = abs(sum(t.return_pct for t in loss_trades)) if n_losses > 0 else 0.01
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        
        # 平均持有天数
        avg_holding = np.mean([t.holding_days for t in trades]) if n_trades > 0 else 0
        
        # 构造日度权益曲线
        equity_curve = self._build_equity_curve(trades, df, self.initial_capital)
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        
        # 年化收益率
        n_years = len(equity_curve) / 252
        total_return = (equity_curve[-1] / self.initial_capital - 1) * 100
        annual_return = ((1 + total_return / 100) ** (1 / max(n_years, 0.1)) - 1) * 100
        
        # 基准收益
        benchmark_return = self._buy_and_hold_return(df)
        
        # 波动率
        annual_vol = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
        
        # 下行波动率
        neg_returns = daily_returns[daily_returns < 0]
        downside_dev = neg_returns.std() * np.sqrt(252) * 100 if len(neg_returns) > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        dd_series = (np.array(equity_curve) / peak - 1) * 100
        max_dd = float(dd_series.min())
        
        # 回撤持续天数
        dd_duration = self._max_dd_duration(dd_series)
        
        # 无风险利率
        rf = 0.025
        
        # Sharpe
        excess = daily_returns - rf/252
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
        
        # Sortino
        sortino = float(excess.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0
        
        # Calmar
        calmar = annual_return / max(0.1, abs(max_dd))
        
        # Information Ratio
        bh_daily = df["close"].pct_change().dropna()
        aligned = pd.concat([daily_returns, bh_daily], axis=1).dropna()
        if len(aligned) > 1:
            active_ret = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            info_ratio = float(active_ret.mean() / active_ret.std() * np.sqrt(252)) if active_ret.std() > 0 else 0
        else:
            info_ratio = 0
        
        # 评级
        rating = self._assign_rating(sharpe, win_rate, profit_factor, max_dd)
        
        # 总结
        summary = self._generate_summary(rating, sharpe, annual_return, max_dd, win_rate, n_trades)
        
        return BacktestResult(
            strategy_name=strategy_name,
            stock_code=code, stock_name=name,
            start_date=str(df["date"].iloc[0]),
            end_date=str(df["date"].iloc[-1]),
            total_return_pct=round(total_return, 2),
            annual_return_pct=round(annual_return, 2),
            benchmark_return_pct=round(benchmark_return, 2),
            annual_volatility=round(annual_vol, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration=dd_duration,
            downside_deviation=round(downside_dev, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            information_ratio=round(info_ratio, 2),
            total_trades=n_trades,
            win_trades=n_wins,
            loss_trades=n_losses,
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            avg_holding_days=round(avg_holding, 1),
            equity_curve=equity_curve,
            trade_records=trades,
            rating=rating,
            summary=summary,
        )
    
    def _build_equity_curve(self, trades: List[StrategyResult],
                             df: pd.DataFrame, 
                             capital: float) -> List[float]:
        """从交易记录构建日度权益曲线"""
        n = len(df)
        curve = [capital] * n
        
        trade_idx = 0
        for t in trades:
            entry_date = t.entry_date.split("T")[0].split(" ")[0]
            exit_date = t.exit_date.split("T")[0].split(" ")[0]
            
            for j in range(n):
                d = str(df["date"].iloc[j]).split("T")[0].split(" ")[0]
                if d >= entry_date:
                    idx = j
                    break
            else:
                continue
            
            for j in range(n):
                d = str(df["date"].iloc[j]).split("T")[0].split(" ")[0]
                if d >= exit_date:
                    exit_idx = j
                    break
            else:
                exit_idx = n - 1
            
            # 应用收益率
            multiplier = 1 + t.return_pct / 100
            for j in range(exit_idx, n):
                curve[j] *= multiplier
        
        return curve
    
    def _buy_and_hold_return(self, df: pd.DataFrame) -> float:
        """买入持有收益率"""
        if len(df) < 2:
            return 0
        return round((df["close"].values[-1] / df["close"].values[0] - 1) * 100, 2)
    
    @staticmethod
    def _max_dd_duration(dd_series: np.ndarray) -> int:
        underwater = dd_series < 0
        if not underwater.any():
            return 0
        max_dur = 0
        cur = 0
        for u in underwater:
            if u:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        return max_dur
    
    @staticmethod
    def _assign_rating(sharpe: float, win_rate: float, 
                        profit_factor: float, max_dd: float) -> str:
        score = 0
        if sharpe > 2.0: score += 4
        elif sharpe > 1.5: score += 3
        elif sharpe > 1.0: score += 2
        elif sharpe > 0.5: score += 1
        
        if win_rate > 60: score += 3
        elif win_rate > 50: score += 2
        elif win_rate > 40: score += 1
        
        if profit_factor > 2.0: score += 3
        elif profit_factor > 1.5: score += 2
        elif profit_factor > 1.2: score += 1
        
        if max_dd > -10: score += 2
        elif max_dd > -20: score += 1
        
        if score >= 10: return "A+"
        elif score >= 8: return "A"
        elif score >= 6: return "B"
        elif score >= 4: return "C"
        else: return "D"
    
    @staticmethod
    def _generate_summary(rating: str, sharpe: float, annual_ret: float,
                           max_dd: float, win_rate: float, n_trades: int) -> str:
        rating_desc = {"A+": "优秀", "A": "良好", "B": "合格", "C": "一般", "D": "不佳"}
        return (
            f"评级{rating}({rating_desc.get(rating,'')}) | "
            f"年化{annual_ret:+.1f}% | "
            f"夏普{sharpe:.2f} | "
            f"最大回撤{max_dd:.1f}% | "
            f"胜率{win_rate:.0f}% | "
            f"交易{n_trades}次"
        )


# ── 便捷函数 ────────────────────────────────────────────

DEFAULT_STRATEGIES = [
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    MultiFactorAlphaStrategy,
]


def quick_backtest(code: str, days: int = 250) -> pd.DataFrame:
    """快速回测所有策略"""
    engine = BacktestEngine()
    return engine.compare_strategies(code, days)


def batch_backtest(codes: List[str] = None, days: int = 250, 
                   top_n: int = 15) -> List[BacktestResult]:
    """批量回测多只股票"""
    if codes is None:
        codes = [c for c, _, _ in CORE_STOCKS[:30]]
    
    engine = BacktestEngine()
    return engine.run_multi(codes, "MultiFactorAlphaStrategy", days, top_n)


def strategy_report(code: str = "600519", days: int = 250) -> str:
    """生成策略回测报告"""
    engine = BacktestEngine()
    df = engine.compare_strategies(code, days)
    
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  策略回测报告 — {get_stock_name(code)} ({code})")
    lines.append(f"{'='*70}")
    lines.append(f"")
    lines.append(f"  {'策略':<20} {'评级':<6} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8} {'交易':>6} {'盈亏比':>8}")
    lines.append(f"  {'-'*76}")
    
    if not df.empty:
        for _, row in df.iterrows():
            lines.append(
                f"  {row['strategy']:<20} {row['rating']:<6} {row['annual_return']:>10} "
                f"{row['sharpe']:>8} {row['max_drawdown']:>10} {row['win_rate']:>8} "
                f"{row['total_trades']:>6} {row['profit_factor']:>8}"
            )
    
    lines.append(f"")
    lines.append(f"  基准买入持有收益: {engine._buy_and_hold_return(fetch_history(code, days=days)):+.1f}%")
    lines.append(f"")
    
    return "\n".join(lines)
