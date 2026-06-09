#!/usr/bin/env python3
"""
Hyperion Pro — 投资决策报告生成器
===================================
生成可直接指导操作的量化投资报告

使用方式：
  python hyperion/cli.py                   # 完整市场分析 + 操作建议
  python hyperion/cli.py --stock 600519    # 个股深度分析
  python hyperion/cli.py --top 10          # 最佳投资标的
  python hyperion/cli.py --risk            # 风险预警
  python hyperion/cli.py --decision         # 投资决策报告
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hyperion.data.market import (
    fetch_realtime_quotes, fetch_index_quotes, fetch_history,
    data_quality_check, CORE_STOCKS, INDICES, get_stock_name, get_stock_industry
)
from hyperion.analysis.market_state import MarketStateAnalyzer
from hyperion.analysis.decision_engine import InvestmentDecisionEngine, InvestmentDecision
from hyperion.engine.backtest import BacktestEngine, strategy_report, quick_backtest
from hyperion.strategy.base import list_strategies
from hyperion.analysis.signal_alerts import SignalAlertSystem, AlertLevel
from hyperion.analysis.trade_journal import TradeJournal, monthly_summary

REPORTS_DIR = Path.home() / ".hyperion_data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_report_lines: List[str] = []   # 收集报告内容用于存盘

# ── 报告持久化 ──────────────────────────────────────────────


# ── 格式化工具 ──────────────────────────────────────────────

def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str: return f"\033[31m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[33m{s}\033[0m"
def _cyan(s: str) -> str: return f"\033[36m{s}\033[0m"
def _bold(s: str) -> str: return f"\033[1m{s}\033[0m"
def _dim(s: str) -> str: return f"\033[2m{s}\033[0m"
def _score_color(score: float) -> str:
    if score > 60: return _green
    elif score > 30: return _cyan
    elif score > -30: return lambda x: x
    elif score > -60: return _yellow
    else: return _red


def _bar(value: float, width: int = 20, max_val: float = 100) -> str:
    """Draw a colored progress bar"""
    pct = max(0, min(1, (value + max_val / 2) / max_val))
    filled = int(pct * width)
    if pct > 0.7:
        color = "\033[42m"
    elif pct > 0.5:
        color = "\033[43m"
    else:
        color = "\033[41m"
    return f"{color}{' ' * filled}\033[0m{' ' * (width - filled)}"


# 劫持 print 以同时记录到报告
import builtins as _builtins
_original_print = _builtins.print

def _print_patched(*args, **kwargs):
    """print() wrapper that also captures output to _report_lines"""
    _original_print(*args, **kwargs)
    # Build the line from args
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    line = sep.join(str(a) for a in args) + ('' if end == '\n' else end)
    if line:
        _report_lines.append(line)

# Replace builtins print in this module
import sys as _sys
_mod = _sys.modules[__name__]
setattr(_mod, 'print', _print_patched)

def _save_report_to_disk(filename: str = None):
    """保存本次会话完整报告为 Markdown"""
    if not _report_lines:
        return None
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = REPORTS_DIR / f"hyperion_brief_{ts}.md"
    else:
        filename = REPORTS_DIR / filename
    import re
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    plain_lines = [ansi_re.sub('', line) for line in _report_lines]
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text('\n'.join(plain_lines), encoding='utf-8')
    return str(filename)



# ── 数据质量检查 ────────────────────────────────────────────

def check_data():
    """数据质量检查"""
    print(f"\n{_bold('═══ 数据质量检查 ═══')}\n")
    report = data_quality_check()
    
    items = [
        ("网络连接", report.get("network_available", False)),
        ("Sina API", report.get("sina_api_available", False)),
        ("股票池", f"{report.get('stocks_in_pool', 0)} 只"),
        ("行业覆盖", f"{report.get('industries', 0)} 个"),
        ("缓存大小", f"{report.get('cache_size_mb', 0):.1f} MB"),
    ]
    for label, value in items:
        if isinstance(value, bool):
            status = _green("✓") if value else _red("✗")
            print(f"  {status} {label}: {'可用' if value else '不可用'}")
        else:
            print(f"  {_cyan('•')} {label}: {value}")
    print()


# ── 市场状态报告 ────────────────────────────────────────────

def market_report():
    """市场状态分析"""
    market = MarketStateAnalyzer.generate_outlook()
    
    state = market.get("market_state", "未知")
    state_colors = {"牛市": _green, "反弹": _green, "震荡": _yellow, "回调": _red, "熊市": _red}
    sc = state_colors.get(state, lambda x: x)
    
    print(f"\n{_bold('═══ 市场状态 ═══')}")
    print(f"  当前阶段: {sc(_bold(state))}")
    print(f"  市场情绪: {market.get('emotion', '未知')}")
    print(f"  风险等级: {market.get('risk_level', '中')} (评分: {market.get('risk_score', 5)}/10)")
    print(f"  置信度:   {market.get('confidence', 0)*100:.0f}%")
    print(f"  推荐仓位: {_bold(market.get('recommended_position', '30-50%'))}")
    
    metrics = market.get("key_metrics", {})
    print(f"\n  涨跌比:   {metrics.get('up_ratio', 0):.1f}% | "
          f"涨停: {metrics.get('limit_up', 0)} | "
          f"跌停: {metrics.get('limit_down', 0)}")
    
    hot = market.get("hot_sectors", [])
    cold = market.get("cold_sectors", [])
    if hot:
        print(f"  热门板块: {_green(', '.join(hot[:3]))}")
    if cold:
        print(f"  回避板块: {_red(', '.join(cold[:3]))}")
    
    print(f"\n  {_bold('操作策略:')}")
    print(f"  {market.get('action_advice', '')}")
    print()


# ── 投资决策报告 ────────────────────────────────────────────

def decision_report(top_n: int = 15):
    """生成投资决策报告"""
    print(f"\n{_bold('═══ 投资决策报告 ═══')}\n")
    
    engine = InvestmentDecisionEngine()
    decisions = engine.analyze_portfolio(top_n=top_n)
    
    # Buy signals
    buys = [d for d in decisions if d.composite_score > 30]
    risk = [d for d in decisions if d.composite_score < -20]
    
    if buys:
        print(f"{_green(_bold('买入信号'))} ({len(buys)} 只)")
        print(f"  {'代码':<8} {'名称':<10} {'评分':>6} {'现价':>8} {'目标':>8} {'止损':>8} {'盈亏比':>6} {'仓位':>5} {'期限':>6}")
        print(f"  {'─'*70}")
        for d in buys[:15]:
            color = _score_color(d.composite_score)
            print(f"  {d.code:<8} {d.name:<10} {color(f'{d.composite_score:+5.1f}'):>6} "
                  f"{d.current_price:>8.2f} {d.target_price_base:>8.2f} {d.stop_loss:>8.2f} "
                  f"{d.reward_risk_ratio:>5.1f}x {d.max_position_pct:>4.0f}% {d.holding_period:>6}")
        print()
    
    if risk:
        print(f"{_red(_bold('风险预警'))} ({len(risk)} 只)")
        print(f"  {'代码':<8} {'名称':<10} {'评分':>6} {'现价':>8} {'建议':<10}")
        print(f"  {'─'*45}")
        for d in risk[:8]:
            color = _score_color(d.composite_score)
            print(f"  {d.code:<8} {d.name:<10} {color(f'{d.composite_score:+5.1f}'):>6} "
                  f"{d.current_price:>8.2f} {_red(d.decision):<10}")
        print()
    
    return decisions


# ── 个股深度分析 ────────────────────────────────────────────

def stock_deep_analysis(codes: List[str]):
    """个股深度分析 — 完整投资决策"""
    engine = InvestmentDecisionEngine()
    
    for code in codes:
        d = engine.analyze(code)
        if d is None:
            print(f"\n{_red(f'✗ {code} 数据不足，无法分析')}")
            continue
        
        sc = _score_color(d.composite_score)
        
        print(f"\n{_bold('═' * 55)}")
        print(f"{_bold(f'  {d.name} ({d.code}) | {d.industry}')}")
        print(f"{_bold('═' * 55)}")
        print()
        
        # Decision header
        print(f"  {_bold('投资决策:')} {sc(_bold(d.decision))}  |  "
              f"综合评分: {sc(f'{d.composite_score:+3.0f}')}  |  "
              f"置信度: {d.confidence*100:.0f}%")
        print()
        
        # Price zone
        print(f"  {_bold('价位区间')}")
        print(f"  当前价:    {_bold(f'¥{d.current_price:.2f}')}")
        print(f"  公允价:    ¥{d.fair_value:.2f}")
        print(f"  乐观目标:  {_green(f'¥{d.target_price_optimistic:.2f}')} ({_green(f'+{(d.target_price_optimistic/d.current_price-1)*100:.1f}%')})")
        print(f"  基准目标:  ¥{d.target_price_base:.2f} (+{(d.target_price_base/d.current_price-1)*100:.1f}%)")
        print(f"  悲观目标:  {_red(f'¥{d.target_price_pessimistic:.2f}')} ({_red(f'{(d.target_price_pessimistic/d.current_price-1)*100:.1f}%')})")
        print()
        
        # Risk control
        print(f"  {_bold('风险控制')}")
        print(f"  硬止损:    {_red(f'¥{d.stop_loss:.2f}')} (跌幅 {(1-d.stop_loss/d.current_price)*100:.1f}%)")
        print(f"  移动止损:  {d.trailing_stop_pct:.0f}%")
        print(f"  推荐仓位:  {d.max_position_pct:.0f}%")
        print(f"  单笔风险:  {d.risk_per_trade:.1f}%")
        print(f"  盈亏比:    {d.reward_risk_ratio:.1f}:1")
        print()
        
        # Score breakdown
        print(f"  {_bold('评分明细')}")
        print(f"  趋势: {d.trend_score:4.0f}/30  {_bar(d.trend_score, 12, 30)}")
        print(f"  动量: {d.momentum_score:4.0f}/25  {_bar(d.momentum_score, 12, 25)}")
        print(f"  波动: {d.volatility_score:4.0f}/15  {_bar(d.volatility_score, 12, 15)}")
        print(f"  估值: {d.fundamental_score:4.0f}/20  {_bar(d.fundamental_score, 12, 20)}")
        print(f"  资金: {d.capital_flow_score:4.0f}/15  {_bar(d.capital_flow_score, 12, 15)}")
        print()
        
        # Catalysts & Risks
        if d.key_catalysts:
            print(f"  {_green(_bold('上涨催化剂'))}")
            for c in d.key_catalysts:
                print(f"  {_green('+')} {c}")
            print()
        
        if d.risk_factors:
            print(f"  {_red(_bold('风险因素'))}")
            for r in d.risk_factors:
                print(f"  {_red('!')} {r}")
            print()
        
        # Scenario
        scn = d.scenario_analysis
        if scn:
            print(f"  {_bold('情景分析')}")
            for case, s in scn.items():
                labels = {"optimistic": _green("乐观"), "base": _cyan("基准"), "pessimistic": _red("悲观")}
                label = labels.get(case, "未知")
                print(f"  [{label}] 概率{s.get('probability','?')} | "
                      f"价格{s.get('price','?')} | 收益{s.get('return','?')}")
            print()
        
        # Action plan
        print(f"  {_bold('操作计划')}")
        print(d.action_plan)
        print()
        
        # Micro-structure insights
        print(f"  {_bold('微观结构')}")
        print(f"  策略共识: {d.strategy_consensus}")
        print(f"  策略评分: {d.strategy_score:.0f}/15")
        print(f"  置信度:   {d.confidence*100:.0f}%")
        print(f"  市场环境: {_yellow('防御优先') if d.market_adjustment < 1.0 else _green('利于操作')}")
        print()
        
        # Investment guidance
        if '持有' in d.decision or '观望' in d.decision:
            print(f"  {_yellow(_bold('持仓建议'))}")
            print(f"  当前持仓者: 维持现有仓位，设置移动止损")
        elif '买入' in d.decision:
            print(f"  {_green(_bold('买入建议'))}")
            print(f"  等待回调至入场区间后建仓，严格执行止损纪律")
        elif '卖出' in d.decision or '减仓' in d.decision or '清仓' in d.decision:
            print(f"  {_red(_bold('卖出建议'))}")
            print(f"  按计划减仓，优先出清高成本仓位")
        print()


# ── Top picks ────────────────────────────────────────────────

def top_picks_report(n: int = 10):
    """最佳投资标的"""
    engine = InvestmentDecisionEngine()
    picks = engine.top_picks(n)
    
    print(f"\n{_bold(f'═══ TOP {n} 投资标的 ═══')}\n")
    
    for i, d in enumerate(picks):
        sc = _score_color(d.composite_score)
        print(f"  {_bold(f'#{i+1}')} {d.name:<10} ({d.code}) | "
              f"{sc(d.decision):>6} | 评分: {sc(f'{d.composite_score:+3.0f}')} | "
              f"现价: ¥{d.current_price:.2f} | 目标: ¥{d.target_price_base:.2f} | "
              f"仓位: {d.max_position_pct:.0f}%")
        if d.summary:
            print(f"     {_dim(d.summary[:100])}")
    print()


# ── 风险预警报告 ────────────────────────────────────────────

def risk_report():
    """风险预警"""
    engine = InvestmentDecisionEngine()
    warnings = engine.risk_warnings(10)
    
    print(f"\n{_bold('═══ 风险预警 ═══')}\n")
    
    if not warnings:
        print(f"  {_green('当前无高风险标的')}")
        return
    
    for d in warnings:
        print(f"  {_red('⚠')} {d.name:<10} ({d.code}) | "
              f"评分: {_red(f'{d.composite_score:+3.0f}')} | "
              f"止损: ¥{d.stop_loss:.2f} | "
              f"建议: {_red(d.decision)}")
        if d.risk_factors:
            for rf in d.risk_factors[:2]:
                print(f"    {_dim(f'- {rf}')}")
    print()


# ── 完整日报 ────────────────────────────────────────────────

def full_daily_brief():
    """完整投资日报"""
    print(f"\n{'='*60}")
    print(f"  {_bold('HYPERION PRO — 每日投资简报')}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    check_data()
    market_report()
    top_picks_report(10)
    risk_report()
    
    # Save to file
    print(f"\n{_dim(f'报告保存至: {REPORTS_DIR}')}\n")


# ── 策略回测报告 ────────────────────────────────────────────

def backtest_report(codes: List[str] = None, strategy_name: str = None, days: int = 250):
    """生成策略回测对比报告"""
    engine = BacktestEngine()
    
    if codes is None or len(codes) == 0:
        codes = ["600519", "000858", "300750"]  # default: 茅台 五粮液 宁德
    
    if strategy_name:
        strategies = [strategy_name]
    else:
        strategies = list_strategies()
    
    print(f"\n{_bold('═══ 策略回测报告 ═══')}")
    print(f"  回测区间: {days} 个交易日")
    print(f"  测试标的: {', '.join(codes)}")
    print(f"  测试策略: {', '.join(strategies)}\n")
    
    for code in codes:
        name = get_stock_name(code)
        if not name or name == code:
            name = ""
        
        print(f"\n  {_bold(f'── {name} ({code}) ──')}")
        df = engine.compare_strategies(code, days)
        
        if df.empty:
            print(f"  {_dim('  无数据')}")
            continue
        
        # Header
        print(f"  {'策略':<22} {'评级':<5} {'年化':>8} {'夏普':>7} {'Sort':>7} {'Calmar':>7} "
              f"{'回撤':>8} {'胜率':>7} {'盈亏比':>7} {'交易':>5}")
        print(f"  {'─'*88}")
        
        for _, row in df.iterrows():
            rating = row['rating']
            rc = _green if rating in ('A+','A') else (_cyan if rating == 'B' else _yellow)
            print(f"  {row['strategy']:<22} {rc(rating):<5} "
                  f"{row['annual_return']:>7} "
                  f"{row['sharpe']:>7} {row['sortino']:>7} {row['calmar']:>7} "
                  f"{row['max_drawdown']:>7} {row['win_rate']:>7} "
                  f"{row['profit_factor']:>7} {row['total_trades']:>4}")
        
        # Summary: pick best strategy
        best = df[df['rating'] != 'N/A']
        if not best.empty:
            # Sort by sharpe
            best = best.sort_values('sharpe', ascending=False)
            best_row = best.iloc[0]
            print(f"\n  {_green(_bold('→ 推荐策略: ' + best_row['strategy']))}")
            print(f"  {_dim(best_row['summary'])}")
    
    # Batch multi-stock backtest
    if len(codes) > 1:
        print(f"\n\n  {_bold('── 多标的综合回测 ──')}")
        all_results = engine.run_multi(codes, "MultiFactorAlphaStrategy", days, top_n=10)
        if all_results:
            print(f"  {'排名':<5} {'标的':<12} {'代码':<8} {'评级':<5} {'年化':>8} {'夏普':>7} {'最大回撤':>10}")
            print(f"  {'─'*60}")
            for i, r in enumerate(all_results):
                rc = _green if r.rating in ('A+','A') else _cyan
                print(f"  {i+1:<5} {r.stock_name:<12} {r.stock_code:<8} {rc(r.rating):<5} "
                      f"{r.annual_return_pct:>7.1f}% {r.sharpe_ratio:>7.2f} {r.max_drawdown_pct:>9.1f}%")
    
    print()


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hyperion Pro — 量化投资决策系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python hyperion/cli.py                   完整投资简报
  python hyperion/cli.py --stock 600519     个股深度分析
  python hyperion/cli.py --stock 600519 000858 300750  多只分析
  python hyperion/cli.py --top 20            最佳投资标的
  python hyperion/cli.py --risk               风险预警
  python hyperion/cli.py --market             仅看市场状态
  python hyperion/cli.py --data               仅数据检查
        """
    )
    
    parser.add_argument("--stock", type=str, nargs="*", help="个股深度分析")
    parser.add_argument("--top", type=int, nargs="?", const=10, help="最佳投资标的")
    parser.add_argument("--risk", action="store_true", help="风险预警")
    parser.add_argument("--market", action="store_true", help="仅市场状态")
    parser.add_argument("--data", action="store_true", help="仅数据检查")
    parser.add_argument("--backtest", type=str, nargs="*", help="策略回测对比（可指定股票代码）")
    parser.add_argument("--bt-strategy", type=str, default=None, help="指定回测策略")
    parser.add_argument("--bt-days", type=int, default=250, help="回测天数")
    parser.add_argument("--alerts", action="store_true", help="扫描实时预警信号")
    parser.add_argument("--alert-level", type=str, default="WARNING", choices=["INFO","WARNING","CRITICAL"], help="预警最低级别")
    parser.add_argument("--journal", action="store_true", help="交易日志概览")
    parser.add_argument("--journal-entry", type=str, nargs="*", help="记录买入: --journal-entry 600519 1850 100 '理由'")
    parser.add_argument("--journal-exit", type=str, nargs="*", help="记录卖出: --journal-exit trade_id 1980 '理由'")
    parser.add_argument("--monthly", type=str, help="月度报告: --monthly 2026-06")
    
    args = parser.parse_args()
    
    has_cmd = any([args.stock, args.top is not None, args.risk, args.market, args.data, args.backtest is not None,
                     args.alerts, args.journal, args.journal_entry is not None, args.journal_exit is not None, args.monthly is not None])
    
    try:
        if args.data:
            check_data()
        
        if args.market:
            market_report()
        
        if args.stock:
            stock_deep_analysis(args.stock)
        
        if args.top is not None:
            top_picks_report(args.top)
        
        if args.risk:
            risk_report()
        
        if args.backtest is not None:
            backtest_report(args.backtest, args.bt_strategy, args.bt_days)
        
        if args.alerts:
            print(f"\n  {_bold('═══ 实时预警扫描 ═══')}\n")
            alerts_sys = SignalAlertSystem()
            alerts = alerts_sys.scan_all()
            alerts_sys.print_alerts(alerts)
            print(f"\n  共 {len(alerts)} 条预警")
        
        if args.journal:
            journal = TradeJournal()
            journal.print_performance_overview()
        
        if args.journal_entry is not None and len(args.journal_entry) >= 3:
            code = args.journal_entry[0]
            price = float(args.journal_entry[1])
            qty = int(args.journal_entry[2])
            reason = ' '.join(args.journal_entry[3:]) if len(args.journal_entry) > 3 else ""
            journal = TradeJournal()
            journal.record_entry(code=code, entry_price=price, quantity=qty, entry_reason=reason)
        
        if args.journal_exit is not None and len(args.journal_exit) >= 2:
            trade_id = args.journal_exit[0]
            exit_price = float(args.journal_exit[1])
            reason = ' '.join(args.journal_exit[2:]) if len(args.journal_exit) > 2 else "手动平仓"
            journal = TradeJournal()
            journal.record_exit(trade_id=trade_id, exit_price=exit_price, exit_reason=reason)
        
        if args.monthly is not None:
            parts = args.monthly.split('-')
            if len(parts) == 2:
                year, month = int(parts[0]), int(parts[1])
                report = monthly_summary(year, month)
                print(f"\n  {_bold(f'═══ {year}年{month}月交易报告 ═══')}\n")
                if report.get('total_trades', 0) == 0:
                    print(f"  📭 {report.get('message', '本月无交易')}")
                else:
                    print(f"  总交易: {report['total_trades']} | 盈利: {report['win_trades']} | 亏损: {report['loss_trades']}")
                    print(f"  胜率: {report['win_rate']:.1f}% | 总盈亏: ¥{report['total_pnl']:+,.2f}")
                    print(f"  最大盈利: +{report.get('max_win',0):.2f}% | 最大亏损: {report.get('max_loss',0):.2f}%")
                    if report.get('recommendation'):
                        print(f"\n  💡 {report['recommendation']}")
        
        if not has_cmd:
            full_daily_brief()
        
        # Auto-save report
        saved = _save_report_to_disk()
        if saved:
            print(f"\n{_dim(f'📄 报告已保存: {saved}')}")
    
    except KeyboardInterrupt:
        print(f"\n{_dim('程序终止')}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{_red(f'错误: {e}')}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
