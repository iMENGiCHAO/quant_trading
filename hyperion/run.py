#!/usr/bin/env python3
"""
Hyperion Pro — 主入口
======================
一键运行：分析市场 → 生成报告 → 输出操作建议

使用方式：
  python hyperion/run.py                  # 完整分析
  python hyperion/run.py --brief          # 仅日报
  python hyperion/run.py --stock 600519   # 个股分析
  python hyperion/run.py --scan           # 全市场扫描
  python hyperion/run.py --portfolio      # 组合分析
  python hyperion/run.py --server         # 启动Web仪表盘
  python hyperion/run.py --data-check     # 数据质量检查
"""
from __future__ import annotations

import sys
import os
import argparse
from datetime import datetime

# 确保项目根在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hyperion.data.market import (
    fetch_realtime_quotes, fetch_index_quotes, fetch_history,
    data_quality_check, scan_market, sector_quotes,
    CORE_STOCKS, INDICES
)
from hyperion.analysis.market_state import MarketStateAnalyzer
from hyperion.analysis.signals import SignalGenerator
from hyperion.reporting.report_generator import ReportGenerator


def cmd_daily_brief():
    """生成每日投资简报"""
    print("=" * 60)
    print("  HYPERION PRO — 生成每日投资简报")
    print("=" * 60)
    print()
    
    print("[1/3] 采集市场数据...")
    report = ReportGenerator.daily_brief()
    
    print("[2/3] 保存报告...")
    report_path = os.path.expanduser(f"~/.hyperion_data/reports/daily_brief_{datetime.now().strftime('%Y%m%d')}.md")
    
    print("[3/3] 完成!")
    print()
    print("=" * 60)
    
    # 打印摘要
    lines = report.split("\n")
    for line in lines[:50]:
        print(line)
    
    print()
    print(f"完整报告已保存: {report_path}")
    print(f"运行 python hyperion/run.py --server 启动Web仪表盘")


def cmd_stock_analysis(codes):
    """个股深度分析"""
    if not codes:
        # 默认分析前5只
        codes = [c for c, _, _ in CORE_STOCKS[:5]]
    
    for code in codes:
        print()
        print("=" * 60)
        report = ReportGenerator.stock_report(code)
        print(report[:2000])
        print(f"... (完整报告已保存)")
        print("=" * 60)


def cmd_full_scan():
    """全市场扫描"""
    print("=" * 60)
    print("  HYPERION PRO — 全市场扫描")
    print("=" * 60)
    print()
    
    print("[1/2] 扫描全市场股票...")
    sig_gen = SignalGenerator()
    
    # 买入信号
    top_buy = sig_gen.top_buy_signals(20)
    print(f"\n📈 **最看好买入 Top 20**:")
    print(f"{'代码':<8} {'名称':<10} {'行业':<12} {'信号':<12} {'评分':<8} {'现价':<8} {'目标价':<8} {'上涨空间':<10}")
    print("-" * 76)
    for sig in top_buy:
        print(f"{sig.code:<8} {sig.name:<10} {sig.industry:<12} {sig.signal:<12} {sig.score:<8.1f} {sig.current_price:<8.2f} {sig.target_price:<8.2f} {sig.upside_potential:<+8.1f}%")
    
    # 卖出信号
    top_sell = sig_gen.top_sell_signals(10)
    if top_sell:
        print(f"\n📉 **建议卖出 Top 10**:")
        print(f"{'代码':<8} {'名称':<10} {'行业':<12} {'信号':<12} {'评分':<8} {'现价':<8}")
        print("-" * 60)
        for sig in top_sell:
            print(f"{sig.code:<8} {sig.name:<10} {sig.industry:<12} {sig.signal:<12} {sig.score:<8.1f} {sig.current_price:<8.2f}")
    
    print()
    print("[2/2] 行业配置建议...")
    recommendations = sig_gen.sector_recommendations()
    print(f"\n📊 **行业配置建议**:")
    print(f"{'行业':<14} {'评分':<8} {'建议':<10} {'推荐标的':<30}")
    print("-" * 62)
    for industry, rec in sorted(recommendations.items(), key=lambda x: x[1]["avg_score"], reverse=True)[:15]:
        top_str = "、".join([s["name"] for s in rec["top_stocks"]])
        print(f"{industry:<14} {rec['avg_score']:<+8.1f} {rec['recommendation']:<10} {top_str:<30}")
    
    print()
    print("=" * 60)
    print("分析完成！如需详细报告: python hyperion/run.py --brief")
    print("=" * 60)


def cmd_portfolio_analysis():
    """组合分析 (用户需自己指定持仓)"""
    print("=" * 60)
    print("  HYPERION PRO — 组合分析")
    print("=" * 60)
    print()
    print("请按格式输入持仓 (代码:仓位比例)，每行一个:")
    print("例如: 600519:0.2")
    print("      000858:0.15")
    print("      300750:0.1")
    print("(输入空行结束)")
    print()
    
    portfolio = {}
    while True:
        try:
            line = input("> ").strip()
            if not line:
                break
            parts = line.split(":")
            code = parts[0].strip()
            weight = float(parts[1].strip()) if len(parts) > 1 else 0.1
            portfolio[code] = weight
        except (EOFError, KeyboardInterrupt):
            break
    
    if portfolio:
        # 归一化
        total = sum(portfolio.values())
        portfolio = {k: v/total for k, v in portfolio.items()}
        
        report = ReportGenerator.portfolio_report(portfolio)
        print(report[:1500])
        print(f"... (完整报告已保存)")
    else:
        print("未输入持仓，跳过组合分析")


def cmd_data_check():
    """数据质量检查"""
    print("=" * 60)
    print("  HYPERION PRO — 数据质量检查")
    print("=" * 60)
    print()
    
    report = data_quality_check()
    for key, val in report.items():
        label = {
            "timestamp": "检测时间",
            "network_available": "网络可用",
            "akshare_available": "akShare可用",
            "cache_size_mb": "缓存大小(MB)",
            "stocks_in_pool": "核心股票池",
            "industries": "行业覆盖",
        }.get(key, key)
        
        if isinstance(val, bool):
            val_str = "✅ 可用" if val else "❌ 不可用"
        elif isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        
        print(f"  {label}: {val_str}")
    
    print()
    
    # 测试数据获取
    print("测试实时行情...")
    quotes = fetch_realtime_quotes()
    if not quotes.empty:
        print(f"  ✅ 获取 {len(quotes)} 只股票行情成功")
    else:
        print(f"  ❌ 行情获取失败")
    
    print("测试指数行情...")
    indices = fetch_index_quotes()
    if not indices.empty:
        print(f"  ✅ 获取 {len(indices)} 个指数行情成功")
    else:
        print(f"  ❌ 指数获取失败")
    
    print("测试历史数据...")
    hist = fetch_history("600519", days=60)
    if not hist.empty:
        print(f"  ✅ 获取 {len(hist)} 天数据成功")
    else:
        print(f"  ❌ 历史数据获取失败")


def cmd_server():
    """启动 Web 仪表盘"""
    print("=" * 60)
    print("  HYPERION PRO — 启动 Web 仪表盘")
    print("=" * 60)
    print()
    print("请确保已安装: pip install dash plotly")
    print("正在启动...")
    print()
    
    try:
        from hyperion.dashboard.app import run_server
        run_server()
    except ImportError as e:
        print(f"❌ 启动失败: {e}")
        print()
        print("请先安装依赖:")
        print("  pip install dash plotly")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperion Pro — 顶级量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python hyperion/run.py                  完整分析 (日报+扫描+组合)
  python hyperion/run.py --brief          仅生成日报
  python hyperion/run.py --stock 600519   个股分析
  python hyperion/run.py --scan           全市场扫描
  python hyperion/run.py --server         启动Web仪表盘
  python hyperion/run.py --data-check     数据质量检查
        """
    )
    
    parser.add_argument("--brief", action="store_true", help="生成每日投资简报")
    parser.add_argument("--stock", type=str, nargs="*", help="个股分析 (指定代码)")
    parser.add_argument("--scan", action="store_true", help="全市场扫描")
    parser.add_argument("--portfolio", action="store_true", help="组合分析")
    parser.add_argument("--server", action="store_true", help="启动Web仪表盘")
    parser.add_argument("--data-check", action="store_true", help="数据质量检查")
    parser.add_argument("--all", action="store_true", help="完整分析")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，执行完整分析
    has_cmd = any([args.brief, args.stock is not None, args.scan, args.portfolio, args.server, args.data_check, args.all])
    
    try:
        if args.data_check or (not has_cmd):
            cmd_data_check()
            print()
        
        if args.all or (not has_cmd):
            cmd_daily_brief()
            print()
            cmd_full_scan()
        
        if args.brief:
            cmd_daily_brief()
        
        if args.scan:
            cmd_full_scan()
        
        if args.stock is not None:
            cmd_stock_analysis(args.stock)
        
        if args.portfolio:
            cmd_portfolio_analysis()
        
        if args.server:
            cmd_server()
    
    except KeyboardInterrupt:
        print("\n\n程序已终止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
