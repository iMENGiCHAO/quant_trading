#!/usr/bin/env python3
"""
Hyperion Quant CLI
===================
Freqtrade-style command line interface.

Usage:
    hyperion download [--symbols SYMBOLS] [--start DATE] [--end DATE]
    hyperion backtest [--config CONFIG]
    hyperion hyperopt [--config CONFIG] [--evals N]
    hyperion analyze [--result RESULT]
    hyperion live [--config CONFIG]
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hyperion")


def cmd_download(args):
    """下载数据"""
    from hyperion.data.server import DataServer
    from hyperion.data.sources import get_source
    
    symbols = args.symbols.split(",") if args.symbols else []
    if not symbols:
        print("请指定 --symbols (逗号分隔)")
        return
    
    print(f"下载 {len(symbols)} 只股票数据...")
    source = get_source("akshare")
    data = source.download_daily(symbols, args.start, args.end)
    
    if data.empty:
        print("未获取到数据")
        return
    
    server = DataServer()
    n = server.store_batch({s: data.xs(s, level="symbol") for s in symbols})
    print(f"已存储 {n} 条记录到 {server.db_path}")


def cmd_backtest(args):
    """运行回测"""
    from hyperion.config import load_config
    from hyperion.data.server import DataServer
    from hyperion.engine.backtest import BacktestEngine
    from hyperion.strategy.ml_strategy import MLMultiFactorStrategy
    
    config = load_config(args.config)
    server = DataServer()
    
    # 加载数据
    symbols = config.data.symbols or server.symbols[:100]
    if not symbols:
        print("无可用数据. 请先运行: hyperion download")
        return
    
    print(f"加载 {len(symbols)} 只股票数据...")
    data_dict = {}
    for sym in symbols:
        df = server.fetch(sym, config.data.start_date, config.data.end_date)
        if not df.empty:
            data_dict[sym] = df
    
    if not data_dict:
        print("未加载到数据")
        return
    
    # 创建策略
    strategy = MLMultiFactorStrategy(
        symbols=list(data_dict.keys()),
        top_k=config.strategy.max_positions,
        mode="momentum"  # 默认动量模式
    )
    
    # 回测
    engine = BacktestEngine(
        initial_capital=config.engine.initial_capital,
        commission=config.engine.commission,
        stamp_duty=config.engine.stamp_duty,
        slippage=config.engine.slippage,
        t_plus_1=config.engine.t_plus_1
    )
    engine.add_data(data_dict)
    engine.add_strategy(strategy)
    
    result = engine.run(progress=True)
    
    # 输出报告
    from hyperion.analysis.report import ReportGenerator
    print(ReportGenerator.to_text(result))
    
    # 保存
    import json
    report_path = Path("backtest_report.json")
    ReportGenerator.to_json(result, str(report_path))
    print(f"报告已保存: {report_path}")


def cmd_hyperopt(args):
    """超参数优化"""
    from hyperion.config import load_config
    from hyperion.engine.hyperopt import HyperoptEngine
    
    config = load_config(args.config)
    
    def objective(trial):
        import random
        return random.random()  # Placeholder
    
    engine = HyperoptEngine(max_evals=args.evals)
    result = engine.optimize(objective, max_evals=args.evals)
    print(result.summary())


def cmd_analyze(args):
    """分析回测结果"""
    import json
    if not args.result:
        args.result = "backtest_report.json"
    
    try:
        with open(args.result) as f:
            report = json.load(f)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print(f"报告文件不存在: {args.result}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperion Quant - Production-Grade Trading Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # download
    p_down = subparsers.add_parser("download", help="Download market data")
    p_down.add_argument("--symbols", type=str, help="Symbols (comma-separated)")
    p_down.add_argument("--start", type=str, default="2024-01-01")
    p_down.add_argument("--end", type=str, default="2024-12-31")
    
    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--config", type=str, default="config.yaml")
    
    # hyperopt
    p_ho = subparsers.add_parser("hyperopt", help="Run hyperparameter optimization")
    p_ho.add_argument("--config", type=str, default="config.yaml")
    p_ho.add_argument("--evals", type=int, default=100)
    
    # analyze
    p_an = subparsers.add_parser("analyze", help="Analyze backtest results")
    p_an.add_argument("--result", type=str)
    
    args = parser.parse_args()
    
    if args.command == "download":
        cmd_download(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "hyperopt":
        cmd_hyperopt(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
