"""
报告生成器
===========
HTML/PDF/JSON 格式的回测报告。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from hyperion.analysis.metrics import PerformanceMetrics
from hyperion.engine.backtest import BacktestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """回测报告生成器"""
    
    @staticmethod
    def generate(result: BacktestResult,
                 benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """生成完整报告
        
        Returns:
            Report dict
        """
        # 计算额外指标
        extra_metrics = {}
        if result.daily_returns is not None:
            extra_metrics = PerformanceMetrics.calculate(
                result.daily_returns, benchmark_returns
            )
        
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "framework": "Hyperion Quant v1.0",
                "start_date": result.start_date,
                "end_date": result.end_date,
                "trading_days": len(result.daily_returns) if result.daily_returns is not None else 0
            },
            "summary": {
                "initial_capital": result.initial_capital,
                "final_value": round(result.final_value, 2),
                "total_return": f"{result.total_return:.2%}",
                "annual_return": f"{result.annual_return:.2%}",
            },
            "performance": {
                "sharpe_ratio": round(result.sharpe_ratio, 4),
                "sortino_ratio": round(result.sortino_ratio, 4),
                "calmar_ratio": round(result.calmar_ratio, 4),
                "volatility": f"{result.volatility:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "max_dd_duration": f"{result.max_drawdown_duration} days",
                "var_95": f"{result.var_95:.2%}"
            },
            "trading": {
                "total_trades": result.total_trades,
                "win_rate": f"{result.win_rate:.2%}",
                "profit_factor": round(result.profit_factor, 2),
                "avg_trade_return": f"{result.avg_trade_return:.2%}"
            },
            "benchmark": {
                "benchmark_return": f"{result.benchmark_return:.2%}",
                "alpha": f"{result.alpha:.2%}",
                "beta": round(result.beta, 4),
                "information_ratio": round(result.information_ratio, 4)
            },
            "extra_metrics": extra_metrics
        }
        
        return report
    
    @staticmethod
    def to_json(result: BacktestResult, path: str,
                benchmark_returns: Optional[pd.Series] = None) -> str:
        """导出JSON报告"""
        report = ReportGenerator.generate(result, benchmark_returns)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
    
    @staticmethod
    def to_text(result: BacktestResult) -> str:
        """文本格式报告"""
        report = ReportGenerator.generate(result)
        lines = [
            "=" * 50,
            "  HYPERION QUANT - BACKTEST REPORT",
            "=" * 50,
            "",
            f"Period:       {result.start_date} → {result.end_date}",
            f"Initial:      ¥{result.initial_capital:,.0f}",
            f"Final:        ¥{result.final_value:,.0f}",
            f"Return:       {result.total_return:.2%}",
            f"Annual:       {result.annual_return:.2%}",
            f"Sharpe:       {result.sharpe_ratio:.2f}",
            f"Max DD:       {result.max_drawdown:.2%}",
            f"Volatility:   {result.volatility:.2%}",
            f"Trades:       {result.total_trades}",
            f"Win Rate:     {result.win_rate:.2%}",
            f"Alpha:        {result.alpha:.2%}",
            f"Beta:         {result.beta:.2f}",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)
