"""
Strategy Layer — 策略定义与管理
=====================================
包含:
1. TopkDropoutStrategy — Qlib-style 选股策略
2. FactorRankStrategy — 因子多空分层
3. StrategyEvaluator — 策略评估器

已在 workflow/engine.py 中实现了核心逻辑，
此处提供独立的便捷接口。
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from hyperion.workflow.engine import (
    TopkDropoutStrategy as _TopkDropoutStrategy,
    PortAnalysisRecord,
    SignalRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """策略执行结果"""
    weights: pd.DataFrame  # MultiIndex [datetime, instrument], weight
    signals: pd.DataFrame
    report: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def summary(self) -> str:
        lines = [
            "=" * 50,
            f"Strategy Result [{self.timestamp}]",
            "=" * 50,
            f"  Positions: {len(self.weights)} ",
            f"  Signals:   {len(self.signals)} ",
            f"  Date range: ",
        ]
        if isinstance(self.weights.index, pd.MultiIndex):
            dates = self.weights.index.get_level_values(0)
            lines[-1] += f"{dates.min()} to {dates.max()}"
        else:
            lines[-1] += "N/A"
        if self.report:
            for k, v in self.report.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4%}")
                else:
                    lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class TopkDropoutStrategy:
    """TopK Dropout 选股策略

    每期选择得分最高的 TopK 个标的，支持 Dropout 排除。
    """

    def __init__(
        self,
        top_k: int = 50,
        dropout_days: int = 0,
        holding_period: int = 1,
        portfolio_size: Optional[int] = None,
    ):
        self.top_k = top_k
        self.dropout_days = dropout_days
        self.holding_period = holding_period
        self.portfolio_size = portfolio_size or top_k
        self._engine = _TopkDropoutStrategy(
            top_k=top_k,
            dropout_days=dropout_days,
            holding_period=holding_period,
        )

    def run(
        self,
        signals: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ) -> StrategyResult:
        """执行策略"""
        if isinstance(signals, pd.Series):
            signals_df = signals.to_frame("score")
        else:
            signals_df = signals

        weights = self._engine.run(signals_df, prices)

        return StrategyResult(
            weights=weights,
            signals=signals_df,
        )


@dataclass
class FactorRankStrategy:
    """因子多空分层策略

    将预测信号分为 N 层，计算各层收益差异。
    """

    n_groups: int = 10
    top_k: int = 50

    def analyze(
        self,
        signals: pd.Series,
        forward_returns: pd.Series,
    ) -> Dict[str, Any]:
        """执行分层分析"""
        pa = PortAnalysisRecord(
            signals=signals,
            forward_returns=forward_returns,
            n_groups=self.n_groups,
            top_k=self.top_k,
        )
        return pa.run()


class StrategyEvaluator:
    """策略评估器 — 对比多个策略的表现"""

    def __init__(self):
        self.results: List[StrategyResult] = []

    def evaluate(
        self,
        strategy: Any,
        signals: pd.Series,
        forward_returns: pd.Series,
        name: str = "",
    ) -> StrategyResult:
        """评估一个策略"""
        result = strategy.run(signals)

        # Portfolio 分析
        pa = PortAnalysisRecord(
            signals=signals,
            forward_returns=forward_returns,
        )
        result.report = pa.run()
        result.report["strategy_name"] = name or strategy.__class__.__name__

        self.results.append(result)
        return result

    def compare(self) -> pd.DataFrame:
        """对比所有策略结果"""
        rows = []
        for r in self.results:
            rows.append({
                "strategy": r.report.get("strategy_name", ""),
                "positions": len(r.weights),
                **{k: v for k, v in r.report.items() if isinstance(v, (int, float))},
            })
        return pd.DataFrame(rows)