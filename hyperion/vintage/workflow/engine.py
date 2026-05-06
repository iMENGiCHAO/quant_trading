"""
Workflow Engine — Qlib-style 工作流自动化
============================================
类似 Qlib 的 `qrun` 命令，通过 YAML 配置文件编排完整流程:

1. Data: 数据加载与因子生成 (Alpha360)
2. Process: 数据处理管道 (Normalize, Fillna, etc.)
3. Dataset: 数据集分割 (Train/Valid/Test)
4. Model: 模型选择与训练
5. Backtest: 回测验证
6. Report: 生成分析报告

Usage:
    from hyperion.workflow import WorkflowEngine
    engine = WorkflowEngine('config.yaml')
    engine.run()
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from hyperion.model_zoo.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """预测信号记录 (Qlib SignalRecord)"""
    signals: pd.DataFrame  # MultiIndex [datetime, instrument], columns=[score]
    model_name: str
    timestamp: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.signals.to_csv(path)
        logger.info(f"Signals saved to {path}")

    def ic_analysis(self, forward_returns: pd.Series) -> Dict[str, float]:
        """计算 Rank IC / ICIR"""
        merged = pd.DataFrame({
            "signal": self.signals.iloc[:, 0],
            "forward_ret": forward_returns,
        }).dropna()

        if len(merged) < 10:
            return {"rank_ic": 0.0, "icir": 0.0}

        rank_ic = merged["signal"].rank().corr(merged["forward_ret"].rank())
        self.metrics["rank_ic"] = rank_ic

        # 简单 ICIR (滚动 Rank IC 的均值/标准差)
        # 按日期分组
        day_groups = merged.groupby(level=0) if isinstance(merged.index, pd.MultiIndex) else [merged]
        ic_series = []
        if isinstance(merged.index, pd.MultiIndex):
            for date, group in merged.groupby(level=0):
                if len(group) >= 5:
                    ic = group["signal"].rank().corr(group["forward_ret"].rank())
                    ic_series.append(ic)
        if ic_series:
            self.metrics["icir"] = np.mean(ic_series) / (np.std(ic_series) + 1e-12)
        else:
            self.metrics["icir"] = 0.0

        return self.metrics


@dataclass
class PortAnalysisRecord:
    """投资组合分析记录 (Qlib PortAnaRecord)

    对预测信号进行分层回测:
    - TopK 多头、TopK 空头、多空对冲
    - 各分层收益分析
    """

    signals: pd.Series
    forward_returns: pd.Series
    n_groups: int = 10
    top_k: int = 50

    def __post_init__(self):
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """执行组合分析"""
        merged = pd.DataFrame({
            "signal": self.signals,
            "return": self.forward_returns,
        }).dropna()

        if len(merged) < self.n_groups:
            logger.warning(f"Not enough samples: {len(merged)} < {self.n_groups}")
            return {}

        # 分层分析
        merged["group"] = pd.qcut(merged["signal"].rank(), self.n_groups, labels=False)

        group_returns = merged.groupby("group")["return"].agg(["mean", "std", "count"])

        # TopK 多头
        top_k = min(self.top_k, len(merged) // self.n_groups)
        merged_sorted = merged.sort_values("signal", ascending=False)
        top_long = merged_sorted.head(top_k)["return"].mean()
        top_short = merged_sorted.tail(top_k)["return"].mean()

        self.results = {
            "group_returns": group_returns.to_dict(),
            "top_k_long_return": float(top_long),
            "top_k_short_return": float(top_short),
            "long_short_spread": float(top_long - top_short),
            "top_group_return": float(group_returns.loc[group_returns.index.max(), "mean"]) if group_returns.index.max() != group_returns.index.min() else 0.0,
            "bottom_group_return": float(group_returns.loc[group_returns.index.min(), "mean"]),
            "spread_10_1": float(group_returns.loc[group_returns.index.max(), "mean"] - group_returns.loc[group_returns.index.min(), "mean"]) if len(group_returns) > 1 else 0.0,
        }
        return self.results

    def summary(self) -> str:
        if not self.results:
            self.run()
        r = self.results
        return (
            f"  TopK Long/Short Spread: {r.get('long_short_spread', 0):.4%}\n"
            f"  TopK Long Return:        {r.get('top_k_long_return', 0):.4%}\n"
            f"  TopK Short Return:       {r.get('top_k_short_return', 0):.4%}\n"
            f"  Spread (Top/Bottom Decile): {r.get('spread_10_1', 0):.4%}"
        )


@dataclass
class TopkDropoutStrategy:
    """TopK Dropout 策略 (Qlib TopkDropoutStrategy)

    每期选择预测得分最高的 TopK 个标的，并持仓至下一期。
    支持 dropout (排除最近 N 天内已选中的标的)。
    """

    top_k: int = 50
    dropout_days: int = 0
    holding_period: int = 1

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """运行 TopK Dropout 策略

        Args:
            signals: MultiIndex [datetime, instrument], column=[score]
            prices: MultiIndex [datetime, instrument], column=[close]

        Returns:
            MultiIndex [datetime, instrument], column=[weight] 持仓权重
        """
        if isinstance(signals.index, pd.MultiIndex):
            dates = signals.index.get_level_values(0).unique().sort_values()
        else:
            dates = signals.index.sort_values()

        weights_list = []
        excluded = set()  # 最近被排除的 (instrument, last_selected_date)

        for i, date in enumerate(dates):
            day_signals = signals.xs(date, level=0) if isinstance(signals.index, pd.MultiIndex) else signals.loc[date]
            day_signals = day_signals.sort_values(signals.columns[0], ascending=False)

            # Dropout: 排除最近被选中的
            if self.dropout_days > 0:
                day_signals = day_signals[
                    ~day_signals.index.isin(
                        [inst for inst, last_date in excluded
                         if (date - last_date).days <= self.dropout_days]
                    )
                ]

            selected = day_signals.head(self.top_k)
            selected_instruments = selected.index.tolist()

            # 更新排除记录
            for inst in selected_instruments:
                excluded.add((inst, date))
                # 清理过期记录
                excluded = {
                    (i, d) for i, d in excluded
                    if (date - d).days <= self.dropout_days
                }

            # 等权持仓
            w = 1.0 / max(len(selected_instruments), 1)
            for inst in selected_instruments:
                weights_list.append({
                    "datetime": date,
                    "instrument": inst,
                    "weight": w,
                })

        weights_df = pd.DataFrame(weights_list)
        if not weights_df.empty:
            weights_df = weights_df.set_index(["datetime", "instrument"])
        return weights_df


class WorkflowEngine:
    """工作流引擎 — 编排量化交易全流程

    支持:
    - 数据加载与因子生成
    - 模型训练与预测
    - 信号分析 (IC, IR)
    - 组合分析 (分层回测)
    - TopK Dropout 策略回测

    Usage:
        engine = WorkflowEngine()
        engine.set_data(data_df, feature_cols=['alpha360...'], label_col='label')
        engine.set_model(LightGBMModel())
        result = engine.run()
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None
        self.label_column: str = "label"
        self.model: Optional[BaseModel] = None
        self.processors: Optional[Any] = None

        self._signal_record: Optional[SignalRecord] = None
        self._port_analysis: Optional[PortAnalysisRecord] = None

        self.results: Dict[str, Any] = {}

    def set_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
    ):
        self.data = data
        self.label_column = label_column
        if feature_columns is None:
            self.feature_columns = [c for c in data.columns if c != label_column]
        else:
            self.feature_columns = feature_columns
        logger.info(f"Data set: {data.shape}, features={len(self.feature_columns)}")
        return self

    def set_model(self, model: BaseModel):
        self.model = model
        logger.info(f"Model set: {model}")
        return self

    def set_processors(self, processors: Any):
        self.processors = processors
        return self

    def run(
        self,
        train_dates: Tuple[str, str],
        test_dates: Tuple[str, str],
    ) -> Dict[str, Any]:
        """执行完整工作流"""
        if self.data is None or self.model is None:
            raise ValueError("Data and model must be set before run()")

        # 1. 数据分割
        from hyperion.data.dataset import DatasetH, DatasetSegment
        handler = DatasetH(
            self.data,
            feature_columns=self.feature_columns,
            label_column=self.label_column,
            processors=self.processors,
        )

        train_seg, valid_seg, test_seg = handler.split(
            train=train_dates,
            valid=None,
            test=test_dates,
        )

        logger.info(
            f"Dataset: train={train_seg.n_samples}, "
            f"valid={valid_seg.n_samples}, "
            f"test={test_seg.n_samples}"
        )

        # 2. 训练
        logger.info(f"Training {self.model}...")
        if valid_seg.n_samples > 0:
            self.model.fit(
                train_seg.features,
                train_seg.labels,
                eval_set=[(valid_seg.features, valid_seg.labels)],
            )
        else:
            self.model.fit(train_seg.features, train_seg.labels)

        # 3. 预测
        logger.info("Predicting...")
        preds = self.model.predict(test_seg.features)

        # 4. 信号记录
        dates = test_seg.dates
        instruments = test_seg.instruments
        # 构建信号 Series
        signal_idx = test_seg.features.index
        signals = pd.Series(preds, index=signal_idx, name="score")
        self._signal_record = SignalRecord(
            signals=signals.to_frame(),
            model_name=self.model.__class__.__name__,
        )

        # 5. IC 分析
        if self.label_column in self.data.columns:
            test_labels = self.data.loc[signal_idx, self.label_column]
            ic_metrics = self._signal_record.ic_analysis(test_labels)
            logger.info(f"Rank IC: {ic_metrics.get('rank_ic', 0):.4f}, ICIR: {ic_metrics.get('icir', 0):.4f}")
            self.results["ic_metrics"] = ic_metrics

        # 6. 组合分析
        if self.label_column in self.data.columns:
            self._port_analysis = PortAnalysisRecord(
                signals=signals,
                forward_returns=test_labels if "test_labels" in dir() else self.data.loc[signal_idx, self.label_column],
            )
            port_results = self._port_analysis.run()
            self.results["portfolio_analysis"] = port_results
            logger.info(f"Portfolio:\n  {self._port_analysis.summary()}")

        self.results["model"] = str(self.model)
        self.results["train_shape"] = train_seg.shape
        self.results["test_shape"] = test_seg.shape

        return self.results

    def report(self) -> str:
        """生成分析报告文本"""
        lines = ["=" * 60, "HYPERION QUANT WORKFLOW REPORT", "=" * 60, ""]

        lines.append(f"Model: {self.results.get('model', 'N/A')}")
        lines.append(f"Train: {self.results.get('train_shape', 'N/A')}")
        lines.append(f"Test:  {self.results.get('test_shape', 'N/A')}")
        lines.append("")

        if "ic_metrics" in self.results:
            ic = self.results["ic_metrics"]
            lines.append("--- IC Analysis ---")
            lines.append(f"  Rank IC:         {ic.get('rank_ic', 0):.4f}")
            lines.append(f"  ICIR:            {ic.get('icir', 0):.4f}")

        if "portfolio_analysis" in self.results:
            pa = self.results["portfolio_analysis"]
            lines.append("")
            lines.append("--- Portfolio Analysis ---")
            lines.append(f"  TopK Long Return:   {pa.get('top_k_long_return', 0):.4%}")
            lines.append(f"  TopK Short Return:  {pa.get('top_k_short_return', 0):.4%}")
            lines.append(f"  Long/Short Spread:  {pa.get('long_short_spread', 0):.4%}")
            lines.append(f"  Spread (D10-D1):    {pa.get('spread_10_1', 0):.4%}")

        return "\n".join(lines)