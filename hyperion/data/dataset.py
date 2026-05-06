"""
DatasetH — Qlib-style 数据集处理器
======================================
从多层级 (instrument × datetime) 数据构建训练/验证/测试集。

Qlib 的 DatasetH 核心设计:
1. Point-in-Time 数据处理: 每个时间点只使用截止该点的数据
2. 支持多股票横截面、多时间序列的分层采样
3. 内置数据预处理管道 (processors)
4. 统一的 data_handler / segment 配置

Usage:
    handler = DatasetH(
        data=X_y_multiindex_df,  # MultiIndex(datetime, instrument)
        task='regression',
        processors=None,  # 可选，使用 DataHandler 处理器
    )
    ds_train, ds_valid, ds_test = handler.split(
        train=(2000, 2018), valid=(2019, 2020), test=(2021, 2022)
    )
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd

from hyperion.alpha.handler import DataHandler, BaseProcessor, ProcessorChain

logger = logging.getLogger(__name__)


@dataclass
class DatasetSegment:
    """数据集分段"""
    features: pd.DataFrame
    labels: pd.Series
    dates: pd.DatetimeIndex
    instruments: pd.Index

    @property
    def shape(self):
        return self.features.shape

    @property
    def n_samples(self):
        return len(self.features)

    def __repr__(self):
        return f"DatasetSegment({self.n_samples} samples, {self.features.shape[1]} features, " \
               f"dates=[{self.dates[0]}, {self.dates[-1]}], {len(self.instruments)} instruments)"


def _ensure_multiindex(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """确保 DataFrame 是 MultiIndex (datetime, instrument)"""
    if isinstance(df.index, pd.MultiIndex):
        if "datetime" not in df.index.names and "date" not in df.index.names:
            logger.warning(f"MultiIndex names: {df.index.names}, expected 'datetime'/'date' level")
        return df
    elif isinstance(df.index, pd.DatetimeIndex):
        # 单一股票，添加 instrument 列作为索引
        raise ValueError(
            "DataFrame must have MultiIndex(datetime, instrument). "
            "Use `df.set_index(['datetime', 'instrument'], inplace=True)`."
        )
    else:
        raise ValueError(f"Unsupported index type: {type(df.index)}")


class DatasetH:
    """Qlib-style 数据集处理器

    参考 Qlib 的 DatasetH + Dataset 设计。

    Args:
        data: 含特征和标签的 DataFrame, MultiIndex(datetime, instrument)
        feature_columns: 特征列名列表。若为 None, 自动检测 (不含 label 列)
        label_column: 标签列名。默认 'label'
        processors: 可选的 DataHandler 或处理器链
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
        processors: Optional[Union[DataHandler, ProcessorChain, List[BaseProcessor]]] = None,
    ):
        data = _ensure_multiindex(data)

        # 自动检测特征列
        if feature_columns is None:
            exclude = {label_column, "instrument", "datetime"}
            feature_columns = [c for c in data.columns if c not in exclude]

        self._raw_data = data
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.processors = processors

        # 获取日期集
        self._dates = data.index.get_level_values(0).unique().sort_values()
        self._instruments = data.index.get_level_values(1).unique()

        logger.info(
            f"DatasetH: {len(self._dates)} dates × {len(self._instruments)} instruments, "
            f"{len(feature_columns)} features"
        )

    @property
    def features(self) -> pd.DataFrame:
        return self._raw_data[self.feature_columns]

    @property
    def labels(self) -> pd.Series:
        if self.label_column in self._raw_data.columns:
            return self._raw_data[self.label_column]
        return pd.Series(index=self._raw_data.index, dtype=float)

    def split(
        self,
        train: Tuple[Union[str, int], Union[str, int]],
        valid: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
        test: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
        valid_fraction: float = 0.2,
        test_fraction: float = 0.2,
    ) -> Tuple[DatasetSegment, DatasetSegment, DatasetSegment]:
        """按日期范围 (或比例) 分割数据集

        Args:
            train: (start, end) 日期范围，例如 (2000, 2018) 或 ("2000-01-01", "2018-12-31")
            valid: (start, end) 验证集日期范围
            test: (start, end) 测试集日期范围
            valid_fraction: 当 valid 为 None 时的验证集比例
            test_fraction: 当 test 为 None 时的测试集比例

        Returns:
            (train_segment, valid_segment, test_segment)
        """
        if valid is None:
            n_dates = len(self._dates)
            train_end_date = self._dates[int(n_dates * (1 - valid_fraction - test_fraction))]
            valid_start_date = train_end_date
            valid_end_date = self._dates[int(n_dates * (1 - test_fraction))]
            valid = (valid_start_date, valid_end_date)
            test = (valid_end_date, self._dates[-1])

        return (
            self._segment("train", *train),
            self._segment("valid", *valid) if valid else self._segment("valid"),
            self._segment("test", *test) if test else self._segment("test"),
        )

    def _segment(
        self,
        name: str,
        start: Optional[Union[str, int]] = None,
        end: Optional[Union[str, int]] = None,
    ) -> DatasetSegment:
        """从数据集中分段"""
        mask = pd.Series(True, index=self._dates)

        if start is not None:
            start_dt = pd.Timestamp(start)
            mask = mask & (self._dates >= start_dt)
        if end is not None:
            end_dt = pd.Timestamp(end)
            mask = mask & (self._dates <= end_dt)

        selected_dates = self._dates[mask.values]
        if len(selected_dates) == 0:
            logger.warning(f"{name}: no dates selected in [{start}, {end}]")
            return DatasetSegment(
                features=pd.DataFrame(columns=self.feature_columns),
                labels=pd.Series(dtype=float),
                dates=pd.DatetimeIndex([]),
                instruments=pd.Index([]),
            )

        df = self._raw_data.loc[self._raw_data.index.get_level_values(0).isin(selected_dates)]

        features = df[self.feature_columns]
        labels = df[self.label_column] if self.label_column in df.columns else pd.Series(
            index=df.index, dtype=float
        )

        # 应用处理器
        if self.processors is not None:
            if isinstance(self.processors, DataHandler):
                if not hasattr(self.processors, "_fitted") or not self._processor_fitted:
                    self.processors.fit(features, labels.to_frame() if isinstance(labels, pd.Series) else labels)
                    self._processor_fitted = True
                features = self.processors.transform_features(features)
            elif isinstance(self.processors, (ProcessorChain, list)):
                chain = self.processors if isinstance(self.processors, ProcessorChain) else ProcessorChain(self.processors)
                features = chain.fit_transform(features)

        logger.info(
            f"{name}: {len(features)} samples, dates=[{selected_dates[0]}, {selected_dates[-1]}]"
        )

        return DatasetSegment(
            features=features,
            labels=labels,
            dates=selected_dates,
            instruments=df.index.get_level_values(1).unique(),
        )

    def get_rolling_window(
        self,
        window: int = 60,
        step: int = 20,
    ) -> List[Tuple[DatasetSegment, DatasetSegment]]:
        """滚动窗口分割 (walk-forward)

        返回 list of (train, test) 时间段
        """
        segments = []
        n = len(self._dates)
        for i in range(0, n - window, step):
            train_dates = self._dates[i:i + window]
            test_dates = self._dates[i + window:min(i + window + step, n)]

            if len(test_dates) == 0:
                break

            train_seg = DatasetSegment(
                features=self._raw_data.loc[
                    self._raw_data.index.get_level_values(0).isin(train_dates),
                    self.feature_columns
                ],
                labels=self._raw_data.loc[
                    self._raw_data.index.get_level_values(0).isin(train_dates),
                    self.label_column
                ] if self.label_column in self._raw_data.columns else pd.Series(dtype=float),
                dates=train_dates,
                instruments=self._instruments,
            )
            test_seg = DatasetSegment(
                features=self._raw_data.loc[
                    self._raw_data.index.get_level_values(0).isin(test_dates),
                    self.feature_columns
                ],
                labels=self._raw_data.loc[
                    self._raw_data.index.get_level_values(0).isin(test_dates),
                    self.label_column
                ] if self.label_column in self._raw_data.columns else pd.Series(dtype=float),
                dates=test_dates,
                instruments=self._instruments,
            )
            segments.append((train_seg, test_seg))

        logger.info(f"Rolling window: {len(segments)} segments (window={window}, step={step})")
        return segments

    def to_qlib_format(self, path: Optional[str] = None) -> pd.DataFrame:
        """转换为 Qlib 格式的 DataFrame"""
        df = self._raw_data.copy()
        df.index = df.index.set_names(["datetime", "instrument"])
        # Qlib 要求 column 结构: [feature_cols, label]
        df = df[self.feature_columns + [self.label_column]]
        if path:
            df.to_parquet(path)
        return df


class RollingDataset:
    """滚动窗口数据集 — 用于 Walk-Forward Analysis

    自动逐时间段创建 DatasetH 并进行训练/评估。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 60,
        step_size: int = 20,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
    ):
        self.base_handler = DatasetH(
            data=data,
            feature_columns=feature_columns,
            label_column=label_column,
        )
        self.window_size = window_size
        self.step_size = step_size
        self.segments = self.base_handler.get_rolling_window(window_size, step_size)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[DatasetSegment, DatasetSegment]:
        return self.segments[idx]

    def __iter__(self):
        return iter(self.segments)

    def __repr__(self):
        return f"RollingDataset({len(self.segments)} folds, window={self.window_size}, step={self.step_size})"