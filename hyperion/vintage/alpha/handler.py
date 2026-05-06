"""
DataHandler 数据处理器 (Qlib-style)
=====================================
数据预处理管道，包括标准化、去极值、填缺失等处理器。

参考 Qlib 的 infer_processors 和 learn_processors 设计。

处理器类型:
- InferProcessor: 在推断时可独立运行 (不需要训练数据统计量)
- LearnProcessor: 需要从训练数据学习统计量 (fit/transform 模式)

Pipeline:
    features → [RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm] → clean features
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """处理器基类"""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseProcessor":
        """从数据中学习统计量"""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据应用变换"""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class RobustZScoreNorm(BaseProcessor):
    """Robust Z-Score 标准化

    使用中位数替代均值，MAD 替代标准差，对异常值更鲁棒。

    z = (x - median) / (MAD * 1.4826)

    Qlib 默认使用此方法进行特征标准化。
    """

    def __init__(self, fields_group: str = "feature", clip_outlier: bool = True):
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.medians_: Optional[pd.Series] = None
        self.mads_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "RobustZScoreNorm":
        self.medians_ = df.median()
        self.mads_ = (df - self.medians_).abs().median()
        # 如果 MAD 为 0，用标准差代替
        zero_mad = self.mads_ == 0
        if zero_mad.any():
            self.mads_[zero_mad] = df.std()[zero_mad]
            self.mads_ = self.mads_.replace(0, 1)  # 如果还是 0 设为 1
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None or self.mads_ is None:
            raise RuntimeError("Must call fit() before transform()")

        result = (df - self.medians_) / (self.mads_ * 1.4826 + 1e-12)

        if self.clip_outlier:
            # 截断异常值到 [-3, 3] 范围
            result = result.clip(-3, 3)

        return result


class CSRankNorm(BaseProcessor):
    """Cross-Sectional Rank Normalization

    横截面排名标准化: 将截面数据转换为均匀分布 / 正态分布。

    适用于 Label 处理 (Qlib 的 DropnaLabel + CSRankNorm).
    """

    def __init__(self, fields_group: str = "label"):
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame) -> "CSRankNorm":
        return self  # 无统计量需要学习

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in df.columns:
            # 排名 → 分位数 → 正态分布
            ranked = df[col].rank(pct=True)
            # 映射到正态分布
            result[col] = ranked.apply(
                lambda x: np.clip(
                    np.sqrt(2) * self._erf_inv(2 * x - 1), -3, 3
                ) if not pd.isna(x) else np.nan
            )
        return result

    @staticmethod
    def _erf_inv(x: float) -> float:
        """近似逆误差函数 (纯 NumPy 实现)"""
        if x <= -1:
            return -float("inf")
        if x >= 1:
            return float("inf")
        # 使用 scipy 的 if available
        try:
            from scipy.special import erfinv
            return float(erfinv(x))
        except ImportError:
            pass
        # 近似: x + x^3/3 + 7x^5/30 + ...
        x3 = x ** 3
        x5 = x ** 5
        return x + x3 / 3 + 7 * x5 / 30 + 127 * x5 * x * x / 630


class Fillna(BaseProcessor):
    """缺失值填充

    Qlib 默认用 0 填充特征、用 NaN 填充 label。
    """

    def __init__(self, fields_group: str = "feature", fill_value: float = 0.0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def fit(self, df: pd.DataFrame) -> "Fillna":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(self.fill_value)


class DropnaLabel(BaseProcessor):
    """删除 Label 为 NaN 的行

    用于训练标签，NaN 标签表示无监督样本。
    """

    def __init__(self, fields_group: str = "label"):
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame) -> "DropnaLabel":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(how="all")


class ClipOutlier(BaseProcessor):
    """截断异常值

    基于百分位数截断。
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "ClipOutlier":
        self.lower_bounds_ = df.quantile(self.lower)
        self.upper_bounds_ = df.quantile(self.upper)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Must call fit() before transform()")
        return df.clip(self.lower_bounds_, self.upper_bounds_, axis=1)


class ProcessorChain:
    """处理器链 — 按顺序执行多个处理器

    Usage:
        chain = ProcessorChain([
            RobustZScoreNorm(fields_group="feature"),
            Fillna(fields_group="feature", fill_value=0.0),
        ])
        chain.fit(train_df)
        train_clean = chain.transform(train_df)
        test_clean = chain.transform(test_df)
    """

    def __init__(self, processors: List[BaseProcessor]):
        self.processors = processors

    def fit(self, df: pd.DataFrame) -> "ProcessorChain":
        for p in self.processors:
            p.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for p in self.processors:
            result = p.transform(result)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        names = [type(p).__name__ for p in self.processors]
        return f"ProcessorChain([{', '.join(names)}])"


class DataHandler:
    """DataHandler — 统一数据处理器

    融合 Qlib 的 infer_processors + learn_processors 设计。

    配置示例:
        handler:
          processors:
            infer:
              - class: RobustZScoreNorm
                kwargs: {fields_group: feature, clip_outlier: true}
              - class: Fillna
                kwargs: {fields_group: feature, fill_value: 0.0}
            learn:
              - class: DropnaLabel
                kwargs: {fields_group: label}
              - class: CSRankNorm
                kwargs: {fields_group: label}
    """

    def __init__(
        self,
        infer_processors: Optional[List[BaseProcessor]] = None,
        learn_processors: Optional[List[BaseProcessor]] = None,
    ):
        self.infer_processors = infer_processors or [
            RobustZScoreNorm(fields_group="feature", clip_outlier=True),
            Fillna(fields_group="feature", fill_value=0.0),
        ]
        self.learn_processors = learn_processors or [
            DropnaLabel(fields_group="label"),
            CSRankNorm(fields_group="label"),
        ]
        self._infer_chain = ProcessorChain(self.infer_processors)
        self._learn_chain = ProcessorChain(self.learn_processors)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> "DataHandler":
        """从训练数据学习"""
        self._infer_chain.fit(features)
        self._learn_chain.fit(labels)
        return self

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """对特征应用 infer 处理器"""
        return self._infer_chain.transform(features)

    def transform_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """对标签应用 learn 处理器"""
        return self._learn_chain.transform(labels)

    def transform(self, features: pd.DataFrame, labels: pd.DataFrame) -> tuple:
        """同时对特征和标签进行变换"""
        return (
            self.transform_features(features),
            self.transform_labels(labels),
        )

    def __repr__(self) -> str:
        return f"DataHandler(infer={self._infer_chain}, learn={self._learn_chain})"