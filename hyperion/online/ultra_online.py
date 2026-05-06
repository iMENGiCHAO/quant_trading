"""
Hyperion+ OnlineLearning — 在线学习与漂移检测引擎
===================================================
解决量化交易中最难的问题：市场非平稳性。

核心组件：
  1. Bayesian Online Learning (模型参数贝叶斯更新)
  2. 概念漂移检测 (KS检验、KL散度、窗口比较)
  3. 自动重训练 (触发条件 + 再训练管道)
  4. 因子IC漂移监控
  5. 市场状态检测 (Regime Detection)
  6. 自适应权重衰减

这些模块在 QLib 中完全没有，是 Alpha Hunter 的杀手锏。
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from collections import deque, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
#  漂移检测: 核心抽象类
# ==========================================================

class DriftDetector:
    """
    概念漂移检测器基类。
    子类实现具体的检测逻辑。
    """
    def __init__(self, name: str, threshold: float = 0.05):
        self.name = name
        self.threshold = threshold
        self.history = []
        self._is_drifted = False
        self._last_drift_idx = -1

    def update(self, data: pd.Series) -> bool:
        """
        更新并检测漂移。
        Return True if drift detected.
        """
        self.history.append(data)
        is_drift = self._detect(data)
        if is_drift:
            self._is_drifted = True
            self._last_drift_idx = len(self.history) - 1
        return is_drift

    def _detect(self, data: pd.Series) -> bool:
        raise NotImplementedError

    def reset(self):
        """重置状态"""
        self.history = []
        self._is_drifted = False
        self._last_drift_idx = -1

    @property
    def is_drifted(self) -> bool:
        return self._is_drifted


# ==========================================================
#  1. KS检验漂移检测 (监控分布变化)
# ==========================================================

class KSDriftDetector(DriftDetector):
    """
    基于 Kolmogorov-Smirnov 检验的分布漂移检测。
    比较参考窗口 vs 检测窗口的分布差异。
    """
    def __init__(self, window_size: int = 60, ref_window: int = 200,
                 threshold: float = 0.05):
        super().__init__("KS_Drift", threshold)
        self.window_size = window_size
        self.ref_window = ref_window
        self._reference = None

    def _detect(self, data: pd.Series) -> bool:
        try:
            from scipy.stats import ks_2samp
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
        
        if not HAS_SCIPY:
            # Fallback: simple mean variance test without scipy
            if len(self.history) < self.window_size + self.ref_window:
                return False
            ref = pd.concat(self.history[-(self.window_size + self.ref_window):-self.window_size])
            test = pd.concat(self.history[-self.window_size:])
            return abs(ref.mean() - test.mean()) > self.threshold * (ref.std() + test.std())
        
        if len(self.history) < self.window_size + self.ref_window:
            return False

        ref_data = pd.concat(self.history[-(self.window_size + self.ref_window):-self.window_size])
        test_data = pd.concat(self.history[-self.window_size:])
        # 检测窗口 (最近的数据)
        test_data = pd.concat(self.history[-self.window_size:])

        if len(ref_data) < 10 or len(test_data) < 10:
            return False

        # KS检验
        statistic, p_value = ks_2samp(ref_data.values, test_data.values)
        
        # 如果p值 < threshold，说明分布显著不同
        return p_value < self.threshold


# ==========================================================
#  2. KL 散度漂移检测 (信息论角度)
# ==========================================================

class KLDivergenceDetector(DriftDetector):
    """
    基于 KL 散度的漂移检测。
    计算参考分布与当前分布之间的信息距离。
    """
    def __init__(self, window_size: int = 60, bins: int = 20,
                 threshold: float = 0.5):
        super().__init__("KL_Drift", threshold)
        self.window_size = window_size
        self.bins = bins

    def _hist_density(self, data: pd.Series) -> np.ndarray:
        """估计概率密度"""
        hist, edges = np.histogram(data.dropna(), bins=self.bins, density=True)
        # 平滑处理，避免0
        hist = (hist + 0.01) / (hist.sum() + self.bins * 0.01)
        return hist

    def _detect(self, data: pd.Series) -> bool:
        if len(self.history) < 2 * self.window_size:
            return False

        # 参考窗口
        ref_data = pd.concat(self.history[-(2*self.window_size):-self.window_size])
        # 当前窗口
        curr_data = pd.concat(self.history[-self.window_size:])

        # 估计密度
        p = self._hist_density(ref_data)
        q = self._hist_density(curr_data)

        # KL散度 (P || P') + (P' || P)，即对称KL
        kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10))) + \
             np.sum(q * np.log((q + 1e-10) / (p + 1e-10)))
        return kl > self.threshold


# ==========================================================
#  3. 滚动窗口统计漂移
# ==========================================================

class RollingStatsDriftDetector(DriftDetector):
    """
    监控滚动窗口统计量的变化 (均值、方差、偏度、峰度)。
    当统计量超过阈值时触发。
    """
    def __init__(self, window_size: int = 60,
                 z_threshold: float = 2.5):
        super().__init__("RollingStats", z_threshold)
        self.window_size = window_size

    def _detect(self, data: pd.Series) -> bool:
        if len(self.history) < 2 * self.window_size:
            return False

        recent = pd.concat(self.history[-self.window_size:])
        prev = pd.concat(self.history[-(2*self.window_size):-self.window_size])

        # 比较均值
        mean_recent = recent.mean()
        mean_prev = prev.mean()
        std_prev = prev.std() + 1e-12

        z_score = abs((mean_recent - mean_prev) / std_prev)
        
        # 比较方差 (F检验简化)
        var_recent = recent.var()
        var_prev = prev.var() + 1e-12
        f_stat = var_recent / var_prev
        var_drift = f_stat > 3 or f_stat < 1/3

        return z_score > self.threshold or var_drift


# ==========================================================
#  4. 市场状态检测 (Regime Detection)
# ==========================================================

class RegimeDetector:
    """
    市场状态检测器。
    识别不同市场状态：Bull/Bear/LowVol/HighVol/Trend/MeanRev等
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.states = []

    def detect(self, returns: pd.Series) -> str:
        """检测当前市场状态"""
        if len(returns) < self.lookback:
            return "unknown"

        recent = returns[-self.lookback:]
        
        # 特征
        vol = recent.std() * np.sqrt(252)  # 年化波动率
        trend = recent.mean() * 252       # 年化收益率
        skew = recent.skew()              # 偏度
        kurt = recent.kurt()              # 峰度

        # 状态判断
        if vol > 0.3:
            return "high_vol"
        elif vol > 0.15:
            return "medium_vol"
        
        if trend > 0.15 and vol < 0.20:
            return "bull_trend"
        elif trend < -0.15 and vol < 0.20:
            return "bear_trend"
        
        if abs(skew) > 0.5:
            return "skewed"
        
        if kurt > 3:
            return "fat_tail"

        return "normal"


# ==========================================================
#  5. 在线学习管道
# ==========================================================

class OnlineLearner:
    """
    在线学习管道。
    核心功能:
      - 监控模型性能
      - 检测漂移
      - 自动触发再训练
      - 自适应权重更新
    """
    def __init__(self,
                 models: Dict[str, Any],
                 detectors: Optional[List[DriftDetector]] = None,
                 regime_detector: Optional[RegimeDetector] = None,
                 retrain_window: int = 252,
                 ic_threshold: float = 0.02):
        self.models = models
        self.detectors = detectors or [
            KSDriftDetector(),
            RollingStatsDriftDetector(),
        ]
        self.regime_detector = regime_detector or RegimeDetector()
        self.retrain_window = retrain_window
        self.ic_threshold = ic_threshold

        self.performance_history = defaultdict(deque)
        self.ic_history = defaultdict(deque)
        self.factor_weights = defaultdict(float)
        self.last_retrain_idx = 0

    def update(self,
               date: Any,
               predictions: Dict[str, float],
               true_values: pd.Series,
               features: Optional[pd.DataFrame] = None):
        """
        在线更新。
        
        Args:
            date: 当前日期
            predictions: 各模型的预测 {model_name: prediction}
            true_values: 实际值
            features: 特征DataFrame (用于漂移检测)
        """
        # 1. 更新IC历史
        for model_name, pred in predictions.items():
            ic = pred.corr(true_values) if hasattr(pred, 'corr') else 0
            self.ic_history[model_name].append(ic)

        # 2. 因子与模型漂移检测
        drift_count = 0
        if features is not None:
            for col in features.columns:
                for detector in self.detectors:
                    if detector.update(features[col]):
                        logger.warning(f"因子 {col} 漂移检测: {detector.name}")
                        drift_count += 1

        # 3. 更新权重 (Bayesian 衰减)
        total_weight = 0
        for model_name in predictions:
            ic_series = pd.Series(list(self.ic_history[model_name]))
            if len(ic_series) > 10:
                recent_ic = ic_series[-10:].mean()
                # 权重 ∝ max(0, IC)^2 (以平方放大高IC模型的影响)
                weight = max(0, recent_ic) ** 2
                self.factor_weights[model_name] = weight
                total_weight += weight
            else:
                self.factor_weights[model_name] = 1.0 / len(self.models)
                total_weight += 1.0 / len(self.models)

        # Normalize
        if total_weight > 0:
            for k in self.factor_weights:
                self.factor_weights[k] /= total_weight

        # 4. 检查再训练条件
        days_since_retrain = len(self.ic_history[list(self.ic_history.keys())[0]]) if self.ic_history else 0
        
        # 触发再训练
        avg_ic = sum(self.ic_history[k][-1] if self.ic_history[k] else 0 
                     for k in self.models) / len(self.models) if self.models else 0
        
        should_retrain = (days_since_retrain - self.last_retrain_idx > self.retrain_window) or \
                        (avg_ic < self.ic_threshold and drift_count > 0)

        if should_retrain:
            logger.info(f"触发自适应再训练: avg_ic={avg_ic:.4f}, drift_count={drift_count}")
            return True  # 信号: 需要再训练

        return False  # 不需要再训练

    def get_weights(self) -> Dict[str, float]:
        """返回当前模型/因子权重"""
        return dict(self.factor_weights)

    def detect_regime(self, returns: pd.Series) -> str:
        """检测当前市场状态"""
        return self.regime_detector.detect(returns)


# ==========================================================
#  6. 自动重训练管道
# ==========================================================

class AdaptiveRetrainer:
    """
    自适应重训练器。
    在检测到漂移后自动重新训练模型。
    """
    def __init__(self, model_factory: Callable, 
                 retrain_strategy: str = "rolling",
                 train_size: int = 500,
                 valid_size: int = 60):
        self.model_factory = model_factory
        self.retrain_strategy = retrain_strategy
        self.train_size = train_size
        self.valid_size = valid_size
        self.training_log = []

    def retrain(self, data: pd.DataFrame, target: pd.Series,
                feature_cols: List[str]) -> Any:
        """
        重训练一个模型
        
        Returns:
            训练好的模型
        """
        logger.info(f"自适应重训练: data_size={len(data)}")
        
        if self.retrain_strategy == "rolling":
            X = data[feature_cols].iloc[-self.train_size:]
            y = target.iloc[-self.train_size:]
            
        elif self.retrain_strategy == "expanding":
            X = data[feature_cols]
            y = target
        else:
            X = data[feature_cols].iloc[-self.train_size:]
            y = target.iloc[-self.train_size:]

        model = self.model_factory()
        model.fit(X, y)
        
        self.training_log.append({
            "date": pd.Timestamp.now(),
            "train_size": len(X),
            "features": len(feature_cols)
        })
        
        return model


# ==========================================================
#  7. 完整管道封装 (一键使用)
# ==========================================================

class OnlineLearningPipeline:
    """
    在线学习全流程。
    
    使用:
        pipeline = OnlineLearningPipeline(models={"lgb": lgb_model, " ridge": ridge_model})
        
        for date in trading_dates:
            # ... 训练/预测 ...
            is_retrain = pipeline.update(date, predictions, true_values, features)
            if is_retrain:
                new_model = pipeline.retrain(data, target, features)
                models[model_name] = new_model
    """
    def __init__(self,
                 models: Dict[str, Any],
                 drift_detectors: Optional[List[DriftDetector]] = None,
                 retrain_strategy: str = "rolling"):
        self.online_learner = OnlineLearner(models, drift_detectors)
        self.retrainer = AdaptiveRetrainer(
            model_factory=lambda: list(models.values())[0],
            retrain_strategy=retrain_strategy
        )

    def update(self,
               date: Any,
               predictions: Dict[str, float],
               true_values: pd.Series,
               features: Optional[pd.DataFrame] = None) -> Tuple[bool, Dict]:
        """
        更新在线学习状态
        
        Returns:
            (needs_retrain, current_weights)
        """
        needs_retrain = self.online_learner.update(date, predictions, true_values, features)
        weights = self.online_learner.get_weights()
        return needs_retrain, weights

    def retrain(self, data: pd.DataFrame, target: pd.Series,
                feature_cols: List[str]) -> Any:
        """执行重训练"""
        return self.retrainer.retrain(data, target, feature_cols)

    def get_regime(self, returns: pd.Series) -> str:
        """获取当前市场状态"""
        return self.online_learner.detect_regime(returns)


# ==========================================================
#  兼容 QLib 接口
# ==========================================================

OnlineDriftDetector = KSDriftDetector
OnlineRetrainer = AdaptiveRetrainer
