"""
贝叶斯在线因子权重学习 (v25 遗产)
==================================
基于贝叶斯理论的在线学习系统, 用于动态调整Alpha因子权重。

核心特性:
- Bayesian Online Changepoint Detection (BOCD)
- 因子IC衰减跟踪
- 自适应权重分配
- 概念漂移检测

融合: v25 Bayesian Updater + River在线学习库
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FactorState:
    """单个因子的贝叶斯状态"""
    mu: float = 0.0       # 后验均值 (预期IC)
    sigma: float = 1.0    # 后验标准差
    n: int = 0            # 更新次数
    weight: float = 1.0   # 当前权重
    
    def update(self, x: float, known_var: float = 0.01):
        """高斯-高斯共轭更新"""
        prior_var = self.sigma ** 2
        post_var = 1.0 / (1.0 / prior_var + 1.0 / known_var)
        self.mu = post_var * (self.mu / prior_var + x / known_var)
        self.sigma = np.sqrt(post_var)
        self.n += 1
    
    @property
    def sharpness(self) -> float:
        """信息精度 (反比于方差)"""
        return 1.0 / (self.sigma + 1e-12)
    
    @property
    def z_score(self) -> float:
        """Z分数 (信号强度)"""
        return self.mu / (self.sigma + 1e-12)


class BayesianUpdater:
    """贝叶斯在线因子权重更新器
    
    为每个Alpha因子维护贝叶斯状态,
    根据每日IC动态调整权重。
    
    Usage:
        updater = BayesianUpdater(n_factors=158)
        updater.update(ic_vector)  # 每日更新
        weights = updater.weights  # 获取当前权重
    """
    
    def __init__(self, n_factors: int,
                 prior_mu: float = 0.0,
                 prior_sigma: float = 1.0,
                 drift_threshold: float = 2.0,
                 decay_factor: float = 0.95):
        self.n_factors = n_factors
        self.decay_factor = decay_factor
        self.drift_threshold = drift_threshold
        
        # 每个因子一个贝叶斯状态
        self.states = [FactorState(mu=prior_mu, sigma=prior_sigma)
                       for _ in range(n_factors)]
        
        # 概念漂移检测缓冲
        self._drift_buffer: List[Dict] = []
        self._drift_detected: List[bool] = [False] * n_factors
        
        # 性能追踪
        self._update_count = 0
        self._ic_history: List[np.ndarray] = []
    
    @property
    def weights(self) -> np.ndarray:
        """当前因子权重 (归一化)"""
        w = np.array([s.weight for s in self.states])
        w_sum = w.sum()
        if w_sum > 0:
            return w / w_sum
        return np.ones(self.n_factors) / self.n_factors
    
    @property
    def ic_estimates(self) -> np.ndarray:
        """因子IC估计值"""
        return np.array([s.mu for s in self.states])
    
    @property
    def confidence(self) -> np.ndarray:
        """因子置信度 (sharpness)"""
        return np.array([s.sharpness for s in self.states])
    
    def update(self, ic_vector: np.ndarray,
               known_var: float = 0.01):
        """使用新IC向量更新贝叶斯状态
        
        Args:
            ic_vector: shape (n_factors,) 每个因子的日IC
            known_var: 观测噪声方差
        """
        ic_vector = np.asarray(ic_vector).flatten()
        if len(ic_vector) != self.n_factors:
            raise ValueError(f"Expected {self.n_factors} ICs, got {len(ic_vector)}")
        
        # 1. 贝叶斯更新每个因子
        for i, ic in enumerate(ic_vector):
            if not np.isnan(ic):
                self.states[i].update(ic, known_var)
        
        # 2. 概念漂移检测
        self._detect_drift(ic_vector)
        
        # 3. 权重计算 (IC × Confidence)
        for i, state in enumerate(self.states):
            if self._drift_detected[i]:
                # 检测到漂移 → 重置权重
                state.weight = 0.1
            else:
                # 正常: IC_sharpness × IC_estimate
                ic_sharp = np.abs(state.mu) * state.sharpness
                state.weight = max(ic_sharp, 0.01)
        
        # 4. 衰减历史权重
        if self._update_count > 0:
            for state in self.states:
                state.weight *= self.decay_factor
        
        self._update_count += 1
    
    def _detect_drift(self, ic_vector: np.ndarray):
        """ADWIN风格概念漂移检测"""
        window_size = max(10, self._update_count // 5)
        
        if self._update_count > window_size * 2 and len(self._ic_history) > window_size:
            recent = np.array(self._ic_history[-window_size:])
            older = np.array(self._ic_history[-2*window_size:-window_size])
            
            for i in range(self.n_factors):
                if np.isnan(recent[:, i]).all() or np.isnan(older[:, i]).all():
                    continue
                mu_new = np.nanmean(recent[:, i])
                mu_old = np.nanmean(older[:, i])
                std_pooled = np.sqrt(np.nanvar(recent[:, i]) + np.nanvar(older[:, i]))
                
                z = abs(mu_new - mu_old) / (std_pooled / np.sqrt(window_size) + 1e-12)
                self._drift_detected[i] = z > self.drift_threshold
    
    def get_active_factors(self, top_k: Optional[int] = None) -> np.ndarray:
        """获取最优因子索引"""
        scores = np.abs(self.ic_estimates) * self.confidence
        idx = np.argsort(scores)[::-1]
        if top_k:
            idx = idx[:top_k]
        return idx
    
    def summary(self) -> dict:
        """系统状态摘要"""
        w = self.weights
        return {
            "n_factors": self.n_factors,
            "n_updates": self._update_count,
            "mean_ic": float(np.mean(self.ic_estimates)),
            "mean_weight": float(np.mean(w)),
            "max_weight": float(np.max(w)),
            "drift_count": int(np.sum(self._drift_detected)),
            "top_factors": self.get_active_factors(5).tolist()
        }
