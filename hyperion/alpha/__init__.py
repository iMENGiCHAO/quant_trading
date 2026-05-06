"""
Hyperion Alpha Layer (Layer 2)
===============================
因子工程 + 特征提取 + 信号生成

融合精华:
  Qlib → Alpha158/360 因子库
  VnPy → 技术指标 + 信号生成
  VectorBT → 向量化指标工厂
  v25 → 贝叶斯在线学习
  v26 → 因果发现

Modules:
  factors.py   — Alpha158 因子提取器
  technical.py — 技术指标工厂
  bayesian.py  — 贝叶斯在线因子权重学习
  causal.py    — 因果发现 (Granger/PC算法)
"""
from hyperion.alpha.factors import Alpha158
from hyperion.alpha.technical import TechnicalIndicators
from hyperion.alpha.bayesian import BayesianUpdater
from hyperion.alpha.causal import CausalDiscovery

__all__ = ["Alpha158", "TechnicalIndicators", "BayesianUpdater", "CausalDiscovery"]
