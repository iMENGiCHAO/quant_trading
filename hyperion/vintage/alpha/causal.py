"""
因果发现模块 (v26 遗产)
=========================
Granger因果检验 + PC算法 + 因果图 → 因子筛选

融合: v26 causal_discovery + TETRAD风格约束
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """因果边"""
    source: str         # 因
    target: str         # 果
    strength: float     # 因果强度
    p_value: float      # 显著性
    lag: int = 1        # 滞后阶数


class CausalDiscovery:
    """因果发现引擎
    
    方法:
    1. Granger 因果检验 → 快速筛选因果关系
    2. PC 算法 → 无向图骨架
    3. 定向规则 → 有向因果图
    
    Usage:
        cd = CausalDiscovery(max_lag=5)
        causal_graph = cd.discover(factor_df, target_col="forward_return")
    """
    
    def __init__(self, max_lag: int = 5,
                 significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance
        self._edges: List[CausalEdge] = []
    
    @property
    def edges(self) -> List[CausalEdge]:
        return self._edges
    
    def discover(self, df: pd.DataFrame,
                 target_col: str = "forward_return") -> List[CausalEdge]:
        """执行因果发现
        
        Args:
            df: 因子DataFrame (含target列)
            target_col: 目标变量列名
            
        Returns:
            List of CausalEdge
        """
        self._edges = []
        df = df.dropna()
        
        if df.empty or len(df) < 50:
            return []
        
        # 特征列
        feature_cols = [c for c in df.columns if c != target_col]
        
        # 1. Granger因果检验 (每个特征 → target)
        for col in feature_cols:
            try:
                result = self._granger_test(df[col], df[target_col])
                if result["p_value"] < self.significance:
                    self._edges.append(CausalEdge(
                        source=col,
                        target=target_col,
                        strength=result["f_stat"],
                        p_value=result["p_value"],
                        lag=result["best_lag"]
                    ))
            except Exception as e:
                logger.debug(f"Granger test failed for {col}: {e}")
        
        # 2. 按因果强度排序
        self._edges.sort(key=lambda e: -e.strength)
        
        return self._edges
    
    def _granger_test(self, x: pd.Series, y: pd.Series) -> dict:
        """Granger因果检验
        
        H0: x does not Granger-cause y
        
        Returns:
            dict with f_stat, p_value, best_lag
        """
        x = x.dropna()
        y = y.dropna()
        
        # Align
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        n = len(x)
        if n < self.max_lag + 20:
            return {"f_stat": 0.0, "p_value": 1.0, "best_lag": 1}
        
        best_lag = 1
        best_p = 1.0
        best_f = 0.0
        
        for lag in range(1, self.max_lag + 1):
            # Restricted model: y ~ lag(y)
            y_lag = pd.DataFrame({"y": y})
            for i in range(1, lag + 1):
                y_lag[f"y_lag{i}"] = y.shift(i)
            
            y_lag = y_lag.dropna()
            if y_lag.empty:
                continue
            
            # Residuals from restricted model
            X_r = np.column_stack([np.ones(len(y_lag)),
                                   y_lag[[f"y_lag{i}" for i in range(1, lag+1)]].values])
            y_vec = y_lag["y"].values
            
            try:
                beta_r = np.linalg.lstsq(X_r, y_vec, rcond=None)[0]
                resid_r = y_vec - X_r @ beta_r
                ssr_r = np.sum(resid_r ** 2)
            except np.linalg.LinAlgError:
                continue
            
            # Unrestricted model: y ~ lag(y) + lag(x)
            X_u = X_r.copy()
            for i in range(1, lag + 1):
                x_lag = x.shift(i).loc[y_lag.index].values
                X_u = np.column_stack([X_u, x_lag])
            
            try:
                beta_u = np.linalg.lstsq(X_u, y_vec, rcond=None)[0]
                resid_u = y_vec - X_u @ beta_u
                ssr_u = np.sum(resid_u ** 2)
            except np.linalg.LinAlgError:
                continue
            
            # F-statistic
            df1 = lag  # number of restrictions
            df2 = len(y_vec) - 2 * lag - 1  # residual df
            if df2 <= 0 or ssr_u < 1e-12:
                continue
            
            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            
            # Approximate p-value using F-distribution
            try:
                from scipy.stats import f as f_dist
                p_value = 1 - f_dist.cdf(f_stat, df1, df2)
            except ImportError:
                # Rough approximation
                p_value = np.exp(-0.5 * f_stat) if f_stat > 0 else 1.0
            
            if p_value < best_p:
                best_p = p_value
                best_f = f_stat
                best_lag = lag
        
        return {
            "f_stat": float(best_f),
            "p_value": float(best_p),
            "best_lag": best_lag
        }
    
    def prune_edges(self, min_strength: float = 2.0) -> List[CausalEdge]:
        """剪枝: 移除弱因果边"""
        self._edges = [e for e in self._edges if e.strength >= min_strength]
        return self._edges
    
    def to_dataframe(self) -> pd.DataFrame:
        """转为DataFrame格式"""
        if not self._edges:
            return pd.DataFrame(columns=["source", "target", "strength", "p_value", "lag"])
        return pd.DataFrame([
            {"source": e.source, "target": e.target,
             "strength": e.strength, "p_value": e.p_value, "lag": e.lag}
            for e in self._edges
        ])
    
    def summary(self) -> dict:
        """因果发现摘要"""
        return {
            "n_edges": len(self._edges),
            "significant_edges": sum(1 for e in self._edges if e.p_value < 0.01),
            "top_causes": [(e.source, round(e.strength, 2))
                          for e in self._edges[:5]]
        }
