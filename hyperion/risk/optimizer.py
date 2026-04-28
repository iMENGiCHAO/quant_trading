"""
组合优化器
===========
- Risk Budgeting (风险平价)
- HRP (分层风险平价, Marcos Lopez de Prado)
- Mean-CVaR
- 等权/市值加权基准

融合: v25 portfolio_optimizer + VectorBT组合分析
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """组合优化引擎"""
    
    @staticmethod
    def equal_weight(n_assets: int) -> np.ndarray:
        """等权组合"""
        return np.ones(n_assets) / n_assets
    
    @staticmethod
    def risk_budgeting(cov_matrix: np.ndarray,
                       risk_budgets: Optional[np.ndarray] = None) -> np.ndarray:
        """风险平价/Risk Budgeting
        
        Args:
            cov_matrix: (n, n) 协方差矩阵
            risk_budgets: (n,) 风险预算, None=等风险贡献
            
        Returns:
            (n,) 权重向量
        """
        n = cov_matrix.shape[0]
        if risk_budgets is None:
            risk_budgets = np.ones(n) / n
        
        def _risk_contributions(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            rc = w * marginal_contrib / (port_vol + 1e-12)
            return rc
        
        def _objective(w):
            rc = _risk_contributions(w)
            risk_target = risk_budgets * (w @ cov_matrix @ w) ** 0.5
            return np.sum((rc - risk_target) ** 2)
        
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(_objective, x0, method="SLSQP",
                         bounds=bounds, constraints=constraints,
                         options={"maxiter": 1000, "ftol": 1e-12})
        
        if result.success:
            return result.x
        else:
            logger.warning(f"Risk budgeting failed: {result.message}")
            return x0
    
    @staticmethod
    def hrp(cov_matrix: np.ndarray) -> np.ndarray:
        """Hierarchical Risk Parity (HRP)
        
        Marcos Lopez de Prado 算法
        
        Args:
            cov_matrix: (n, n) 协方差矩阵
            
        Returns:
            (n,) 权重向量
        """
        n = cov_matrix.shape[0]
        
        # 1. 计算相关系数矩阵和距离矩阵
        std = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        dist = np.sqrt(0.5 * (1 - corr))
        # Ensure diagonal is exactly zero for scipy linkage
        np.fill_diagonal(dist, 0.0)
        
        # 2. 层次聚类 (简化版: 使用scipy)
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            # 使用Ward方法
            condensed = squareform(dist)
            link = linkage(condensed, method="ward")
            sort_idx = leaves_list(link)
        except ImportError:
            # Fallback: 按波动率排序
            sort_idx = np.argsort(std)
        
        # 3. 递归二分分配权重
        sorted_cov = cov_matrix[np.ix_(sort_idx, sort_idx)]
        weights = PortfolioOptimizer._hrp_recursive_bisection(sorted_cov)
        
        # 4. 恢复原始顺序
        final_weights = np.zeros(n)
        final_weights[sort_idx] = weights
        
        return final_weights / final_weights.sum()
    
    @staticmethod
    def _hrp_recursive_bisection(cov: np.ndarray) -> np.ndarray:
        """HRP递归二分"""
        n = cov.shape[0]
        
        if n == 1:
            return np.array([1.0])
        if n == 2:
            var = np.diag(cov)
            w = (1.0 / var) / np.sum(1.0 / var)
            return w
        
        # 分裂成两组 (按方差排序)
        var = np.diag(cov)
        split = n // 2
        
        # 两组各自递归
        left_cov = cov[:split, :split]
        right_cov = cov[split:, split:]
        
        left_w = PortfolioOptimizer._hrp_recursive_bisection(left_cov)
        right_w = PortfolioOptimizer._hrp_recursive_bisection(right_cov)
        
        # 计算组间权重 (反比于组方差)
        left_var = left_w @ left_cov @ left_w
        right_var = right_w @ right_cov @ right_w
        
        alpha_left = 1.0 / (left_var + 1e-12)
        alpha_right = 1.0 / (right_var + 1e-12)
        total_alpha = alpha_left + alpha_right
        
        left_scale = alpha_left / total_alpha
        right_scale = alpha_right / total_alpha
        
        return np.concatenate([left_w * left_scale, right_w * right_scale])
    
    @staticmethod
    def mean_variance(returns: pd.DataFrame,
                      target_return: Optional[float] = None) -> np.ndarray:
        """Markowitz均值-方差优化
        
        Args:
            returns: (n_days, n_assets) 收益率矩阵
            target_return: 目标收益率
            
        Returns:
            (n,) 权重
        """
        n = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        
        if target_return is None:
            target_return = np.median(mu)
        
        def _portfolio_variance(w):
            return w @ cov @ w
        
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: w @ mu - target_return}
        ]
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(_portfolio_variance, x0, method="SLSQP",
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    @staticmethod
    def max_sharpe(returns: pd.DataFrame,
                   risk_free_rate: float = 0.03) -> np.ndarray:
        """最大夏普比率组合
        
        Args:
            returns: (n_days, n_assets)
            risk_free_rate: 无风险利率(年化)
            
        Returns:
            (n,) 权重
        """
        n = returns.shape[1]
        mu = returns.mean().values * 252 - risk_free_rate
        cov = returns.cov().values * 252
        
        def _neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -port_return / (port_vol + 1e-12)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(_neg_sharpe, x0, method="SLSQP",
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
