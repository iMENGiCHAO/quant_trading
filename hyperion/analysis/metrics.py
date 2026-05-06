"""
性能指标计算 (Backtrader Analyzer + VectorBT 融合)
======================================================
- 收益: 总收益/年化收益/超额收益
- 风险: 波动率/最大回撤/VaR/CVaR
- 比率: Sharpe/Sortino/Calmar/Omega/Information Ratio
- 交易: 胜率/盈亏比/平均持仓期
- 因子: IC/IR/IC衰减
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PerformanceMetrics:
    """综合性能指标计算器"""
    
    @staticmethod
    def calculate(returns: pd.Series,
                  benchmark_returns: Optional[pd.Series] = None,
                  risk_free_rate: float = 0.03) -> Dict[str, float]:
        """计算全部性能指标
        
        Args:
            returns: 日收益率
            benchmark_returns: 基准日收益率
            risk_free_rate: 年化无风险利率
            
        Returns:
            指标字典
        """
        returns = returns.dropna()
        if len(returns) < 20:
            return {"error": "Insufficient data (min 20 days)"}
        
        metrics = {}
        rf_daily = risk_free_rate / 252
        
        # === 收益指标 ===
        cumulative = (1 + returns).cumprod()
        metrics["total_return"] = float(cumulative.iloc[-1] - 1)
        n_years = len(returns) / 252
        metrics["annual_return"] = float((1 + metrics["total_return"]) ** (1 / max(0.01, n_years)) - 1)
        metrics["daily_mean"] = float(returns.mean())
        metrics["daily_std"] = float(returns.std())
        
        # === 风险指标 ===
        metrics["volatility"] = float(returns.std() * np.sqrt(252))
        metrics["max_drawdown"] = float(PerformanceMetrics._max_drawdown(returns))
        
        # VaR & CVaR
        metrics["var_95"] = float(returns.quantile(0.05))
        metrics["cvar_95"] = float(returns[returns <= metrics["var_95"]].mean())
        
        # === 比率 ===
        excess = returns - rf_daily
        metrics["sharpe_ratio"] = float(excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        metrics["sortino_ratio"] = float(excess.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        metrics["calmar_ratio"] = float(metrics["annual_return"] / abs(metrics["max_drawdown"])) if metrics["max_drawdown"] != 0 else 0
        
        # Omega ratio
        threshold = rf_daily
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        metrics["omega_ratio"] = float(gains / losses) if losses > 0 else float("inf")
        
        # === 基准对比 ===
        if benchmark_returns is not None:
            bench = benchmark_returns.dropna()
            common = returns.index.intersection(bench.index)
            if len(common) > 20:
                r = returns.loc[common]
                b = bench.loc[common]
                metrics["benchmark_return"] = float((1 + b).cumprod().iloc[-1] - 1)
                metrics["tracking_error"] = float((r - b).std() * np.sqrt(252))
                metrics["information_ratio"] = float((r.mean() - b.mean()) / (r - b).std() * np.sqrt(252)) if (r - b).std() > 0 else 0
                
                # Alpha & Beta
                cov = np.cov(r, b)
                metrics["beta"] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0
                metrics["alpha"] = float(metrics["annual_return"] - risk_free_rate - metrics["beta"] * (metrics["benchmark_return"] - risk_free_rate))
        
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    
    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    @staticmethod
    def ic_analysis(predictions: pd.Series, returns: pd.Series) -> Dict:
        """IC/IR分析 (Qlib style)
        
        Args:
            predictions: 预测分数
            returns: 实际收益
            
        Returns:
            ic_mean, ic_std, ir, rank_ic
        """
        aligned = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
        if len(aligned) < 20:
            return {}
        
        # Pearson IC
        ic = aligned["pred"].corr(aligned["ret"])
        
        # Rank IC
        rank_ic = aligned["pred"].rank().corr(aligned["ret"].rank())
        
        return {
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "ir": round(ic / aligned["pred"].std() if aligned["pred"].std() > 0 else 0, 4),
            "n_samples": len(aligned)
        }
